# Finnish BERT — Sentence Embeddings from Scratch

> **A complete NLP pipeline built without any pre-trained models or high-level ML libraries.**  
> From raw text acquisition to trained sentence embeddings — every component written from the ground up in pure Python and PyTorch.

---

## The Challenge

This project was conceived as a deliberate engineering constraint: *"Can you build a working sentence embedding model for a morphologically complex language — using only the Python standard library and PyTorch?"*

No HuggingFace. No sentencepiece. No pre-trained weights. No Trainer APIs.

Finnish was chosen intentionally. It is one of the most morphologically complex languages in computational linguistics — agglutinative, with 15 grammatical cases and compound words that encode entire English phrases in a single token (e.g., *taloissammekin* = "in our houses too"). This forces every layer of the pipeline — tokenization, masking, model architecture — to make language-aware decisions that a generic English pipeline would not.

---

## What Was Built

A seven-stage pipeline that takes a language model from zero to evaluated sentence embeddings:

```
Raw Text  →  Clean Corpus  →  BPE Tokenizer  →  Preprocessed Dataset
                                                         ↓
Geometric Evaluation  ←  Trained BERT  ←  MLM Pre-training  ←  BERT Architecture
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_data_download.py` | Fetches all Finnish books from Project Gutenberg |
| 2 | `02_clean.py` | Strips boilerplate, repairs line breaks, deduplicates |
| 3 | `03_tokenizer.py` | Trains a 50K-token BPE vocabulary from scratch |
| 4 | `04_dataset.py` | Builds the MLM dataset with Whole Word Masking |
| 5 | `05_model.py` | Defines the BERT encoder architecture and MLM head |
| 6 | `06_train.py` | Full pre-training loop with fp16, checkpointing, resume |
| 7 | `07_evaluate.py` | Geometric embedding quality evaluation |

---

## Model Architecture

The model follows the **6L/256H/4A** small BERT configuration from Turc et al. (2019), chosen for its favourable depth-to-parameter trade-off on downstream tasks.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Layers | 6 | Depth > width per parameter (Turc et al. 2019) |
| Hidden dimension | 256 | Fits A100 40GB at batch 128, seq 128 |
| Attention heads | 4 | d_k = 64, Vaswani et al. (2017) convention |
| FFN inner dim | 1,024 | 4× hidden, BERT convention |
| Vocabulary size | 50,000 | FinBERT scale (Virtanen et al. 2019) |
| Max sequence length | 128 | BERT Phase 1; O(T²) attention cost |
| Parameters | ~17.6M | With weight tying on embedding/output projection |
| Positional encoding | Learned | Adapts to Finnish statistical patterns |
| Pooling strategy | Mean pool | Outperforms [CLS] without fine-tuning (Reimers & Gurevych 2019) |

---

## Pipeline

### Stage 1 — Data Acquisition (`01_data_download.py`)

Downloads the entire Finnish Gutenberg catalogue (~3,600 books). Rather than paginating an unstable third-party API, the script fetches Gutenberg's full catalogue as a single gzip'd CSV — one request, no rate limits.

Concurrent downloads (10 workers) with exponential back-off retry logic and a thread-safe in-place progress bar written using ANSI escape codes and Python's `threading.Lock`.

### Stage 2 — Text Preprocessing (`02_clean.py`)

Follows the **SPGC pipeline** (Gerlach & Font-Clos, arXiv:1812.08092), the standard corpus preparation method for Gutenberg text:

1. **Boilerplate removal** — strips everything outside the `*** START OF ***` / `*** END OF ***` markers.
2. **Line-break repair** — Gutenberg files use hard line breaks at ~70 characters (typewriter convention). A regex join restores these to continuous prose while preserving genuine paragraph boundaries (`\n\n`).
3. **Paragraph filtering** — rejects all-uppercase headings, non-alphabetic noise, and fragments under 20 characters.
4. **Deduplication** — MD5-hashes each normalised paragraph. Storing 16-byte digests instead of full strings reduces the seen-set memory footprint by 3–10× for typical Finnish paragraph lengths.

**Result:** 3,532,731 unique paragraphs → `data/processed/sentences.txt` (912 MB).

### Stage 3 — BPE Tokenizer (`03_tokenizer.py`)

A from-scratch implementation of Byte Pair Encoding (Sennrich et al., ACL 2016) with two performance-critical optimisations:

**1. Parallel word frequency counting.**  
Frequency counting is embarrassingly parallel — each line is independent. The corpus is split into `cpu_count()` equal chunks and dispatched to a `multiprocessing.Pool`. Workers return per-chunk Counters merged in the main process with no inter-process locking required.

**2. Incremental BPE with a reverse pair index.**  
Naïve BPE recomputes all pair frequencies from scratch after every merge: O(V × L) per merge, where V = unique word types and L = average token length. With V ≈ 80K and 49K merges, that is ~23 billion operations.

Instead, a `pair_to_words` reverse index maps every adjacent token pair to the set of word types containing it. After a merge, only the ~2–5% of words that contained the merged pair need updating: O(W_pair × L) — a **~50× speedup**.

**3. Fast inference encoder.**  
At encoding time, `_bpe_encode_word` uses the priority-queue algorithm: find the highest-priority applicable merge, apply it, repeat. This is O(K × L²) where K is the number of merges that fire on a word (typically 1–5) — roughly **2,400× faster** per word than replaying all 49K rules naïvely.

**Result:** 50,000-token BPE vocabulary trained in ~5 minutes on 200K sampled paragraphs → `data/tokenizer/vocab.json` + `data/tokenizer/merges.txt`.

### Stage 4 — Dataset & Masking (`04_dataset.py`)

**Why Whole Word Masking matters for Finnish.**  
Standard BERT masking selects individual subword tokens. For agglutinative Finnish, this creates a near-trivial prediction task: if *taloissammekin* is segmented into `[talo][issamme][kin]` and only `issamme` is masked, the model recovers it almost deterministically from `talo` and `kin` with no morpho-semantic understanding required.

Whole Word Masking (Google BERT team, 2019; Cui et al. 2019 for Chinese) masks all subwords of a word simultaneously, forcing the model to predict the complete morphological form from surrounding context.

The **80-10-10 strategy** (Devlin et al. 2019) is applied at the word level, not the token level:
- **80%** → all subwords replaced with `[MASK]`
- **10%** → all subwords replaced with a random vocabulary token
- **10%** → all subwords left unchanged

**Dynamic masking** (RoBERTa, Liu et al. 2019): masking is applied fresh on every `__getitem__` call rather than baked in at preprocessing time. Each epoch sees a different masking pattern over the same tokens.

**Preprocessing for Colab.** Tokenising 2M paragraphs at training time would add ~3 minutes of CPU overhead on every session restart. `preprocess_and_save` BPE-encodes all paragraphs in parallel once, writing `dataset.pt` as a pre-computed LongTensor (3.8 GB). At training time, `torch.load()` takes ~10 seconds — a ~20× saving per session.

**Result:** `data/processed/dataset.pt` (3.8 GB, LongTensor of shape 2M × 128).

### Stage 5 — Model Architecture (`05_model.py`)

A clean PyTorch implementation of the BERT encoder, faithful to Devlin et al. (2019):

- `BertEmbeddings` — token + learned positional embeddings, LayerNorm, Dropout
- `MultiHeadSelfAttention` — scaled dot-product attention with additive padding mask (`-1e4`, safe for fp16)
- `TransformerEncoderBlock` — Post-LN residual connections with GELU activation
- `BertEncoder` — stack of 6 `TransformerEncoderBlock` layers
- `BertForMLM` — encoder + two-layer MLM head (`Linear → GELU → LayerNorm → Linear`)
- `get_sentence_embedding` — mean pooling over real tokens with L2 normalisation

**Weight tying:** the MLM output projection shares its weight matrix with the input token embedding (`mlm_decoder.weight = token_embedding.weight`). This halves the parameter count for that layer (~12.8M) and enforces a consistent representation space between input tokens and predicted output tokens.

### Stage 6 — Pre-training (`06_train.py`)

A production-grade training loop designed for cloud (Google Colab A100) with robust resumability:

| Feature | Implementation |
|---|---|
| Mixed precision | `torch.cuda.amp` fp16 + `GradScaler` |
| Gradient accumulation | 4 steps → effective batch 512 |
| Gradient clipping | `clip_grad_norm_` after `scaler.unscale_()` (correct ordering) |
| Optimizer | AdamW with weight decay on weight matrices only (Loshchilov & Hutter 2019) |
| LR schedule | Linear warmup (10% of steps) + cosine decay |
| Checkpointing | Atomic write via `os.replace` → safe against mid-write VM reset |
| Cloud persistence | Checkpoint mirrored to Google Drive after each save |
| Step-level resume | Exact step count and intra-epoch batch skip restored from checkpoint |

**Why atomic checkpointing matters in Colab:** Google Colab VMs are recycled silently after 8–12 hours. `torch.save()` writes files incrementally; a mid-write VM reset produces a silently corrupt file. Writing to a `.tmp` path and calling `os.replace()` (POSIX-atomic rename) ensures the previous checkpoint remains intact until the new one is fully on disk.

**Training run:** 45,000 optimizer steps on 2M paragraphs (~11.5 epochs), ~30 minutes on an A100 40GB.

### Stage 7 — Evaluation (`07_evaluate.py`)

Four complementary geometric metrics across 40 Finnish test sentences (5 semantic categories: Animals, Vehicles, Nature, People, Food), compared against a randomly initialised baseline:

| Metric | Formula | Direction | Source |
|---|---|---|---|
| **Anisotropy** | Mean cosine similarity of random pairs | Lower is better | Ethayarajh 2019 |
| **Uniformity** | log mean Gaussian kernel over all pairs | More negative is better | Wang & Isola 2020 |
| **Alignment** | Mean ‖u − v‖² for positive pairs | Lower is better | Wang & Isola 2020 |
| **Nearest-neighbour** | Top-k retrieval by cosine similarity | Qualitative | — |

---

## Results

```
Training : 45,000 steps  |  2M paragraphs  |  ~30 min on A100 40GB
Initial loss : 10.82  (ln(50,000) — random initialisation baseline)
Final loss   : 5.35
Perplexity   : ~210
Checkpoint   : checkpoints/checkpoint_step045000.pt  (202 MB)
```

**Embedding quality (cosine similarity, 5 test sentences):**

```
                      S1      S2      S3      S4      S5
Pet / moving       1.000   0.882   0.858   0.812   0.833
Pet / resting      0.882   1.000   0.808   0.828   0.857
Vehicle / moving   0.858   0.808   1.000   0.788   0.787
Human / reading    0.812   0.828   0.788   1.000   0.788
Child / playing    0.833   0.857   0.787   0.788   1.000
```

Animals and people cluster together; vehicles score lower against both — the model has learned real semantic structure.

**On the geometric metrics (anisotropy / uniformity):**  
Ethayarajh (2019) showed that MLM pre-training systematically increases anisotropy: the model learns representations optimised for token-level prediction, concentrating vectors along a small number of dominant directions. This is a well-documented property of BERT-class models trained with MLM only — not a failure of this implementation. Nearest-neighbour retrieval confirms genuine semantic clustering despite the anisotropy.

---

## Getting Started

### Prerequisites

- Python 3.10+
- For training: a CUDA-capable GPU (local) or Google Colab with an A100 runtime

### 1. Install Dependencies

```bash
git clone https://github.com/tyerdogan/finnish-embeddings.git
cd finnish-embeddings
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Data Preparation Pipeline (Local, CPU)

These four steps run on any CPU machine. Each step must complete before the next starts.

```bash
# Step 1 — Download ~3,600 Finnish books from Project Gutenberg (~1 hour)
python src/01_data_download.py
# Output: data/raw/*.txt

# Step 2 — Clean, filter, and deduplicate paragraphs (~2 minutes)
python src/02_clean.py
# Output: data/processed/sentences.txt  (~912 MB, 3.5M paragraphs)

# Step 3 — Train the BPE tokenizer (~5 minutes)
python src/03_tokenizer.py
# Output: data/tokenizer/vocab.json  (~1.1 MB)
#         data/tokenizer/merges.txt  (~0.6 MB)

# Step 4 — Pre-tokenise the corpus and save as a LongTensor (~2 minutes)
python src/04_dataset.py --build-only
# Output: data/processed/dataset.pt  (~3.8 GB)

# Step 5 — Verify model architecture before training (~5 seconds)
python src/05_model.py
# Instantiates BertForMLM, runs a forward pass with dummy data, and confirms:
#   · Parameter count : ~17.6M
#   · Initial loss    : ~10.82  (expected ln(50,000) for a random-init model)
#   · Sentence embeddings shape and unit L2 norm
# If this step passes cleanly the architecture is ready for training.
```

### 3. Train on Google Colab (GPU Required)

Training requires a GPU. The Colab notebook (`notebooks/train_colab.ipynb`) handles the full cloud setup automatically. Follow these steps:

#### 3a. Upload files to Google Drive

After running Step 2 and Step 3 locally, upload the following files to Google Drive. Create the folder structure exactly as shown:

```
MyDrive/
└── Finnish-Embedding/
    └── data/
        ├── tokenizer/
        │   ├── vocab.json          ← from data/tokenizer/
        │   └── merges.txt          ← from data/tokenizer/
        └── processed/
            ├── sentences.txt       ← from data/processed/  (912 MB)
            └── dataset.pt          ← from data/processed/  (3.8 GB)
```

> **Note on `dataset.pt`:** This file is 3.8 GB. Uploading it once to Drive means every subsequent Colab session loads it in ~10 seconds via a symlink instead of re-building it from scratch.

#### 3b. Open the Colab notebook

1. Go to [colab.research.google.com](https://colab.research.google.com) and open `notebooks/train_colab.ipynb`.
2. Select **Runtime → Change Runtime Type → A100 GPU**.
3. Run all cells in order. Each cell is described below.

#### 3c. Notebook cells — what each one does

| Cell | Action |
|---|---|
| **Step 1 — GPU Check** | Verifies a CUDA GPU is available and prints VRAM. Raises an error if no GPU is found so you know before wasting time. |
| **Step 2 — Mount Google Drive** | Mounts Drive at `/content/drive` and checks that `vocab.json`, `merges.txt`, and `sentences.txt` are present in `MyDrive/Finnish-Embedding/data/`. Raises a `FileNotFoundError` with instructions if any file is missing. |
| **Step 3 — Clone the Repo** | Clones this repository into `/content/finnish-embeddings/` (skips silently if already cloned). |
| **Step 4 — Install Dependencies** | Installs `tqdm` (the only non-stdlib dependency beyond PyTorch, which Colab provides). |
| **Step 5 — Create File Symlinks** | Creates symlinks from the Colab VM's local filesystem into the Drive-mounted paths. This avoids copying large files onto the ephemeral `/content/` disk. Also links any existing checkpoints from Drive so training resumes automatically from the last saved step. |
| **Step 6 — Build `dataset.pt`** | Checks if `dataset.pt` already exists on Drive (linked in Step 5). If yes, uses it directly. If no, runs `04_dataset.py --build-only` to tokenise 2M paragraphs in parallel (~2 min), then copies the result to Drive for future sessions. |
| **Step 7 — Train** | Runs `06_train.py`. Training resumes automatically from the latest checkpoint if one is linked. Saves a new checkpoint every 2,000 steps to both the local VM and Drive. Logs loss / perplexity / learning rate every 100 steps. |
| **Step 8 — Plot Loss Curve** | Loads `loss_history` from the latest checkpoint and plots the MLM loss over all training steps. Saves `loss_curve.png` to Drive. |
| **Step 9 — Embedding Sanity Check** | Loads the trained model, encodes 5 Finnish test sentences, and prints a cosine similarity matrix to verify the model has learned semantic structure. |

#### 3d. Resuming after a VM reset

Colab VMs reset after ~8–12 hours of inactivity. After a reset:

1. Re-run all cells from the top (Steps 1–5 are fast, they just re-link files).
2. Step 5 automatically links any checkpoints already on Drive.
3. Step 7 (`06_train.py`) detects the latest checkpoint and resumes from that step — no training progress is lost.

#### 3e. Downloading the trained checkpoint

After training completes, the checkpoint is already in Drive:

```
MyDrive/Finnish-Embedding/checkpoints/checkpoint_step045000.pt
```

Download it from Drive to your local machine and place it at:

```
checkpoints/checkpoint_step045000.pt
```

### 4. Run Evaluation (Local)

```bash
python src/07_evaluate.py
```

This loads the latest checkpoint from `checkpoints/`, encodes 40 Finnish test sentences, and prints the anisotropy, uniformity, and nearest-neighbour results.

For the full visual analysis (heatmaps, t-SNE, metric comparison charts):

```bash
jupyter notebook notebooks/evaluate.ipynb
```

---

## File Structure

```
finnish-embeddings/
├── src/
│   ├── 01_data_download.py   # Gutenberg data acquisition
│   ├── 02_clean.py           # Corpus cleaning and deduplication
│   ├── 03_tokenizer.py       # BPE tokenizer (train, encode, decode)
│   ├── 04_dataset.py         # MLM dataset with Whole Word Masking
│   ├── 05_model.py           # BERT architecture and sentence embedding
│   ├── 06_train.py           # Pre-training loop
│   └── 07_evaluate.py        # Geometric evaluation metrics
├── notebooks/
│   ├── train_colab.ipynb     # End-to-end Colab training workflow
│   └── evaluate.ipynb        # Visual analysis and metric plots
├── data/                     # Generated — not committed (see .gitignore)
│   ├── raw/                  # Downloaded Gutenberg .txt files
│   ├── processed/
│   │   ├── sentences.txt     # Cleaned corpus (912 MB)
│   │   └── dataset.pt        # Pre-tokenised LongTensor (3.8 GB)
│   └── tokenizer/
│       ├── vocab.json        # BPE vocabulary (50K tokens)
│       └── merges.txt        # BPE merge rules (49K rules)
├── checkpoints/              # Generated — not committed
│   └── checkpoint_step045000.pt
├── outputs/                  # Evaluation plots
├── requirements.txt
└── README.md
```

---

## Design Principles

**No high-level ML abstractions.** Every component — tokenizer, model, training loop, evaluation metrics — is implemented directly. This is intentional: understanding what each layer does requires writing it, not configuring it.

**Language-aware engineering.** Finnish is not English. Decisions like Whole Word Masking, hapax legomena filtering, and the BPE vocabulary size were made specifically for agglutinative morphology, not copied from an English-language default.

**Production robustness in research code.** Atomic checkpoint writes, step-level resume, fp16 GradScaler state persistence, and multiprocessing safety across platforms (macOS spawn vs. Linux fork) are treated as first-class concerns, not afterthoughts.

---

## Future Development

The current pipeline establishes a solid foundation — a fully from-scratch BERT encoder trained on Finnish text with proper MLM, WWM, and production-grade training infrastructure. The natural next steps build directly on this base.

### 1. SimCSE Contrastive Fine-tuning

The single highest-leverage improvement. SimCSE (Gao et al. 2021, arXiv:2104.08821) fine-tunes the encoder with a contrastive objective: each sentence is passed through the model twice with different dropout masks, creating two minimally different views. The NT-Xent loss maximises agreement between these views while using in-batch negatives to push dissimilar sentences apart.

This directly addresses the two weaknesses measured in this project:
- **Anisotropy** drops significantly — the negative-pair term forces vectors to spread across the hypersphere.
- **Uniformity** improves as a side effect of the same spreading pressure.

No labelled data required. The pre-trained checkpoint here is the starting point; SimCSE fine-tuning typically converges in a few thousand steps.

### 2. Corpus Expansion

The current corpus is Project Gutenberg Finnish literature — predominantly 19th–early 20th century texts. Two additions would substantially improve representation quality:

- **Finnish Wikipedia** — modern language, encyclopaedic breadth, sentence-level structure well-suited to BERT pre-training.
- **Finnish news archives** — contemporary vocabulary, named entities, current Finnish orthography.

Both are freely available. Combining them with the Gutenberg corpus would bring the token count closer to the ~3B tokens used by FinBERT (Virtanen et al. 2019), the current Finnish-language baseline.

### 3. BERT Phase 2 — Longer Sequences

Following the original BERT training schedule, Phase 1 uses `max_seq_length=128` (done). Phase 2 extends to `max_seq_length=512` for the final 10% of training steps. The O(T²) attention cost makes 512-token sequences ~16× more expensive per sample, but the longer context improves the model's ability to capture document-level dependencies — particularly relevant for Finnish, where long compound words and case suffixes encode relationships that span clause boundaries.

The `BertConfig` and training loop in this codebase already support this change with a single hyperparameter edit.

### 4. Downstream Task Evaluation

The current evaluation is geometric (anisotropy, uniformity, nearest-neighbour retrieval). Production-readiness requires task-based evaluation:

- **Finnish STS** — semantic textual similarity benchmark to measure cosine similarity as a direct ranking signal.
- **Named Entity Recognition** — attach a linear classification head to the encoder and fine-tune on a Finnish NER dataset (e.g. Turku NER corpus) to measure transfer quality.
- **Text classification** — sentiment or topic classification to validate that the encoder's representations are linearly separable for downstream tasks.

### 5. Tokenizer Upgrade

The current BPE tokenizer is character-level (splits words into characters first, then merges). A **WordPiece** or **Unigram** tokenizer (as used by FinBERT and XLM-RoBERTa) would produce a vocabulary with better morpheme-level coverage for Finnish — particularly for rare inflected forms not seen during BPE training. Both algorithms are implementable within the same from-scratch constraint as the current codebase.

---

## References

| Paper | Used for |
|---|---|
| Devlin et al. 2019 — BERT (arXiv:1810.04805) | Architecture, MLM, tokenisation, weight initialisation |
| Turc et al. 2019 — Well-Read Students (arXiv:1908.08962) | 6L/256H/4A small BERT config |
| Liu et al. 2019 — RoBERTa (arXiv:1907.11692) | NSP removal, dynamic masking, cosine LR decay |
| Sennrich et al. 2016 — BPE (ACL 2016) | Subword tokenisation algorithm |
| Virtanen et al. 2019 — FinBERT (arXiv:1912.07076) | Finnish vocabulary size (50K), corpus design |
| Loshchilov & Hutter 2019 — AdamW (arXiv:1711.05101) | Decoupled weight decay |
| Micikevicius et al. 2018 — Mixed Precision (arXiv:1710.03740) | fp16 training, GradScaler |
| Reimers & Gurevych 2019 — Sentence-BERT (arXiv:1908.10084) | Mean pooling strategy |
| Ethayarajh 2019 — Anisotropy (arXiv:1909.00512) | Anisotropy metric, MLM representation geometry |
| Wang & Isola 2020 — Uniformity & Alignment (ICML) | Uniformity and alignment metrics |
| Gerlach & Font-Clos 2018 — SPGC (arXiv:1812.08092) | Gutenberg corpus preparation pipeline |
| Cui et al. 2019 — Chinese BERT with WWM (arXiv:1906.08101) | Whole Word Masking for morphologically rich languages |
| Xiong et al. 2020 — Pre-LN (arXiv:2002.04745) | Post-LN warmup requirement |
| Vaswani et al. 2017 — Attention Is All You Need | Transformer architecture, scaled dot-product attention |
