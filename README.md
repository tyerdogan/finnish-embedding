# Finnish BERT — Sentence Embeddings from Scratch

> **A complete, NLP pipeline built without any pre-trained models or high-level ML libraries.**  
> From raw text acquisition to trained sentence embeddings — every component written from the ground up in pure Python and PyTorch.

---

## The Challenge

This project was conceived as a deliberate engineering constraint: *"Can you build a working sentence embedding model for a morphologically complex language — using only the Python standard library and PyTorch?"*

No HuggingFace. No sentencepiece. No pre-trained weights. No Trainer APIs.

Finnish was chosen intentionally. It is one of the most morphologically complex languages in computational linguistics — agglutinative, with 15 grammatical cases and compound words that encode entire English phrases in a single token (e.g., *taloissammekin* = "not even in our houses"). This forces every layer of the pipeline — tokenization, masking, model architecture — to make language-aware decisions that a generic English pipeline would not.

---

## What Was Built

A seven-stage pipeline that takes a language model from zero to evaluated sentence embeddings:

```
Raw Text  →  Clean Corpus  →  BPE Tokenizer  →  Preprocessed Dataset
    ↓
Trained BERT  ←  MLM Pre-training  ←  BERT Architecture
    ↓
Geometric Evaluation (Anisotropy · Uniformity · Alignment · Nearest-Neighbour)
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_data_download.py` | Fetches all Finnish books from Project Gutenberg |
| 2 | `02_clean.py` | Strips boilerplate, repairs line breaks, deduplicates |
| 3 | `03_tokenizer.py` | Trains a 50K-token BPE vocabulary from scratch |
| 4 | `04_dataset.py` | Builds the MLM dataset with Whole Word Masking |
| 5 | `05_model.py` | Defines the BERT encoder and MLM head |
| 6 | `06_train.py` | Full pre-training loop with checkpointing |
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

## Pipeline Deep Dive

### Stage 1 — Data Acquisition (`01_data_download.py`)

Downloads the entire Finnish Gutenberg catalogue (~3,600 books). Rather than paginating an unstable third-party API, the script fetches Gutenberg's full catalogue as a single gzip'd CSV — one request, no rate limits.

Concurrent downloads (10 workers) with exponential back-off retry logic and a thread-safe in-place progress bar written using ANSI escape codes and Python's `threading.Lock`.

### Stage 2 — Text Preprocessing (`02_clean.py`)

Follows the **SPGC pipeline** (Gerlach & Font-Clos, arXiv:1812.08092), the standard corpus preparation method for Gutenberg text:

1. **Boilerplate removal** — strips everything outside the `*** START OF ***` / `*** END OF ***` markers.
2. **Line-break repair** — Gutenberg files use hard line breaks at ~70 characters (typewriter convention). A regex join restores these to continuous prose while preserving genuine paragraph boundaries (`\n\n`).
3. **Paragraph filtering** — rejects all-uppercase headings, non-alphabetic noise, and fragments under 20 characters.
4. **Deduplication** — MD5-hashes each normalised paragraph. Storing 16-byte digests instead of full strings reduces the seen-set memory footprint by 3–10× for typical Finnish paragraph lengths.

**Result:** 3,532,731 unique paragraphs.

### Stage 3 — BPE Tokenizer (`03_tokenizer.py`)

A from-scratch implementation of Byte Pair Encoding (Sennrich et al., ACL 2016) with two performance-critical optimisations:

**1. Parallel word frequency counting.**  
Frequency counting is embarrassingly parallel — each line is independent. The corpus is split into `cpu_count()` equal chunks and dispatched to a `multiprocessing.Pool`. Workers return per-chunk Counters merged in the main process. No inter-process locking required.

**2. Incremental BPE with a reverse pair index.**  
Naïve BPE recomputes all pair frequencies from scratch after every merge: O(V × L) per merge where V = unique word types and L = average token length. With V ≈ 80K and 49K merges, that is ~23 billion operations.

Instead, a `pair_to_words` reverse index maps every adjacent token pair to the set of word types that contain it. After a merge, only the ~2–5% of words that contained the merged pair need updating: O(W_pair × L) — a **~50× speedup**.

**3. Fast inference encoder.**  
At encoding time, `_bpe_encode_word` applies the priority-queue algorithm: find the highest-priority applicable merge, apply it, repeat. This is O(K × L²) where K is the number of merges that actually fire on a given word (typically 1–5) — roughly **2,400× faster** per word than replaying all 49K rules naïvely.

**Result:** 50,000-token BPE vocabulary trained in ~5 minutes on 200K sampled paragraphs.

### Stage 4 — Dataset & Masking (`04_dataset.py`)

**Why Whole Word Masking matters for Finnish.**  
Standard BERT masking selects individual subword tokens. For agglutinative Finnish, this creates a near-trivial prediction task: if *taloissammekin* is segmented into `[talo][issamme][kin]` and only `issamme` is masked, the model recovers it almost deterministically from `talo` and `kin` with no morpho-semantic understanding required.

Whole Word Masking (Google BERT team, 2019; Cui et al. 2019 for Chinese) masks all subwords of a word simultaneously, forcing the model to predict the complete morphological form from surrounding context. For Finnish — with 15+ grammatical forms per word — this is essential.

The **80-10-10 strategy** (Devlin et al. 2019) is applied at the word level, not the token level:
- **80%** → all subwords replaced with `[MASK]`
- **10%** → all subwords replaced with a random vocabulary token
- **10%** → all subwords left unchanged

**Dynamic masking** (RoBERTa, Liu et al. 2019): masking is applied fresh on every `__getitem__` call rather than baked in at preprocessing time. Each epoch sees a different masking pattern over the same tokens — multiplying effective data diversity without growing the dataset.

**Preprocessing for Colab.** Tokenising 2M paragraphs at training time adds ~3 minutes of CPU overhead on every session restart. `preprocess_and_save` encodes all paragraphs in parallel once, writing `dataset.pt` as a pre-computed LongTensor (3.8 GB). At training time, `torch.load()` takes ~10 seconds.

### Stage 5 — Model Architecture (`05_model.py`)

A clean PyTorch implementation of the BERT encoder, faithful to Devlin et al. (2019):

- `BertEmbeddings` — token + learned positional embeddings, LayerNorm, Dropout
- `MultiHeadSelfAttention` — scaled dot-product attention with additive padding mask (`-1e4`, safe for fp16)
- `TransformerEncoderBlock` — Post-LN residual connections with GELU activation
- `BertForMLM` — encoder + two-layer MLM head (`Linear → GELU → LayerNorm → Linear`)
- `get_sentence_embedding` — mean pooling over real tokens with L2 normalisation

**Weight tying:** the MLM output projection shares its weight matrix with the input token embedding (`mlm_decoder.weight = token_embedding.weight`). This halves the parameter count for that layer (~12.8M), reduces overfitting on the large vocabulary projection, and enforces a consistent representation space between input tokens and predicted output tokens.

### Stage 6 — Pre-training (`06_train.py`)

A production-grade training loop designed for cloud (Google Colab A100) with robust resumability:

| Feature | Implementation |
|---|---|
| Mixed precision | `torch.cuda.amp` fp16 + `GradScaler` |
| Gradient accumulation | 4 steps → effective batch 512 |
| Gradient clipping | `clip_grad_norm_` after `scaler.unscale_()` (correct order) |
| Optimizer | AdamW with weight decay on weight matrices only (Loshchilov & Hutter 2019) |
| LR schedule | Linear warmup (10% of steps) + cosine decay |
| Checkpointing | Atomic write via `os.replace` → safe against mid-write VM reset |
| Cloud persistence | Checkpoint mirrored to Google Drive after each save |
| Step-level resume | Exact step count and intra-epoch batch skip restored from checkpoint |

**Why atomic checkpointing matters in Colab:** Google Colab VMs are recycled silently after 8–12 hours of inactivity. `torch.save()` writes files incrementally; a mid-write VM reset produces a silently corrupt file. Writing to a `.tmp` path and calling `os.replace()` (POSIX-atomic rename) ensures the previous checkpoint remains intact until the new one is fully on disk.

**Training run:** 45,000 optimizer steps on 2M paragraphs (~11.5 epochs), ~30 minutes on an A100 40GB.

### Stage 7 — Evaluation (`07_evaluate.py`)

Four complementary geometric metrics across 40 Finnish test sentences (5 semantic categories: Animals, Vehicles, Nature, People, Food), compared against a randomly initialised baseline:

| Metric | Formula | Direction | Source |
|---|---|---|---|
| **Anisotropy** | Mean cosine similarity of random pairs | Lower = less cone-shaped = better | Ethayarajh 2019 |
| **Uniformity** | log mean Gaussian kernel over all pairs | More negative = more spread = better | Wang & Isola 2020 |
| **Alignment** | Mean ‖u − v‖² for positive pairs | Lower = more stable = better | Wang & Isola 2020 |
| **Nearest-neighbour** | Top-k retrieval by cosine similarity | Qualitative | — |

**Results after 45K steps:**

| Metric | Random Init | Trained | Note |
|---|---|---|---|
| Anisotropy | 0.63 | 0.79 | ↑ expected for MLM-only BERT |
| Uniformity | −1.46 | −0.82 | ↑ expected for MLM-only BERT |
| Intra/inter-category ratio | ~1.0 | ~1.05 | Weak but real semantic clustering |
| Nearest-neighbour | random | mostly same-category | Qualitative improvement |

**Why the metrics "worsen" — and why that is expected.**  
Ethayarajh (2019) showed that MLM pre-training systematically increases anisotropy: the model learns to concentrate representations along a small number of dominant directions that are useful for token-level prediction, not for sentence-level similarity. This is a well-documented property of BERT-class models and is not a failure of this implementation.

The trained model *does* learn real semantic structure — nearest-neighbour retrieval returns same-category results — but the representation space is not yet optimised for cosine similarity as a sentence similarity metric. The natural next step is SimCSE-style contrastive fine-tuning, which directly optimises the alignment/uniformity trade-off.

---

## Results Summary

```
Training: 45,000 steps  |  2M paragraphs  |  ~30 min on A100 40GB
Initial loss : 10.82  (ln(50,000) — random initialisation baseline)
Final loss   : 5.35
Perplexity   : ~210
Checkpoint   : checkpoints/checkpoint_step045000.pt  (202 MB)
```

---

## Getting Started

### Requirements

```bash
git clone https://github.com/tyerdogan/finnish-embeddings.git
cd finnish-embeddings
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Local Pipeline (Data Prep + Tokenizer)

Run the scripts in order. Steps 1–4 run comfortably on a CPU-only machine.

```bash
python src/01_data_download.py   # ~1 h — downloads ~3,600 Finnish books
python src/02_clean.py           # ~2 min — produces data/processed/sentences.txt
python src/03_tokenizer.py       # ~5 min — trains BPE, writes data/tokenizer/
python src/04_dataset.py         # ~2 min — builds data/processed/dataset.pt
python src/07_evaluate.py        # evaluate embedding quality
```

### Cloud Training (Google Colab + A100)

Steps 1–4 are run locally (CPU). Training runs on Colab.

1. Run steps 1–4 locally.
2. Upload `data/tokenizer/` and `data/processed/dataset.pt` to Google Drive under `Finnish-Embedding/data/`.
3. Open `notebooks/train_colab.ipynb` in Colab and select an A100 runtime.
4. Mount Drive and run all cells. Checkpoints sync to Drive every 2,000 steps and survive VM resets.

### Evaluation Notebook

`notebooks/evaluate.ipynb` provides visual analysis:
- Cosine similarity heatmap across semantic categories
- t-SNE projection of the 40 test sentence embeddings
- Anisotropy and uniformity comparison (trained vs. random baseline)
- Per-category nearest-neighbour retrieval

---

## Design Principles

**No high-level ML abstractions.** Every component — tokenizer, model, training loop, evaluation metrics — is implemented directly. This is intentional: understanding what each layer does requires writing it, not configuring it.

**Language-aware engineering.** Finnish is not English. Decisions like Whole Word Masking, hapax legomena filtering, and the BPE vocabulary size were made specifically for agglutinative morphology, not copied from an English-language default.

**Production robustness in research code.** Atomic checkpoint writes, step-level resume, fp16 GradScaler state persistence, and multiprocessing safety across platforms (macOS spawn vs. Linux fork) are treated as first-class concerns, not afterthoughts.

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
