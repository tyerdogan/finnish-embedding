# Finnish BERT: From-Scratch Sentence Embeddings

A complete, independent pipeline for training Finnish language embeddings from the ground up. This project implements a BERT-style architecture without relying on high-level libraries like HuggingFace or Tokenizers, focusing on raw PyTorch implementation, custom subword algorithms, and fundamental NLP principles.

## Project Philosophy

This repository demonstrates a "post-apocalyptic" NLP approach: building a robust embedding model using only the Python standard library and PyTorch. It covers the entire lifecycle of a language model, from automated data acquisition and custom tokenizer training to Masked Language Modeling (MLM) with Whole Word Masking (WWM) and advanced geometric evaluation.

The project is designed to handle the unique challenges of the **Finnish language**, which is morphologically rich and agglutinative, requiring specific strategies like WWM and custom BPE to learn meaningful representations.

## Technical Specifications

### Model Architecture
The model follows a refined BERT configuration optimized for efficiency and representational power (based on Turc et al. 2019):
- **Architecture:** BERT Encoder-only Transformer.
- **Configuration:** 6 Layers, 256 Hidden Dimension, 4 Attention Heads (6L/256H/4A).
- **Vocabulary:** 50,000 tokens (Custom Byte Pair Encoding).
- **Parameters:** ~17.6M (Utilizing weight tying between input and output embeddings to reduce memory footprint and improve generalization).
- **Positionality:** Learned positional embeddings (Devlin et al. 2019) capped at 128 tokens.
- **Pooling:** Mean pooling over non-padding tokens for stable sentence-level representations (Reimers & Gurevych 2019).

### Training Strategy
- **Objective:** Masked Language Modeling (MLM) with Whole Word Masking.
- **Optimizer:** AdamW with linear learning rate warmup (10% of total steps).
- **Precision:** Mixed Precision (FP16/AMP) for accelerated training on modern GPUs.
- **Batching:** Micro-batch size of 128 with Gradient Accumulation (4 steps) for an **effective batch size of 512**.

## Pipeline Deep Dive

### 1. Data Acquisition (`01_data_download.py`)
- **Source:** Automatically fetches Finnish titles from the Project Gutenberg catalog (~3,600 books).
- **Mechanism:** Uses a gzip'd catalog fetch instead of API pagination for maximum reliability.
- **Robustness:** Implements thread-safe concurrent downloads (10 workers) with exponential back-off retry logic for network stability.

### 2. Text Preprocessing (`02_clean.py`)
- **Cleaning:** Follows the SPGC (Standard Project Gutenberg Corpus) pipeline standards (Gerlach & Font-Clos 2018).
- **Structure:** Identifies standard Gutenberg headers/footers to discard metadata and license text.
- **Normalization:** Fixes "typewriter-style" hard line breaks while preserving genuine paragraph boundaries.
- **Validation:** Filters out all-uppercase headers, short fragments (<20 chars), and non-alphabetic noise.

### 3. BPE Tokenization (`03_tokenizer.py`)
- **Implementation:** A native, parallel implementation of Byte Pair Encoding (Sennrich et al. 2016).
- **Parallelism:** Uses multiprocessing for "embarrassingly parallel" word frequency counting across all CPU cores.
- **Efficiency:** Filters hapax legomena (words appearing only once) to ensure stable merge decisions and reduce vocabulary noise.

### 4. Dataset & Whole Word Masking (`04_dataset.py`)
- **WWM Logic:** In agglutinative languages like Finnish, masking individual subwords often makes the task too trivial (e.g., predicting "issamme" from "talo-___-kin"). WWM masks all subwords of a word simultaneously, forcing the model to learn deep morphological dependencies.
- **Strategy:** Implements the 80-10-10 masking strategy (80% mask, 10% random, 10% identity) applied at the word level.
- **Preprocessing:** Includes a `--build-only` mode to pre-tokenize the entire corpus into a single `dataset.pt` tensor for instant loading during training.

### 5. Pre-training (`06_train.py`)
- **Engine:** Pure PyTorch training loop with `torch.cuda.amp` for mixed precision.
- **Checkpointing:** Dual-save system (Local + Google Drive) to survive cloud VM resets.
- **Monitoring:** Logs Loss, Perplexity (PPL), and Learning Rate every 100 steps.

### 6. Evaluation (`07_evaluate.py`)
- **Anisotropy:** Measures if embeddings occupy a narrow cone (lower is better for expressiveness).
- **Uniformity:** Measures how evenly embeddings are distributed on the hypersphere (Wang & Isola 2020).
- **Alignment:** Measures closeness of related concepts.
- **Qualitative:** Includes nearest-neighbor search across 5 Finnish semantic categories (Animals, Vehicles, Nature, People, Food).

## Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/tyerdogan/finnish-embeddings.git
cd finnish-embeddings
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Local Execution
Run the scripts in sequence. This is recommended for data preparation:
```bash
python src/01_data_download.py   # Downloads ~3,600 books
python src/02_clean.py           # Cleans and segments into sentences.txt
python src/03_tokenizer.py       # Trains BPE (50k vocab)
python src/04_dataset.py         # Builds dataset.pt
python src/06_train.py           # Starts MLM training (Requires GPU)
python src/07_evaluate.py        # Runs geometric metrics
```

### 3. Cloud Training (Google Colab)
For heavy training, use the provided notebook:
1. Run local steps 1-4 to generate `data/tokenizer/` and `data/processed/dataset.pt`.
2. Upload the `data/` folder to your Google Drive under `Finnish-Embedding/`.
3. Open `notebooks/train_colab.ipynb` in Colab.
4. Mount Drive and run all cells. Checkpoints will sync back to your Drive every 2,000 steps.

## Analysis & Visualization
The `notebooks/evaluate.ipynb` notebook provides a comprehensive visual analysis:
- **Cosine Similarity Heatmaps:** Visualize clustering of Finnish semantic categories.
- **t-SNE Projections:** See how the model separates "Animals" from "Vehicles" in high-dimensional space.
- **Metric Comparison:** Automatic comparison between the trained model and a random baseline.

## References
- **BERT:** Devlin et al. 2019 (arXiv:1810.04805)
- **Small BERT:** Turc et al. 2019 (arXiv:1908.08962)
- **Sentence-BERT:** Reimers & Gurevych 2019 (arXiv:1908.10084)
- **Anisotropy:** Ethayarajh 2019 (arXiv:1909.00512)
- **Uniformity & Alignment:** Wang & Isola 2020 (arXiv:2005.10242)
- **SPGC Pipeline:** Gerlach & Font-Clos 2018 (arXiv:1812.08092)
- **WWM (Chinese/General):** Cui et al. 2019 (arXiv:1906.08101)
