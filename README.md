# Finnish BERT — From Scratch

Finnish sentence embedding model, from scratch, without HuggingFace.

## Scenario
Post-apocalyptic NLP: no HuggingFace, no pretrained models,
only Python stdlib + PyTorch. Training an embedding model from scratch
on 3,601 Finnish Gutenberg books.

## Pipeline
| Script | What it does | Output |
|--------|-------------|--------|
| 01_download.py | Download 3,601 Finnish books from Project Gutenberg | data/raw/ |
| 02_clean.py | Clean texts and split into paragraphs | data/processed/sentences.txt |
| 03_tokenizer.py | Train BPE tokenizer (Sennrich et al. 2016) | data/tokenizer/ |
| 04_dataset.py | Build MLM dataset with Whole Word Masking | data/processed/dataset.pt |
| 05_model.py | BERT 6L/256H/4A architecture (17.6M params) | — |
| 06_train.py | MLM pre-training, fp16, checkpointing | checkpoints/ |

## Model
- **Architecture:** BERT encoder, 6 layers, 256 hidden, 4 heads (Turc et al. 2019)
- **Vocab:** 50K BPE (FinBERT scale, Virtanen et al. 2019)
- **Masking:** Whole Word Masking (Google BERT, Cui et al. 2019)
- **Parameters:** 17.6M (weight tying included)

## Dependencies
Only: `torch`, `requests`, `tqdm`, `matplotlib`, `sklearn`

## Colab Training
`notebooks/train_colab.ipynb`

## References
- Devlin et al. 2019 — BERT (arXiv:1810.04805)
- Sennrich et al. 2016 — BPE (ACL 2016)
- Turc et al. 2019 — Small BERT (arXiv:1908.08962)
- Liu et al. 2019 — RoBERTa (arXiv:1907.11692)
- Loshchilov & Hutter 2019 — AdamW (arXiv:1711.05101)
- Micikevicius et al. 2018 — Mixed Precision (arXiv:1710.03740)
- Reimers & Gurevych 2019 — Sentence-BERT (arXiv:1908.10084)
- Cui et al. 2019 — Chinese BERT WWM (arXiv:1906.08101)
