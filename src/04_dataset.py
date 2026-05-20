import random
import sys
import importlib.util
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT_DIR      = Path(__file__).resolve().parent.parent
CORPUS_PATH   = ROOT_DIR / "data" / "processed" / "sentences.txt"
TOKENIZER_DIR = ROOT_DIR / "data" / "tokenizer"

# 03_tokenizer.py starts with a digit so a regular import statement is invalid.
# importlib lets us load it by file path without renaming the file.
_spec = importlib.util.spec_from_file_location(
    "tokenizer",
    Path(__file__).parent / "03_tokenizer.py",
)
_tok = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tok)
encode               = _tok.encode
load_tokenizer       = _tok.load_tokenizer
_apply_merges_to_word = _tok._apply_merges_to_word


# ── MASKING ───────────────────────────────────────────────────────────────────

def apply_masking(token_ids, vocab, end_of_word_ids=None):
    """
    Applies BERT's 80-10-10 masked language modelling strategy using
    Whole Word Masking (WWM).

    Why Whole Word Masking:
      Standard subword masking (Devlin et al. 2019 original) selects individual
      BPE tokens independently.  For agglutinative languages like Finnish this
      creates a trivial prediction task: if "taloissammekin" is segmented into
      ["talo", "issamme", "kin</w>"] and only "issamme" is masked, the model
      can recover it almost deterministically from "talo" and "kin".  No
      real morpho-semantic understanding is needed.

      Whole Word Masking (first introduced for English BERT by Google in
      May 2019 as an improvement to the original pre-processing code,
      documented in the BERT GitHub repository; adapted for Chinese by
      Cui et al. 2019, arXiv:1906.08101, published in IEEE/ACM TASLP 2021) masks all
      subword tokens belonging to the same word simultaneously.  This forces
      the model to predict the entire morphological form from surrounding
      context, encouraging deeper representation learning.

      For Finnish — ranked among the most morphologically complex languages in
      NLP benchmarks — WWM is especially important: Finnish words can have
      15+ grammatical forms, each tokenised differently.  Masking the whole
      word ensures the model learns to represent the stem and its inflections
      as semantically related.

    Masking strategy per selected word (Devlin et al. 2019, 80-10-10):
      80% → all subword tokens replaced with [MASK]
      10% → all subword tokens replaced with random vocabulary tokens
      10% → all subword tokens left unchanged

    The 80-10-10 split is applied per word, not per token, to preserve the
    training signal: replacing some subwords with [MASK] and others randomly
    within the same word would produce inconsistent inputs.

    end_of_word_ids:
      A frozenset of token IDs whose string form ends with '</w>', marking the
      final subword of each word.  Pre-computed once in FinnishMLMDataset.__init__
      and passed here to avoid rebuilding it on every __getitem__ call (which
      would cost O(vocab_size) = O(50,000) per sample and slow training).

    Arguments:
      token_ids       : list[int] — the full padded token sequence
      vocab           : dict[str, int] — token-to-id mapping
      end_of_word_ids : frozenset[int] — IDs of tokens that end a word

    Returns (masked_ids, labels):
      masked_ids : list[int] — input with some tokens replaced
      labels     : list[int] — original IDs at masked positions, -100 elsewhere
    """
    special_ids = {
        vocab["[PAD]"],
        vocab["[CLS]"],
        vocab["[SEP]"],
        vocab["[MASK]"],
    }
    vocab_size = len(vocab)

    if end_of_word_ids is None:
        end_of_word_ids = frozenset(
            tid for tok, tid in vocab.items() if tok.endswith("</w>")
        )

    # ── Group token indices by word ───────────────────────────────────────────
    # A word spans one or more consecutive non-special tokens; the last token
    # of the word has its ID in end_of_word_ids (its string ends with '</w>').
    words: list = []
    current_word: list = []
    for i, tid in enumerate(token_ids):
        if tid in special_ids:
            if current_word:
                words.append(current_word)
                current_word = []
            continue
        current_word.append(i)
        if tid in end_of_word_ids:
            words.append(current_word)
            current_word = []
    if current_word:
        words.append(current_word)

    if not words:
        return list(token_ids), [-100] * len(token_ids)

    # ── Select 15% of words (not tokens) for masking ─────────────────────────
    n_mask       = max(1, round(len(words) * 0.15))
    selected     = random.sample(words, min(n_mask, len(words)))

    masked_ids = list(token_ids)
    labels     = [-100] * len(token_ids)

    for word_positions in selected:
        # All subwords of this word share the same 80-10-10 decision
        r = random.random()
        for idx in word_positions:
            labels[idx] = token_ids[idx]
            if r < 0.80:
                masked_ids[idx] = vocab["[MASK]"]
            elif r < 0.90:
                masked_ids[idx] = random.randint(0, vocab_size - 1)
            # else: keep original (10%)

    return masked_ids, labels


# ── DATASET ───────────────────────────────────────────────────────────────────

class FinnishMLMDataset(Dataset):
    """
    PyTorch Dataset for Finnish masked language model pre-training.

    Each item is a single paragraph from the cleaned corpus, tokenized,
    truncated, padded, and masked according to the BERT MLM protocol.

    Design decisions:

    max_seq_length=128:
      BERT (Devlin et al. 2019) trains in two phases: Phase 1 uses
      max_seq_length=128 for 90% of steps because self-attention is
      O(n²) in sequence length — doubling the length quadruples the
      compute cost.  128 is long enough to capture paragraph-level context
      while keeping GPU memory usage tractable for consumer hardware.

    mask_prob=0.15:
      The BERT paper selects 15% of words as masking candidates (WWM).
      Lower values give the model too little training signal per sequence;
      higher values destroy too much context, making predictions trivial.

    RoBERTa (Liu et al. 2019):
      NSP (Next Sentence Prediction) is omitted.  RoBERTa (Liu et al. 2019)
      found that removing NSP matches or slightly improves downstream task
      performance.  MLM alone is sufficient.

    Token layout per sample:
      [CLS] t₁ t₂ … tₙ [SEP] [PAD] … [PAD]
       id=2               id=3  id=0

    Truncation removes tokens from the right of the encoded sequence,
    keeping the beginning of the paragraph (which typically contains the
    topic sentence).

    Expected sequence length distribution:
      The measured average real-token count is approximately 38–40 tokens
      out of the 128-token budget, leaving roughly 88–90 padding positions
      per sample on average.  This is not a design flaw — it is a direct
      consequence of two properties of the source corpus:

      (a) Project Gutenberg Finnish literature is heavily dialogue-oriented.
          Finnish novels frequently contain short quoted utterances and stage
          directions such as:
            "-- Kyllä."                   (~2–3 tokens)
            "Hän sanoi: »Mitä?«"          (~4–6 tokens)
          These short lines become their own paragraphs after the cleaning
          pipeline splits on blank lines, and they constitute a large fraction
          of the 3.5M corpus paragraphs.

      (b) Finnish agglutinative morphology produces longer words than English.
          A single Finnish word like 'taloissammekin' (not even in our houses)
          encodes what requires a full English phrase, so Finnish paragraphs
          tend to be shorter in word count than their English equivalents,
          resulting in fewer tokens per paragraph despite the per-word density.

      The heavy padding does not harm training because the attention mask
      prevents [PAD] tokens from contributing to self-attention or to the
      cross-entropy loss.  GPU compute wasted on padding positions can be
      recovered at training time via dynamic padding in the collate function
      (see collate_fn for trade-offs).
    """

    def __init__(self, corpus_path, vocab, merges,
                 max_seq_length=128, mask_prob=0.15):
        """
        Arguments:
          corpus_path    : path to the cleaned corpus (one paragraph per line)
          vocab          : dict[str, int] from load_tokenizer
          merges         : list of (str, str) merge rules from load_tokenizer
          max_seq_length : total sequence length including [CLS] and [SEP]
          mask_prob      : fraction of tokens selected for masking (BERT: 0.15)
        """
        self.vocab          = vocab
        self.merges         = merges
        self.max_seq_length = max_seq_length
        self.mask_prob      = mask_prob

        # Pre-compute once: IDs of tokens that end a word (token string ends
        # with '</w>').  Passed to apply_masking on every __getitem__ call to
        # avoid rebuilding this frozenset O(vocab_size) times during training.
        self.end_of_word_ids = frozenset(
            tid for tok, tid in vocab.items() if tok.endswith("</w>")
        )

        with open(corpus_path, encoding="utf-8") as f:
            # Strip blank lines; each non-empty line is one training paragraph
            self.paragraphs = [ln.rstrip("\n") for ln in f if ln.strip()]

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        """
        Tokenizes, truncates, pads, and masks one paragraph.

        Returns a dict of three tensors of shape (max_seq_length,):
          input_ids      : masked token IDs fed to the model
          attention_mask : 1 for real tokens, 0 for padding
          labels         : original IDs at masked positions, -100 elsewhere
                           (-100 is CrossEntropyLoss.ignore_index, so unmasked
                           positions contribute zero loss, following BERT)

        Why right-truncation:
          Tokens beyond position max_seq_length-2 are dropped from the right.
          In Gutenberg narrative prose, the first sentence of a paragraph
          typically establishes the topic or speaker, so preserving the left
          side of a long paragraph retains the most semantically dense content.
          This mirrors BERT's own pre-training convention for long documents.
        """
        paragraph = self.paragraphs[idx]

        # ── Tokenise ──────────────────────────────────────────────────────────
        token_ids = encode(paragraph, self.merges, self.vocab)

        # ── Truncate: leave 2 positions for [CLS] and [SEP] ──────────────────
        token_ids = token_ids[: self.max_seq_length - 2]

        # ── Add special boundary tokens (BERT convention) ─────────────────────
        cls_id = self.vocab["[CLS]"]   # id=2
        sep_id = self.vocab["[SEP]"]   # id=3
        pad_id = self.vocab["[PAD]"]   # id=0

        token_ids = [cls_id] + token_ids + [sep_id]
        seq_len   = len(token_ids)

        # ── Pad to max_seq_length ─────────────────────────────────────────────
        n_pad     = self.max_seq_length - seq_len
        token_ids = token_ids + [pad_id] * n_pad

        # ── Attention mask: 1 for real tokens, 0 for padding ─────────────────
        attention_mask = [1] * seq_len + [0] * n_pad

        # ── Apply Whole Word Masking (WWM) ────────────────────────────────────
        masked_ids, labels = apply_masking(token_ids, self.vocab, self.end_of_word_ids)

        return {
            "input_ids":      torch.tensor(masked_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,  dtype=torch.long),
            "labels":         torch.tensor(labels,          dtype=torch.long),
        }


# ── DATALOADER HELPER ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Collates a list of __getitem__ dicts into a single batched dict of tensors.

    Because every sample is already padded to max_seq_length in __getitem__,
    all tensors in the batch have identical shapes and torch.stack suffices —
    no further padding is needed.

    Returns a dict with keys input_ids, attention_mask, labels, each of shape
    (batch_size, max_seq_length).

    Why static padding rather than dynamic padding:
      Dynamic padding — padding each batch only to its longest sequence —
      can reduce wasted self-attention compute significantly given that the
      average real-token count (~38) is well below max_seq_length (128).
      Two reasons favour static padding here:

      1. Compatibility with dataset.pt.  preprocess_and_save writes
         pre-padded LongTensors of fixed shape (N, 128).  When the
         DataLoader reads from the saved file, all tensors are already
         128 tokens wide; re-trimming per batch requires an extra slice
         operation that introduces more overhead than it removes at the
         batch sizes typical of MLM pre-training (32–256 samples).

      2. Simplicity.  FinnishMLMDataset.__getitem__ produces fixed-shape
         outputs by design, making torch.stack always valid.  Switching to
         dynamic padding would require (a) removing padding from __getitem__,
         (b) re-implementing pad_sequence for three parallel tensors
         (input_ids, attention_mask, labels), and (c) regenerating dataset.pt
         in a variable-length format — a substantial refactor for a modest
         compute saving at pre-training scale.

      To adopt dynamic padding for fine-tuning or production use, replace
      this function with one that calls torch.nn.utils.rnn.pad_sequence on
      each key and recomputes attention_mask from the new lengths.
    """
    return {
        "input_ids":      torch.stack([s["input_ids"]      for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "labels":         torch.stack([s["labels"]         for s in batch]),
    }


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

# Per-process globals set by _init_worker so that _encode_chunk does not have
# to carry vocab/merges inside every chunk argument (those objects are large).
_WORKER_VOCAB       = None
_WORKER_MERGES      = None
_WORKER_MAX_SEQ     = None
_WORKER_MERGE_RANK  = None


def _init_worker(vocab, merges, max_seq_length):
    """
    Initialises per-process global state for multiprocessing workers.

    Called once per worker process by Pool(initializer=...).  Storing vocab,
    merges, and max_seq_length as module-level globals avoids pickling and
    transmitting these large objects with every chunk, reducing IPC overhead.

    merge_rank is built here rather than passed as an argument: it can be
    derived from merges in O(M) time and is ~4× larger than the merges list
    itself, so it is cheaper to construct once per worker process than to
    pickle and transmit it alongside every chunk dispatch.

    Note: _WORKER_MERGES stores the raw merge list for completeness and for
    any future code that needs the ordered list directly.  _encode_chunk itself
    uses _WORKER_MERGE_RANK (the dict form) for O(1) pair lookups via
    _bpe_encode_word — it does not read _WORKER_MERGES.
    """
    global _WORKER_VOCAB, _WORKER_MERGES, _WORKER_MAX_SEQ, _WORKER_MERGE_RANK
    _WORKER_VOCAB      = vocab
    _WORKER_MERGES     = merges
    _WORKER_MAX_SEQ    = max_seq_length
    # Lower rank = higher priority (earlier merge rule wins)
    _WORKER_MERGE_RANK = {pair: i for i, pair in enumerate(merges)}


def _bpe_encode_word(word_chars, merge_rank):
    """
    Applies BPE merges to one word using the priority-queue algorithm.

    Instead of iterating through all M merge rules in training order (naive
    O(M × L) per word, where M=49,779 and L≈6), this algorithm:
      1. Finds all adjacent token pairs in the current word.
      2. Looks up each pair in merge_rank (O(1) dict lookup).
      3. Applies only the highest-priority applicable merge.
      4. Repeats until no merge rule matches any adjacent pair.

    Cost: O(K × L²) where K is the number of merges that actually apply to
    this word (typically 1–5 for short Finnish words) and L is current token
    count.  For K=5, L=5: 125 operations vs the naive 300,000 — a ~2,400×
    speedup per word call (Sennrich et al. 2016 algorithm described in
    HuggingFace tokenizers documentation, 2020).

    Must be a top-level function so it can be used inside _encode_chunk
    which runs in worker processes on macOS/Windows (spawn start method).
    """
    tokens = list(word_chars)
    while len(tokens) > 1:
        pairs      = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        applicable = [(merge_rank[p], i, p) for i, p in enumerate(pairs) if p in merge_rank]
        if not applicable:
            break
        _, idx, (a, b) = min(applicable)
        tokens = tokens[:idx] + [a + b] + tokens[idx + 2:]
    return tokens


def _encode_chunk(paragraphs):
    """
    Encodes a list of raw text paragraphs into padded integer sequences,
    ready to be assembled into the final dataset tensors.

    For each paragraph:
      1. BPE-encode with the vocabulary and merge rules loaded at worker init.
      2. Truncate to max_seq_length - 2 (reserving one slot each for [CLS]
         and [SEP]).
      3. Prepend [CLS] (id=2) and append [SEP] (id=3) — BERT convention.
      4. Right-pad with [PAD] (id=0) to max_seq_length.
      5. Construct the attention mask: 1 for every real token, 0 for padding.

    Why this is embarrassingly parallel:
      Each paragraph is encoded independently — the encoding of paragraph i
      does not depend on the result of paragraph j.  There is no shared mutable
      state between iterations, satisfying the definition of an embarrassingly
      parallel workload (Herlihy & Shavit, "The Art of Multiprocessor
      Programming", 2008).  The only synchronisation point is the final
      assembly of results in the main process, which is a simple sequential
      write into a pre-allocated tensor.

    This function must be top-level (not nested) so that Python's multiprocessing
    module can pickle it for transmission to worker processes on macOS/Windows,
    where the 'spawn' start method requires all pickled objects to be importable
    from the module's top-level namespace.

    Arguments:
      paragraphs : list[str] — raw text paragraphs assigned to this worker

    Returns a list of (input_ids, attention_mask) tuples, both as list[int]
    of length max_seq_length.
    """
    vocab          = _WORKER_VOCAB
    max_seq_length = _WORKER_MAX_SEQ
    merge_rank     = _WORKER_MERGE_RANK

    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    pad_id = vocab["[PAD]"]
    unk_id = vocab.get("[UNK]", 1)

    # Word-level cache combined with the fast O(K×L²) encoder.
    # Each unique word form is encoded at most once per chunk; subsequent
    # occurrences are served from the dict in O(1).
    word_cache: dict = {}

    def _encode_cached(text):
        token_ids = []
        for word in text.strip().split():
            if word not in word_cache:
                chars  = list(word) + ["</w>"]
                tokens = _bpe_encode_word(chars, merge_rank)
                word_cache[word] = [vocab.get(t, unk_id) for t in tokens]
            token_ids.extend(word_cache[word])
        return token_ids

    results = []
    for paragraph in paragraphs:
        token_ids = _encode_cached(paragraph)
        token_ids = token_ids[: max_seq_length - 2]
        token_ids = [cls_id] + token_ids + [sep_id]
        seq_len   = len(token_ids)
        n_pad     = max_seq_length - seq_len
        token_ids = token_ids + [pad_id] * n_pad
        attention_mask = [1] * seq_len + [0] * n_pad
        results.append((token_ids, attention_mask))

    return results


def preprocess_and_save(corpus_path, vocab, merges,
                        output_path, max_seq_length=128,
                        max_samples=1_000_000):
    """
    Tokenizes up to max_samples paragraphs in parallel and saves them as a
    single PyTorch tensor file at output_path.

    max_samples parameter:
      The default of 1_000_000 is a conservative baseline: SimCSE (Gao et al.
      2021) achieved state-of-the-art sentence embeddings on 1M English
      Wikipedia sentences, establishing 1M as a sufficient lower bound.
      main() overrides this to 2_000_000 for three reasons specific to Finnish:
        (a) Finnish morphological diversity is higher than English — the same
            number of sentences yields more unique word forms, so more data
            directly expands vocabulary coverage during MLM pre-training.
        (b) The cleaned corpus contains 3.5M paragraphs; 2M uses only 57% of
            available data while doubling the training signal over the baseline.
        (c) With the fast BPE encoder (_bpe_encode_word, O(K×L²) per word),
            preprocessing 2M paragraphs takes ~1.6 minutes — a one-time cost
            acceptable before repeated Colab training sessions.

    Why masking is not applied here:
      RoBERTa (Liu et al. 2019) found that dynamic masking — generating
      a fresh random mask for each training step — performs comparably to
      or marginally better than static masking baked in at preprocessing time.  Keeping masking out of
      this function means FinnishMLMDataset.__getitem__ applies apply_masking()
      fresh on every access, giving each epoch a distinct view of the data.

    Why preprocess at all:
      Encoding 2M paragraphs with the BPE tokenizer at training time would add
      ~3 minutes of CPU overhead before the first gradient step on every Colab
      session restart.  Preprocessing once, uploading dataset.pt to Google
      Drive, and loading it in torch.load() takes ~10 seconds — a 20× saving
      per session for iterative training runs.

    Multiprocessing strategy:
      Paragraphs are divided into (cpu_count × 8) chunks and dispatched via
      Pool.imap_unordered with chunksize=1, so completed chunks are returned
      to the main process immediately and the progress bar updates every
      ~10,000 paragraphs rather than waiting for all workers to finish.
      vocab and merges are transmitted to workers once via Pool initializer,
      avoiding the cost of pickling large objects with every chunk.

    Output format (torch.save):
      {
          "input_ids":      LongTensor(N, max_seq_length),
          "attention_mask": LongTensor(N, max_seq_length),
      }
      Labels are intentionally omitted — apply_masking() generates them
      dynamically in FinnishMLMDataset.__getitem__ so each training epoch
      sees a different masking pattern over the same tokens.

    Displays a live progress bar updated after each chunk completes.
    """
    import time
    import multiprocessing

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, encoding="utf-8") as f:
        paragraphs = [ln.rstrip("\n") for ln in f if ln.strip()]

    paragraphs = paragraphs[:max_samples]
    n          = len(paragraphs)

    n_cores    = multiprocessing.cpu_count()

    # Bora et al. 2022 (arXiv:2202.11464): k > l tasks
    # improves load balancing. 8x gives ~96 chunks,
    # progress bar updates ~every 10k paragraphs.

    chunk_size = max(1, n // (n_cores * 8))
    chunks     = [paragraphs[i: i + chunk_size] for i in range(0, n, chunk_size)]
    n_chunks   = len(chunks)

    # Pre-allocate output tensors — order is filled as imap_unordered completes
    all_ids   = torch.zeros(n, max_seq_length, dtype=torch.long)
    all_masks = torch.zeros(n, max_seq_length, dtype=torch.long)

    print(f"Preprocessing {n:,} paragraphs  "
          f"({n_chunks} chunks across {n_cores} cores)")

    BAR_W      = 40
    para_count = 0
    start      = time.time()

    with multiprocessing.Pool(
        processes   = n_cores,
        initializer = _init_worker,
        initargs    = (vocab, merges, max_seq_length),
    ) as pool:
        for chunk_idx, chunk_results in enumerate(
            # chunksize=1 ensures each completed chunk is returned
            # to main process immediately, enabling live progress bar.
            pool.imap_unordered(_encode_chunk, chunks, chunksize=1), start=1
        ):
            for ids, mask in chunk_results:
                all_ids[para_count]   = torch.tensor(ids,  dtype=torch.long)
                all_masks[para_count] = torch.tensor(mask, dtype=torch.long)
                para_count += 1

            elapsed  = time.time() - start
            rate     = para_count / elapsed if elapsed > 0 else 0
            eta_secs = (n - para_count) / rate if rate > 0 else 0
            filled   = int(BAR_W * chunk_idx / n_chunks)
            bar      = "█" * filled + "░" * (BAR_W - filled)
            print(
                f"\r  [{bar}] {chunk_idx}/{n_chunks} chunks  "
                f"{para_count:,}/{n:,} paragraphs  "
                f"{elapsed / 60:.1f} min  ETA: {eta_secs / 60:.1f} min  ",
                end="", flush=True,
            )

    print()
    torch.save({"input_ids": all_ids, "attention_mask": all_masks}, output_path)
    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"  Saved {output_path}  ({size_mb:.1f} MB)")


# ── BUILD ENTRY POINT ─────────────────────────────────────────────────────────

def build_dataset():
    """
    Builds dataset.pt and nothing else.

    Called from Colab via `python src/04_dataset.py --build-only`.
    Does not construct FinnishMLMDataset, so it never loads the full
    3.5M-paragraph corpus into RAM.
    """
    vocab, merges = load_tokenizer(TOKENIZER_DIR)
    output_path = ROOT_DIR / "data" / "processed" / "dataset.pt"
    preprocess_and_save(CORPUS_PATH, vocab, merges, output_path, max_samples=2_000_000)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    """
    End-to-end validation and preprocessing pipeline.

      1. Load the trained BPE tokenizer from data/tokenizer/.
      2. Build FinnishMLMDataset over the full 3.5M-paragraph corpus.
      3. Sample two DataLoader batches as a smoke-test — print tensor shapes,
         masked token counts, and a decoded sample showing which whole words
         were masked by the WWM strategy.
      4. Compute mean sequence length and masking rate over 50 random samples
         (50 random samples provide a quick sanity check; this is not a
         statistically rigorous estimate — confidence intervals at n=50 would
         be wide for high-variance metrics like sequence length).
      5. Run preprocess_and_save to BPE-encode 2M paragraphs in parallel and
         write data/processed/dataset.pt for use in Colab training sessions.
    """
    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    vocab, merges = load_tokenizer(TOKENIZER_DIR)
    print(f"  Vocab size : {len(vocab):,}")
    print(f"  Merge rules: {len(merges):,}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\nBuilding dataset...")
    dataset = FinnishMLMDataset(
        corpus_path    = CORPUS_PATH,
        vocab          = vocab,
        merges         = merges,
        max_seq_length = 128,
        mask_prob      = 0.15,
    )
    print(f"  Dataset size: {len(dataset):,} paragraphs")

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size  = 8,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    # ── Sample two batches ────────────────────────────────────────────────────
    id_to_token = {v: k for k, v in vocab.items()}

    print("\n── Batch preview ──────────────────────────────────────")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 2:
            break

        ids   = batch["input_ids"]      # (B, 128)
        mask  = batch["attention_mask"] # (B, 128)
        lbls  = batch["labels"]         # (B, 128)

        n_masked = (lbls != -100).sum().item()
        n_tokens = mask.sum().item()

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  input_ids      shape : {list(ids.shape)}")
        print(f"  attention_mask shape : {list(mask.shape)}")
        print(f"  labels         shape : {list(lbls.shape)}")
        print(f"  Real tokens in batch : {n_tokens}  |  Masked tokens: {n_masked}")

        # Decode the first sample, marking masked positions with <MASK>
        sample_ids  = ids[0].tolist()
        sample_lbls = lbls[0].tolist()
        tokens = []
        for tid, lbl in zip(sample_ids, sample_lbls):
            if tid == vocab["[PAD]"]:
                break
            if lbl != -100:
                original = id_to_token.get(lbl, "?")
                tokens.append(f"[{original}→MASK]")
            else:
                tokens.append(id_to_token.get(tid, "?"))
        print(f"\n  Sample 0 decoded:\n  {''.join(tokens).replace('</w>', ' ')}")

    # ── Dataset statistics ────────────────────────────────────────────────────
    # 50 random samples provide a quick sanity check on masking rate
    # and sequence length. This is not a statistically rigorous estimate
    # (confidence intervals would be wide at n=50 for high-variance
    # metrics), but sufficient to verify the pipeline is behaving as
    # expected before full training.

    print("\n── Statistics (sampled over 50 items) ─────────────────")
    sample_size   = min(50, len(dataset))
    indices       = random.sample(range(len(dataset)), sample_size)
    total_len     = 0
    total_masked  = 0

    for i in indices:
        item = dataset[i]
        real_len      = item["attention_mask"].sum().item()
        n_masked_item = (item["labels"] != -100).sum().item()
        total_len    += real_len
        total_masked += n_masked_item

    avg_len    = total_len    / sample_size
    avg_masked = total_masked / sample_size

    print(f"  Dataset size            : {len(dataset):,}")
    print(f"  Avg sequence length     : {avg_len:.1f} / 128 tokens")
    print(f"  Avg masked tokens / seq : {avg_masked:.1f}  "
          f"(≈{avg_masked / avg_len * 100:.1f}% of real tokens, target 15%)")

    # ── Preprocess and save ───────────────────────────────────────────────────
    output_path = ROOT_DIR / "data" / "processed" / "dataset.pt"
    print()
    preprocess_and_save(CORPUS_PATH, vocab, merges, output_path, max_samples=2_000_000)



if __name__ == "__main__":
    if "--build-only" in sys.argv:
        # Colab: build dataset.pt only, skip smoke test
        build_dataset()
    else:
        # Local: full smoke test + dataset.pt
        main()
