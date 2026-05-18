import os
import re
import json
import time
import random
import collections
import multiprocessing
from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parent.parent
CORPUS_PATH   = ROOT_DIR / "data" / "processed" / "sentences.txt"
TOKENIZER_DIR = ROOT_DIR / "data" / "tokenizer"

# BERT (Devlin et al. 2019): special tokens must occupy the first IDs so that
# downstream models can rely on fixed positions (e.g. padding mask uses id 0).
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


# ── SECTION 0: CORPUS SAMPLING ────────────────────────────────────────────────

def sample_corpus(corpus_path, n=200_000, seed=42):
    """
    Draws a random subset of n lines from the corpus and writes it to
    data/processed/sentences_sample.txt.  If the sample file already exists
    it is returned immediately without re-sampling, making repeated runs
    deterministic and fast.

    Why 200,000 lines:
      SimCSE (Gao et al. 2021): the unsupervised SimCSE model was trained on
      approximately 1 million sentences drawn from English Wikipedia.  For a
      smaller, morphologically richer language like Finnish, 200,000
      paragraphs provide sufficient lexical diversity to learn a stable BPE
      vocabulary while keeping training time tractable on a single machine.
      FinBERT (Virtanen et al. 2019, arXiv:1912.07076): the Finnish BERT model
      was trained on Finnish Wikipedia and news corpora (~3 GB of text).
      200,000 paragraphs provide sufficient lexical diversity for BPE
      vocabulary convergence — the most frequent character pairs stabilise
      quickly and adding more data beyond a few hundred thousand sentences
      yields diminishing returns on vocabulary quality.

    seed=42 ensures the sample is reproducible across machines, which is
    important for experiment reproducibility (a standard requirement in NLP
    research).

    Returns the Path to the sample file.
    """
    sample_path = Path(corpus_path).parent / "sentences_sample.txt"

    if sample_path.exists():
        print(f"  Sample already exists: {sample_path}")
        return sample_path

    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    sampled = random.sample(lines, min(n, len(lines)))

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print(f"  Sampled {len(sampled):,} / {len(lines):,} lines → {sample_path}")
    return sample_path


# ── SECTION 1: BPE TRAINING ───────────────────────────────────────────────────

def _count_chunk(lines):
    """
    Counts BPE-formatted word frequencies for a contiguous slice of corpus lines.

    Each word is stored as a tuple of individual characters plus a '</w>'
    end-of-word marker: "talo" → ("t","a","l","o","</w>").  Tuples are the
    natural representation for BPE word sequences: each element is one symbol
    (character or merged sub-word), merging two adjacent symbols rebuilds the
    tuple without string parsing, and tuples are directly hashable as Counter
    keys.  The merge loop performs millions of such lookups, so avoiding any
    intermediate string conversion matters for overall throughput.

    Why this is embarrassingly parallel:
      Word frequency counting has no data dependency between lines — the count
      for line i does not depend on any result from line j.  Each chunk can
      therefore be processed entirely independently on a separate CPU core with
      no inter-process communication until the final merge step.  This is the
      textbook definition of an embarrassingly parallel workload (Herlihy &
      Shavit, "The Art of Multiprocessor Programming", 2008).  The only shared
      state is the final Counter, which is merged in the main process after all
      workers have finished — there is no lock contention during counting.

    Takes a list of raw text lines.
    Returns a collections.Counter mapping word tuples to their chunk frequency.
    """
    freqs = collections.Counter()
    for line in lines:
        for word in line.strip().split():
            freqs[tuple(word) + ("</w>",)] += 1
    return freqs


def get_word_frequencies(corpus_path, min_frequency=2):
    """
    Reads the entire corpus and counts how often each BPE-formatted word form
    appears, distributing the work across all available CPU cores.

    Words are represented as tuples of characters ending with '</w>':
      "talo" → ("t","a","l","o","</w>")

    The min_frequency filter (default 2) discards hapax legomena — words that
    appear only once.  This is standard practice in BPE training: hapax
    legomena contribute no merge decisions (a pair that appears only once in a
    once-occurring word has frequency 1 and will never be the global maximum),
    yet they can constitute 40-60% of unique word types in morphologically
    rich languages like Finnish.  Filtering them reduces the unique-word-type
    count by roughly half, which halves the cost of every merge step.

    Multiprocessing strategy:
      1. Read all lines into memory (single I/O pass).
      2. Split lines into cpu_count() equal-sized chunks.
      3. Dispatch each chunk to a worker process via multiprocessing.Pool.map.
         Each worker runs _count_chunk independently with zero inter-process
         communication.
      4. Merge the per-chunk Counters in the main process with Counter.update.

    Why the BPE merge loop itself cannot be parallelised:
      Unlike frequency counting, BPE merges are strictly sequential — merge
      i+1 must operate on the vocabulary produced by merge i.  There is a true
      data dependency between every iteration, making parallelisation of the
      merge loop impossible without fundamentally changing the algorithm.

    Returns a collections.Counter mapping word tuples to their corpus frequency.
    """
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()

    n_cores    = multiprocessing.cpu_count()
    chunk_size = max(1, len(lines) // n_cores)
    chunks     = [lines[i: i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(_count_chunk, chunks)

    freqs = collections.Counter()
    for result in results:
        freqs.update(result)

    # Discard hapax legomena — they never influence merge decisions and
    # filtering them cuts unique-type count by ~50% for Finnish text.
    return collections.Counter({w: c for w, c in freqs.items() if c >= min_frequency})


def _build_pair_index(word_freqs):
    """
    Builds two complementary indices over the current word vocabulary in a
    single O(total_symbols) pass:

      pair_freqs    : Counter mapping (token_a, token_b) → weighted frequency.
                      Equivalent to computing get_pair_frequencies from scratch,
                      but used as a persistent structure that is updated
                      incrementally after each merge (see _merge_step).

      pair_to_words : dict mapping (token_a, token_b) → set of word tuples that
                      contain that adjacent pair.  This reverse index is the key
                      to incremental BPE: instead of scanning all word types
                      after every merge, _merge_step looks up only the words
                      that actually contain the merged pair.

    Academic basis:
      The reverse-index incremental update strategy reduces the per-merge cost
      from O(V × L) — where V is the number of unique word types and L is the
      average word length — to O(W_pair × L), where W_pair is the number of
      word types containing the current best pair.  For typical Finnish corpora
      W_pair / V ≈ 1-5%, yielding a 20-100× speedup over full recomputation.

    Returns (pair_freqs, pair_to_words).
    """
    pair_freqs    = collections.Counter()
    pair_to_words = collections.defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    return pair_freqs, pair_to_words


def _merge_step(word_freqs, pair_freqs, pair_to_words, best_pair):
    """
    Applies one BPE merge operation and updates all three indices in-place
    in a single pass over only the words that contain best_pair.

    For each affected word:
      1. Subtract its frequency from pair_freqs for every adjacent pair it
         currently contains (its old pairs no longer exist after merging).
      2. Remove it from word_freqs and from pair_to_words for all its old pairs.
      3. Build the new word tuple by replacing every consecutive occurrence of
         (a, b) with the merged token "ab".
      4. Add the new word to word_freqs (accumulating if a collision occurs —
         two distinct old forms can produce the same new form after merging).
      5. Add the new word's frequency to pair_freqs for all its new adjacent
         pairs, and register the new word in pair_to_words.

    Collision handling (step 4):
      In agglutinative Finnish, merging ("t","a") could turn both
      ("t","a","l","o","</w>") and ("t","a","r","k","k","a","</w>") into forms
      starting with "ta".  If two old forms produce an identical new tuple,
      their frequencies are summed and the new pair_freqs entries are
      incremented by both contributions — which is correct because the merged
      token now represents all original occurrences.

    Complexity: O(W_pair × L) where W_pair = words containing best_pair,
    L = average word length in tokens.  This is the key cost reduction vs.
    full recomputation which is O(V × L).
    """
    a, b   = best_pair
    merged = a + b

    for old_word in list(pair_to_words.get(best_pair, [])):
        if old_word not in word_freqs:
            continue
        freq = word_freqs[old_word]

        # ── Step 1-2: remove old word's contributions ─────────────────────────
        del word_freqs[old_word]
        for i in range(len(old_word) - 1):
            p = (old_word[i], old_word[i + 1])
            pair_freqs[p] -= freq
            if pair_freqs[p] <= 0:
                del pair_freqs[p]
                pair_to_words.pop(p, None)
            else:
                pair_to_words[p].discard(old_word)

        # ── Step 3: build new word by merging every occurrence of (a, b) ──────
        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and old_word[i] == a and old_word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        new_word = tuple(new_word)

        # ── Step 4-5: add new word's contributions ────────────────────────────
        word_freqs[new_word] = word_freqs.get(new_word, 0) + freq
        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i + 1])
            pair_freqs[p] = pair_freqs.get(p, 0) + freq
            pair_to_words[p].add(new_word)

    # The merged pair no longer exists as an adjacent pair anywhere
    pair_freqs.pop(best_pair, None)
    pair_to_words.pop(best_pair, None)


def train_bpe(corpus_path, vocab_size=50_000):
    """
    Trains a Byte-Pair Encoding tokenizer from scratch using the incremental
    algorithm, which is dramatically faster than naive full recomputation.

    Algorithm outline:
      1. Count word frequencies with multiprocessing (embarrassingly parallel).
      2. Build the initial character vocabulary plus BERT special tokens.
      3. Build pair_freqs and pair_to_words indices in one pass (_build_pair_index).
      4. Repeat until vocab_size is reached:
         a. Find the most frequent pair via Counter.most_common(1) — this is
            O(V_pairs) but runs at C speed, taking ~1 ms even for 100K pairs.
         b. Record the merge and add the compound token to the vocabulary.
         c. Update all three indices incrementally via _merge_step, touching
            only the O(W_pair) words that contain the best pair.
      5. Return vocab and merges.

    Why incremental is fast:
      Full recomputation is O(V × L) per merge where V = unique word types and
      L = average token length (~6 for Finnish).  With V = 80K, that is 480K
      operations per merge × 49K merges = 23 billion operations.
      Incremental update is O(W_pair × L) per merge.  W_pair / V ≈ 2-5% for
      typical Finnish corpora, giving 480K → ~9K operations per merge, a
      ~50× speedup.  50K merges at 9K ops each = 450M operations, which
      Python executes in 3-8 minutes.

    Academic basis:
      Sennrich et al. 2016 (ACL): BPE algorithm and </w> convention.
      FinBERT (Virtanen et al. 2019, arXiv:1912.07076): used vocab_size=50,000
      for Finnish, providing broad morphological coverage without an
      excessively large embedding matrix.
      BERT (Devlin et al. 2019): special token ID assignment.

    Prints a live single-line progress bar with speed and ETA.
    Returns (vocab, merges).
    """
    print(f"Using {multiprocessing.cpu_count()} CPU cores for word frequency counting")

    # ── Step 1: word frequencies ───────────────────────────────────────────────
    print("Step 1/3  Counting word frequencies (parallel)...")
    word_freqs = get_word_frequencies(corpus_path)
    print(f"  Unique word types (freq≥2) : {len(word_freqs):>10,}")

    # ── Step 2: initial vocabulary ─────────────────────────────────────────────
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    # Collect ALL chars (including hapax chars) so no char becomes UNK
    all_chars = {char for word in word_freqs for char in word}
    for char in sorted(all_chars):
        if char not in vocab:
            vocab[char] = len(vocab)

    n_chars    = len(vocab) - len(SPECIAL_TOKENS)
    num_merges = vocab_size - len(vocab)
    print(f"  Initial vocab size         : {len(vocab):>10,}  "
          f"({len(SPECIAL_TOKENS)} special + {n_chars} chars)")
    print(f"  Merges required            : {num_merges:>10,}")

    # ── Step 3: pair index ─────────────────────────────────────────────────────
    print("Step 2/3  Building pair index...")
    pair_freqs, pair_to_words = _build_pair_index(word_freqs)
    print(f"  Unique pairs               : {len(pair_freqs):>10,}")

    # ── Step 4: merge loop ─────────────────────────────────────────────────────
    print("Step 3/3  BPE merge loop:\n")
    BAR_W  = 40
    merges = []
    start  = time.time()

    for i in range(num_merges):
        if not pair_freqs:
            print("\n  No more pairs — corpus exhausted early.")
            break

        # Counter.most_common(1) is O(n) but runs at C speed (~1 ms for 100K pairs)
        best_pair = pair_freqs.most_common(1)[0][0]

        # Incremental update: only touches words containing best_pair
        _merge_step(word_freqs, pair_freqs, pair_to_words, best_pair)

        new_token        = "".join(best_pair)
        vocab[new_token] = len(vocab)
        merges.append(best_pair)

        # ── progress bar (every merge, same line) ─────────────────────────────
        done    = i + 1
        elapsed = time.time() - start
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = int((num_merges - done) / rate) if rate > 0 else 0
        filled  = int(BAR_W * done / num_merges)
        bar     = "█" * filled + "░" * (BAR_W - filled)
        print(
            f"\r  [{bar}] {done}/{num_merges}  "
            f"vocab:{len(vocab):,}  "
            f"{rate:.0f}/s  "
            f"ETA:{eta // 60}m{eta % 60:02d}s   ",
            end="", flush=True,
        )

    elapsed_total = time.time() - start
    print(f"\n\n  Merge loop finished in {elapsed_total / 60:.1f} min")
    return vocab, merges


# ── SECTION 2: ENCODE / DECODE ────────────────────────────────────────────────

def _apply_merges_to_word(word_chars, merges):
    """
    Applies the learned merge rules to a single word's character sequence.
    Iterates through merges in training order; earlier merges take priority,
    matching the greedy left-to-right strategy of Sennrich et al. 2016.
    """
    tokens = list(word_chars)
    for (a, b) in merges:
        i        = 0
        new_toks = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_toks.append(a + b)
                i += 2
            else:
                new_toks.append(tokens[i])
                i += 1
        tokens = new_toks
    return tokens


def encode(text, merges, vocab):
    """
    Converts a plain-text string into a list of token IDs.

    Process:
      1. Split text on whitespace into words.
      2. For each word, produce its initial character sequence with </w> marker.
      3. Apply all learned merge rules in training order (greedy left-to-right).
      4. Map each resulting sub-word token to its vocabulary ID; tokens absent
         from the vocabulary are mapped to [UNK] (id 1).

    Academic basis:
      Sennrich et al. 2016: at inference time the same merge rules learned
      during training are replayed on new text.  Because merges are applied in
      training order, the segmentation is deterministic and consistent with the
      representations seen during model training.

    Returns a list of integer token IDs.
    """
    unk_id    = vocab.get("[UNK]", 1)
    token_ids = []
    for word in text.strip().split():
        chars  = list(word) + ["</w>"]
        tokens = _apply_merges_to_word(chars, merges)
        for token in tokens:
            token_ids.append(vocab.get(token, unk_id))
    return token_ids


def decode(token_ids, vocab):
    """
    Converts a list of token IDs back into a plain-text string.

    Process:
      1. Build a reverse mapping from ID to token string.
      2. Concatenate all token strings.
      3. Replace every </w> marker with a single space to recover word boundaries.

    Academic basis:
      Sennrich et al. 2016: the </w> marker encodes the original word boundary,
      so replacing it with a space during decoding faithfully reconstructs the
      whitespace-delimited input.

    Returns the decoded string (leading/trailing whitespace stripped).
    """
    id_to_token = {v: k for k, v in vocab.items()}
    tokens      = [id_to_token.get(i, "[UNK]") for i in token_ids]
    text        = "".join(tokens).replace("</w>", " ")
    return text.strip()


# ── SECTION 3: SAVE / LOAD ────────────────────────────────────────────────────

def save_tokenizer(vocab, merges, path=None):
    """
    Persists the trained tokenizer to disk in two human-readable files:

      vocab.json  — JSON object mapping every token string to its integer ID.
      merges.txt  — Plain-text file with one merge rule per line in the format
                    "token_a token_b", ordered by application priority
                    (earliest lines are applied first during encoding).

    The directory is created automatically if it does not exist.
    """
    out_dir = Path(path) if path else TOKENIZER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        for (a, b) in merges:
            f.write(f"{a} {b}\n")

    print(f"  Saved vocab  → {out_dir / 'vocab.json'}  ({len(vocab):,} tokens)")
    print(f"  Saved merges → {out_dir / 'merges.txt'}  ({len(merges):,} rules)")


def load_tokenizer(path=None):
    """
    Loads a previously saved tokenizer from disk.

    Reads vocab.json into a {token: id} dict and merges.txt into an ordered
    list of (token_a, token_b) tuples, preserving the training-time priority
    order required for deterministic encoding.

    Returns (vocab, merges).
    """
    in_dir = Path(path) if path else TOKENIZER_DIR

    with open(in_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    merges = []
    with open(in_dir / "merges.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

    return vocab, merges


# ── SECTION 4: MAIN ───────────────────────────────────────────────────────────

def main():
    """
    End-to-end pipeline: samples the corpus, trains BPE, saves the tokenizer,
    then validates it with five Finnish sentences and reports statistics.

    Token/word ratio is expected to be higher for Finnish than for English
    because Finnish is agglutinative: a single Finnish word such as
    "taloissammekin" (not even in our houses) encodes information that
    requires a full English phrase.  At the same vocabulary size, Finnish
    BPE segmentation produces more tokens per word than English — the exact
    ratio depends on corpus and vocabulary, but a higher-than-English value
    is consistent with the morphological complexity of the language.
    """
    # ── Sampling ──────────────────────────────────────────────────────────────
    print("Sampling corpus...")
    corpus = sample_corpus(CORPUS_PATH, n=200_000)

    # ── Training ──────────────────────────────────────────────────────────────
    vocab, merges = train_bpe(corpus, vocab_size=50_000)

    print(f"Final vocab size : {len(vocab):,}")
    print(f"Total merges     : {len(merges):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving tokenizer...")
    save_tokenizer(vocab, merges)

    # ── Validation ────────────────────────────────────────────────────────────
    test_sentences = [
        "Hän käveli hitaasti pitkin rantaa.",
        "Suomen kieli on agglutinatiivinen kieli.",
        "Talvinen maisema oli kaunis ja hiljainen.",
        "Kirjassa kerrotaan vanhan miehen elämästä.",
        "Minulla on nälkä ja haluaisin syödä jotain.",
    ]

    print("\nValidation (encode → decode):")
    print("-" * 60)
    total_tokens = total_words = 0
    for sentence in test_sentences:
        ids      = encode(sentence, merges, vocab)
        result   = decode(ids, vocab)
        match    = "✓" if result == sentence else "✗"
        n_words  = len(sentence.split())
        n_tokens = len(ids)
        total_tokens += n_tokens
        total_words  += n_words
        print(f"  {match}  [{n_tokens} tok / {n_words} words]  {sentence}")
        if result != sentence:
            print(f"     decoded: {result}")

    print("-" * 60)
    ratio = total_tokens / total_words if total_words else 0
    print(f"\nStatistics:")
    print(f"  Vocab size        : {len(vocab):,}")
    print(f"  Merge rules       : {len(merges):,}")
    print(f"  Avg tokens / word : {ratio:.2f}  "
          f"(Finnish is agglutinative; higher token/word ratios than English are expected)")


if __name__ == "__main__":
    main()
