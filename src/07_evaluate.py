"""
07_evaluate.py — Embedding quality evaluation for the Finnish BERT model.

Implements four complementary metrics from the sentence embedding literature:
  - Anisotropy        (Ethayarajh 2019, arXiv:1909.00512)
  - Uniformity        (Wang & Isola 2020, ICML)
  - Alignment         (Wang & Isola 2020, ICML)
  - Nearest-neighbour search for qualitative inspection
"""

import math
import sys
import random
import importlib.util
import collections
from pathlib import Path

import torch


# ── Module-level constants ────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = Path(__file__).resolve().parent

# Special token IDs match SPECIAL_TOKENS order in 03_tokenizer.py:
# ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
_PAD_ID = 0
_CLS_ID = 2
_SEP_ID = 3


# ── Dynamic module loading (filenames begin with a digit) ─────────────────────

def _load_src_module(name: str, filename: str):
    """Load a sibling src module whose filename is not a valid identifier."""
    spec = importlib.util.spec_from_file_location(name, _SRC_DIR / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tok_mod   = _load_src_module("tokenizer_03", "03_tokenizer.py")
_model_mod = _load_src_module("model_05",     "05_model.py")

encode                 = _tok_mod.encode           # encode(text, merges, vocab) -> list[int]
load_tokenizer         = _tok_mod.load_tokenizer   # load_tokenizer(path) -> (vocab, merges)
BertConfig             = _model_mod.BertConfig
BertForMLM             = _model_mod.BertForMLM
get_sentence_embedding = _model_mod.get_sentence_embedding


# ── SECTION 9: FINNISH TEST SENTENCES ────────────────────────────────────────
# Defined at module level so both main() and the notebook can import directly.

CATEGORIES: dict[str, list[str]] = {
    "Eläimet (Animals)": [
        "Koira juoksi puistossa.",
        "Kissa nukkui sohvalla.",
        "Lintu lauloi oksalla.",
        "Hevonen juoksi pellolla.",
        "Kala ui järvessä.",
        "Karhu nukkui talvella.",
        "Koira haukkui kovasti.",
        "Kissa katsoi ikkunasta.",
    ],
    "Ajoneuvot (Vehicles)": [
        "Auto ajoi nopeasti tiellä.",
        "Juna saapui asemalle.",
        "Laiva lähti satamasta.",
        "Lentokone lensi pilvien yli.",
        "Polkupyörä seisoi pihalla.",
        "Bussi pysähtyi liikennevaloissa.",
        "Moottoripyörä ajoi ohi.",
        "Auto parkattiin autotalliin.",
    ],
    "Luonto (Nature)": [
        "Aurinko paistoi kirkkaasti.",
        "Meri oli tyyni ja sininen.",
        "Lumi peitti maan valkoiseksi.",
        "Tuuli puhalsi voimakkaasti.",
        "Sade kaatoi läpi yön.",
        "Metsä oli hiljainen aamulla.",
        "Järvi heijasti taivaan.",
        "Pilvet liikkuivat hitaasti.",
    ],
    "Ihmiset (People)": [
        "Hän luki kirjaa hiljaa.",
        "Lapsi leikki ulkona.",
        "Mies käveli kaupungilla.",
        "Nainen lauloi kauniisti.",
        "Lapset juoksivat puistossa.",
        "Vanha mies istui penkillä.",
        "Tyttö piirsi kuvaa.",
        "Poika pelasi jalkapalloa.",
    ],
    "Ruoka (Food)": [
        "Leipä oli tuoretta ja pehmeää.",
        "Kahvi maistui hyvältä aamulla.",
        "Keitto oli lämmin ja maukas.",
        "Hedelmät olivat makeita.",
        "Juusto sopi hyvin leivän kanssa.",
        "Kala oli paistettu hyvin.",
        "Tee oli kuuma ja rauhoittava.",
        "Marjat olivat happamia.",
    ],
}

# Flat list of all 40 test sentences, preserving category order.
FINNISH_TEST_SENTENCES: list[str] = [
    sentence
    for sentences in CATEGORIES.values()
    for sentence in sentences
]


# ── SECTION 1: LOAD MODEL ─────────────────────────────────────────────────────

def load_model(
    checkpoint_path: Path | None,
    device: torch.device,
) -> BertForMLM:
    """
    Instantiate BertForMLM and optionally restore weights from a checkpoint.

    Why map_location='cpu' before moving to device:
      Checkpoints saved on GPU embed the source device in the file.  Loading
      directly with map_location=device fails when the checkpoint's source
      device differs from the current device (e.g. 'cuda:1' vs 'cuda:0' on
      a multi-GPU machine).  Loading to CPU first and then calling .to(device)
      is fully device-agnostic.  It also prevents GPU out-of-memory errors:
      the optimizer state tensors stored alongside the model weights can be
      2× the model size; a direct GPU load would momentarily hold both the
      on-disk tensors and the newly allocated model in GPU memory at once.

    Why eval() mode:
      eval() disables Dropout and ensures that any normalisation layers use
      their running statistics rather than batch statistics.  Without eval(),
      sentence embeddings differ between calls on the same input because each
      forward pass applies a different random dropout mask, making all
      metrics non-reproducible.

    Args:
        checkpoint_path: Path to a .pt checkpoint produced by 06_train.py,
                         or None for freshly initialised (random) weights.
        device:          torch.device to run inference on.

    Returns:
        BertForMLM in eval mode on the specified device.
    """
    config = BertConfig()
    model  = BertForMLM(config)

    if checkpoint_path is not None:
        # Load to CPU first to avoid cross-device GPU OOM (see docstring).
        state = torch.load(checkpoint_path, map_location="cpu",
                           weights_only=False)
        # weights_only=False: checkpoint contains non-tensor objects
        # (loss_history list, config dict). PyTorch 2.x requires
        # explicit flag to suppress FutureWarning.
        # 06_train.py wraps weights under 'model_state_dict' key.
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ── SECTION 2: LOAD TOKENIZER FILES ──────────────────────────────────────────

def load_tokenizer_files() -> tuple[dict, list]:
    """
    Load the BPE vocabulary and merge rules from the canonical tokenizer
    directory at ROOT_DIR/data/tokenizer/.

    Returns:
        (vocab, merges) — the same structures returned by load_tokenizer().
          vocab  : dict[str, int]  — token → ID mapping
          merges : list[tuple[str, str]] — ordered merge rules
    """
    tokenizer_dir = ROOT_DIR / "data" / "tokenizer"
    return load_tokenizer(str(tokenizer_dir))


# ── SECTION 3: ENCODE SENTENCES ──────────────────────────────────────────────

def encode_sentences(
    sentences: list[str],
    model: BertForMLM,
    vocab: dict,
    merges: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a list of plain-text sentences into L2-normalised sentence
    embeddings using the supplied BertForMLM model.

    Why the 126-token content limit:
      BertConfig.max_seq_length is 128.  Two positions are permanently
      reserved for [CLS] at index 0 and [SEP] at the last real-token
      position, leaving exactly 126 positions for content tokens.  Truncating
      content at 126 ensures every sequence — regardless of input length —
      fits within both the 128-slot positional embedding table and the
      model's maximum pre-training context.  This mirrors the BERT
      pre-processing convention (Devlin et al. 2019, Appendix A): truncate
      the longer sequence rather than raising an error.

    Why batch processing is more efficient than per-sentence inference:
      GPU SIMD units achieve peak throughput only when the batch dimension
      saturates the parallel execution units.  Sending one sentence at a time
      causes repeated kernel-launch overhead and leaves most CUDA cores idle
      during each forward pass.  Batching all N sentences together performs a
      single set of matrix multiplications at full hardware throughput.  For
      N=40 sentences of length 128 and hidden_size=256, the entire batch
      occupies roughly 5 MB of GPU memory — well within any modern device.

    Args:
        sentences: list of raw Finnish sentences.
        model:     BertForMLM instance in eval mode.
        vocab:     BPE vocabulary dict {token: id}.
        merges:    BPE merge rules list of (str, str) pairs.
        device:    target torch.device.

    Returns:
        Tensor of shape (N, hidden_size) with L2-normalised embeddings.
    """
    max_len  = 128
    all_ids  = []
    all_mask = []

    for sentence in sentences:
        token_ids = encode(sentence, merges, vocab)
        token_ids = token_ids[:126]                           # reserve 2 slots for CLS + SEP

        seq = [_CLS_ID] + token_ids + [_SEP_ID]
        n   = len(seq)
        pad = max_len - n

        all_ids.append(seq  + [_PAD_ID] * pad)
        all_mask.append([1] * n + [0] * pad)

    input_ids      = torch.tensor(all_ids,  dtype=torch.long, device=device)  # (N, 128)
    attention_mask = torch.tensor(all_mask, dtype=torch.long, device=device)  # (N, 128)

    return get_sentence_embedding(model, input_ids, attention_mask)            # (N, H)


# ── SECTION 4: ANISOTROPY ─────────────────────────────────────────────────────

def compute_anisotropy(
    embeddings: torch.Tensor,
    n_pairs: int = 1000,
    seed: int = 42,
) -> float:
    """
    Estimate anisotropy as the average cosine similarity between randomly
    sampled pairs of embeddings (Ethayarajh 2019, arXiv:1909.00512).

    What anisotropy measures:
      An embedding space is anisotropic when vectors concentrate in a narrow
      cone rather than spreading uniformly across all directions.  In a highly
      anisotropic space every pair of embeddings — regardless of semantic
      content — has a high cosine similarity, making cosine an unreliable
      discriminator for retrieval and STS tasks.

    Metric formula:
      anisotropy = (1 / M) * Σ_m  cos(u_m, v_m)
      where M = n_pairs random pairs (u_m, v_m), m = 1 … M.

    Because embeddings are L2-normalised (||e|| = 1), cosine similarity
    reduces to a dot product:
      cos(u, v) = (u · v) / (||u|| · ||v||) = u · v

    Reference values:
      These values vary significantly by model architecture, training
      data, and layer depth (Ethayarajh 2019, Figure 1).  Rather than
      citing absolute thresholds, we compare trained vs random-init
      models side by side — a relative improvement is meaningful
      regardless of the absolute scale.

      Direction: lower anisotropy score = less anisotropic = better.

    Why anisotropy matters:
      Ethayarajh 2019 showed that high anisotropy is the primary reason
      off-the-shelf BERT embeddings perform poorly on semantic similarity
      tasks without fine-tuning: the effective dimensionality is low and
      most of the variance lies in a handful of dominant directions rather
      than encoding content-relevant features.

    Args:
        embeddings: (N, H) unit-norm tensor.
        n_pairs:    number of random pairs to average over.
        seed:       random seed for reproducibility.

    Returns:
        float — average cosine similarity; lower means less anisotropic.
    """
    # random.Random(seed) creates an isolated instance so this function
    # does not mutate the global random state shared by the rest of the
    # program.  Calling random.seed() globally would interfere with any
    # other code that relies on the module-level RNG (e.g. dataset
    # shuffling, dropout, augmentation pipelines).
    rng = random.Random(seed)
    emb = embeddings.cpu()
    n   = emb.shape[0]

    total = 0.0
    for _ in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        # dot product == cosine similarity for unit-norm vectors
        total += float(torch.dot(emb[i], emb[j]))

    return total / n_pairs


# ── SECTION 5: UNIFORMITY ─────────────────────────────────────────────────────

def compute_uniformity(embeddings: torch.Tensor) -> float:
    """
    Measure how uniformly embeddings are distributed across the unit
    hypersphere (Wang & Isola 2020, ICML).

    Metric formula (Wang & Isola 2020, Equation 2):
      uniformity = log( mean_{i<j} exp(-2 * ||u_i - v_j||²) )

    The Gaussian kernel exp(-2 * ||u - v||²) has bandwidth t = 2 and maps
    pairwise squared distances to the interval (0, 1].  The log pushes the
    score to (−∞, 0], where:
      - Values near 0 indicate collapsed embeddings (all distances ≈ 0).
      - More negative values indicate larger pairwise distances, i.e. a
        more uniformly distributed set of embeddings.

    Implementation note:
      Because embeddings are L2-normalised, ||u - v||² = 2 − 2(u · v),
      which we compute as a vectorised subtraction from the dot-product
      matrix rather than computing explicit Euclidean distances.  Only the
      upper triangle (i < j) is used to avoid counting each pair twice.

    Reference values:
      These values vary by model size, training data, and
      number of sentences used for computation.
      Wang & Isola 2020 do not report universal thresholds.
      Use relative comparison: trained vs random-init is
      more informative than absolute values.
      Direction: more negative = more uniform = better.

    Args:
        embeddings: (N, H) unit-norm tensor.

    Returns:
        float — log-Gaussian uniformity; more negative is better.
    """
    emb = embeddings.cpu().float()
    n   = emb.shape[0]

    # Pairwise squared distances: ||u-v||² = 2 - 2(u·v)  (unit-norm identity)
    dot    = torch.mm(emb, emb.T)              # (N, N)
    sq_d   = 2.0 - 2.0 * dot                  # (N, N)

    # Upper-triangle mask to select each pair exactly once (i < j)
    mask   = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    kernel = torch.exp(-2.0 * sq_d[mask])     # shape: (N*(N-1)/2,)

    return float(torch.log(kernel.mean()))


# ── SECTION 6: ALIGNMENT ──────────────────────────────────────────────────────

def compute_alignment(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    alpha: int = 2,
) -> float:
    """
    Measure how closely positive pairs of embeddings map to each other
    (Wang & Isola 2020, ICML).

    Metric formula (Wang & Isola 2020, Equation 1):
      alignment = (1 / N) * Σ_i ||u_i − v_i||^alpha

    where (u_i, v_i) are positive pairs and alpha=2 is the paper default.
    Lower alignment means positive pairs are mapped closer together,
    indicating better representation quality.

    What positive pairs are:
      In supervised settings (e.g. NLI entailment) positive pairs are
      semantically equivalent sentences.  In SimCSE (Gao et al. 2021,
      arXiv:2104.08821), each sentence is paired with a second forward pass
      under a different dropout mask — treating stochastic dropout noise as
      a free data augmentation that creates minimally different views of the
      same input.

    Usage without labelled pairs:
      When no external positive pairs are available, pass the same sentence
      encoded twice under different model dropout realisations.  If
      embeddings_a and embeddings_b are identical (self-alignment), the
      result is exactly 0.0 — the theoretical lower bound — confirming
      correct implementation.  With two independent dropout passes the
      result measures how stable the model's representations are under
      stochastic perturbation.

    Args:
        embeddings_a: (N, H) unit-norm tensor — first view of each sentence.
        embeddings_b: (N, H) unit-norm tensor — positive-pair view.
        alpha:        distance exponent (Wang & Isola 2020 default = 2).

    Returns:
        float — mean ||u − v||^alpha; lower is better.
    """
    diff  = embeddings_a.cpu().float() - embeddings_b.cpu().float()  # (N, H)
    norms = diff.norm(p=2, dim=-1)                                    # (N,)
    return float((norms ** alpha).mean())


# ── SECTION 7: NEAREST-NEIGHBOUR SEARCH ──────────────────────────────────────

def nearest_neighbor_search(
    query_embs: torch.Tensor,
    corpus_embs: torch.Tensor,
    corpus_sentences: list[str],
    top_k: int = 5,
) -> list[list[tuple[str, float]]]:
    """
    Retrieve the top-k most similar corpus sentences for each query embedding
    using exact brute-force dot-product search.

    Why dot product equals cosine similarity here:
      Both query_embs and corpus_embs are L2-normalised (unit norm), so:
        cos(u, v) = (u · v) / (||u|| · ||v||) = u · v
      This lets a single torch.mm call compute the full (Q × N) similarity
      matrix without an explicit per-pair cosine computation.

    Why brute-force is sufficient for this scale (N ≤ 1 000):
      Exact nearest-neighbour via a dense matrix multiplication is O(Q × N × H).
      For Q=5, N=40, H=256: 5 × 40 × 256 = 51 200 multiply-adds — negligible
      on any hardware.  Approximate methods such as FAISS amortise an
      O(N log N) index build over many queries; this overhead is only
      cost-effective when N > 100 K and query volume is high.

    Args:
        query_embs:       (Q, H) unit-norm query embeddings.
        corpus_embs:      (N, H) unit-norm corpus embeddings.
        corpus_sentences: N plain-text sentences matching corpus_embs rows.
        top_k:            number of results to return per query.

    Returns:
        List of Q lists, each containing top_k (sentence, score) tuples
        sorted by descending similarity score.
    """
    sim = torch.mm(query_embs.cpu(), corpus_embs.cpu().T)  # (Q, N)
    results: list[list[tuple[str, float]]] = []

    for q in range(sim.shape[0]):
        row     = sim[q]                                   # (N,)
        indices = torch.topk(row, k=min(top_k, row.shape[0])).indices
        hits    = [(corpus_sentences[i], float(row[i])) for i in indices.tolist()]
        results.append(hits)

    return results


# ── SECTION 8: CHECKPOINT DISCOVERY ──────────────────────────────────────────

def get_latest_checkpoint() -> Path | None:
    """
    Locate the most recently saved training checkpoint by searching
    ROOT_DIR/checkpoints/ and ROOT_DIR/models/ in that order.

    Searches both checkpoints/ (06_train.py default) and models/
    (manual download — e.g. a checkpoint copied from Colab Drive).
    The first directory that contains at least one matching file wins;
    within that directory the lexicographically last file is returned.

    Why lexicographic sort is safe:
      06_train.py writes checkpoint filenames as checkpoint_step{N:06d}.pt,
      zero-padding the step counter to exactly 6 digits
      (e.g. checkpoint_step000500.pt, checkpoint_step001000.pt).
      Zero-padding guarantees that alphabetical (lexicographic) order
      matches numeric order for any step count up to 999 999.  Without
      zero-padding, "step10" < "step9" lexicographically, giving a wrong
      result; the fixed-width format eliminates that ambiguity entirely.

    Returns:
        Path to the latest checkpoint file, or None if no checkpoints exist
        in either directory.
    """
    for search_dir in (ROOT_DIR / "checkpoints", ROOT_DIR / "models"):
        if not search_dir.exists():
            continue
        candidates = sorted(search_dir.glob("checkpoint_step*.pt"))
        if candidates:
            return candidates[-1]
    return None


# ── SECTION 10: MAIN ──────────────────────────────────────────────────────────

def main() -> dict:
    """
    End-to-end evaluation pipeline: encode the 40 Finnish test sentences,
    compute anisotropy and uniformity for both a trained model and a randomly
    initialised baseline, run nearest-neighbour retrieval, and print a
    summary table.

    Returns a dict for downstream use in the companion notebook:
      {
        "trained_embeddings": Tensor (40, 256),
        "random_embeddings":  Tensor (40, 256),
        "sentences":          list[str],           # FINNISH_TEST_SENTENCES
        "categories":         dict[str, list[str]], # CATEGORIES
        "metrics": {
            "trained_anisotropy": float,
            "random_anisotropy":  float,
            "trained_uniformity": float,
            "random_uniformity":  float,
        },
      }
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    # ── Step 1: Load trained model and a fresh random-init baseline ───────────
    checkpoint = get_latest_checkpoint()
    if checkpoint is not None:
        print(f"Checkpoint found: {checkpoint.name}", flush=True)
    else:
        print("No checkpoint found — trained model uses random initialisation.", flush=True)

    trained_model = load_model(checkpoint, device)
    random_model  = load_model(None,       device)

    n_params = sum(p.numel() for p in trained_model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f} M\n", flush=True)

    # ── Step 2: Encode all 40 sentences with both models ─────────────────────
    print("Loading tokenizer...", flush=True)
    vocab, merges = load_tokenizer_files()
    print(f"Vocabulary size: {len(vocab):,}  Merge rules: {len(merges):,}\n", flush=True)

    print(f"Encoding {len(FINNISH_TEST_SENTENCES)} sentences...", flush=True)
    trained_embs = encode_sentences(FINNISH_TEST_SENTENCES, trained_model, vocab, merges, device)
    random_embs  = encode_sentences(FINNISH_TEST_SENTENCES, random_model,  vocab, merges, device)
    print(f"Embedding shape: {list(trained_embs.shape)}\n", flush=True)  # [40, 256]

    # ── Step 3: Anisotropy (Ethayarajh 2019) ──────────────────────────────────
    print("Computing anisotropy (1 000 random pairs)...", flush=True)
    trained_aniso = compute_anisotropy(trained_embs)
    random_aniso  = compute_anisotropy(random_embs)

    # ── Step 4: Uniformity (Wang & Isola 2020) ────────────────────────────────
    print("Computing uniformity (all 780 pairs)...", flush=True)
    trained_unif = compute_uniformity(trained_embs)
    random_unif  = compute_uniformity(random_embs)

    # ── Step 5: Nearest-neighbour retrieval (trained model, top-3) ────────────
    # One query per category — first sentence of each.
    category_names  = list(CATEGORIES.keys())
    query_sentences = [sents[0] for sents in CATEGORIES.values()]
    query_indices   = [FINNISH_TEST_SENTENCES.index(s) for s in query_sentences]
    query_embs      = trained_embs[query_indices]          # (5, H)

    nn_results = nearest_neighbor_search(
        query_embs, trained_embs, FINNISH_TEST_SENTENCES, top_k=3
    )

    print("\nNearest-neighbour search — trained model, top-3 per category:", flush=True)
    print("=" * 66, flush=True)
    for cat_name, query_sent, hits in zip(category_names, query_sentences, nn_results):
        print(f"\nCategory : {cat_name}", flush=True)
        print(f"  Query  : {query_sent}", flush=True)
        for rank, (sent, score) in enumerate(hits, 1):
            tag = " <-- query" if sent == query_sent else ""
            print(f"  {rank}. [{score:.4f}]  {sent}{tag}", flush=True)

    # ── Step 6: Summary table ─────────────────────────────────────────────────
    # Arrow direction: both metrics improve as they become more negative.
    aniso_arrow = "↓" if trained_aniso < random_aniso else "↑"
    unif_arrow  = "↓" if trained_unif  < random_unif  else "↑"

    print("\n", flush=True)
    print("─" * 56, flush=True)
    print(f"{'Metric':<16} {'Random Init':>14}  {'Trained':>14}  Δ", flush=True)
    print("─" * 56, flush=True)
    print(f"{'Anisotropy':<16} {random_aniso:>14.4f}  {trained_aniso:>14.4f}  {aniso_arrow}", flush=True)
    print(f"{'Uniformity':<16} {random_unif:>14.4f}  {trained_unif:>14.4f}  {unif_arrow}", flush=True)
    print("─" * 56, flush=True)
    print("Note: for both metrics ↓ is better (less anisotropic / more uniform).", flush=True)
    sys.stdout.flush()

    # ── Step 7: Return data for notebook ─────────────────────────────────────
    return {
        "trained_embeddings": trained_embs.cpu(),
        "random_embeddings":  random_embs.cpu(),
        "sentences":          FINNISH_TEST_SENTENCES,
        "categories":         CATEGORIES,
        "metrics": collections.OrderedDict([
            ("trained_anisotropy", trained_aniso),
            ("random_anisotropy",  random_aniso),
            ("trained_uniformity", trained_unif),
            ("random_uniformity",  random_unif),
        ]),
    }


if __name__ == "__main__":
    main()
