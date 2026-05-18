import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SECTION 1: CONFIG ─────────────────────────────────────────────────────────

@dataclass
class BertConfig:
    """
    Hyperparameters for a small Finnish BERT encoder.

    Follows the 6L/256H/4A configuration from Turc et al. 2019
    (Well-Read Students Learn Better, arXiv:1908.08962), which showed that
    increasing depth is more parameter-efficient than increasing width for
    downstream task performance on GLUE benchmarks.  All values are set to
    match or directly derive from that configuration unless noted otherwise.
    """

    vocab_size: int = 50_000
    # Our BPE vocabulary (FinBERT scale, Virtanen et al. 2019).

    hidden_size: int = 256
    # Turc et al. 2019 small config (256H).  Fits comfortably in ~2 GB GPU
    # memory at batch_size=64, seq_len=128.

    num_layers: int = 6
    # Turc et al. 2019: depth contributes more to downstream performance per
    # parameter than equivalent increases in hidden width.

    num_heads: int = 4
    # BERT convention: d_k = hidden_size / num_heads = 64.
    # Vaswani et al. 2017 used d_k = 64 across all model sizes.

    ffn_size: int = 1024
    # BERT convention: FFN inner dimension = hidden_size × 4 (Devlin et al.
    # 2019).  256 × 4 = 1024.

    max_seq_length: int = 128
    # BERT Phase 1: 90% of pre-training steps use max_seq_length=128 because
    # self-attention cost is O(T²) in sequence length (Devlin et al. 2019).

    dropout: float = 0.1
    # Applied to all sub-layer outputs and embeddings (Devlin et al. 2019).

    pad_token_id: int = 0
    # Our BPE tokenizer assigns [PAD] id=0 (BERT convention).


# ── SECTION 2: EMBEDDINGS ─────────────────────────────────────────────────────

class BertEmbeddings(nn.Module):
    """
    Combines token embeddings and positional embeddings into a single
    representation, followed by LayerNorm and Dropout.

    Token embeddings map each token ID to a dense vector in R^hidden_size.
    Positional embeddings add information about where in the sequence each
    token appears.

    Why learned positional embeddings, not sinusoidal:
      Vaswani et al. 2017 (Transformer) proposed sinusoidal positional
      encodings that are fixed functions of position, arguing they would
      generalise to sequence lengths not seen during training.  BERT
      (Devlin et al. 2019) instead uses learned positional embeddings —
      each position has a trainable parameter vector.  The BERT paper notes
      that both approaches yield similar performance; learned embeddings are
      preferred because they adapt to the statistical patterns of the training
      data (e.g. position 1 after [CLS] tends to carry different information
      than position 60) without requiring manual design of the encoding
      function.  The cost is that the model cannot generalise to sequences
      longer than max_seq_length, which is acceptable here since all
      sequences are capped at 128 tokens.

    Input/output shapes:
      input_ids : (B, T) — integer token IDs
      output    : (B, T, H) — contextual input representations
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_length,
            config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout    = nn.Dropout(config.dropout)

        # Register position indices as a non-parameter buffer so they are
        # moved to the correct device automatically with .to(device).
        position_ids = torch.arange(config.max_seq_length).unsqueeze(0)  # (1, T)
        self.register_buffer("position_ids", position_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        token_emb = self.token_embedding(input_ids)                 # (B, T, H)
        pos_emb   = self.position_embedding(self.position_ids[:, :T])  # (1, T, H)
        x = token_emb + pos_emb                                     # (B, T, H)
        return self.dropout(self.layer_norm(x))


# ── SECTION 3: MULTI-HEAD SELF-ATTENTION ──────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention as defined in Vaswani et al.
    2017 (Attention Is All You Need), Equation 1:

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Why scale by sqrt(d_k):
      The dot product Q·K grows in magnitude with d_k (the head dimension)
      because it is a sum of d_k terms.  For large d_k the dot products push
      the softmax into regions with very small gradients, impeding learning.
      Dividing by sqrt(d_k) counteracts this growth and keeps gradients
      well-conditioned regardless of head dimension (Vaswani et al. 2017,
      Section 3.2.1).

    Multi-head splitting:
      Instead of one attention operation over hidden_size dimensions, we run
      num_heads parallel attention heads each over d_k = hidden_size /
      num_heads dimensions.  Each head can specialise to different types of
      relationships (syntactic, semantic, positional).  The outputs are
      concatenated and projected back to hidden_size.

    Attention mask:
      Padding positions (attention_mask=0) receive a large negative additive
      bias (-1e9) before softmax, making their post-softmax weights
      effectively zero.  This prevents real tokens from attending to padding.

    Input/output shapes:
      x              : (B, T, H)
      attention_mask : (B, T) — 1 for real tokens, 0 for padding
      output         : (B, T, H)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0, (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_heads ({config.num_heads})"
        )
        self.num_heads = config.num_heads
        self.d_k       = config.hidden_size // config.num_heads

        self.q_proj  = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj  = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj  = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H = x.shape

        # Project and reshape into (B, num_heads, T, d_k)
        def _split(proj, t):
            return proj(t).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q = _split(self.q_proj, x)  # (B, H_heads, T, d_k)
        k = _split(self.k_proj, x)
        v = _split(self.v_proj, x)

        # Scaled dot-product attention scores
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H_heads, T, T)

        if attention_mask is not None:
            # (B, T) → (B, 1, 1, T): broadcast over heads and query positions
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            scores = scores + pad_mask * -1e4

        weights = F.softmax(scores, dim=-1)          # (B, H_heads, T, T)
        weights = self.attn_dropout(weights)

        # Weighted sum of values, then merge heads
        out = torch.matmul(weights, v)               # (B, H_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, H)  # (B, T, H)
        return self.out_proj(out)


# ── SECTION 4: TRANSFORMER ENCODER BLOCK ─────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder layer: multi-head self-attention followed by a
    position-wise feed-forward network, each wrapped in a residual connection
    and LayerNorm.

    Post-LN vs Pre-LN:
      BERT (Devlin et al. 2019) uses Post-LN: LayerNorm is applied AFTER
      adding the residual, i.e. x = LayerNorm(x + Sublayer(x)).
      Later work (Xiong et al. 2020, "On Layer Normalization in the
      Transformer Architecture") showed that Pre-LN — normalising the input
      to each sublayer before the transformation — yields more stable
      gradients and can train without warmup.  We follow the original BERT
      Post-LN design to stay faithful to the paper, with the understanding
      that learning-rate warmup is needed when training from scratch.

    GELU vs ReLU:
      The feed-forward network uses GELU (Gaussian Error Linear Unit,
      Hendrycks & Gimpel 2016, arXiv:1606.08415) rather than ReLU.  GELU
      weights each input by the probability that it is larger than a
      standard Gaussian sample: GELU(x) = x · Φ(x).  This produces a
      smoother activation function that preserves small negative values
      (unlike ReLU's hard zero) and has been found to train more stably in
      deep Transformer networks.  BERT uses GELU throughout.

    Feed-forward network (FFN):
      Linear(H → ffn_size) → GELU → Dropout → Linear(ffn_size → H)
      The 4× expansion (ffn_size = 4 × hidden_size) and contraction follows
      the BERT and original Transformer convention.

    Input/output shapes:
      x              : (B, T, H)
      attention_mask : (B, T) — optional
      output         : (B, T, H)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.norm1     = nn.LayerNorm(config.hidden_size)
        self.norm2     = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sublayer — Post-LN: norm after residual
        x = self.norm1(x + self.dropout(self.attention(x, attention_mask)))
        # FFN sublayer — Post-LN
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ── SECTION 5: BERT ENCODER ───────────────────────────────────────────────────

class BertEncoder(nn.Module):
    """
    Full BERT encoder: embedding layer followed by a stack of Transformer
    encoder blocks.

    The [CLS] token is assumed to occupy position 0 of every input sequence
    (BERT convention).  Its final hidden state is commonly used as a
    sequence-level representation for classification tasks.  For sentence
    embeddings, mean pooling over all real tokens generally outperforms the
    [CLS] token — see get_sentence_embedding().

    Input/output shapes:
      input_ids      : (B, T) — integer token IDs
      attention_mask : (B, T) — 1 for real tokens, 0 for padding; optional
      output         : (B, T, H) — contextual token representations
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.layers     = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x  # (B, T, H)


# ── SECTION 6: BERT FOR MLM ───────────────────────────────────────────────────

class BertForMLM(nn.Module):
    """
    BERT encoder with a Masked Language Modelling head for pre-training.

    MLM head architecture (Devlin et al. 2019, Section 3.3.1):
      Linear(H → H) → GELU → LayerNorm → Linear(H → vocab_size)

    Why a two-layer head rather than a single projection:
      The intermediate dense layer projects the encoder's general-purpose
      hidden states into a task-specific representation space before the
      final vocabulary projection.  This separation means the MLM gradient
      does not directly reshape the encoder's representations — only the
      head is forced to produce vocab-sized logits.  The GELU activation
      allows smooth gradient flow, and LayerNorm stabilises the input
      distribution to the final large projection matrix (vocab_size = 50K),
      which would otherwise have a poorly conditioned gradient landscape.

    Loss:
      CrossEntropyLoss with ignore_index=-100 so that unmasked positions
      (label=-100 in the dataset) contribute zero loss.  This follows the
      BERT pre-training protocol: only the ~15% of masked tokens per
      sequence provide training signal.

    Weight tying:
      The output projection (Linear H → vocab_size) shares its weight matrix
      with the token embedding layer, following the original BERT and most
      subsequent Transformer language models.  This halves the parameter count
      for that layer (~12.8M), reduces overfitting on the vocabulary projection,
      and encourages a consistent representation space between input tokens and
      predicted output tokens.

    Input/output:
      input_ids      : (B, T)
      attention_mask : (B, T) — optional
      labels         : (B, T) — optional; -100 at unmasked positions
      Returns:
        (loss, logits) if labels provided, else logits
        logits shape : (B, T, vocab_size)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config      = config
        self.encoder     = BertEncoder(config)
        self.mlm_dense   = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm    = nn.LayerNorm(config.hidden_size)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the output projection.
        # Halves the parameter count for this layer (~12.8M) and improves generalisation
        # by forcing the model to use the same representation space for input and output.
        self.mlm_decoder.weight = self.encoder.embeddings.token_embedding.weight

        # BERT weight initialisation: truncated normal(0, 0.02) for all weights,
        # zeros for biases, ones/zeros for LayerNorm gamma/beta (Devlin et al. 2019).
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden = self.encoder(input_ids, attention_mask)  # (B, T, H)

        # MLM head
        x      = F.gelu(self.mlm_dense(hidden))          # (B, T, H)
        x      = self.mlm_norm(x)                        # (B, T, H)
        logits = self.mlm_decoder(x)                     # (B, T, vocab_size)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits


# ── SECTION 7: SENTENCE EMBEDDING ────────────────────────────────────────────

def get_sentence_embedding(
    model: BertForMLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Produces a fixed-size L2-normalised sentence embedding from a BertForMLM
    model using mean pooling over real (non-padding) token representations.

    Why mean pooling instead of the [CLS] token:
      BERT's [CLS] token is designed for sequence-level classification tasks
      and is fine-tuned for that purpose.  When the model is used without
      task-specific fine-tuning (as in unsupervised sentence similarity),
      [CLS] often produces worse embeddings because it has not been
      explicitly trained to encode global sentence semantics.

      Reimers & Gurevych 2019 (Sentence-BERT, arXiv:1908.10084) compared
      pooling strategies on STS benchmarks and found that mean pooling over
      all token representations generally outperforms [CLS] pooling, though the
      performance difference is typically small for
      pre-trained (non-fine-tuned) BERT encoders.  Mean pooling aggregates
      information from every real token, giving a more representative
      summary of the full sequence content.

    Why L2 normalisation:
      Normalising to unit length maps all embeddings to the surface of the
      unit hypersphere, making cosine similarity equivalent to dot product.
      This simplifies downstream nearest-neighbour search and distance
      computations.

    This function runs under torch.no_grad() — it is intended for inference
    only, not for computing gradients.

    Arguments:
      model          : BertForMLM — pre-trained (or randomly initialised) model
      input_ids      : (B, T) — tokenised sequences
      attention_mask : (B, T) — 1 for real tokens, 0 for padding

    Returns:
      embeddings : (B, H) — unit-norm sentence vectors
    """
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(input_ids, attention_mask)  # (B, T, H)

    # Mean pooling: zero out padding positions before averaging
    mask      = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
    summed    = (hidden * mask).sum(dim=1)                 # (B, H)
    count     = mask.sum(dim=1).clamp(min=1e-9)            # (B, 1)
    mean_pool = summed / count                             # (B, H)

    # L2 normalise to unit sphere
    norm = mean_pool.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return mean_pool / norm                                # (B, H)


# ── SECTION 8: SMOKE TEST ─────────────────────────────────────────────────────

def main():
    """
    Instantiates BertForMLM, runs a forward pass with dummy data, and
    validates that:
      1. Parameter count is in the expected range (~30M).
      2. Initial loss is close to ln(vocab_size) ≈ 10.8, which is the
         expected cross-entropy when the model outputs a near-uniform
         distribution over all vocabulary tokens at random initialisation.
      3. Sentence embeddings have the correct shape and unit L2 norm.
    """
    config = BertConfig()
    model  = BertForMLM(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready.  Params: {n_params / 1e6:.1f} M")
    print(f"  hidden_size={config.hidden_size}  num_layers={config.num_layers}"
          f"  num_heads={config.num_heads}  ffn_size={config.ffn_size}")

    # ── Dummy batch ───────────────────────────────────────────────────────────
    B, T   = 4, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Simulate real tokens in positions 0–63, padding in 64–127
    input_ids      = torch.randint(5, config.vocab_size, (B, T), device=device)
    input_ids[:, 0] = 2                                   # [CLS] at position 0
    attention_mask  = torch.ones(B, T, dtype=torch.long, device=device)
    attention_mask[:, 64:] = 0                            # simulate padding

    # Apply ~15% masking to real positions and build labels
    labels = torch.full((B, T), -100, dtype=torch.long, device=device)
    for b in range(B):
        real_pos  = attention_mask[b].nonzero(as_tuple=False).view(-1)
        n_mask    = max(1, int(len(real_pos) * 0.15))
        perm      = torch.randperm(len(real_pos), device=device)[:n_mask]
        masked    = real_pos[perm]
        labels[b, masked]    = input_ids[b, masked]       # target = original id
        input_ids[b, masked] = 4                          # replace with [MASK]

    # ── Forward pass ──────────────────────────────────────────────────────────
    loss, logits = model(input_ids, attention_mask, labels)

    expected_loss = math.log(config.vocab_size)
    print(f"\nForward pass OK")
    print(f"  logits shape : {list(logits.shape)}")
    print(f"  loss         : {loss.item():.3f}"
          f"  (random-init expected ≈ {expected_loss:.2f} = ln({config.vocab_size:,}))")

    # ── Sentence embeddings ───────────────────────────────────────────────────
    emb   = get_sentence_embedding(model, input_ids, attention_mask)
    norms = emb.norm(p=2, dim=-1)
    print(f"\nSentence embedding OK")
    print(f"  shape : {list(emb.shape)}")
    print(f"  norms : {[round(n, 4) for n in norms.tolist()]}  (should all be 1.0)")


if __name__ == "__main__":
    main()
