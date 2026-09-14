# gpt2_models.py
#
# Self-contained GPT-2-style model initialization and parameter counting.
#
# Run:
#   python gpt2_models.py
#
# Requirements:
#   pip install torch

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def get_config(base_config, model_name="gpt2-small"):
    """Return a copied GPT-2 configuration for the selected model size."""
    config = base_config.copy()

    model_settings = {
        "gpt2-small": {
            "emb_dim": 768,
            "n_layers": 12,
            "n_heads": 12,
        },
        "gpt2-medium": {
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16,
        },
        "gpt2-large": {
            "emb_dim": 1280,
            "n_layers": 36,
            "n_heads": 20,
        },
        "gpt2-xl": {
            "emb_dim": 1600,
            "n_layers": 48,
            "n_heads": 25,
        },
    }

    if model_name not in model_settings:
        valid_names = ", ".join(model_settings.keys())
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Choose one of: {valid_names}"
        )

    config.update(model_settings[model_name])

    if config["emb_dim"] % config["n_heads"] != 0:
        raise ValueError(
            "emb_dim must be evenly divisible by n_heads: "
            f"{config['emb_dim']} / {config['n_heads']}"
        )

    return config


# ------------------------------------------------------------
# Model components
# ------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalization implemented with learnable scale and shift."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized + self.shift


class GELU(nn.Module):
    """GELU activation used by GPT-2."""

    def forward(self, x):
        return 0.5 * x * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """GPT-2 MLP: emb_dim -> 4 * emb_dim -> emb_dim."""

    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg["emb_dim"]

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out ({d_out}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)

        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf,
        )

        attention_weights = torch.softmax(
            attention_scores / (self.head_dim ** 0.5),
            dim=-1,
        )

        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(1, 2)

        context_vectors = context_vectors.contiguous().view(
            batch_size,
            num_tokens,
            self.d_out,
        )

        return self.out_proj(context_vectors)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized GPT-2 transformer block."""

    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """GPT-2-style decoder-only transformer model."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
        )

        self.pos_emb = nn.Embedding(
            cfg["context_length"],
            cfg["emb_dim"],
        )

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        # This is deliberately separate so the program can report:
        # (1) all stored parameters, and
        # (2) the effective count if tied to tok_emb.weight.
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds the configured "
                f"context length ({self.cfg['context_length']})."
            )

        token_embeddings = self.tok_emb(in_idx)

        position_embeddings = self.pos_emb(
            torch.arange(
                seq_len,
                device=in_idx.device,
            )
        )

        x = token_embeddings + position_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits


# ------------------------------------------------------------
# Parameter reporting
# ------------------------------------------------------------

def calculate_size(model):
    """
    Print:
      - all parameters physically stored in this implementation
      - effective trainable parameters if tok_emb and out_head are tied
      - FP32 model size
    """
    total_params = sum(parameter.numel() for parameter in model.parameters())

    output_head_params = sum(
        parameter.numel()
        for parameter in model.out_head.parameters()
    )

    tied_weight_params = total_params - output_head_params

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_bytes / (1024 ** 3)

    print(f"Total number of parameters: {total_params:,}")
    print(
        "Number of trainable parameters considering weight tying: "
        f"{tied_weight_params:,}"
    )
    print(f"Total size in FP32: {total_size_mb:,.2f} MB ({total_size_gb:.2f} GB)")


# ------------------------------------------------------------
# Run all GPT-2 configurations
# ------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(123)

    for model_abbrev in ("small", "medium", "large", "xl"):
        model_name = f"gpt2-{model_abbrev}"

        config = get_config(
            GPT_CONFIG_124M,
            model_name=model_name,
        )

        model = GPTModel(config)

        print("\n" + "=" * 72)
        print(model_name.upper())
        print("=" * 72)
        print(
            f"Embedding dimension: {config['emb_dim']:,} | "
            f"Transformer blocks: {config['n_layers']} | "
            f"Attention heads: {config['n_heads']}"
        )

        calculate_size(model)

        del model