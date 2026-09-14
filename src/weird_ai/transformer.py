import torch
from torch import nn

from weird_ai.layer_norm import LayerNorm
from weird_ai.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool,
    ):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.context_length = context_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        x = residual + self.dropout1(attention_output)

        residual = x
        x = residual + self.dropout2(self.feed_forward(self.norm2(x)))

        return x