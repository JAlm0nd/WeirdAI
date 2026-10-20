import torch
import torch.nn as nn

torch.manual_seed(123)

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


mha = MultiHeadAttentionWrapper(
    d_in=768,
    d_out=768,
    context_length=1024,
    dropout=0.0,
    num_heads=12
)