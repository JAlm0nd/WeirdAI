import torch
import torch.nn as nn

# Input embeddings: 6 tokens, with 3 features per token
inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]
])

d_in = inputs.shape[1]  # 3
d_out = 2


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        # Matrices stored directly as (d_in, d_out)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context_vecs = attn_weights @ values
        return context_vecs


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        # nn.Linear stores weight tensors as (d_out, d_in)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context_vecs = attn_weights @ values
        return context_vecs


# Create both versions
torch.manual_seed(789)

sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v1 = SelfAttention_v1(d_in, d_out)

# Copy and transpose the v2 weights into v1
with torch.no_grad():
    sa_v1.W_query.copy_(sa_v2.W_query.weight.T)
    sa_v1.W_key.copy_(sa_v2.W_key.weight.T)
    sa_v1.W_value.copy_(sa_v2.W_value.weight.T)

# Both outputs should now match
output_v2 = sa_v2(inputs)
output_v1 = sa_v1(inputs)

print("SelfAttention_v2 output:")
print(output_v2)

print("\nSelfAttention_v1 output:")
print(output_v1)

print("\nOutputs match:", torch.allclose(output_v2, output_v1))