import math
import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """
        # Each row i contains dot products between token i and every token j.
        attention_scores = x @ x.T

        # Normalize across tokens being attended to.
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Weighted sum of the original token vectors.
        context_vectors = attention_weights @ x

        return context_vectors, attention_weights


class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Scale by sqrt(key/query dimension) before softmax.
        attention_scores = (queries @ keys.T) / math.sqrt(keys.shape[-1])
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = attention_weights @ values

        return context_vectors, attention_weights


class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(
        self,
        embedding_dim,
        output_dim,
        context_length,
        dropout=0.0,
        qkv_bias=False,
    ):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """
        _, num_tokens, _ = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Shape: (batch_size, num_tokens, num_tokens)
        attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(
            keys.shape[-1]
        )

        # Set all future-token scores to negative infinity before softmax.
        causal_mask = self.mask[:num_tokens, :num_tokens].bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values
        return context_vectors