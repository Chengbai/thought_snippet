import torch
import torch.nn as nn
from typing import Optional, Tuple


class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = (
            x.shape
        )  # 3D tensor of shape (batch_size, seq_len, d_model)

        # Project queries, keys, values
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # 4D tensor of shape (batch_size, n_heads, seq_len, head_dim)
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # 4D tensor of shape (batch_size, n_heads, seq_len, head_dim)
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # 4D tensor of shape (batch_size, n_heads, seq_len, head_dim)

        # Concatenate with cached KV if exists
        # Notes: the logic bellow is specially setup for the chunked-prefill for model inference.
        # At this case, each token is a valid `final` token!
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Both k_cache and v_cache have shape: (batch_size, n_heads, seq_len, head_dim)
            k = torch.cat([k_cache, k], dim=2)  # concat on seq dim
            v = torch.cat([v_cache, v], dim=2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()  # this is needed because the `stride` logic is `broken` after the above transpose!
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.o_proj(attn_output)

        # Return new cache if requested
        new_cache = (k, v) if use_cache else None
        return output, new_cache


class TransformerLayerWithCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = AttentionWithKVCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ):
        # Self-attention with residual
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache, use_cache)
        x = x + attn_out

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x, new_cache


def chunked_prefill(
    model_layers: nn.ModuleList,
    input_ids: torch.Tensor,
    chunk_size: int = 512,
    embed_layer: Optional[nn.Module] = None,
) -> list:
    """
    Perform chunked prefill to generate KV cache.

    Args:
        model_layers: List of transformer layers
        input_ids: Input token IDs [batch_size, seq_len]
        chunk_size: Number of tokens to process per chunk
        embed_layer: Optional embedding layer

    Returns:
        List of KV caches for each layer
    """
    batch_size, total_seq_len = input_ids.shape

    # Initialize embedding
    if embed_layer is not None:
        hidden_states = embed_layer(input_ids)
    else:
        # Assume input_ids are already embeddings
        hidden_states = input_ids

    # Initialize cache for each layer
    kv_caches = [None] * len(model_layers)

    # Process in chunks
    for chunk_start in range(0, total_seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_seq_len)
        chunk = hidden_states[:, chunk_start:chunk_end, :]

        print(f"Processing chunk [{chunk_start}:{chunk_end}]")

        # Pass through all layers
        for layer_idx, layer in enumerate(model_layers):
            chunk, new_cache = layer(
                chunk, kv_cache=kv_caches[layer_idx], use_cache=True
            )
            kv_caches[layer_idx] = new_cache

    return kv_caches


# Example usage
if __name__ == "__main__":
    # Model configuration
    d_model = 768
    n_heads = 12
    d_ff = 3072
    n_layers = 12
    vocab_size = 50257

    # Create model layers
    embedding = nn.Embedding(vocab_size, d_model)
    layers = nn.ModuleList(
        [TransformerLayerWithCache(d_model, n_heads, d_ff) for _ in range(n_layers)]
    )

    # Create long input sequence
    batch_size = 2
    seq_len = 2048
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Embed tokens
    embeddings = embedding(input_ids)

    # Perform chunked prefill with 512-token chunks
    print(f"Starting chunked prefill for {seq_len} tokens...")
    kv_caches = chunked_prefill(
        model_layers=layers, input_ids=embeddings, chunk_size=512
    )

    print(f"\nKV cache generated!")
    print(f"Number of layers: {len(kv_caches)}")

    # Inspect first layer's cache
    k_cache, v_cache = kv_caches[0]
    print(f"Key cache shape: {k_cache.shape}")  # [batch, n_heads, seq_len, head_dim]
    print(f"Value cache shape: {v_cache.shape}")

    # Now you can use these caches for decode phase
    # Decode example: generate next token
    print("\n--- Starting decode phase ---")
    new_token_embedding = torch.randn(batch_size, 1, d_model)  # Single new token

    for layer_idx, layer in enumerate(layers):
        new_token_embedding, updated_cache = layer(
            new_token_embedding, kv_cache=kv_caches[layer_idx], use_cache=True
        )
        kv_caches[layer_idx] = updated_cache

    # Updated cache now includes the new token
    k_cache, v_cache = kv_caches[0]
    print(f"Updated key cache shape: {k_cache.shape}")  # seq_len increased by 1
