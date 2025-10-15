import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Optional tuple of (past_keys, past_values)
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)  # [batch, seq_len, d_model]
        v = self.v_proj(x)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]
        # Shape: [batch, num_heads, seq_len, head_dim]

        # Use KV-cache if provided (inference mode)
        if kv_cache is not None:
            past_k, past_v = kv_cache  # Shape: [batch, num_heads, seq_len, head_dim]
            k = torch.cat([past_k, k], dim=2)  # Concatenate along seq_len
            v = torch.cat([past_v, v], dim=2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        # Return cache if requested
        if use_cache:
            return output, (k, v)
        return output


# ============================================
# DEMONSTRATION: Training vs Inference
# ============================================


def training_example():
    """Training: No KV-cache needed, process all tokens at once"""
    print("=== TRAINING MODE (No KV-Cache) ===")

    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    model = MultiHeadAttention(d_model, num_heads)

    # All tokens available at once (teacher forcing)
    x = torch.randn(batch_size, seq_len, d_model)

    # Single forward pass, no caching
    output = model(x, kv_cache=None, use_cache=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Processed all tokens in one pass\n")


def inference_without_cache():
    """Inference WITHOUT KV-cache: Recomputes everything each step"""
    print("=== INFERENCE WITHOUT KV-CACHE (Inefficient) ===")

    d_model = 512
    num_heads = 8
    batch_size = 1
    max_new_tokens = 5

    model = MultiHeadAttention(d_model, num_heads)
    model.eval()

    # Start with initial prompt
    prompt = torch.randn(batch_size, 3, d_model)  # 3 prompt tokens
    generated = prompt

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Process ENTIRE sequence every time (inefficient!)
            output = model(generated, kv_cache=None, use_cache=False)

            # Get last token's output and "generate" next token
            next_token = torch.randn(batch_size, 1, d_model)
            generated = torch.cat([generated, next_token], dim=1)

            print(f"Step {i+1}: Processing {generated.shape[1]} tokens")

    print(f"Final sequence length: {generated.shape[1]}")
    print("✗ Recomputed K,V for all previous tokens at each step\n")


def inference_with_cache():
    """Inference WITH KV-cache: Efficient, only computes new token"""
    print("=== INFERENCE WITH KV-CACHE (Efficient) ===")

    d_model = 512
    num_heads = 8
    batch_size = 1
    max_new_tokens = 5

    model = MultiHeadAttention(d_model, num_heads)
    model.eval()

    # Start with initial prompt
    prompt = torch.randn(batch_size, 3, d_model)  # 3 prompt tokens

    with torch.no_grad():
        # First pass: process prompt and initialize cache
        output, kv_cache = model(prompt, kv_cache=None, use_cache=True)
        past_k, past_v = kv_cache
        print(f"Initial prompt: {prompt.shape[1]} tokens")
        print(f"Cache initialized: K/V shape = {past_k.shape}")

        # Generate new tokens
        current_token = torch.randn(batch_size, 1, d_model)

        for i in range(max_new_tokens):
            # Only process the NEW token, reuse cached K,V
            output, kv_cache = model(current_token, kv_cache=kv_cache, use_cache=True)

            past_k, past_v = kv_cache
            print(
                f"Step {i+1}: Processing 1 new token, cache has {past_k.shape[2]} tokens"
            )

            # "Generate" next token
            current_token = torch.randn(batch_size, 1, d_model)

    print(f"✓ Only computed K,V for new tokens, reused cache\n")


# ============================================
# PRACTICAL MEMORY COMPARISON
# ============================================


def memory_comparison():
    """Show memory savings from KV-cache"""
    print("=== MEMORY SAVINGS ANALYSIS ===")

    d_model = 4096  # GPT-3 size
    num_heads = 32
    num_layers = 32
    batch_size = 1
    seq_len = 2048

    # Bytes per parameter (float16)
    bytes_per_param = 2

    # Memory for one layer's KV cache
    memory_per_layer = 2 * seq_len * d_model * bytes_per_param  # 2 for K and V
    total_kv_memory = memory_per_layer * num_layers / (1024**2)  # MB

    print(f"Model: {num_layers} layers, {d_model} dims, {num_heads} heads")
    print(f"Sequence length: {seq_len} tokens")
    print(f"KV-cache memory: {total_kv_memory:.1f} MB")

    # Speed comparison
    print(f"\nWithout cache: O(n²) where n = sequence length")
    print(f"With cache: O(n) - only process new token")
    print(f"At 2048 tokens, cache is ~2048x faster per new token!")


if __name__ == "__main__":
    training_example()
    inference_without_cache()
    inference_with_cache()
    memory_comparison()
