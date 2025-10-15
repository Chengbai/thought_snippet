# Thought: KV-cache
---

**KV-cache** (Key-Value cache) is a memory optimization technique used in transformer models to avoid redundant computations during autoregressive generation.

## How It Works
In transformers with self-attention, each token attends to all previous tokens. Without KV-cache, you'd recompute the keys (K) and values (V) for all previous tokens at every generation step. KV-cache stores these computed K and V tensors so they can be reused.

**The key insight**: During generation, past tokens' K and V projections never change - only the query (Q) for the new token is needed.

## Training vs Inference
KV-cache is **NOT used during training - only during inference**:
- **Training**: All tokens are known upfront (teacher forcing). You process the entire sequence in parallel with causal masking. No need for caching since it's a single forward pass.
- **Inference**: Tokens are generated one at a time autoregressively. KV-cache avoids recomputing K,V for all previous tokens at each step, reducing computation from O(n²) to O(n).

| Aspect          | **Training**                         | **Inference**                                   |
| --------------- | ------------------------------------ | ----------------------------------------------- |
| **Input**       | Whole sequence available at once     | One token at a time (auto-regressive)           |
| **Computation** | Compute attention over full sequence | Compute attention incrementally                 |
| **KV Cache**    | Not used — recomputed every batch    | Used — to avoid recomputing for previous tokens |
| **Goal**        | Learn weights                        | Generate efficiently                            |
