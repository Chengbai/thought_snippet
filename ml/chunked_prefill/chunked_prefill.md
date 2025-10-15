# Thought: Chunked-prefill
---
**Chunked prefill** is a technique for processing **long prompt sequences** in large language models (LLMs) by breaking the prefill phase into smaller chunks rather than processing the entire prompt at once.

## Background
---
In LLM inference, there are two phases:
- Prefill: Processing the input prompt to generate the KV cache (compute-bound)
- Decode: Generating tokens one at a time (memory-bound)

## The problem
Long prompts create issues:
- Memory spikes: Processing a 10K token prompt at once requires massive memory for intermediate activations
- Latency: Users wait a long time before seeing the first token
- Batch inefficiency: Can't easily mix prefill (long sequences) and decode (single tokens) in the same batch

## Chunked prefill solution
Instead of processing all tokens at once, split the prompt into chunks (e.g., 512 or 1024 tokens) and process iteratively:
```
Prompt: [10,000 tokens]
→ Chunk 1: [tokens 0-512]    → generate KV cache
→ Chunk 2: [tokens 512-1024]  → append to KV cache  
→ Chunk 3: [tokens 1024-1536] → append to KV cache
...
→ Start decode phase
```

## Benefits
- **Lower memory footprint**: Smaller activation memory per step
- **Better batching**: Mix prefill chunks with decode requests in the same batch (similar computational intensity)
- **Faster time-to-first-token (TTFT)**: Can start generating sooner in some implementations
- I**mproved throughput**: Better GPU utilization by mixing workloads

This technique is used in **inference systems like vLLM, TensorRT-LLM, and other high-performance serving frameworks**.


## Example
Example implementation of chunked prefill with KV cache generation in PyTorch:
```
ml/chunked_prefill/chunked_prefill.py
```
### Key points:
- Chunk processing: The prompt is split into chunks (e.g., 512 tokens) and processed sequentially
- KV cache accumulation: Each chunk's keys and values are concatenated to the existing cache
- Memory efficiency: Only one chunk's activations are in memory at a time
- Seamless decode: After prefill, the cache is ready for autoregressive generation

This approach is especially useful for **serving systems handling variable-length prompts efficiently**!