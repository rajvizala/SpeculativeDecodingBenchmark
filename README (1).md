# Speculative Decoding Across Two GPUs

Implementation of simplified speculative decoding with **distilgpt2** (draft) on GPU0 and **gpt2** (target) on GPU1.

## Quick Start (Kaggle)

1. Create a Kaggle Notebook
2. **Settings → Accelerator → GPU T4 x2**
3. Upload `main.py` and run:
   ```bash
   python main.py
   ```

## Results

Run on Kaggle with 2× Tesla T4 GPUs, 10 prompts, 100 tokens each.

| Method | tok/s | Latency (ms/token) | Total Time (s) | Speedup |
|--------|-------|-------------------|----------------|---------|
| Greedy (baseline) | 102.47 | 9.76 | 9.76 | 1.00× |
| Speculative k=2 | 131.38 | 7.61 | 7.61 | 1.28× |
| Speculative k=4 | 122.73 | 8.15 | 8.15 | 1.20× |
| Speculative k=8 | 104.20 | 9.60 | 9.60 | 1.02× |

k=2 gives the best speedup. k=8 performs nearly the same as baseline because the draft model (distilgpt2) is only moderately smaller than the target, so generating 8 draft tokens sequentially costs almost as much as running the target model directly.

## Design Decisions

- **Custom loops** instead of `model.generate()` — fair apples-to-apples comparison
- **Single target forward pass** per iteration — verifies all k draft tokens at once
- **Greedy acceptance** — accept draft token if `argmax(target_logits) == draft_token`; reject at first mismatch and use target's prediction
- **Bonus token** — if all k draft tokens are accepted, emit one extra token from the target logits already computed
- **No KV-cache** — keeps the code focused on the speculative decoding logic; both baseline and speculative are equally affected

## Bottlenecks

1. **Draft model is not much smaller** — distilgpt2 (82M) vs gpt2 (124M). On T4, inference is memory-bandwidth bound, so the speedup from halving layers is modest.
2. **No KV-cache** — each draft token requires a full forward pass recomputing attention over the entire prefix. At k=8, 8 sequential draft passes nearly cancel out the benefit.
3. **Batch size = 1** — T4 Tensor Cores are underutilized. Larger batches would better saturate memory bandwidth.
4. **Cross-GPU transfer** — small (k integers) but non-zero overhead per iteration.

## Next Optimizations

1. **KV-cache** — cache keys/values so each forward pass only computes the new token. Biggest impact.
2. **Larger draft/target gap** — use a much smaller draft model (e.g., 2-layer custom model) so draft generation is 5–10× faster.
3. **Batching** — process multiple prompts concurrently to improve GPU utilization.
4. **CUDA graphs / torch.compile** — reduce CPU launch overhead for the repetitive loop.

## Requirements

```
torch>=2.0
transformers>=4.30
```

## Reference

Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192.
