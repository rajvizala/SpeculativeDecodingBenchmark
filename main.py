#!/usr/bin/env python3
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The future of artificial intelligence is",
    "In the year 2050, humanity will",
    "The quantum computer revolutionized",
    "Once upon a time in a distant galaxy",
    "The scientist discovered that",
    "Climate change affects our planet by",
    "The neural network architecture called",
    "During the Renaissance, artists",
    "The economic theory suggests that",
    "Space exploration has always been",
]


def load_models():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    draft = AutoModelForCausalLM.from_pretrained("distilgpt2")
    draft.to("cuda:0")
    draft.eval()

    target = AutoModelForCausalLM.from_pretrained("gpt2")
    target.to("cuda:1")
    target.eval()

    return draft, target, tokenizer


def greedy_generate(target, tokenizer, prompt, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:1")
    for _ in range(max_new):
        out = target(ids)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def speculative_generate(draft, target, tokenizer, prompt, max_new=100, k=4):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:1")
    generated = 0

    while generated < max_new:
        actual_k = min(k, max_new - generated)
        if actual_k <= 0:
            break

        draft_ids = ids.to("cuda:0")
        draft_tokens = []
        with torch.no_grad():
            for _ in range(actual_k):
                out = draft(draft_ids)
                nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(nxt)
                draft_ids = torch.cat([draft_ids, nxt], dim=1)

        draft_seq = torch.cat(draft_tokens, dim=1).to("cuda:1")
        full = torch.cat([ids, draft_seq], dim=1)

        with torch.no_grad():
            logits = target(full).logits

        orig_len = ids.shape[1]
        accepted = 0
        for i in range(actual_k):
            pred = torch.argmax(logits[:, orig_len + i - 1, :], dim=-1)
            if pred.item() == draft_seq[0, i].item():
                accepted += 1
            else:
                break

        if accepted > 0:
            ids = torch.cat([ids, draft_seq[:, :accepted]], dim=1)
            generated += accepted

        if accepted < actual_k:
            nxt = torch.argmax(logits[:, orig_len + accepted - 1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            generated += 1
        else:
            if max_new - generated > 0:
                bonus = torch.argmax(logits[:, orig_len + actual_k - 1, :], dim=-1, keepdim=True)
                ids = torch.cat([ids, bonus], dim=1)
                generated += 1

    return ids[:, : tokenizer(prompt, return_tensors="pt").input_ids.shape[1] + max_new]


def benchmark(name, gen_fn, prompts, tokenizer, max_new=100, warmup=2):
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")

    for _ in range(warmup):
        gen_fn(prompts[0])
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")

    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        t0 = time.perf_counter()
        out = gen_fn(prompt)
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        t1 = time.perf_counter()

        inp_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        new_tok = out.shape[1] - inp_len
        dt = t1 - t0
        total_tokens += new_tok
        total_time += dt
        print(f"  Prompt {i+1:2d}: {new_tok:3d} tokens in {dt:6.3f}s ({new_tok/dt:6.1f} tok/s)")

    tok_s = total_tokens / total_time
    lat = total_time / total_tokens
    print(f"\n  Summary:")
    print(f"    Throughput:  {tok_s:.2f} tok/s")
    print(f"    Latency:     {lat*1000:.2f} ms/token")
    print(f"    Total time:  {total_time:.3f}s")

    return {"method": name, "tok_s": tok_s, "lat_ms": lat * 1000, "time": total_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}, CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs. On Kaggle: Settings -> Accelerator -> GPU T4 x2")
        return

    draft, target, tokenizer = load_models()

    baseline = benchmark(
        "Greedy (baseline)",
        lambda p: greedy_generate(target, tokenizer, p, args.max_new_tokens),
        PROMPTS, tokenizer, args.max_new_tokens, args.warmup,
    )

    results = [baseline]
    for k in args.k_values:
        res = benchmark(
            f"Speculative (k={k})",
            lambda p, k=k: speculative_generate(draft, target, tokenizer, p, args.max_new_tokens, k),
            PROMPTS, tokenizer, args.max_new_tokens, args.warmup,
        )
        results.append(res)

    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'tok/s':>10} {'Latency(ms)':>12} {'Total(s)':>10} {'Speedup':>8}")
    print("-" * 70)
    for r in results:
        sp = r["tok_s"] / baseline["tok_s"]
        print(f"{r['method']:<25} {r['tok_s']:>10.2f} {r['lat_ms']:>12.2f} {r['time']:>10.2f} {sp:>8.2f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
