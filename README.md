# trainmath

[![npm version](https://img.shields.io/npm/v/trainmath)](https://www.npmjs.com/package/trainmath) [![CI](https://github.com/qwertymuzaffar/trainmath/actions/workflows/ci.yml/badge.svg)](https://github.com/qwertymuzaffar/trainmath/actions/workflows/ci.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Plan and sanity-check LLM pretraining runs - **zero dependencies**, TypeScript, runs in Node, browsers and edge runtimes.

It answers the questions that come up before and during every training run:

- **What throughput should I see?** Expected tokens/sec for a model on N devices at a target MFU, and the MFU behind a number you measured.
- **Does it fit?** Per-device memory for weights, gradients, master weights, optimizer states, activations and logits under DDP, ZeRO-1/2/3 or FSDP.
- **What will it cost?** Hours, device-hours and dollars for a token budget, with Chinchilla budgets as the default.
- **Why is my MFU low?** An ordered list of likely causes from the flags you pass.

Every number comes from a published formula, listed below with its source, and the tests check them against GPT-2 124M, GPT-3 175B, Llama 3 8B, PaLM 540B and Chinchilla 70B.

## Install

```bash
npm install trainmath
```

Node 18 or newer. ESM and CommonJS builds, types included. Releases are published from GitHub Actions with npm provenance.

## Quick start

```ts
import { trainingPlan, mfu, memoryPerGpu, planToMarkdown } from 'trainmath';

// How long and how much: a 1B model at Chinchilla tokens on 8 x H100 at 40% MFU.
const plan = trainingPlan({ model: { params: 1e9 }, gpu: 'h100', nGpus: 8, mfu: 0.4, pricePerGpuHour: 2.5 });
plan.tokens;    // 20e9
plan.hours;     // 10.5
plan.gpuHours;  // 84.3
plan.cost;      // 211

// What does my measured throughput mean?
const m = mfu({ tokensPerSec: 420_000, gpu: 'h100', nGpus: 8, model: { params: 1e9, nLayers: 16, dModel: 2048, seqLen: 2048 } });
m.mfu;          // 0.36

// Does Llama 3 8B fit on 8 x A100 80GB with FSDP at sequence length 8192?
const mem = memoryPerGpu({ model: models['llama3-8b'], gpu: 'a100-80', nGpus: 8, sharding: 'fsdp', microBatch: 1 });
mem.perGpuGiB.total;  // about 55
mem.fits;             // true

console.log(planToMarkdown([plan]));
```

## CLI

```bash
npx trainmath table --sizes 1B,2B,4B,14B --gpu h100 --gpus 8 --mfu 0.4 --price 2.5
```

```
## Training plan: 8 x H100 SXM 80GB at 40% MFU, $2.50 per device-hour

| Params | Tokens | FLOPs   | Tokens/sec | Wall-clock | Device-hours | Cost    |
| ---    | ---    | ---     | ---        | ---        | ---          | ---     |
| 1B     | 20B    | 1.2e20  | 527k       | 10.5 h     | 84.3         | $211    |
| 2B     | 40B    | 4.8e20  | 264k       | 42.1 h     | 337          | $843    |
| 4B     | 80B    | 1.92e21 | 132k       | 7 d        | 1.35k        | $3,370  |
| 14B    | 280B   | 2.35e22 | 37.7k      | 86 d       | 16.5k        | $41,287 |
```

Other commands:

```bash
trainmath params --model llama3-8b                       # exact parameter count and breakdown
trainmath flops --model gpt3-175b --tokens 300B           # FLOPs per token and for the run
trainmath mfu --params 1B --tokens-per-sec 420k --gpu h100 --gpus 8
trainmath throughput --params 1B --gpu h100 --gpus 8 --mfu 0.4
trainmath chinchilla --params 1B                          # tokens, FLOPs and predicted loss
trainmath memory --model llama3-8b --gpu a100-80 --gpus 8 --sharding fsdp --micro-batch 1
trainmath plan --params 1B --gpu h100 --gpus 8 --mfu 0.4 --price 2.5
trainmath gap --mfu 0.22 --gpus 8 --sharding fsdp --precision fp32 --no-compile
trainmath suggest --params 1B                             # a plausible shape for "about 1B"
trainmath gpus                                            # the built-in accelerator table
trainmath models                                          # the built-in model presets
```

A model is `--model <preset>`, `--params 1B` (with optional `--layers`, `--d-model`, `--heads`, `--vocab`, `--seq-len` for the attention and activation terms), or a full shape `--vocab --d-model --layers --heads` with `--kv-heads`, `--d-ff`, `--mlp gelu|swiglu`, `--untied`, `--bias`, `--learned-positions`. Numbers accept `1e9`, `1B`, `124M`, `20B`, `2.5k`. Add `--json` for machine-readable output.

## What it computes

| Function | Computes | Formula and source |
| --- | --- | --- |
| `transformerParams(arch)` | Exact parameter count from the shape: attention, MLP, norms, embeddings, positions, output head. | Matrix sizes. Checked against GPT-2 124M (124,439,808), GPT-3 175B and Llama 3 8B (8,030,261,248). |
| `flopsPerToken(model)` | Training FLOPs per token, dense and attention terms. | `6N + 12 L d T` per token, where N counts the blocks and the output head. PaLM, Appendix B. |
| `trainingFlops(model, tokens)` | FLOPs for a whole run. | FLOPs per token x tokens. Gives 3.14e23 for GPT-3 175B on 300B tokens, as in the paper. |
| `mfu({ tokensPerSec, gpu, nGpus, model })` | Model FLOPs utilization from a measured throughput. | `tokens/sec x FLOPs per token / (devices x peak dense FLOP/s)`. PaLM, Appendix B. Reproduces PaLM's 46% on 6144 TPU v4. |
| `expectedThroughput({ gpu, nGpus, mfu, model })` | Tokens/sec you should see at a target MFU, total and per device. | The same formula solved for tokens/sec. |
| `chinchilla({ params })`, `({ tokens })`, `({ flops })` | Compute-optimal tokens for a model size, the reverse, or the split of a FLOP budget. Predicted loss. | `D = 20 N`; `C = 6 N D`; `L(N, D) = E + A / N^0.34 + B / D^0.28` with the published fit. Hoffmann et al. 2022. |
| `memoryPerGpu({ model, gpu, nGpus, sharding, precision, optimizer, microBatch, seqLen, attention, activationCheckpointing })` | Per-device GiB by part, and whether it fits. | 16 bytes per parameter for AdamW mixed precision (bf16 weights + bf16 grads + fp32 master + two fp32 moments), sharded by ZeRO stage. Rajbhandari et al. 2020. Activations per block `s b h (34 + 5 a s / h)` bytes with materialized scores, `34 s b h` with flash attention or selective checkpointing, `2 s b h` with full checkpointing. Korthikanti et al. 2022. Logits `s b V x 4` bytes. |
| `trainingPlan({ model, tokens, gpu, nGpus, mfu, pricePerGpuHour })` | Seconds, hours, device-hours and dollars. | `hours = tokens / tokens per sec / 3600`; `cost = device-hours x price`. |
| `planTable(sizes, options)` | One plan per model size. | Same. |
| `explainGap({ mfu, ...flags })` | A band (poor, low, ok, good, suspicious) and an ordered list of likely causes and checks. | Heuristics from published H100 and A100 runs (llm.c, TorchTitan, the nanoGPT speedrun, PaLM). Not a formula. |
| `suggestArch(params)` | A plausible dense shape for a parameter count, so a bare "1B" gets attention FLOPs and activation memory. | `N = 12 L d^2` with `d = 128 L` and head size 128. Kaplan et al. 2020 found the aspect ratio flat over a wide range. |
| `gpus`, `resolveGpu(id)`, `peakFlops(gpu, precision)` | Peak dense TFLOPS by precision, memory and bandwidth for H100 SXM/PCIe/NVL, H200, A100 80/40, B200, L40S, RTX 4090, MI300X, TPU v4/v5e/v5p/v6e. Aliases like `h100`, `a100`. | Vendor data sheets, dense numbers only (sparsity figures halved), checked September 2026. Pass your own `GpuSpec` to override. |
| `models`, `resolveModel(name)` | Presets: `gpt2-124m`, `gpt2-350m`, `gpt2-774m`, `gpt2-1.6b`, `gpt3-175b`, `llama3-8b`, `llama3-70b`. | Public model cards. |
| `*ToMarkdown(...)` | GitHub-flavored Markdown for every result, and `markdownTable(headers, rows)`. | |
| `parseNumber`, `formatCount`, `formatSci`, `formatDuration`, `formatMoney`, `formatGiB` | `"20B"` to `2e10` and back. | |

## The formulas, in plain words

**Parameters.** A transformer block has four attention matrices (`d x d` for queries and outputs, `d x kv` for keys and values) and two or three MLP matrices (`d x d_ff`, three for SwiGLU). With `d_ff = 4 d` and full attention that is about `12 d^2` per block, so a model is roughly `12 L d^2` plus `V d` for the embeddings. The library counts every matrix, bias and norm exactly instead of using the approximation.

**FLOPs.** A forward pass costs about 2 FLOPs per parameter per token (one multiply, one add), and the backward pass costs twice that, so training is `6 N` per token. Attention adds `12 L d T` per token for the score and value products, which is a few percent at sequence length 2048 and much more at 32k. The embedding lookup is free; the output head is a real matmul and is counted whether or not it shares weights with the embeddings.

**MFU.** Model FLOPs utilization divides the FLOPs the model needs by the FLOPs the hardware could do at its dense peak. It ignores recomputation, so a run with activation checkpointing shows a lower MFU than its hardware utilization. Dense transformers land between 30% and 55% on H100 class hardware; PaLM reported 46% on TPU v4. Numbers above 65% usually mean the peak or the FLOPs were counted wrong.

**Chinchilla.** With a fixed compute budget, the best trade-off between model size and training tokens is about 20 tokens per parameter. That is the budget for a fair comparison between two training methods at the same size. Production models train far past it on purpose, because a smaller model that saw more tokens is cheaper to serve.

**Memory.** Mixed-precision AdamW keeps bf16 weights and gradients (2 + 2 bytes), fp32 master weights (4) and two fp32 moments (8): 16 bytes per parameter. ZeRO-1 shards the optimizer states across data-parallel devices, ZeRO-2 also the gradients, ZeRO-3 and FSDP also the weights. Activations depend on the batch shape, not the parameter count, and dominate at long sequence lengths; flash attention removes the `a s^2` term, and full checkpointing keeps only block inputs.

## Reading an MFU

| MFU | Reads as | Usually |
| --- | --- | --- |
| above 65% | suspicious | Sparsity peak used, FLOPs overcounted, or tokens/sec measured over a few warm steps. |
| 45% to 65% | good | Well tuned. Further gains need kernel or communication work. |
| 30% to 45% | ok | A working FSDP run at 1B to 10B. 10% to 20% usually left. |
| 15% to 30% | low | fp32, materialized attention, tiny micro-batch, no compiler, or an input pipeline stall. |
| below 15% | poor | The devices are mostly waiting. Look at the step trace and the loader first. |

`explainGap` turns the flags you pass (precision, attention kernel, compiler, micro-batch, checkpointing, sharding, multi-node) into an ordered list of causes and the measurements to take next.

## Units

Memory is in GiB (2^30 bytes), the unit `nvidia-smi` shows, so a 1B-parameter AdamW model is 16e9 bytes = 14.9 GiB. Peaks are dense TFLOPS: NVIDIA data sheets lead with the sparsity figure, which is double. Prices are whatever you pass per device-hour.

## Browser demo

`demo/index.html` is a single page with sliders for model size, hardware, MFU and price that shows the plan, the memory breakdown and the size table live. It loads the library's ESM build and nothing else.

```bash
npm run build && npm run build:demo   # assembles _site/
npx serve _site                        # or any static server
```

## Limitations

- Dense decoder-only transformers. No mixture-of-experts, no encoder-decoder, no multimodal towers.
- The FLOPs and memory formulas assume standard attention blocks. Sliding-window attention, MLA and other variants change the attention term.
- Memory under tensor or pipeline parallelism is not modeled, only data-parallel sharding. Multi-node communication is not modeled at all; the MFU you target has to include it.
- The activation formula is the Megatron accounting for bf16; frameworks differ in what they keep, chunk or recompute, so treat it as plus or minus 20%.
- The Chinchilla loss fit was made on Chinchilla's data and tokenizer. Use it to compare two budgets, not to predict your loss.
- Peak numbers and prices go stale. The table says when it was checked; pass your own `GpuSpec` when it matters.
- `explainGap` is a checklist, not a profiler. The profiler trace is the ground truth.

## Alternatives

- [EleutherAI cookbook](https://github.com/EleutherAI/cookbook) has Python scripts for parameter counts, FLOPs and memory that this package mirrors, with the same sources.
- Hugging Face's `accelerate estimate-memory` estimates inference and training memory for a Hub model from its config.
- Web calculators for VRAM and training cost exist; trainmath is the same math as a typed, tested package you can call from a script, a CI check or a dashboard.

## License

MIT
