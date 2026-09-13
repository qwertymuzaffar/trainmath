import type { Attention, Sharding, TrainingPrecision } from './memory.js';

export type MfuBand = 'suspicious' | 'good' | 'ok' | 'low' | 'poor';

export interface GapInput {
  /** Measured MFU, 0 to 1 (from mfu()). */
  mfu: number;
  nGpus?: number;
  sharding?: Sharding;
  precision?: TrainingPrecision;
  /** torch.compile or an equivalent graph compiler in use. */
  compiled?: boolean;
  attention?: Attention | 'unknown';
  /** Sequences per device per micro-step. */
  microBatch?: number;
  /** Activation checkpointing in use. */
  activationCheckpointing?: boolean;
  /** Multi-node run. */
  multiNode?: boolean;
}

export interface GapReport {
  band: MfuBand;
  headline: string;
  /** Likely causes, most likely first, given the flags you passed. */
  causes: string[];
  /** What to measure next. */
  checks: string[];
}

/**
 * Heuristics for reading a measured MFU. The bands come from published dense-transformer runs on
 * H100 and A100 class hardware (llm.c, TorchTitan, the nanoGPT speedrun, PaLM). They are a starting
 * point for an audit, not a diagnosis.
 */
export function explainGap(input: GapInput): GapReport {
  const m = input.mfu;
  if (typeof m !== 'number' || !Number.isFinite(m) || m < 0) throw new Error(`mfu must be a non-negative fraction (0.4 for 40%), got ${String(m)}`);
  const causes: string[] = [];
  const checks: string[] = [];

  let band: MfuBand;
  let headline: string;
  if (m > 0.65) {
    band = 'suspicious';
    headline =
      m > 1
        ? 'Above 100%: physically impossible. The throughput, the FLOPs per token or the peak is wrong.'
        : 'Above what dense transformers reach on real hardware. Check the measurement before celebrating.';
    if (m > 1) causes.push('tokens/sec may be per device while nGpus is also set, or counted in sequences or batches instead of tokens.');
    causes.push('The FLOPs per token may be overcounted: make sure N excludes the input embedding lookup and that seqLen is the real one.');
    causes.push('The peak may be the sparsity number from the data sheet (2x the dense peak). trainmath uses dense peaks.');
    causes.push('tokens/sec may have been measured over a few warm steps instead of a whole interval with data loading and logging.');
    causes.push('If matmuls run in FP8, divide by the FP8 peak (precision: "fp8").');
  } else if (m >= 0.45) {
    band = 'good';
    headline = 'Well tuned. Large gains from here need kernel-level or communication work.';
  } else if (m >= 0.3) {
    band = 'ok';
    headline = 'Typical for a working FSDP run at 1B to 10B. There is usually 10 to 20% left.';
  } else if (m >= 0.15) {
    band = 'low';
    headline = 'Something systematic is wrong: precision, attention kernel, batch shape or the input pipeline.';
  } else {
    band = 'poor';
    headline = 'The GPUs are mostly waiting. Look at the input pipeline and the step trace first.';
  }

  if (band !== 'suspicious') {
    if (input.precision === 'fp32') {
      causes.push('fp32 matmuls run at the TF32 peak at best, about half the bf16 peak. Move to bf16 mixed precision.');
    }
    if (input.attention === 'standard') {
      causes.push('Materialized attention scores: use scaled_dot_product_attention or FlashAttention. Big win at long sequences.');
    } else if (input.attention === 'unknown' || input.attention === undefined) {
      checks.push('Confirm the attention kernel in a profiler trace (look for flash or mem-efficient SDPA kernels).');
    }
    if (input.compiled === false) {
      causes.push('No graph compiler: torch.compile usually adds 1.2x to 1.5x on small and mid-size models by fusing elementwise ops.');
    }
    if (input.microBatch !== undefined && input.microBatch <= 2) {
      causes.push('Tiny per-device micro-batch: matmuls are too small to fill the SMs. Raise it until memory is nearly full, then use gradient accumulation for the rest.');
    }
    if (input.activationCheckpointing) {
      causes.push('Activation checkpointing recomputes the forward pass: model FLOPs stay the same, so MFU drops by up to a third. Try selective checkpointing or a smaller micro-batch without it.');
    }
    if ((input.nGpus ?? 1) > 1) {
      if (input.sharding === 'fsdp' || input.sharding === 'zero3') {
        causes.push('FSDP/ZeRO-3 all-gathers every block: check that prefetching overlaps communication with compute and that blocks are wrapped at the right granularity.');
      } else {
        causes.push('Gradient all-reduce not overlapped with the backward pass: check bucket sizes and that the reducer runs during backward, not after.');
      }
      if (input.multiNode) {
        causes.push('Multi-node: inter-node bandwidth is far below NVLink. Check NCCL uses the fast fabric (RDMA, not TCP) and prefer hybrid sharding that keeps all-gathers inside a node.');
      }
    }
    causes.push('Input pipeline stalls: tokenization or decoding on the fly, too few loader workers, or a network filesystem. Pre-tokenize to memory-mapped shards.');
    causes.push('Host-side overhead between steps: Python-side logging, .item() calls that synchronize, or checkpointing too often.');
    checks.push('Profile 5 to 10 steps with torch.profiler and read the GPU timeline: gaps between kernels mean the host or the loader is the bottleneck.');
    checks.push('Measure tokens/sec over at least 100 steps after warmup, including data loading and logging.');
    checks.push('Run one step with synthetic data: if throughput jumps, the input pipeline is the problem.');
    checks.push('Compare per-device throughput at 1 GPU and at N GPUs: the ratio is your scaling efficiency.');
  } else {
    checks.push('Recompute MFU with the dense bf16 peak and FLOPs per token from trainmath, then re-measure over a long interval.');
  }

  return { band, headline, causes, checks };
}
