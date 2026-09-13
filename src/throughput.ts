import { flopsPerToken } from './flops.js';
import { peakFlops, resolveGpu, type ComputePrecision, type GpuSpec } from './gpus.js';
import { assertPositive, assertPositiveInt, type ModelSpec } from './types.js';

export interface MfuInput {
  /** Measured training throughput across all devices, in tokens per second. */
  tokensPerSec: number;
  gpu: string | GpuSpec;
  /** Number of devices the throughput was measured on. Default 1. */
  nGpus?: number;
  model: ModelSpec;
  seqLen?: number;
  /** Which peak to divide by. Default 'bf16'. Use 'fp8' only if the matmuls actually ran in FP8. */
  precision?: ComputePrecision;
}

export interface MfuResult {
  /** Model FLOPs utilization, 0 to 1. */
  mfu: number;
  flopsPerToken: number;
  /** Achieved model FLOP/s across all devices. */
  achievedFlops: number;
  achievedTflopsPerGpu: number;
  /** Peak FLOP/s across all devices at the chosen precision. */
  peakFlops: number;
  peakTflopsPerGpu: number;
  tokensPerSec: number;
  nGpus: number;
  gpu: GpuSpec;
  precision: ComputePrecision;
}

/**
 * Model FLOPs utilization (PaLM, Appendix B): achieved model FLOP/s divided by peak dense FLOP/s.
 * Model FLOPs count only the math the model needs (6N + attention), not recomputation, so a run
 * with activation checkpointing shows a lower MFU than its hardware utilization.
 */
export function mfu(input: MfuInput): MfuResult {
  assertPositive(input.tokensPerSec, 'tokensPerSec');
  const nGpus = input.nGpus ?? 1;
  assertPositiveInt(nGpus, 'nGpus');
  const precision = input.precision ?? 'bf16';
  const gpu = resolveGpu(input.gpu);
  const perToken = flopsPerToken(input.model, { seqLen: input.seqLen }).total;
  const achievedFlops = input.tokensPerSec * perToken;
  const peakPerGpu = peakFlops(gpu, precision);
  const peak = peakPerGpu * nGpus;
  return {
    mfu: achievedFlops / peak,
    flopsPerToken: perToken,
    achievedFlops,
    achievedTflopsPerGpu: achievedFlops / nGpus / 1e12,
    peakFlops: peak,
    peakTflopsPerGpu: peakPerGpu / 1e12,
    tokensPerSec: input.tokensPerSec,
    nGpus,
    gpu,
    precision,
  };
}

export interface ThroughputInput {
  gpu: string | GpuSpec;
  nGpus?: number;
  /** Target MFU, 0 to 1. Default 0.4, a realistic figure for a well-tuned FSDP run at 1B to 10B on H100s. */
  mfu?: number;
  model: ModelSpec;
  seqLen?: number;
  precision?: ComputePrecision;
}

export interface ThroughputResult {
  /** Expected tokens per second across all devices. */
  tokensPerSec: number;
  tokensPerSecPerGpu: number;
  flopsPerToken: number;
  mfu: number;
  peakFlops: number;
  nGpus: number;
  gpu: GpuSpec;
  precision: ComputePrecision;
}

/** The throughput a run should reach at a given MFU. Compare it with what you measure. */
export function expectedThroughput(input: ThroughputInput): ThroughputResult {
  const nGpus = input.nGpus ?? 1;
  assertPositiveInt(nGpus, 'nGpus');
  const target = input.mfu ?? 0.4;
  if (!(target > 0 && target <= 1)) throw new Error(`mfu must be between 0 and 1, got ${String(target)}`);
  const precision = input.precision ?? 'bf16';
  const gpu = resolveGpu(input.gpu);
  const perToken = flopsPerToken(input.model, { seqLen: input.seqLen }).total;
  const peak = peakFlops(gpu, precision) * nGpus;
  const tokensPerSec = (peak * target) / perToken;
  return {
    tokensPerSec,
    tokensPerSecPerGpu: tokensPerSec / nGpus,
    flopsPerToken: perToken,
    mfu: target,
    peakFlops: peak,
    nGpus,
    gpu,
    precision,
  };
}
