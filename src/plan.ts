import { CHINCHILLA_RATIO, chinchillaTokens } from './chinchilla.js';
import { flopsPerToken, modelParams } from './flops.js';
import { resolveGpu, type ComputePrecision, type GpuSpec } from './gpus.js';
import { transformerParams, type ParamCount } from './params.js';
import { expectedThroughput } from './throughput.js';
import { assertPositive, type Arch, type ModelSpec } from './types.js';

export interface PlanInput {
  model: ModelSpec;
  /** Training tokens. Defaults to the Chinchilla budget, tokensPerParam x params. */
  tokens?: number;
  /** Used when tokens is not given. Default 20. */
  tokensPerParam?: number;
  gpu: string | GpuSpec;
  nGpus?: number;
  /** Target MFU, 0 to 1. Default 0.4. */
  mfu?: number;
  /** Price of one device-hour in dollars. Cost is left undefined without it. */
  pricePerGpuHour?: number;
  seqLen?: number;
  precision?: ComputePrecision;
}

export interface Plan {
  params: number;
  tokens: number;
  tokensPerParam: number;
  flopsPerToken: number;
  /** Total training FLOPs. */
  flops: number;
  tokensPerSec: number;
  tokensPerSecPerGpu: number;
  seconds: number;
  hours: number;
  days: number;
  gpuHours: number;
  cost: number | undefined;
  pricePerGpuHour: number | undefined;
  mfu: number;
  nGpus: number;
  gpu: GpuSpec;
  precision: ComputePrecision;
}

/** Hours, device-hours and dollars for a run at a target MFU. */
export function trainingPlan(input: PlanInput): Plan {
  const params = modelParams(input.model);
  const ratio = input.tokensPerParam ?? CHINCHILLA_RATIO;
  const tokens = input.tokens ?? chinchillaTokens(params, ratio);
  assertPositive(tokens, 'tokens');
  if (input.pricePerGpuHour !== undefined) assertPositive(input.pricePerGpuHour, 'pricePerGpuHour');
  const nGpus = input.nGpus ?? 1;
  const thr = expectedThroughput({
    gpu: input.gpu,
    nGpus,
    mfu: input.mfu,
    model: input.model,
    seqLen: input.seqLen,
    precision: input.precision,
  });
  const perToken = flopsPerToken(input.model, { seqLen: input.seqLen }).total;
  const seconds = tokens / thr.tokensPerSec;
  const hours = seconds / 3600;
  const gpuHours = hours * nGpus;
  return {
    params,
    tokens,
    tokensPerParam: tokens / params,
    flopsPerToken: perToken,
    flops: perToken * tokens,
    tokensPerSec: thr.tokensPerSec,
    tokensPerSecPerGpu: thr.tokensPerSecPerGpu,
    seconds,
    hours,
    days: hours / 24,
    gpuHours,
    cost: input.pricePerGpuHour === undefined ? undefined : gpuHours * input.pricePerGpuHour,
    pricePerGpuHour: input.pricePerGpuHour,
    mfu: thr.mfu,
    nGpus,
    gpu: resolveGpu(input.gpu),
    precision: thr.precision,
  };
}

/** One plan per model size (a parameter count or a full model spec), same hardware and budget rule for all. */
export function planTable(models: Array<number | ModelSpec>, input: Omit<PlanInput, 'model' | 'tokens'>): Plan[] {
  return models.map((m) => trainingPlan({ ...input, model: typeof m === 'number' ? { params: m } : m }));
}

export interface SuggestArchOptions {
  vocab?: number;
  seqLen?: number;
  /** Head size. Default 128. */
  dHead?: number;
  /** Target dModel / nLayers. Default 128, in the range Kaplan et al. found to be flat. */
  aspect?: number;
  mlp?: 'gelu' | 'swiglu';
  tiedEmbeddings?: boolean;
}

export interface SuggestedArch {
  arch: Arch;
  count: ParamCount;
  /** How far the non-embedding count is from the target, as a fraction. */
  error: number;
}

/**
 * A plausible dense shape for a target non-embedding parameter count: N is about 12 L d^2 and d is about
 * aspect x L, so L = cbrt(N / (12 aspect^2)). The result is a starting point for FLOPs and memory
 * estimates when you only know "about 1B", not a recommendation for a real model.
 */
export function suggestArch(targetParams: number, opts: SuggestArchOptions = {}): SuggestedArch {
  assertPositive(targetParams, 'targetParams');
  const vocab = opts.vocab ?? 32000;
  const seqLen = opts.seqLen ?? 2048;
  const dHead = opts.dHead ?? 128;
  const aspect = opts.aspect ?? 128;
  const mlp = opts.mlp ?? 'gelu';
  const tiedEmbeddings = opts.tiedEmbeddings ?? true;
  const l0 = Math.cbrt(targetParams / (12 * aspect * aspect));
  const candidates = new Set([Math.max(1, Math.floor(l0)), Math.max(1, Math.ceil(l0))]);
  let best: SuggestedArch | undefined;
  for (const nLayers of candidates) {
    const dModel = Math.max(dHead, Math.round((aspect * nLayers) / dHead) * dHead);
    const arch: Arch = { vocab, dModel, nLayers, nHeads: dModel / dHead, seqLen, mlp, tiedEmbeddings };
    const count = transformerParams(arch);
    const error = Math.abs(count.nonEmbedding - targetParams) / targetParams;
    if (!best || error < best.error) best = { arch, count, error };
  }
  return best as SuggestedArch;
}
