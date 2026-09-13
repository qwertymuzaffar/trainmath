export type { Arch, ModelSpec, ParamsSpec } from './types.js';
export { isArch } from './types.js';

export { normalizeArch, transformerParams } from './params.js';
export type { NormalizedArch, ParamBreakdown, ParamCount } from './params.js';

export { flopsPerToken, modelParams, trainingFlops } from './flops.js';
export type { FlopsOptions, FlopsPerToken } from './flops.js';

export { gpus, listGpus, peakFlops, resolveGpu } from './gpus.js';
export type { ComputePrecision, GpuSpec } from './gpus.js';

export { expectedThroughput, mfu } from './throughput.js';
export type { MfuInput, MfuResult, ThroughputInput, ThroughputResult } from './throughput.js';

export { CHINCHILLA_FIT, CHINCHILLA_RATIO, chinchilla, chinchillaLoss, chinchillaParams, chinchillaTokens, computeOptimal } from './chinchilla.js';
export type { ChinchillaFit, ChinchillaInput, ChinchillaResult } from './chinchilla.js';

export { memoryPerGpu } from './memory.js';
export type { Attention, BytesPerParam, Checkpointing, MemoryEstimate, MemoryInput, Optimizer, Sharding, TrainingPrecision } from './memory.js';

export { planTable, suggestArch, trainingPlan } from './plan.js';
export type { Plan, PlanInput, SuggestArchOptions, SuggestedArch } from './plan.js';

export { explainGap } from './gap.js';
export type { GapInput, GapReport, MfuBand } from './gap.js';

export { listModels, models, resolveModel } from './models.js';

export { chinchillaToMarkdown, gapToMarkdown, gpusToMarkdown, markdownTable, memoryToMarkdown, mfuToMarkdown, paramsToMarkdown, planToMarkdown, throughputToMarkdown } from './report.js';

export { GIB, formatCount, formatDuration, formatGiB, formatMoney, formatPercent, formatSci, parseNumber } from './format.js';
