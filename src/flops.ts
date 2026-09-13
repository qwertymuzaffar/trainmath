import { transformerParams } from './params.js';
import { assertPositive, isArch, type ModelSpec } from './types.js';

export interface FlopsPerToken {
  /** dense + attention. */
  total: number;
  /** 6 x parameters that take part in matrix multiplies (blocks + output head). */
  dense: number;
  /** 12 x nLayers x dModel x seqLen: the attention-score and attention-value products, forward and backward. Zero when the shape is unknown. */
  attention: number;
  /** The parameter count behind the dense term. */
  paramsCounted: number;
  /** Sequence length used for the attention term, if known. */
  seqLen: number | undefined;
}

export interface FlopsOptions {
  /** Overrides the model's sequence length for the attention term. */
  seqLen?: number;
}

/**
 * Training FLOPs per token. Dense term 6N (forward 2N, backward 4N) plus the attention term
 * 12 x L x d x T from PaLM, Appendix B. For an Arch, N counts the blocks and the output head
 * (the embedding lookup is free; the head matmul is not, tied or untied). For a ParamsSpec,
 * N is the given parameter count.
 */
export function flopsPerToken(model: ModelSpec, opts: FlopsOptions = {}): FlopsPerToken {
  if (isArch(model)) {
    const p = transformerParams(model);
    const seqLen = opts.seqLen ?? p.arch.seqLen;
    assertPositive(seqLen, 'seqLen');
    const paramsCounted = p.nonEmbedding + p.arch.vocab * p.arch.dModel;
    const dense = 6 * paramsCounted;
    const attention = 12 * p.arch.nLayers * p.arch.dModel * seqLen;
    return { total: dense + attention, dense, attention, paramsCounted, seqLen };
  }
  assertPositive(model.params, 'params');
  const seqLen = opts.seqLen ?? model.seqLen;
  const dense = 6 * model.params;
  const hasShape = model.nLayers !== undefined && model.dModel !== undefined && seqLen !== undefined;
  const attention = hasShape ? 12 * (model.nLayers as number) * (model.dModel as number) * (seqLen as number) : 0;
  return { total: dense + attention, dense, attention, paramsCounted: model.params, seqLen: hasShape ? seqLen : undefined };
}

/** Total training FLOPs for a run: FLOPs per token x tokens. */
export function trainingFlops(model: ModelSpec, tokens: number, opts: FlopsOptions = {}): number {
  assertPositive(tokens, 'tokens');
  return flopsPerToken(model, opts).total * tokens;
}

/** Parameter count for either kind of model spec. */
export function modelParams(model: ModelSpec): number {
  return isArch(model) ? transformerParams(model).total : model.params;
}
