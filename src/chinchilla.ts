import { assertPositive } from './types.js';

/** Parametric loss fit from Hoffmann et al. 2022 ("Training Compute-Optimal Large Language Models"), Approach 3. */
export interface ChinchillaFit {
  E: number;
  A: number;
  B: number;
  alpha: number;
  beta: number;
}

export const CHINCHILLA_FIT: Readonly<ChinchillaFit> = Object.freeze({ E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 });

/** Default tokens per parameter for a compute-optimal run. Hoffmann et al. found roughly 20. */
export const CHINCHILLA_RATIO = 20;

/** Compute-optimal training tokens for a parameter count: ratio x params. */
export function chinchillaTokens(params: number, ratio = CHINCHILLA_RATIO): number {
  assertPositive(params, 'params');
  assertPositive(ratio, 'ratio');
  return params * ratio;
}

/** The parameter count a token budget is compute-optimal for: tokens / ratio. */
export function chinchillaParams(tokens: number, ratio = CHINCHILLA_RATIO): number {
  assertPositive(tokens, 'tokens');
  assertPositive(ratio, 'ratio');
  return tokens / ratio;
}

/**
 * Predicted final training loss L(N, D) = E + A / N^alpha + B / D^beta.
 * The fit was made on the Chinchilla data and tokenizer, so treat the absolute value as a rough guide
 * and the differences between two (N, D) pairs as the useful part.
 */
export function chinchillaLoss(params: number, tokens: number, fit: ChinchillaFit = CHINCHILLA_FIT): number {
  assertPositive(params, 'params');
  assertPositive(tokens, 'tokens');
  return fit.E + fit.A / params ** fit.alpha + fit.B / tokens ** fit.beta;
}

/** Splits a FLOP budget C = 6 N D into the compute-optimal N and D with D = ratio x N. */
export function computeOptimal(flops: number, ratio = CHINCHILLA_RATIO): { params: number; tokens: number } {
  assertPositive(flops, 'flops');
  assertPositive(ratio, 'ratio');
  const params = Math.sqrt(flops / (6 * ratio));
  return { params, tokens: params * ratio };
}

export type ChinchillaInput = { params: number; tokens?: undefined; flops?: undefined } | { tokens: number; params?: undefined; flops?: undefined } | { flops: number; params?: undefined; tokens?: undefined };

export interface ChinchillaResult {
  params: number;
  tokens: number;
  /** 6 N D. */
  flops: number;
  ratio: number;
  /** Predicted loss from the Hoffmann fit. */
  loss: number;
  /** Which value was given. */
  from: 'params' | 'tokens' | 'flops';
}

/** One entry point for the three directions: give params, tokens or a FLOP budget. */
export function chinchilla(input: ChinchillaInput, opts: { ratio?: number; fit?: ChinchillaFit } = {}): ChinchillaResult {
  const ratio = opts.ratio ?? CHINCHILLA_RATIO;
  const fit = opts.fit ?? CHINCHILLA_FIT;
  let params: number;
  let tokens: number;
  let from: ChinchillaResult['from'];
  if (typeof input.params === 'number') {
    params = input.params;
    tokens = chinchillaTokens(params, ratio);
    from = 'params';
  } else if (typeof input.tokens === 'number') {
    tokens = input.tokens;
    params = chinchillaParams(tokens, ratio);
    from = 'tokens';
  } else if (typeof input.flops === 'number') {
    ({ params, tokens } = computeOptimal(input.flops, ratio));
    from = 'flops';
  } else {
    throw new Error('chinchilla needs one of params, tokens or flops');
  }
  return { params, tokens, flops: 6 * params * tokens, ratio, loss: chinchillaLoss(params, tokens, fit), from };
}
