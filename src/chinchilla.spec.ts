import { describe, expect, it } from 'vitest';
import { chinchilla, chinchillaLoss, chinchillaParams, chinchillaTokens, computeOptimal } from './chinchilla.js';

describe('chinchilla', () => {
  it('uses 20 tokens per parameter by default', () => {
    expect(chinchillaTokens(1e9)).toBe(20e9);
    expect(chinchillaParams(20e9)).toBe(1e9);
    expect(chinchillaTokens(1e9, 25)).toBe(25e9);
  });

  it('splits a FLOP budget so that D = 20 N and 6ND = C', () => {
    const { params, tokens } = computeOptimal(1.2e20);
    expect(params).toBeCloseTo(1e9, -3);
    expect(tokens).toBeCloseTo(20e9, -3);
    expect(6 * params * tokens).toBeCloseTo(1.2e20, -12);
  });

  it('predicts about 1.94 loss for Chinchilla 70B on 1.4T tokens and falls with N and D', () => {
    const l = chinchillaLoss(70e9, 1.4e12);
    expect(l).toBeGreaterThan(1.9);
    expect(l).toBeLessThan(1.97);
    expect(chinchillaLoss(140e9, 1.4e12)).toBeLessThan(l);
    expect(chinchillaLoss(70e9, 2.8e12)).toBeLessThan(l);
  });

  it('one entry point for all three directions', () => {
    expect(chinchilla({ params: 1e9 })).toMatchObject({ params: 1e9, tokens: 20e9, flops: 1.2e20, ratio: 20, from: 'params' });
    expect(chinchilla({ tokens: 40e9 })).toMatchObject({ params: 2e9, from: 'tokens' });
    expect(chinchilla({ flops: 1.2e20 }).params).toBeCloseTo(1e9, -3);
    expect(chinchilla({ flops: 1.2e20 }).from).toBe('flops');
    expect(chinchilla({ params: 1e9 }, { ratio: 10 }).tokens).toBe(10e9);
    expect(() => chinchilla({} as never)).toThrow(/one of/);
    expect(() => chinchillaTokens(-1)).toThrow(/params/);
  });
});
