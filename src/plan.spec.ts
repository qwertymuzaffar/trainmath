import { describe, expect, it } from 'vitest';
import { models } from './models.js';
import { planTable, suggestArch, trainingPlan } from './plan.js';

describe('trainingPlan', () => {
  it('1B at Chinchilla tokens on 8 x H100 at 40% MFU: about 10.5 hours and $211 at $2.50', () => {
    const p = trainingPlan({ model: { params: 1e9 }, gpu: 'h100', nGpus: 8, mfu: 0.4, pricePerGpuHour: 2.5 });
    expect(p.tokens).toBe(20e9);
    expect(p.flops).toBe(1.2e20);
    expect(p.hours).toBeCloseTo(10.53, 1);
    expect(p.gpuHours).toBeCloseTo(84.3, 0);
    expect(p.cost).toBeCloseTo(210.6, 0);
    expect(p.tokensPerSec).toBeCloseTo(527_467, -2);
    expect(p.gpu.id).toBe('h100-sxm');
  });

  it('accepts explicit tokens and a tokens-per-parameter ratio, cost undefined without a price', () => {
    const explicit = trainingPlan({ model: { params: 1e9 }, tokens: 1e9, gpu: 'h100' });
    expect(explicit.tokensPerParam).toBe(1);
    expect(explicit.cost).toBeUndefined();
    const ratio = trainingPlan({ model: { params: 1e9 }, tokensPerParam: 100, gpu: 'h100' });
    expect(ratio.tokens).toBe(100e9);
    expect(ratio.hours).toBeCloseTo(explicit.hours * 100, 6);
  });

  it('planTable scales linearly with N^2 under the Chinchilla rule', () => {
    const plans = planTable([1e9, 2e9, 4e9], { gpu: 'h100', nGpus: 8, mfu: 0.4, pricePerGpuHour: 2.5 });
    expect(plans).toHaveLength(3);
    expect((plans[1] as never as { hours: number }).hours / (plans[0] as never as { hours: number }).hours).toBeCloseTo(4, 6);
    expect((plans[2] as never as { cost: number }).cost).toBeCloseTo(3370, -1);
  });

  it('works with a full shape, where the attention term makes it a little slower', () => {
    const shape = trainingPlan({ model: models['gpt2-124m'] as never, gpu: 'h100', tokens: 10e9 });
    const bare = trainingPlan({ model: { params: 124_439_808 }, gpu: 'h100', tokens: 10e9 });
    expect(shape.flopsPerToken).toBeGreaterThan(6 * 123_653_376);
    expect(shape.params).toBe(124_439_808);
    expect(bare.flopsPerToken).toBe(6 * 124_439_808);
  });

  it('validates', () => {
    expect(() => trainingPlan({ model: { params: 1e9 }, gpu: 'h100', tokens: -1 })).toThrow(/tokens/);
    expect(() => trainingPlan({ model: { params: 1e9 }, gpu: 'h100', pricePerGpuHour: 0 })).toThrow(/pricePerGpuHour/);
  });
});

describe('suggestArch', () => {
  it('finds a shape near the target non-embedding count with head size 128', () => {
    const s = suggestArch(1e9);
    expect(s.error).toBeLessThan(0.25);
    expect(s.arch.dModel % 128).toBe(0);
    expect(s.arch.nHeads).toBe(s.arch.dModel / 128);
    expect(s.count.arch.vocab).toBe(32000);
  });

  it('respects options', () => {
    const s = suggestArch(124e6, { vocab: 50257, seqLen: 1024, dHead: 64, aspect: 64, mlp: 'swiglu', tiedEmbeddings: false });
    expect(s.arch.vocab).toBe(50257);
    expect(s.arch.dModel % 64).toBe(0);
    expect(s.count.arch.mlp).toBe('swiglu');
    expect(s.count.breakdown.head).toBeGreaterThan(0);
    expect(() => suggestArch(0)).toThrow(/targetParams/);
  });
});
