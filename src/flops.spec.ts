import { describe, expect, it } from 'vitest';
import { flopsPerToken, modelParams, trainingFlops } from './flops.js';
import { models } from './models.js';
import { expectedThroughput, mfu } from './throughput.js';

describe('flopsPerToken', () => {
  it('reproduces the GPT-3 paper: 3.14e23 FLOPs for 300B tokens (dense term)', () => {
    const f = flopsPerToken(models['gpt3-175b'] as never);
    const dense = f.dense * 300e9;
    expect(Math.abs(dense - 3.14e23) / 3.14e23).toBeLessThan(0.01);
    // The attention term adds a few percent at seq 2048.
    expect(f.attention).toBe(12 * 96 * 12288 * 2048);
    expect(f.total).toBe(f.dense + f.attention);
  });

  it('counts blocks plus the output head, never the position table', () => {
    const f = flopsPerToken(models['gpt2-124m'] as never);
    expect(f.paramsCounted).toBe(123_653_376 - 50257 * 768 + 50257 * 768); // non-embedding + head = nanoGPT's N
    expect(f.dense).toBe(6 * f.paramsCounted);
    expect(f.seqLen).toBe(1024);
  });

  it('accepts a bare parameter count, with an attention term only when the shape is given', () => {
    expect(flopsPerToken({ params: 1e9 })).toMatchObject({ dense: 6e9, attention: 0, total: 6e9, seqLen: undefined });
    const f = flopsPerToken({ params: 1e9, nLayers: 16, dModel: 2048, seqLen: 2048 });
    expect(f.attention).toBe(12 * 16 * 2048 * 2048);
    expect(f.seqLen).toBe(2048);
  });

  it('lets seqLen be overridden', () => {
    const a = flopsPerToken(models['gpt2-124m'] as never, { seqLen: 4096 });
    expect(a.attention).toBe(12 * 12 * 768 * 4096);
  });

  it('trainingFlops multiplies by tokens and modelParams reads either spec', () => {
    expect(trainingFlops({ params: 1e9 }, 20e9)).toBe(1.2e20);
    expect(modelParams({ params: 5 })).toBe(5);
    expect(modelParams(models['gpt2-124m'] as never)).toBe(124_439_808);
    expect(() => trainingFlops({ params: 1e9 }, 0)).toThrow(/tokens/);
  });
});

describe('mfu and expectedThroughput', () => {
  it('reproduces PaLM 540B on 6144 TPU v4 chips: about 46% MFU at 238k tokens/sec', () => {
    const r = mfu({
      tokensPerSec: 238.3e3,
      gpu: 'tpu-v4',
      nGpus: 6144,
      model: { params: 540e9, nLayers: 118, dModel: 18432, seqLen: 2048 },
    });
    expect(r.mfu).toBeGreaterThan(0.45);
    expect(r.mfu).toBeLessThan(0.475);
    expect(r.peakTflopsPerGpu).toBe(275);
  });

  it('round-trips through expectedThroughput', () => {
    const t = expectedThroughput({ gpu: 'h100', nGpus: 8, mfu: 0.4, model: { params: 1e9 } });
    expect(t.tokensPerSec).toBeCloseTo((8 * 989e12 * 0.4) / 6e9, 0);
    expect(t.tokensPerSecPerGpu * 8).toBeCloseTo(t.tokensPerSec, 6);
    const back = mfu({ tokensPerSec: t.tokensPerSec, gpu: 'h100', nGpus: 8, model: { params: 1e9 } });
    expect(back.mfu).toBeCloseTo(0.4, 10);
  });

  it('uses the fp8 peak when asked and refuses it where it does not exist', () => {
    const bf16 = mfu({ tokensPerSec: 1e5, gpu: 'h100', model: { params: 1e9 } });
    const fp8 = mfu({ tokensPerSec: 1e5, gpu: 'h100', model: { params: 1e9 }, precision: 'fp8' });
    expect(fp8.mfu).toBeCloseTo(bf16.mfu * (989 / 1979), 9);
    expect(() => mfu({ tokensPerSec: 1e5, gpu: 'a100', model: { params: 1e9 }, precision: 'fp8' })).toThrow(/fp8/);
  });

  it('validates inputs', () => {
    expect(() => mfu({ tokensPerSec: 0, gpu: 'h100', model: { params: 1 } })).toThrow(/tokensPerSec/);
    expect(() => expectedThroughput({ gpu: 'h100', mfu: 1.5, model: { params: 1 } })).toThrow(/mfu/);
    expect(() => expectedThroughput({ gpu: 'h100', nGpus: 0, model: { params: 1 } })).toThrow(/nGpus/);
  });
});
