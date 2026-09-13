import { describe, expect, it } from 'vitest';
import { models } from './models.js';
import { normalizeArch, transformerParams } from './params.js';

describe('transformerParams', () => {
  it('matches GPT-2 124M exactly (124,439,808 incl. learned positions)', () => {
    const p = transformerParams(models['gpt2-124m'] as never);
    expect(p.total).toBe(124_439_808);
    // nanoGPT reports 123.65M when it drops the position embeddings.
    expect(p.total - p.breakdown.positions).toBe(123_653_376);
    expect(p.breakdown.embeddings).toBe(50257 * 768);
    expect(p.breakdown.head).toBe(0);
    expect(p.perLayer).toBe(7_087_872);
  });

  it('matches GPT-3 175B within 0.5%', () => {
    const p = transformerParams(models['gpt3-175b'] as never);
    expect(Math.abs(p.total - 175e9) / 175e9).toBeLessThan(0.005);
  });

  it('matches Llama 3 8B (8.03B, untied, GQA, SwiGLU)', () => {
    const p = transformerParams(models['llama3-8b'] as never);
    expect(p.total).toBe(8_030_261_248);
    expect(p.breakdown.head).toBe(128256 * 4096);
    expect(p.breakdown.positions).toBe(0);
  });

  it('matches Llama 3 70B within 0.5%', () => {
    const p = transformerParams(models['llama3-70b'] as never);
    expect(Math.abs(p.total - 70.6e9) / 70.6e9).toBeLessThan(0.005);
  });

  it('defaults dFf to 4d for gelu and to 8/3 d rounded up to 256 for swiglu', () => {
    expect(normalizeArch({ vocab: 100, dModel: 768, nLayers: 1, nHeads: 12 }).dFf).toBe(3072);
    expect(normalizeArch({ vocab: 100, dModel: 4096, nLayers: 1, nHeads: 32, mlp: 'swiglu' }).dFf).toBe(11008);
  });

  it('counts a single tiny block by hand', () => {
    // d=4, heads=2, kv=1 (dHead 2, kvDim 2), gelu dFf 16, no bias, tied, vocab 10.
    const p = transformerParams({ vocab: 10, dModel: 4, nLayers: 1, nHeads: 2, nKvHeads: 1, dFf: 16 });
    const attention = 4 * 4 + 2 * 4 * 2 + 4 * 4; // q, k+v, o
    const mlp = 2 * 4 * 16;
    const norms = 2 * 4 + 4; // two per block + final
    expect(p.breakdown.attention).toBe(attention);
    expect(p.breakdown.mlp).toBe(mlp);
    expect(p.breakdown.norms).toBe(norms);
    expect(p.nonEmbedding).toBe(attention + mlp + norms);
    expect(p.total).toBe(attention + mlp + norms + 10 * 4);
  });

  it('rejects impossible shapes', () => {
    expect(() => transformerParams({ vocab: 10, dModel: 10, nLayers: 1, nHeads: 3 })).toThrow(/divisible/);
    expect(() => transformerParams({ vocab: 10, dModel: 8, nLayers: 1, nHeads: 4, nKvHeads: 3 })).toThrow(/divisible/);
    expect(() => transformerParams({ vocab: 0, dModel: 8, nLayers: 1, nHeads: 4 })).toThrow(/vocab/);
    expect(() => transformerParams({ vocab: 10, dModel: 8, nLayers: 1.5, nHeads: 4 })).toThrow(/integer/);
    expect(() => transformerParams({ vocab: 10, dModel: 8, nLayers: 1, nHeads: 4, mlp: 'relu' as never })).toThrow(/mlp/);
  });
});
