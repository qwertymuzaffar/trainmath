import { describe, expect, it } from 'vitest';
import { GIB } from './format.js';
import { memoryPerGpu } from './memory.js';
import { models } from './models.js';

describe('memoryPerGpu', () => {
  it('AdamW mixed precision is 16 bytes per parameter before sharding', () => {
    const e = memoryPerGpu({ model: { params: 1e9 }, gpu: 'h100' });
    expect(e.bytesPerParam).toEqual({ weights: 2, grads: 2, master: 4, optimizer: 8, total: 16 });
    const states = e.perGpuGiB.weights + e.perGpuGiB.grads + e.perGpuGiB.master + e.perGpuGiB.optimizer;
    expect(states).toBeCloseTo(16e9 / GIB, 6);
    expect(e.perGpuGiB.activations).toBe(0);
    expect(e.perGpuGiB.overhead).toBe(2);
    expect(e.fits).toBe(true);
    expect(e.notes.some((n) => /No shape/.test(n))).toBe(true);
  });

  it('other precisions and optimizers change the bytes per parameter', () => {
    expect(memoryPerGpu({ model: { params: 1 }, precision: 'fp32' }).bytesPerParam.total).toBe(16);
    expect(memoryPerGpu({ model: { params: 1 }, precision: 'bf16' }).bytesPerParam.total).toBe(12);
    expect(memoryPerGpu({ model: { params: 1 }, optimizer: 'sgd' }).bytesPerParam.total).toBe(8);
    expect(memoryPerGpu({ model: { params: 1 }, optimizer: 'adamw-8bit' }).bytesPerParam.total).toBe(10);
    expect(memoryPerGpu({ model: { params: 1 }, optimizer: 'muon' }).bytesPerParam.total).toBe(12);
    expect(memoryPerGpu({ model: { params: 1 }, optimizer: 'muon' }).notes[0]).toMatch(/Muon/);
  });

  it('shards states by ZeRO stage', () => {
    const base = { model: { params: 8e9 }, nGpus: 8 };
    const ddp = memoryPerGpu({ ...base, sharding: 'ddp' }).perGpuGiB;
    const z1 = memoryPerGpu({ ...base, sharding: 'zero1' }).perGpuGiB;
    const z2 = memoryPerGpu({ ...base, sharding: 'zero2' }).perGpuGiB;
    const z3 = memoryPerGpu({ ...base, sharding: 'zero3' }).perGpuGiB;
    expect(z1.weights).toBe(ddp.weights);
    expect(z1.grads).toBe(ddp.grads);
    expect(z1.master).toBeCloseTo(ddp.master / 8, 9);
    expect(z1.optimizer).toBeCloseTo(ddp.optimizer / 8, 9);
    expect(z2.grads).toBeCloseTo(ddp.grads / 8, 9);
    expect(z3.weights).toBeCloseTo(ddp.weights / 8, 9);
    expect(z3.total).toBeLessThan(z2.total);
    expect(z2.total).toBeLessThan(z1.total);
    expect(z1.total).toBeLessThan(ddp.total);
  });

  it('Llama 3 8B: does not fit one A100 80GB replicated, fits with FSDP over 8 at seq 8192', () => {
    const ddp = memoryPerGpu({ model: models['llama3-8b'] as never, gpu: 'a100-80', sharding: 'ddp' });
    expect(ddp.fits).toBe(false);
    expect(ddp.perGpuGiB.weights + ddp.perGpuGiB.grads + ddp.perGpuGiB.master + ddp.perGpuGiB.optimizer).toBeCloseTo((8_030_261_248 * 16) / GIB, 3);
    const fsdp = memoryPerGpu({ model: models['llama3-8b'] as never, gpu: 'a100-80', sharding: 'fsdp', nGpus: 8 });
    expect(fsdp.fits).toBe(true);
    // 32 layers x 34 x s b h bytes with flash attention and no checkpointing.
    expect(fsdp.perGpuGiB.activations).toBeCloseTo((32 * 34 * 8192 * 4096) / GIB, 3);
    // fp32 logits: s x b x vocab x 4 bytes.
    expect(fsdp.perGpuGiB.logits).toBeCloseTo((8192 * 128256 * 4) / GIB, 3);
    expect(fsdp.notes.some((n) => /unsharded block/.test(n))).toBe(true);
  });

  it('activation formula follows Korthikanti et al. for standard attention and checkpointing modes', () => {
    const model = { params: 1e9, nLayers: 16, dModel: 2048, nHeads: 16, seqLen: 2048 };
    const sbh = 2048 * 2048;
    const standard = memoryPerGpu({ model, attention: 'standard' }).perGpuGiB.activations * GIB;
    expect(standard).toBeCloseTo(16 * sbh * (34 + (5 * 16 * 2048) / 2048), -3);
    const flash = memoryPerGpu({ model }).perGpuGiB.activations * GIB;
    expect(flash).toBeCloseTo(16 * 34 * sbh, -3);
    const selective = memoryPerGpu({ model, attention: 'standard', activationCheckpointing: 'selective' }).perGpuGiB.activations * GIB;
    expect(selective).toBeCloseTo(flash, -3);
    const full = memoryPerGpu({ model, activationCheckpointing: 'full' }).perGpuGiB.activations * GIB;
    expect(full).toBeCloseTo(16 * 2 * sbh + 34 * sbh, -3);
    const bigger = memoryPerGpu({ model, microBatch: 4 }).perGpuGiB.activations;
    expect(bigger).toBeCloseTo(4 * (flash / GIB), 6);
  });

  it('validates inputs', () => {
    expect(() => memoryPerGpu({ model: { params: 1 }, nGpus: 0 })).toThrow(/nGpus/);
    expect(() => memoryPerGpu({ model: { params: 1 }, microBatch: 0 })).toThrow(/microBatch/);
    expect(() => memoryPerGpu({ model: { params: 1 }, sharding: 'tp' as never })).toThrow(/sharding/);
    expect(() => memoryPerGpu({ model: { params: 1 }, precision: 'int4' as never })).toThrow(/precision/);
    expect(() => memoryPerGpu({ model: { params: 1 }, optimizer: 'rmsprop' as never })).toThrow(/optimizer/);
    expect(() => memoryPerGpu({ model: { params: 1 }, gpu: 'gtx-1080' })).toThrow(/unknown gpu/);
  });
});
