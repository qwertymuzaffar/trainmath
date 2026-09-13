import { describe, expect, it } from 'vitest';
import { gpus, listGpus, peakFlops, resolveGpu } from './gpus.js';

describe('gpus', () => {
  it('resolves ids, aliases and custom specs', () => {
    expect(resolveGpu('h100').id).toBe('h100-sxm');
    expect(resolveGpu('H100 SXM').id).toBe('h100-sxm');
    expect(resolveGpu('a100').memoryGiB).toBe(80);
    expect(resolveGpu('4090').id).toBe('rtx-4090');
    expect(resolveGpu('trillium').id).toBe('tpu-v6e');
    const custom = { id: 'x', name: 'X', vendor: 'custom' as const, peakTflops: { bf16: 100 }, memoryGiB: 10, bandwidthTBs: 1, source: 'me' };
    expect(resolveGpu(custom)).toBe(custom);
    expect(() => resolveGpu('gtx-1080')).toThrow(/unknown gpu/);
    expect(() => resolveGpu({} as never)).toThrow(/custom gpu/);
  });

  it('lists dense peaks and converts to FLOP/s', () => {
    expect(peakFlops('h100')).toBe(989e12);
    expect(peakFlops('h100', 'fp8')).toBe(1979e12);
    expect(peakFlops('b200')).toBe(2250e12);
    expect(() => peakFlops('tpu-v4', 'fp8')).toThrow(/no fp8/);
    expect(listGpus().length).toBeGreaterThan(10);
    expect(listGpus().map((g) => g.id)).toContain('mi300x');
    expect(Object.isFrozen(gpus)).toBe(true);
  });
});
