import { describe, expect, it } from 'vitest';
import { explainGap } from './gap.js';

describe('explainGap', () => {
  it('bands the MFU', () => {
    expect(explainGap({ mfu: 0.8 }).band).toBe('suspicious');
    expect(explainGap({ mfu: 0.5 }).band).toBe('good');
    expect(explainGap({ mfu: 0.35 }).band).toBe('ok');
    expect(explainGap({ mfu: 0.2 }).band).toBe('low');
    expect(explainGap({ mfu: 0.05 }).band).toBe('poor');
    expect(() => explainGap({ mfu: -1 })).toThrow(/mfu/);
  });

  it('suspicious values point at the measurement, not the run', () => {
    const r = explainGap({ mfu: 0.9 });
    expect(r.causes.join(' ')).toMatch(/sparsity/);
    expect(r.checks).toHaveLength(1);
    expect(r.headline).not.toMatch(/impossible/);
  });

  it('values above 100% are impossible, not an error', () => {
    const r = explainGap({ mfu: 2.9 });
    expect(r.band).toBe('suspicious');
    expect(r.headline).toMatch(/impossible/);
    expect(r.causes[0]).toMatch(/per device/);
    expect(() => explainGap({ mfu: Number.NaN })).toThrow(/mfu/);
  });

  it('adds causes for the flags you pass, most specific first', () => {
    const r = explainGap({
      mfu: 0.18,
      nGpus: 16,
      sharding: 'fsdp',
      precision: 'fp32',
      compiled: false,
      attention: 'standard',
      microBatch: 1,
      activationCheckpointing: true,
      multiNode: true,
    });
    const text = r.causes.join('\n');
    expect(r.causes[0]).toMatch(/fp32/);
    expect(text).toMatch(/FlashAttention/);
    expect(text).toMatch(/torch\.compile/);
    expect(text).toMatch(/micro-batch/);
    expect(text).toMatch(/checkpointing/);
    expect(text).toMatch(/all-gather/);
    expect(text).toMatch(/Multi-node/);
    expect(r.checks.length).toBeGreaterThanOrEqual(4);
  });

  it('mentions the all-reduce for plain DDP and asks about the attention kernel when unknown', () => {
    const r = explainGap({ mfu: 0.3, nGpus: 8, sharding: 'ddp' });
    expect(r.causes.join('\n')).toMatch(/all-reduce/);
    expect(r.checks.join('\n')).toMatch(/attention kernel/);
  });
});
