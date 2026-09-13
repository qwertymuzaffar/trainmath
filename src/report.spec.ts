import { describe, expect, it } from 'vitest';
import { chinchilla } from './chinchilla.js';
import { explainGap } from './gap.js';
import { memoryPerGpu } from './memory.js';
import { models } from './models.js';
import { transformerParams } from './params.js';
import { planTable } from './plan.js';
import { chinchillaToMarkdown, gapToMarkdown, gpusToMarkdown, markdownTable, memoryToMarkdown, mfuToMarkdown, paramsToMarkdown, planToMarkdown, throughputToMarkdown } from './report.js';
import { expectedThroughput, mfu } from './throughput.js';

describe('markdown reports', () => {
  it('markdownTable escapes pipes', () => {
    const t = markdownTable(['a', 'b'], [['x|y', 1]]);
    expect(t.split('\n')).toHaveLength(3);
    expect(t).toContain('x\\|y');
  });

  it('renders every report without throwing and with the headline numbers', () => {
    expect(paramsToMarkdown(transformerParams(models['gpt2-124m'] as never))).toContain('124M total');
    expect(mfuToMarkdown(mfu({ tokensPerSec: 4e5, gpu: 'h100', model: { params: 1e9 } }))).toMatch(/## MFU: \d+\.\d%/);
    expect(throughputToMarkdown(expectedThroughput({ gpu: 'h100', nGpus: 8, model: { params: 1e9 } }))).toContain('40% MFU');
    expect(chinchillaToMarkdown(chinchilla({ params: 1e9 }))).toContain('20B');
    const fits = memoryToMarkdown(memoryPerGpu({ model: { params: 1e9 }, gpu: 'h100' }));
    expect(fits).toContain('fits H100 SXM 80GB');
    const noFit = memoryToMarkdown(memoryPerGpu({ model: { params: 8e9 }, gpu: 'a100-40' }));
    expect(noFit).toContain('does NOT fit');
    const noGpu = memoryToMarkdown(memoryPerGpu({ model: { params: 8e9 } }));
    expect(noGpu).not.toContain('fit');
    const table = planToMarkdown(planTable([1e9, 2e9], { gpu: 'h100', nGpus: 8, pricePerGpuHour: 2.5 }));
    expect(table).toContain('$211');
    expect(table).toContain('Cost');
    expect(planToMarkdown(planTable([1e9], { gpu: 'h100' }))).not.toContain('Cost');
    expect(planToMarkdown([])).toBe('');
    expect(gapToMarkdown(explainGap({ mfu: 0.2 }), 0.2)).toContain('20.0%');
    expect(gapToMarkdown(explainGap({ mfu: 0.9 }))).toContain('suspicious');
    expect(gpusToMarkdown()).toContain('h100-sxm');
  });
});
