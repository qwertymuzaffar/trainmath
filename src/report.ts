import type { ChinchillaResult } from './chinchilla.js';
import { formatCount, formatDuration, formatGiB, formatMoney, formatPercent, formatSci } from './format.js';
import type { GapReport } from './gap.js';
import { listGpus } from './gpus.js';
import type { MemoryEstimate } from './memory.js';
import type { ParamCount } from './params.js';
import type { Plan } from './plan.js';
import type { MfuResult, ThroughputResult } from './throughput.js';

function escapeCell(value: string): string {
  return value.replace(/\|/g, '\\|');
}

/** A GitHub-flavored Markdown table. */
export function markdownTable(headers: string[], rows: Array<Array<string | number>>): string {
  const head = `| ${headers.map(escapeCell).join(' | ')} |`;
  const rule = `| ${headers.map(() => '---').join(' | ')} |`;
  const body = rows.map((r) => `| ${r.map((c) => escapeCell(String(c))).join(' | ')} |`);
  return [head, rule, ...body].join('\n');
}

export function paramsToMarkdown(count: ParamCount): string {
  const b = count.breakdown;
  const a = count.arch;
  const lines = [
    `## Parameters: ${formatCount(count.total)} total, ${formatCount(count.nonEmbedding)} non-embedding`,
    '',
    `Shape: vocab ${a.vocab}, d_model ${a.dModel}, ${a.nLayers} layers, ${a.nHeads} heads (${a.nKvHeads} kv), d_ff ${a.dFf} ${a.mlp}, seq ${a.seqLen}, ${a.tiedEmbeddings ? 'tied' : 'untied'} embeddings${a.bias ? ', biases' : ''}${a.learnedPositions ? ', learned positions' : ''}.`,
    '',
    markdownTable(
      ['Part', 'Parameters', 'Share'],
      [
        ['Attention', formatCount(b.attention), formatPercent(b.attention / count.total)],
        ['MLP', formatCount(b.mlp), formatPercent(b.mlp / count.total)],
        ['Norms', formatCount(b.norms), formatPercent(b.norms / count.total)],
        ['Embeddings', formatCount(b.embeddings), formatPercent(b.embeddings / count.total)],
        ['Positions', formatCount(b.positions), formatPercent(b.positions / count.total)],
        ['Output head', formatCount(b.head), formatPercent(b.head / count.total)],
        ['Per layer', formatCount(count.perLayer), ''],
      ],
    ),
  ];
  return lines.join('\n');
}

export function mfuToMarkdown(r: MfuResult): string {
  return [
    `## MFU: ${formatPercent(r.mfu)}`,
    '',
    markdownTable(
      ['Item', 'Value'],
      [
        ['Measured tokens/sec', formatCount(r.tokensPerSec)],
        ['FLOPs per token', formatSci(r.flopsPerToken)],
        ['Achieved model FLOP/s', formatSci(r.achievedFlops)],
        ['Achieved TFLOPS per device', r.achievedTflopsPerGpu.toFixed(1)],
        ['Peak TFLOPS per device', `${r.peakTflopsPerGpu} (${r.gpu.name}, ${r.precision} dense)`],
        ['Devices', r.nGpus],
      ],
    ),
  ].join('\n');
}

export function throughputToMarkdown(r: ThroughputResult): string {
  return [
    `## Expected throughput at ${formatPercent(r.mfu, 0)} MFU: ${formatCount(r.tokensPerSec)} tokens/sec`,
    '',
    markdownTable(
      ['Item', 'Value'],
      [
        ['Per device', `${formatCount(r.tokensPerSecPerGpu)} tokens/sec`],
        ['FLOPs per token', formatSci(r.flopsPerToken)],
        ['Hardware', `${r.nGpus} x ${r.gpu.name}, ${r.precision} dense peak`],
      ],
    ),
  ].join('\n');
}

export function chinchillaToMarkdown(r: ChinchillaResult): string {
  return [
    `## Chinchilla budget (${r.ratio} tokens per parameter, from ${r.from})`,
    '',
    markdownTable(
      ['Item', 'Value'],
      [
        ['Parameters', formatCount(r.params)],
        ['Tokens', formatCount(r.tokens)],
        ['Training FLOPs (6ND)', formatSci(r.flops)],
        ['Predicted loss (Hoffmann fit)', r.loss.toFixed(3)],
      ],
    ),
  ].join('\n');
}

export function memoryToMarkdown(e: MemoryEstimate): string {
  const g = e.perGpuGiB;
  const verdict =
    e.fits === undefined ? '' : e.fits ? ` - fits ${e.gpu?.name} with ${formatGiB(e.headroomGiB ?? 0)} to spare` : ` - does NOT fit ${e.gpu?.name} (${formatGiB(-(e.headroomGiB ?? 0))} over)`;
  const lines = [
    `## Memory per device: ${formatGiB(g.total)}${verdict}`,
    '',
    `${formatCount(e.params)} parameters, ${e.precision}, ${e.optimizer}, ${e.sharding} across ${e.nGpus} device${e.nGpus === 1 ? '' : 's'}; ${e.bytesPerParam.total} bytes per parameter before sharding.`,
    '',
    markdownTable(
      ['Part', 'GiB'],
      [
        ['Weights', formatGiB(g.weights)],
        ['Gradients', formatGiB(g.grads)],
        ['Master weights', formatGiB(g.master)],
        ['Optimizer states', formatGiB(g.optimizer)],
        ['Activations', formatGiB(g.activations)],
        ['Logits', formatGiB(g.logits)],
        ['Overhead', formatGiB(g.overhead)],
        ['Total', formatGiB(g.total)],
      ],
    ),
  ];
  if (e.notes.length) lines.push('', ...e.notes.map((n) => `- ${n}`));
  return lines.join('\n');
}

export function planToMarkdown(plans: Plan[]): string {
  if (plans.length === 0) return '';
  const first = plans[0] as Plan;
  const hasCost = plans.some((p) => p.cost !== undefined);
  const headers = ['Params', 'Tokens', 'FLOPs', 'Tokens/sec', 'Wall-clock', 'Device-hours'];
  if (hasCost) headers.push('Cost');
  const rows = plans.map((p) => {
    const row: Array<string | number> = [
      formatCount(p.params),
      formatCount(p.tokens),
      formatSci(p.flops),
      formatCount(p.tokensPerSec),
      formatDuration(p.seconds),
      formatCount(p.gpuHours),
    ];
    if (hasCost) row.push(p.cost === undefined ? '' : formatMoney(p.cost));
    return row;
  });
  const price = first.pricePerGpuHour === undefined ? '' : `, ${formatMoney(first.pricePerGpuHour)} per device-hour`;
  return [
    `## Training plan: ${first.nGpus} x ${first.gpu.name} at ${formatPercent(first.mfu, 0)} MFU${price}`,
    '',
    markdownTable(headers, rows),
  ].join('\n');
}

export function gapToMarkdown(r: GapReport, mfuValue?: number): string {
  const lines = [`## MFU ${mfuValue === undefined ? '' : formatPercent(mfuValue) + ' '}reads as: ${r.band}`, '', r.headline, ''];
  if (r.causes.length) lines.push('Likely causes, most likely first:', '', ...r.causes.map((c, i) => `${i + 1}. ${c}`), '');
  if (r.checks.length) lines.push('Checks:', '', ...r.checks.map((c) => `- ${c}`));
  return lines.join('\n').trimEnd();
}

export function gpusToMarkdown(): string {
  return markdownTable(
    ['Id', 'Name', 'BF16 TFLOPS', 'FP8 TFLOPS', 'Memory GiB', 'TB/s'],
    listGpus().map((g) => [g.id, g.name, g.peakTflops.bf16, g.peakTflops.fp8 ?? '', g.memoryGiB, g.bandwidthTBs]),
  );
}
