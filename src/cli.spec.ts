import { describe, expect, it } from 'vitest';
import { runCli } from './cli.js';

async function run(args: string): Promise<{ code: number; out: string; err: string }> {
  let out = '';
  let err = '';
  const code = await runCli(args.split(/\s+/).filter(Boolean), { stdout: (t) => (out += t + '\n'), stderr: (t) => (err += t + '\n') });
  return { code, out, err };
}

describe('cli', () => {
  it('prints usage without a command and for help', async () => {
    expect((await run('')).code).toBe(1);
    expect((await run('')).out).toContain('Usage:');
    expect((await run('help')).code).toBe(0);
    expect((await run('plan --help')).code).toBe(0);
    const unknown = await run('nope');
    expect(unknown.code).toBe(1);
    expect(unknown.err).toContain('unknown command');
  });

  it('params, flops, mfu, throughput', async () => {
    expect((await run('params --model gpt2-124m')).out).toContain('124M total');
    const json = await run('params --vocab 50257 --d-model 768 --layers 12 --heads 12 --seq-len 1024 --bias --learned-positions --json');
    expect(JSON.parse(json.out).total).toBe(124_439_808);
    expect((await run('params --params 1B')).err).toContain('full shape');
    expect((await run('flops --params 1B --tokens 20B')).out).toContain('1.2e20');
    expect((await run('flops --model llama3-8b --seq-len 4096 --json')).out).toContain('"seqLen": 4096');
    const m = await run('mfu --params 1B --tokens-per-sec 527k --gpu h100 --gpus 8');
    expect(m.out).toMatch(/## MFU: 40\.\d%/);
    expect((await run('mfu --params 1B --gpu h100')).err).toContain('--tokens-per-sec is required');
    expect((await run('throughput --params 1B --gpu h100 --gpus 8 --mfu 0.4')).out).toContain('527k tokens/sec');
  });

  it('chinchilla, memory, plan, table', async () => {
    expect((await run('chinchilla --params 1B')).out).toContain('20B');
    expect((await run('chinchilla --tokens 20B --json')).out).toContain('"from": "tokens"');
    expect((await run('chinchilla --flops 1.2e20')).out).toContain('1B');
    expect((await run('chinchilla')).err).toContain('needs --params');
    const mem = await run('memory --model llama3-8b --gpu a100-80 --gpus 8 --sharding fsdp --micro-batch 1');
    expect(mem.out).toContain('fits A100 SXM 80GB');
    expect((await run('memory --params 1B --sharding tp')).err).toContain('--sharding must be one of');
    const plan = await run('plan --params 1B --gpu h100 --gpus 8 --mfu 0.4 --price 2.5');
    expect(plan.out).toContain('$211');
    const table = await run('table --sizes 1B,2B,4B,14B --gpu h100 --gpus 8 --mfu 0.4 --price 2.5');
    expect(table.out.split('\n').filter((l) => l.startsWith('| ')).length).toBe(6);
    expect(table.out).toContain('$41,');
    expect((await run('table --gpu h100 --json')).out).toContain('"gpuHours"');
  });

  it('gap, suggest, gpus, models', async () => {
    const gap = await run('gap --mfu 0.22 --gpus 8 --sharding fsdp --precision fp32 --no-compile --micro-batch 1 --checkpointing-on --multi-node');
    expect(gap.out).toContain('reads as: low');
    expect(gap.out).toContain('fp32');
    expect((await run('gap')).err).toContain('--mfu is required');
    expect((await run('suggest --params 1B --mlp swiglu --untied')).out).toContain('non-embedding');
    expect((await run('suggest --params 1B --mlp relu')).err).toContain('--mlp must be');
    expect((await run('gpus')).out).toContain('| h100-sxm |');
    expect((await run('gpus --json')).out).toContain('"id": "h100-sxm"');
    expect((await run('models')).out).toContain('gpt2-124m');
    expect((await run('models --json')).out).toContain('"llama3-8b"');
  });

  it('flag parsing: --name=value, boolean flags, bad numbers, missing model', async () => {
    expect((await run('chinchilla --params=2B')).out).toContain('40B');
    expect((await run('flops --params lots')).err).toContain('cannot parse');
    expect((await run('flops')).err).toContain('give a model');
    expect((await run('flops --params')).err).toContain('needs a value');
    expect((await run('params --model gpt9')).err).toContain('unknown model');
    expect((await run('params --vocab 100 --d-model 8 --layers 1 --heads 2 --mlp relu')).err).toContain('--mlp must be');
  });
});
