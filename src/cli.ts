import { realpathSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import {
  chinchilla,
  chinchillaToMarkdown,
  explainGap,
  expectedThroughput,
  flopsPerToken,
  gapToMarkdown,
  gpus,
  gpusToMarkdown,
  listModels,
  memoryPerGpu,
  memoryToMarkdown,
  mfu,
  mfuToMarkdown,
  paramsToMarkdown,
  parseNumber,
  planTable,
  planToMarkdown,
  resolveModel,
  suggestArch,
  throughputToMarkdown,
  trainingPlan,
  transformerParams,
  formatSci,
  formatCount,
  type Attention,
  type Checkpointing,
  type ComputePrecision,
  type ModelSpec,
  type Optimizer,
  type Sharding,
  type TrainingPrecision,
} from './index.js';

export interface CliIO {
  stdout: (text: string) => void;
  stderr: (text: string) => void;
}

interface Parsed {
  command: string | undefined;
  positionals: string[];
  flags: Record<string, string | true>;
}

const BOOLEAN_FLAGS = new Set(['json', 'untied', 'bias', 'learned-positions', 'compile', 'no-compile', 'checkpointing-on', 'multi-node', 'help']);

const USAGE = `trainmath - plan and sanity-check LLM pretraining runs

Usage:
  trainmath params   <model>
  trainmath flops    <model> [--tokens 20B] [--seq-len 2048]
  trainmath mfu      <model> --tokens-per-sec 420k --gpu h100 [--gpus 8] [--precision bf16|fp8]
  trainmath throughput <model> --gpu h100 [--gpus 8] [--mfu 0.4]
  trainmath chinchilla (--params 1B | --tokens 20B | --flops 1e20) [--ratio 20]
  trainmath memory   <model> [--gpu h100] [--gpus 8] [--sharding ddp|zero1|zero2|zero3|fsdp]
                     [--precision bf16-mixed|fp16-mixed|fp32|bf16] [--optimizer adamw|adamw-8bit|adafactor|sgd-momentum|sgd|muon|lion]
                     [--micro-batch 1] [--seq-len 2048] [--attention flash|standard] [--checkpointing none|selective|full]
  trainmath plan     <model> --gpu h100 [--gpus 8] [--mfu 0.4] [--price 2.50] [--tokens 20B | --tokens-per-param 20]
  trainmath table    --sizes 1B,2B,4B,14B --gpu h100 [--gpus 8] [--mfu 0.4] [--price 2.50] [--tokens-per-param 20]
  trainmath gap      --mfu 0.22 [--gpus 8] [--sharding fsdp] [--precision fp32] [--attention standard]
                     [--no-compile] [--micro-batch 1] [--checkpointing-on] [--multi-node]
  trainmath suggest  --params 1B [--vocab 32000] [--seq-len 2048] [--aspect 128] [--mlp gelu|swiglu]
  trainmath gpus
  trainmath models

<model> is one of:
  --model gpt2-124m | gpt2-350m | gpt2-774m | gpt2-1.6b | gpt3-175b | llama3-8b | llama3-70b
  --params 1B [--layers 16 --d-model 2048 --heads 16 --vocab 32000 --seq-len 2048]
  --vocab 50257 --d-model 768 --layers 12 --heads 12 [--kv-heads 4] [--d-ff 3072] [--mlp gelu|swiglu]
      [--seq-len 1024] [--untied] [--bias] [--learned-positions]

Numbers accept 1e9, 1B, 124M, 20B, 2.5k. Add --json for machine-readable output.`;

function parseArgs(argv: string[]): Parsed {
  const flags: Record<string, string | true> = {};
  const positionals: string[] = [];
  let command: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    if (arg.startsWith('--')) {
      const eq = arg.indexOf('=');
      if (eq !== -1) {
        flags[arg.slice(2, eq)] = arg.slice(eq + 1);
        continue;
      }
      const name = arg.slice(2);
      const next = argv[i + 1];
      if (BOOLEAN_FLAGS.has(name) || next === undefined || next.startsWith('--')) {
        flags[name] = true;
      } else {
        flags[name] = next;
        i++;
      }
    } else if (command === undefined) {
      command = arg;
    } else {
      positionals.push(arg);
    }
  }
  return { command, positionals, flags };
}

function num(flags: Parsed['flags'], name: string): number | undefined {
  const v = flags[name];
  if (v === undefined) return undefined;
  if (v === true) throw new Error(`--${name} needs a value`);
  return parseNumber(v, `--${name}`);
}

function str(flags: Parsed['flags'], name: string): string | undefined {
  const v = flags[name];
  if (v === undefined) return undefined;
  if (v === true) throw new Error(`--${name} needs a value`);
  return v;
}

function need(value: number | undefined, name: string): number {
  if (value === undefined) throw new Error(`--${name} is required`);
  return value;
}

function modelFromFlags(flags: Parsed['flags']): ModelSpec {
  const preset = str(flags, 'model');
  if (preset) {
    const arch = resolveModel(preset);
    const seqLen = num(flags, 'seq-len');
    return seqLen ? { ...arch, seqLen } : arch;
  }
  const dModel = num(flags, 'd-model');
  const nLayers = num(flags, 'layers');
  const nHeads = num(flags, 'heads');
  const vocab = num(flags, 'vocab');
  const params = num(flags, 'params');
  if (params !== undefined) {
    return { params, nLayers, dModel, nHeads, vocab, seqLen: num(flags, 'seq-len') };
  }
  if (dModel !== undefined && nLayers !== undefined && nHeads !== undefined && vocab !== undefined) {
    const mlp = str(flags, 'mlp');
    if (mlp !== undefined && mlp !== 'gelu' && mlp !== 'swiglu') throw new Error('--mlp must be gelu or swiglu');
    return {
      vocab,
      dModel,
      nLayers,
      nHeads,
      nKvHeads: num(flags, 'kv-heads'),
      dFf: num(flags, 'd-ff'),
      mlp: mlp as 'gelu' | 'swiglu' | undefined,
      seqLen: num(flags, 'seq-len'),
      tiedEmbeddings: flags.untied ? false : undefined,
      bias: flags.bias ? true : undefined,
      learnedPositions: flags['learned-positions'] ? true : undefined,
    };
  }
  throw new Error('give a model: --model <preset>, --params <n>, or --vocab --d-model --layers --heads');
}

function oneOf<T extends string>(value: string | undefined, allowed: readonly T[], name: string): T | undefined {
  if (value === undefined) return undefined;
  if (!(allowed as readonly string[]).includes(value)) throw new Error(`--${name} must be one of ${allowed.join(', ')}`);
  return value as T;
}

const SHARDINGS = ['none', 'ddp', 'zero1', 'zero2', 'zero3', 'fsdp'] as const;
const TRAINING_PRECISIONS = ['bf16-mixed', 'fp16-mixed', 'fp32', 'bf16'] as const;
const COMPUTE_PRECISIONS = ['bf16', 'fp16', 'tf32', 'fp8', 'fp32'] as const;
const OPTIMIZERS = ['adamw', 'adam', 'adamw-8bit', 'adafactor', 'sgd-momentum', 'sgd', 'muon', 'lion'] as const;
const ATTENTIONS = ['flash', 'standard'] as const;
const CHECKPOINTINGS = ['none', 'selective', 'full'] as const;

function emit(io: CliIO, flags: Parsed['flags'], json: unknown, markdown: () => string): void {
  io.stdout(flags.json ? JSON.stringify(json, null, 2) : markdown());
}

export async function runCli(argv: string[], io: CliIO = { stdout: (t) => console.log(t), stderr: (t) => console.error(t) }): Promise<number> {
  const { command, flags } = parseArgs(argv);
  if (!command || command === 'help' || flags.help) {
    io.stdout(USAGE);
    return command || flags.help ? 0 : 1;
  }
  try {
    switch (command) {
      case 'params': {
        const model = modelFromFlags(flags);
        if (!('vocab' in model && 'dModel' in model && 'nLayers' in model && 'nHeads' in model) || 'params' in model) {
          throw new Error('params needs a full shape (--model <preset> or --vocab --d-model --layers --heads)');
        }
        const count = transformerParams(model);
        emit(io, flags, count, () => paramsToMarkdown(count));
        return 0;
      }
      case 'flops': {
        const model = modelFromFlags(flags);
        const perToken = flopsPerToken(model, { seqLen: num(flags, 'seq-len') });
        const tokens = num(flags, 'tokens');
        const total = tokens === undefined ? undefined : perToken.total * tokens;
        emit(io, flags, { ...perToken, tokens, total }, () =>
          [
            `## FLOPs per token: ${formatSci(perToken.total)} (dense ${formatSci(perToken.dense)} + attention ${formatSci(perToken.attention)})`,
            '',
            `Counted parameters: ${formatCount(perToken.paramsCounted)}${perToken.seqLen ? `, sequence length ${perToken.seqLen}` : ''}.`,
            total === undefined ? '' : `Training FLOPs for ${formatCount(tokens as number)} tokens: ${formatSci(total)}.`,
          ]
            .filter((l) => l !== '')
            .join('\n'),
        );
        return 0;
      }
      case 'mfu': {
        const model = modelFromFlags(flags);
        const result = mfu({
          tokensPerSec: need(num(flags, 'tokens-per-sec'), 'tokens-per-sec'),
          gpu: str(flags, 'gpu') ?? 'h100-sxm',
          nGpus: num(flags, 'gpus'),
          model,
          seqLen: num(flags, 'seq-len'),
          precision: oneOf<ComputePrecision>(str(flags, 'precision'), COMPUTE_PRECISIONS, 'precision'),
        });
        emit(io, flags, result, () => mfuToMarkdown(result));
        return 0;
      }
      case 'throughput': {
        const model = modelFromFlags(flags);
        const result = expectedThroughput({
          gpu: str(flags, 'gpu') ?? 'h100-sxm',
          nGpus: num(flags, 'gpus'),
          mfu: num(flags, 'mfu'),
          model,
          seqLen: num(flags, 'seq-len'),
          precision: oneOf<ComputePrecision>(str(flags, 'precision'), COMPUTE_PRECISIONS, 'precision'),
        });
        emit(io, flags, result, () => throughputToMarkdown(result));
        return 0;
      }
      case 'chinchilla': {
        const params = num(flags, 'params');
        const tokens = num(flags, 'tokens');
        const flops = num(flags, 'flops');
        const opts = { ratio: num(flags, 'ratio') };
        const result =
          params !== undefined ? chinchilla({ params }, opts) : tokens !== undefined ? chinchilla({ tokens }, opts) : flops !== undefined ? chinchilla({ flops }, opts) : undefined;
        if (!result) throw new Error('chinchilla needs --params, --tokens or --flops');
        emit(io, flags, result, () => chinchillaToMarkdown(result));
        return 0;
      }
      case 'memory': {
        const model = modelFromFlags(flags);
        const estimate = memoryPerGpu({
          model,
          gpu: str(flags, 'gpu'),
          nGpus: num(flags, 'gpus'),
          sharding: oneOf<Sharding>(str(flags, 'sharding'), SHARDINGS, 'sharding'),
          precision: oneOf<TrainingPrecision>(str(flags, 'precision'), TRAINING_PRECISIONS, 'precision'),
          optimizer: oneOf<Optimizer>(str(flags, 'optimizer'), OPTIMIZERS, 'optimizer'),
          microBatch: num(flags, 'micro-batch'),
          seqLen: num(flags, 'seq-len'),
          attention: oneOf<Attention>(str(flags, 'attention'), ATTENTIONS, 'attention'),
          activationCheckpointing: oneOf<Checkpointing>(str(flags, 'checkpointing'), CHECKPOINTINGS, 'checkpointing'),
          overheadGiB: num(flags, 'overhead'),
        });
        emit(io, flags, estimate, () => memoryToMarkdown(estimate));
        return 0;
      }
      case 'plan': {
        const model = modelFromFlags(flags);
        const plan = trainingPlan({
          model,
          tokens: num(flags, 'tokens'),
          tokensPerParam: num(flags, 'tokens-per-param'),
          gpu: str(flags, 'gpu') ?? 'h100-sxm',
          nGpus: num(flags, 'gpus'),
          mfu: num(flags, 'mfu'),
          pricePerGpuHour: num(flags, 'price'),
          seqLen: num(flags, 'seq-len'),
          precision: oneOf<ComputePrecision>(str(flags, 'precision'), COMPUTE_PRECISIONS, 'precision'),
        });
        emit(io, flags, plan, () => planToMarkdown([plan]));
        return 0;
      }
      case 'table': {
        const sizesText = str(flags, 'sizes') ?? '1B,2B,4B,14B';
        const sizes = sizesText.split(',').map((s) => parseNumber(s.trim(), '--sizes'));
        const plans = planTable(sizes, {
          tokensPerParam: num(flags, 'tokens-per-param'),
          gpu: str(flags, 'gpu') ?? 'h100-sxm',
          nGpus: num(flags, 'gpus'),
          mfu: num(flags, 'mfu'),
          pricePerGpuHour: num(flags, 'price'),
          seqLen: num(flags, 'seq-len'),
          precision: oneOf<ComputePrecision>(str(flags, 'precision'), COMPUTE_PRECISIONS, 'precision'),
        });
        emit(io, flags, plans, () => planToMarkdown(plans));
        return 0;
      }
      case 'gap': {
        const value = need(num(flags, 'mfu'), 'mfu');
        const report = explainGap({
          mfu: value,
          nGpus: num(flags, 'gpus'),
          sharding: oneOf<Sharding>(str(flags, 'sharding'), SHARDINGS, 'sharding'),
          precision: oneOf<TrainingPrecision>(str(flags, 'precision'), TRAINING_PRECISIONS, 'precision'),
          compiled: flags['no-compile'] ? false : flags.compile ? true : undefined,
          attention: oneOf<Attention>(str(flags, 'attention'), ATTENTIONS, 'attention'),
          microBatch: num(flags, 'micro-batch'),
          activationCheckpointing: flags['checkpointing-on'] ? true : undefined,
          multiNode: flags['multi-node'] ? true : undefined,
        });
        emit(io, flags, report, () => gapToMarkdown(report, value));
        return 0;
      }
      case 'suggest': {
        const mlp = str(flags, 'mlp');
        if (mlp !== undefined && mlp !== 'gelu' && mlp !== 'swiglu') throw new Error('--mlp must be gelu or swiglu');
        const result = suggestArch(need(num(flags, 'params'), 'params'), {
          vocab: num(flags, 'vocab'),
          seqLen: num(flags, 'seq-len'),
          aspect: num(flags, 'aspect'),
          dHead: num(flags, 'd-head'),
          mlp: mlp as 'gelu' | 'swiglu' | undefined,
          tiedEmbeddings: flags.untied ? false : undefined,
        });
        emit(io, flags, result, () => paramsToMarkdown(result.count));
        return 0;
      }
      case 'gpus':
        emit(io, flags, Object.values(gpus), () => gpusToMarkdown());
        return 0;
      case 'models':
        io.stdout(flags.json ? JSON.stringify(listModels()) : listModels().join('\n'));
        return 0;
      default:
        io.stderr(`unknown command "${command}"\n\n${USAGE}`);
        return 1;
    }
  } catch (error) {
    io.stderr(`trainmath: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}

const isMain = (() => {
  try {
    const entry = process.argv[1];
    return entry !== undefined && pathToFileURL(realpathSync(entry)).href === import.meta.url;
  } catch {
    return false;
  }
})();

if (isMain) {
  runCli(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
