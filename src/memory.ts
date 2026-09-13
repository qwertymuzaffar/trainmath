import { GIB } from './format.js';
import { resolveGpu, type GpuSpec } from './gpus.js';
import { transformerParams } from './params.js';
import { assertPositive, assertPositiveInt, isArch, type ModelSpec } from './types.js';

export type Sharding = 'none' | 'ddp' | 'zero1' | 'zero2' | 'zero3' | 'fsdp';
export type TrainingPrecision = 'bf16-mixed' | 'fp16-mixed' | 'fp32' | 'bf16';
export type Optimizer = 'adamw' | 'adam' | 'adamw-8bit' | 'adafactor' | 'sgd-momentum' | 'sgd' | 'muon' | 'lion';
export type Attention = 'flash' | 'standard';
export type Checkpointing = 'none' | 'selective' | 'full';

export interface MemoryInput {
  model: ModelSpec;
  /** Needed for the fits verdict. */
  gpu?: string | GpuSpec;
  /** Data-parallel devices the states are sharded across. Default 1. */
  nGpus?: number;
  /** How the states are split across devices. Default 'ddp' (everything replicated). */
  sharding?: Sharding;
  /** Default 'bf16-mixed': bf16 weights and grads with fp32 master weights and optimizer states. */
  precision?: TrainingPrecision;
  /** Default 'adamw'. */
  optimizer?: Optimizer;
  /** Sequences per device per micro-step. Default 1. */
  microBatch?: number;
  seqLen?: number;
  /** 'flash' (SDPA or FlashAttention, scores never materialized) or 'standard'. Default 'flash'. */
  attention?: Attention;
  /** Activation checkpointing. Default 'none'. */
  activationCheckpointing?: Checkpointing;
  /** CUDA context, allocator fragmentation and kernel workspaces, in GiB. Default 2. */
  overheadGiB?: number;
}

export interface BytesPerParam {
  weights: number;
  grads: number;
  master: number;
  optimizer: number;
  total: number;
}

export interface MemoryEstimate {
  perGpuGiB: {
    weights: number;
    grads: number;
    master: number;
    optimizer: number;
    activations: number;
    logits: number;
    overhead: number;
    total: number;
  };
  /** Unsharded bytes per parameter for the chosen precision and optimizer. */
  bytesPerParam: BytesPerParam;
  params: number;
  nGpus: number;
  sharding: Sharding;
  precision: TrainingPrecision;
  optimizer: Optimizer;
  gpu?: GpuSpec;
  capacityGiB?: number;
  fits?: boolean;
  headroomGiB?: number;
  notes: string[];
}

const PRECISION: Record<TrainingPrecision, { weights: number; grads: number; master: number }> = {
  'bf16-mixed': { weights: 2, grads: 2, master: 4 },
  'fp16-mixed': { weights: 2, grads: 2, master: 4 },
  fp32: { weights: 4, grads: 4, master: 0 },
  bf16: { weights: 2, grads: 2, master: 0 },
};

const OPTIMIZER: Record<Optimizer, { bytes: number; note?: string }> = {
  adamw: { bytes: 8 },
  adam: { bytes: 8 },
  'adamw-8bit': { bytes: 2, note: '8-bit Adam keeps both moments in one byte each (bitsandbytes).' },
  adafactor: { bytes: 0.1, note: 'Adafactor factors the second moment into row and column vectors; 0.1 byte per parameter is a rough allowance.' },
  'sgd-momentum': { bytes: 4 },
  sgd: { bytes: 0 },
  muon: { bytes: 4, note: 'Muon keeps one momentum buffer. Embeddings, head and norms usually stay on AdamW; that share is not counted.' },
  lion: { bytes: 4 },
};

function divisors(sharding: Sharding, n: number): { weights: number; grads: number; master: number; optimizer: number } {
  switch (sharding) {
    case 'none':
    case 'ddp':
      return { weights: 1, grads: 1, master: 1, optimizer: 1 };
    case 'zero1':
      return { weights: 1, grads: 1, master: n, optimizer: n };
    case 'zero2':
      return { weights: 1, grads: n, master: n, optimizer: n };
    case 'zero3':
    case 'fsdp':
      return { weights: n, grads: n, master: n, optimizer: n };
    default:
      throw new Error(`unknown sharding "${String(sharding)}"`);
  }
}

/**
 * Per-device training memory. States follow the ZeRO accounting (Rajbhandari et al. 2020): with AdamW and
 * mixed precision, 16 bytes per parameter before sharding. Activations follow Korthikanti et al. 2022
 * ("Reducing Activation Recomputation in Large Transformer Models"): per block, s x b x h x (34 + 5 a s / h)
 * bytes with materialized attention scores, 34 s b h with flash attention or selective checkpointing,
 * and 2 s b h with full checkpointing. Logits are counted in fp32. Everything is an estimate.
 */
export function memoryPerGpu(input: MemoryInput): MemoryEstimate {
  const nGpus = input.nGpus ?? 1;
  assertPositiveInt(nGpus, 'nGpus');
  const sharding = input.sharding ?? 'ddp';
  const precision = input.precision ?? 'bf16-mixed';
  const optimizer = input.optimizer ?? 'adamw';
  const microBatch = input.microBatch ?? 1;
  assertPositiveInt(microBatch, 'microBatch');
  const attention = input.attention ?? 'flash';
  const ckpt = input.activationCheckpointing ?? 'none';
  const overheadGiB = input.overheadGiB ?? 2;
  const notes: string[] = [];

  const prec = PRECISION[precision];
  if (!prec) throw new Error(`unknown precision "${String(precision)}"`);
  const opt = OPTIMIZER[optimizer];
  if (!opt) throw new Error(`unknown optimizer "${String(optimizer)}"`);
  if (opt.note) notes.push(opt.note);

  const bytesPerParam: BytesPerParam = {
    weights: prec.weights,
    grads: prec.grads,
    master: prec.master,
    optimizer: opt.bytes,
    total: prec.weights + prec.grads + prec.master + opt.bytes,
  };

  let params: number;
  let shape: { nLayers: number; dModel: number; nHeads: number; seqLen: number; vocab?: number; perLayer: number } | undefined;
  if (isArch(input.model)) {
    const p = transformerParams(input.model);
    params = p.total;
    shape = {
      nLayers: p.arch.nLayers,
      dModel: p.arch.dModel,
      nHeads: p.arch.nHeads,
      seqLen: input.seqLen ?? p.arch.seqLen,
      vocab: p.arch.vocab,
      perLayer: p.perLayer,
    };
  } else {
    assertPositive(input.model.params, 'params');
    params = input.model.params;
    const m = input.model;
    const seqLen = input.seqLen ?? m.seqLen;
    if (m.nLayers && m.dModel && m.nHeads && seqLen) {
      shape = { nLayers: m.nLayers, dModel: m.dModel, nHeads: m.nHeads, seqLen, vocab: m.vocab, perLayer: params / m.nLayers };
    }
  }

  const div = divisors(sharding, nGpus);
  let weightsBytes = (params * prec.weights) / div.weights;
  const gradsBytes = (params * prec.grads) / div.grads;
  const masterBytes = (params * prec.master) / div.master;
  const optimizerBytes = (params * opt.bytes) / div.optimizer;
  if ((sharding === 'fsdp' || sharding === 'zero3') && nGpus > 1 && shape) {
    weightsBytes += shape.perLayer * prec.weights;
    notes.push('FSDP/ZeRO-3 gathers one block at a time; one unsharded block of weights is added.');
  }

  let activationsBytes = 0;
  let logitsBytes = 0;
  if (shape) {
    const { nLayers, dModel, nHeads, seqLen } = shape;
    const sbh = seqLen * microBatch * dModel;
    const fullBlock = attention === 'standard' ? sbh * (34 + (5 * nHeads * seqLen) / dModel) : 34 * sbh;
    let perBlock: number;
    if (ckpt === 'full') perBlock = 2 * sbh;
    else if (ckpt === 'selective') perBlock = 34 * sbh;
    else perBlock = fullBlock;
    activationsBytes = nLayers * perBlock;
    if (ckpt === 'full') {
      activationsBytes += fullBlock;
      notes.push('Full checkpointing keeps only block inputs; one block is recomputed at a time, so one full block is added.');
    }
    if (shape.vocab) {
      logitsBytes = seqLen * microBatch * shape.vocab * 4;
      notes.push('Logits counted once in fp32; frameworks that keep a bf16 copy or chunk the loss differ.');
    }
  } else {
    notes.push('No shape given (nLayers, dModel, nHeads, seqLen), so activations and logits are not estimated.');
  }

  const perGpuGiB = {
    weights: weightsBytes / GIB,
    grads: gradsBytes / GIB,
    master: masterBytes / GIB,
    optimizer: optimizerBytes / GIB,
    activations: activationsBytes / GIB,
    logits: logitsBytes / GIB,
    overhead: overheadGiB,
    total: 0,
  };
  perGpuGiB.total =
    perGpuGiB.weights + perGpuGiB.grads + perGpuGiB.master + perGpuGiB.optimizer + perGpuGiB.activations + perGpuGiB.logits + perGpuGiB.overhead;

  const estimate: MemoryEstimate = { perGpuGiB, bytesPerParam, params, nGpus, sharding, precision, optimizer, notes };
  if (input.gpu !== undefined) {
    const gpu = resolveGpu(input.gpu);
    estimate.gpu = gpu;
    estimate.capacityGiB = gpu.memoryGiB;
    estimate.headroomGiB = gpu.memoryGiB - perGpuGiB.total;
    estimate.fits = perGpuGiB.total <= gpu.memoryGiB;
  }
  return estimate;
}
