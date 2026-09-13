import { assertPositiveInt, type Arch } from './types.js';

export type NormalizedArch = Required<Arch>;

export interface ParamBreakdown {
  /** Q, K, V and output projections in all blocks (with their biases). */
  attention: number;
  /** Feed-forward matrices in all blocks (with their biases). */
  mlp: number;
  /** Per-block norms plus the final norm. */
  norms: number;
  /** Input token embeddings (vocab x dModel). */
  embeddings: number;
  /** Learned positional embeddings, if any. */
  positions: number;
  /** Separate output head when embeddings are untied. */
  head: number;
}

export interface ParamCount {
  /** All trainable parameters. */
  total: number;
  /** Parameters in the transformer blocks and the final norm. This is N in Kaplan-style scaling laws. */
  nonEmbedding: number;
  /** Input embeddings, learned positions and the untied output head. */
  embedding: number;
  /** Parameters in one transformer block. */
  perLayer: number;
  breakdown: ParamBreakdown;
  arch: NormalizedArch;
}

/** Fills in defaults and validates the shape. */
export function normalizeArch(arch: Arch): NormalizedArch {
  assertPositiveInt(arch.vocab, 'vocab');
  assertPositiveInt(arch.dModel, 'dModel');
  assertPositiveInt(arch.nLayers, 'nLayers');
  assertPositiveInt(arch.nHeads, 'nHeads');
  if (arch.dModel % arch.nHeads !== 0) throw new Error(`dModel (${arch.dModel}) must be divisible by nHeads (${arch.nHeads})`);
  const nKvHeads = arch.nKvHeads ?? arch.nHeads;
  assertPositiveInt(nKvHeads, 'nKvHeads');
  if (arch.nHeads % nKvHeads !== 0) throw new Error(`nHeads (${arch.nHeads}) must be divisible by nKvHeads (${nKvHeads})`);
  const mlp = arch.mlp ?? 'gelu';
  if (mlp !== 'gelu' && mlp !== 'swiglu') throw new Error(`mlp must be 'gelu' or 'swiglu', got ${String(mlp)}`);
  const dFf = arch.dFf ?? (mlp === 'swiglu' ? Math.ceil(((8 / 3) * arch.dModel) / 256) * 256 : 4 * arch.dModel);
  assertPositiveInt(dFf, 'dFf');
  const seqLen = arch.seqLen ?? 2048;
  assertPositiveInt(seqLen, 'seqLen');
  return {
    vocab: arch.vocab,
    dModel: arch.dModel,
    nLayers: arch.nLayers,
    nHeads: arch.nHeads,
    nKvHeads,
    dFf,
    mlp,
    tiedEmbeddings: arch.tiedEmbeddings ?? true,
    seqLen,
    bias: arch.bias ?? false,
    learnedPositions: arch.learnedPositions ?? false,
  };
}

/**
 * Counts parameters of a dense decoder-only transformer exactly from its shape.
 * Checked against GPT-2 124M (124,439,808), GPT-3 175B and Llama 3 8B (8.03B).
 */
export function transformerParams(arch: Arch): ParamCount {
  const a = normalizeArch(arch);
  const d = a.dModel;
  const dHead = d / a.nHeads;
  const kvDim = a.nKvHeads * dHead;

  const attentionWeights = d * d + 2 * d * kvDim + d * d;
  const attentionBias = a.bias ? d + 2 * kvDim + d : 0;
  const mlpWeights = a.mlp === 'swiglu' ? 3 * d * a.dFf : 2 * d * a.dFf;
  const mlpBias = a.bias ? (a.mlp === 'swiglu' ? 2 * a.dFf + d : a.dFf + d) : 0;
  const normPerBlock = a.bias ? 4 * d : 2 * d;
  const finalNorm = a.bias ? 2 * d : d;

  const attention = a.nLayers * (attentionWeights + attentionBias);
  const mlp = a.nLayers * (mlpWeights + mlpBias);
  const norms = a.nLayers * normPerBlock + finalNorm;
  const perLayer = attentionWeights + attentionBias + mlpWeights + mlpBias + normPerBlock;
  const embeddings = a.vocab * d;
  const positions = a.learnedPositions ? a.seqLen * d : 0;
  const head = a.tiedEmbeddings ? 0 : a.vocab * d;

  const nonEmbedding = attention + mlp + norms;
  const embedding = embeddings + positions + head;
  return {
    total: nonEmbedding + embedding,
    nonEmbedding,
    embedding,
    perLayer,
    breakdown: { attention, mlp, norms, embeddings, positions, head },
    arch: a,
  };
}
