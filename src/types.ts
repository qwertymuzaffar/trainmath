/** A dense decoder-only transformer, described by its shape. */
export interface Arch {
  /** Vocabulary size. */
  vocab: number;
  /** Hidden size (d_model). */
  dModel: number;
  /** Number of transformer blocks. */
  nLayers: number;
  /** Number of attention (query) heads. */
  nHeads: number;
  /** Key/value heads for grouped-query attention. Defaults to nHeads (plain multi-head attention). */
  nKvHeads?: number;
  /** Feed-forward hidden size. Defaults to 4 * dModel for 'gelu' and to (8/3) * dModel rounded up to a multiple of 256 for 'swiglu'. */
  dFf?: number;
  /** MLP kind: 'gelu' has two weight matrices (GPT-2, GPT-3), 'swiglu' has three (Llama). Default 'gelu'. */
  mlp?: 'gelu' | 'swiglu';
  /** Whether the output head reuses the input embedding matrix. Default true (GPT-2). Llama 3 8B uses false. */
  tiedEmbeddings?: boolean;
  /** Training sequence length. Default 2048. */
  seqLen?: number;
  /** Biases on linear layers and norms. Default false. GPT-2 and GPT-3 use true. */
  bias?: boolean;
  /** Learned positional embeddings (seqLen x dModel). Default false (rotary or none). GPT-2 and GPT-3 use true. */
  learnedPositions?: boolean;
}

/** A model known only by its parameter count, with optional shape hints for the attention FLOPs and activation memory. */
export interface ParamsSpec {
  /** Total trainable parameters. All of them are counted in the FLOPs estimate. */
  params: number;
  nLayers?: number;
  dModel?: number;
  nHeads?: number;
  vocab?: number;
  seqLen?: number;
}

export type ModelSpec = Arch | ParamsSpec;

export function isArch(model: ModelSpec): model is Arch {
  const m = model as Partial<Arch> & Partial<ParamsSpec>;
  return (
    typeof m.params !== 'number' &&
    typeof m.vocab === 'number' &&
    typeof m.dModel === 'number' &&
    typeof m.nLayers === 'number' &&
    typeof m.nHeads === 'number'
  );
}

export function assertPositive(value: number, name: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive number, got ${String(value)}`);
  }
}

export function assertPositiveInt(value: number, name: string): void {
  assertPositive(value, name);
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer, got ${value}`);
}
