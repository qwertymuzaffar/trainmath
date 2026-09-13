import type { Arch } from './types.js';

/** Well-known dense shapes, for reference and tests. Counts are exact for GPT-2 and Llama 3, within 0.5% for GPT-3. */
export const models: Readonly<Record<string, Readonly<Arch>>> = Object.freeze({
  'gpt2-124m': Object.freeze({ vocab: 50257, dModel: 768, nLayers: 12, nHeads: 12, seqLen: 1024, mlp: 'gelu', bias: true, learnedPositions: true, tiedEmbeddings: true }),
  'gpt2-350m': Object.freeze({ vocab: 50257, dModel: 1024, nLayers: 24, nHeads: 16, seqLen: 1024, mlp: 'gelu', bias: true, learnedPositions: true, tiedEmbeddings: true }),
  'gpt2-774m': Object.freeze({ vocab: 50257, dModel: 1280, nLayers: 36, nHeads: 20, seqLen: 1024, mlp: 'gelu', bias: true, learnedPositions: true, tiedEmbeddings: true }),
  'gpt2-1.6b': Object.freeze({ vocab: 50257, dModel: 1600, nLayers: 48, nHeads: 25, seqLen: 1024, mlp: 'gelu', bias: true, learnedPositions: true, tiedEmbeddings: true }),
  'gpt3-175b': Object.freeze({ vocab: 50257, dModel: 12288, nLayers: 96, nHeads: 96, seqLen: 2048, mlp: 'gelu', bias: true, learnedPositions: true, tiedEmbeddings: true }),
  'llama3-8b': Object.freeze({ vocab: 128256, dModel: 4096, nLayers: 32, nHeads: 32, nKvHeads: 8, dFf: 14336, seqLen: 8192, mlp: 'swiglu', tiedEmbeddings: false }),
  'llama3-70b': Object.freeze({ vocab: 128256, dModel: 8192, nLayers: 80, nHeads: 64, nKvHeads: 8, dFf: 28672, seqLen: 8192, mlp: 'swiglu', tiedEmbeddings: false }),
});

export function listModels(): string[] {
  return Object.keys(models);
}

export function resolveModel(name: string): Arch {
  const key = name.trim().toLowerCase();
  const found = models[key];
  if (!found) throw new Error(`unknown model "${name}"; known: ${listModels().join(', ')}`);
  return { ...found };
}
