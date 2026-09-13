export type ComputePrecision = 'bf16' | 'fp16' | 'tf32' | 'fp8' | 'fp32';

export interface GpuSpec {
  id: string;
  name: string;
  vendor: 'NVIDIA' | 'AMD' | 'Google' | 'custom';
  /** Peak dense tensor-core TFLOPS by precision. Sparsity numbers from marketing sheets are halved to get these. */
  peakTflops: { bf16: number; fp16?: number; tf32?: number; fp8?: number; fp32?: number };
  /** Device memory in GiB, as marketed. */
  memoryGiB: number;
  /** Memory bandwidth in TB/s. */
  bandwidthTBs: number;
  /** Where the numbers come from and when they were checked. */
  source: string;
  note?: string;
}

const NVIDIA = 'NVIDIA data sheet, dense (no sparsity), checked 2026-09';
const AMD = 'AMD data sheet, dense (no sparsity), checked 2026-09';
const GOOGLE = 'Google Cloud TPU documentation, checked 2026-09';

const TABLE: GpuSpec[] = [
  { id: 'h100-sxm', name: 'H100 SXM 80GB', vendor: 'NVIDIA', peakTflops: { bf16: 989, fp16: 989, tf32: 495, fp8: 1979, fp32: 67 }, memoryGiB: 80, bandwidthTBs: 3.35, source: NVIDIA },
  { id: 'h100-pcie', name: 'H100 PCIe 80GB', vendor: 'NVIDIA', peakTflops: { bf16: 756, fp16: 756, tf32: 378, fp8: 1513, fp32: 51 }, memoryGiB: 80, bandwidthTBs: 2.0, source: NVIDIA },
  { id: 'h100-nvl', name: 'H100 NVL 94GB', vendor: 'NVIDIA', peakTflops: { bf16: 835, fp16: 835, tf32: 417, fp8: 1671, fp32: 60 }, memoryGiB: 94, bandwidthTBs: 3.9, source: NVIDIA },
  { id: 'h200-sxm', name: 'H200 SXM 141GB', vendor: 'NVIDIA', peakTflops: { bf16: 989, fp16: 989, tf32: 495, fp8: 1979, fp32: 67 }, memoryGiB: 141, bandwidthTBs: 4.8, source: NVIDIA },
  { id: 'a100-sxm-80', name: 'A100 SXM 80GB', vendor: 'NVIDIA', peakTflops: { bf16: 312, fp16: 312, tf32: 156, fp32: 19.5 }, memoryGiB: 80, bandwidthTBs: 2.04, source: NVIDIA, note: 'No FP8 tensor cores.' },
  { id: 'a100-sxm-40', name: 'A100 SXM 40GB', vendor: 'NVIDIA', peakTflops: { bf16: 312, fp16: 312, tf32: 156, fp32: 19.5 }, memoryGiB: 40, bandwidthTBs: 1.56, source: NVIDIA, note: 'No FP8 tensor cores.' },
  { id: 'a100-pcie-80', name: 'A100 PCIe 80GB', vendor: 'NVIDIA', peakTflops: { bf16: 312, fp16: 312, tf32: 156, fp32: 19.5 }, memoryGiB: 80, bandwidthTBs: 1.94, source: NVIDIA, note: 'No FP8 tensor cores.' },
  { id: 'b200-sxm', name: 'B200 SXM 180GB', vendor: 'NVIDIA', peakTflops: { bf16: 2250, fp16: 2250, tf32: 1125, fp8: 4500, fp32: 75 }, memoryGiB: 180, bandwidthTBs: 7.7, source: NVIDIA, note: 'HGX B200 configuration.' },
  { id: 'l40s', name: 'L40S 48GB', vendor: 'NVIDIA', peakTflops: { bf16: 362, fp16: 362, tf32: 183, fp8: 733, fp32: 91.6 }, memoryGiB: 48, bandwidthTBs: 0.864, source: NVIDIA },
  { id: 'rtx-4090', name: 'GeForce RTX 4090 24GB', vendor: 'NVIDIA', peakTflops: { bf16: 165, fp16: 165, tf32: 83, fp8: 330, fp32: 82.6 }, memoryGiB: 24, bandwidthTBs: 1.01, source: NVIDIA, note: 'Consumer card, no NVLink.' },
  { id: 'mi300x', name: 'Instinct MI300X 192GB', vendor: 'AMD', peakTflops: { bf16: 1307, fp16: 1307, tf32: 653, fp8: 2615, fp32: 163 }, memoryGiB: 192, bandwidthTBs: 5.3, source: AMD },
  { id: 'tpu-v4', name: 'TPU v4', vendor: 'Google', peakTflops: { bf16: 275 }, memoryGiB: 32, bandwidthTBs: 1.2, source: GOOGLE, note: 'Per chip.' },
  { id: 'tpu-v5e', name: 'TPU v5e', vendor: 'Google', peakTflops: { bf16: 197, fp8: 394 }, memoryGiB: 16, bandwidthTBs: 0.82, source: GOOGLE, note: 'Per chip. The fp8 figure is INT8.' },
  { id: 'tpu-v5p', name: 'TPU v5p', vendor: 'Google', peakTflops: { bf16: 459 }, memoryGiB: 95, bandwidthTBs: 2.76, source: GOOGLE, note: 'Per chip.' },
  { id: 'tpu-v6e', name: 'TPU v6e (Trillium)', vendor: 'Google', peakTflops: { bf16: 918, fp8: 1836 }, memoryGiB: 32, bandwidthTBs: 1.64, source: GOOGLE, note: 'Per chip. The fp8 figure is INT8.' },
];

const ALIASES: Record<string, string> = {
  h100: 'h100-sxm',
  'h100-80': 'h100-sxm',
  'h100-sxm5': 'h100-sxm',
  h200: 'h200-sxm',
  a100: 'a100-sxm-80',
  'a100-80': 'a100-sxm-80',
  'a100-40': 'a100-sxm-40',
  b200: 'b200-sxm',
  '4090': 'rtx-4090',
  rtx4090: 'rtx-4090',
  mi300: 'mi300x',
  v4: 'tpu-v4',
  v5e: 'tpu-v5e',
  v5p: 'tpu-v5p',
  v6e: 'tpu-v6e',
  trillium: 'tpu-v6e',
};

/** Built-in accelerator table, keyed by id. Override or extend by passing your own GpuSpec anywhere a gpu is accepted. */
export const gpus: Readonly<Record<string, GpuSpec>> = Object.freeze(
  Object.fromEntries(TABLE.map((g) => [g.id, Object.freeze({ ...g, peakTflops: Object.freeze({ ...g.peakTflops }) })])),
);

export function listGpus(): GpuSpec[] {
  return TABLE.map((g) => gpus[g.id] as GpuSpec);
}

/** Accepts an id ("h100-sxm"), an alias ("h100", "a100") or a full GpuSpec of your own. */
export function resolveGpu(gpu: string | GpuSpec): GpuSpec {
  if (typeof gpu !== 'string') {
    if (!gpu || typeof gpu.peakTflops?.bf16 !== 'number' || typeof gpu.memoryGiB !== 'number') {
      throw new Error('custom gpu needs at least { id, name, peakTflops: { bf16 }, memoryGiB }');
    }
    return gpu;
  }
  const key = gpu.trim().toLowerCase().replace(/[\s_]+/g, '-');
  const found = gpus[key] ?? gpus[ALIASES[key] ?? ''];
  if (!found) {
    throw new Error(`unknown gpu "${gpu}"; known ids: ${TABLE.map((g) => g.id).join(', ')}`);
  }
  return found;
}

/** Peak dense throughput in FLOP/s (not TFLOPS) for one device at the given precision. */
export function peakFlops(gpu: string | GpuSpec, precision: ComputePrecision = 'bf16'): number {
  const spec = resolveGpu(gpu);
  const tflops = spec.peakTflops[precision];
  if (typeof tflops !== 'number') {
    throw new Error(`${spec.id} has no ${precision} peak listed (available: ${Object.keys(spec.peakTflops).join(', ')})`);
  }
  return tflops * 1e12;
}
