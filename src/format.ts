const SUFFIX: Record<string, number> = { k: 1e3, m: 1e6, b: 1e9, g: 1e9, t: 1e12 };

/** Parses "1e9", "1B", "124M", "20B", "2.5k", "1,000" or a plain number. Throws on anything else. */
export function parseNumber(value: number | string, name = 'value'): number {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${name} must be a finite number`);
    return value;
  }
  const text = String(value).trim().replace(/[,_\s]/g, '');
  const m = /^([+-]?(?:\d+\.?\d*|\.\d+))(?:[eE]([+-]?\d+))?([kKmMbBgGtT])?$/.exec(text);
  if (!m) throw new Error(`${name}: cannot parse "${value}" (use 1e9, 1B, 124M, 20B or a plain number)`);
  const base = Number(m[1]) * (m[2] ? 10 ** Number(m[2]) : 1);
  const suffix = m[3] ? (SUFFIX[m[3].toLowerCase()] ?? 1) : 1;
  return base * suffix;
}

function trimZeros(s: string): string {
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** 1.2e20 -> "1.2e20", 3.14159e23 -> "3.14e23". */
export function formatSci(n: number, digits = 2): string {
  if (n === 0) return '0';
  const [mantissa, exponent] = n.toExponential(digits).split('e');
  return `${trimZeros(mantissa ?? '0')}e${Number(exponent)}`;
}

/** 124439808 -> "124M", 2e10 -> "20B", 1.2e20 -> "1.2e20", 527467 -> "527k". */
export function formatCount(n: number, digits = 3): string {
  if (!Number.isFinite(n)) return String(n);
  const abs = Math.abs(n);
  if (abs === 0) return '0';
  if (abs >= 1e15) return formatSci(n, Math.max(0, digits - 1));
  const units: Array<[number, string]> = [
    [1e12, 'T'],
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'k'],
  ];
  for (const [factor, symbol] of units) {
    if (abs >= factor) return trimZeros((n / factor).toPrecision(digits)) + symbol;
  }
  return trimZeros(n.toPrecision(digits));
}

/** Seconds -> "42 s", "9.7 min", "10.5 h", "7.0 d", "1.2 y". */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return String(seconds);
  if (seconds < 60) return `${trimZeros(seconds.toFixed(seconds < 10 ? 1 : 0))} s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${trimZeros(minutes.toFixed(1))} min`;
  const hours = minutes / 60;
  if (hours < 48) return `${trimZeros(hours.toFixed(1))} h`;
  const days = hours / 24;
  if (days < 365) return `${trimZeros(days.toFixed(1))} d`;
  return `${trimZeros((days / 365).toFixed(1))} y`;
}

/** US dollars: 210.6 -> "$211", 3370.4 -> "$3,370", 2.5 -> "$2.50". */
export function formatMoney(usd: number): string {
  if (!Number.isFinite(usd)) return String(usd);
  if (Math.abs(usd) >= 100) return `$${Math.round(usd).toLocaleString('en-US')}`;
  return `$${usd.toFixed(2)}`;
}

/** 14.90123 -> "14.9 GiB". */
export function formatGiB(gib: number): string {
  if (!Number.isFinite(gib)) return String(gib);
  if (gib >= 100) return `${Math.round(gib).toLocaleString('en-US')} GiB`;
  if (gib >= 10) return `${trimZeros(gib.toFixed(1))} GiB`;
  return `${trimZeros(gib.toFixed(2))} GiB`;
}

/** 0.4123 -> "41.2%". */
export function formatPercent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export const GIB = 2 ** 30;
