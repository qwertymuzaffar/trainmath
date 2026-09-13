import { describe, expect, it } from 'vitest';
import { formatCount, formatDuration, formatGiB, formatMoney, formatPercent, formatSci, parseNumber } from './format.js';

describe('parseNumber', () => {
  it('reads plain, scientific and suffixed numbers', () => {
    expect(parseNumber('1e9')).toBe(1e9);
    expect(parseNumber('1B')).toBe(1e9);
    expect(parseNumber('124M')).toBe(124e6);
    expect(parseNumber('20b')).toBe(20e9);
    expect(parseNumber('2.5k')).toBe(2500);
    expect(parseNumber('1.4T')).toBe(1.4e12);
    expect(parseNumber('1,000')).toBe(1000);
    expect(parseNumber(' 42 ')).toBe(42);
    expect(parseNumber(7)).toBe(7);
    expect(parseNumber('.5')).toBe(0.5);
  });

  it('rejects junk', () => {
    expect(() => parseNumber('lots')).toThrow(/cannot parse/);
    expect(() => parseNumber('1e')).toThrow(/cannot parse/);
    expect(() => parseNumber(Number.NaN)).toThrow(/finite/);
  });
});

describe('formatters', () => {
  it('formatCount', () => {
    expect(formatCount(124_439_808)).toBe('124M');
    expect(formatCount(2e10)).toBe('20B');
    expect(formatCount(527_467)).toBe('527k');
    expect(formatCount(1.2e20)).toBe('1.2e20');
    expect(formatCount(1.4e12)).toBe('1.4T');
    expect(formatCount(0)).toBe('0');
    expect(formatCount(42)).toBe('42');
    expect(formatCount(Number.POSITIVE_INFINITY)).toBe('Infinity');
  });

  it('formatSci', () => {
    expect(formatSci(1.2e20)).toBe('1.2e20');
    expect(formatSci(3.14159e23)).toBe('3.14e23');
    expect(formatSci(0)).toBe('0');
    expect(formatSci(6e9, 0)).toBe('6e9');
  });

  it('formatDuration', () => {
    expect(formatDuration(5.25)).toBe('5.3 s');
    expect(formatDuration(42)).toBe('42 s');
    expect(formatDuration(583)).toBe('9.7 min');
    expect(formatDuration(37_917)).toBe('10.5 h');
    expect(formatDuration(606_635)).toBe('7 d');
    expect(formatDuration(7.42e6)).toBe('85.9 d');
    expect(formatDuration(4e7)).toBe('1.3 y');
    expect(formatDuration(-1)).toBe('-1');
  });

  it('formatMoney, formatGiB, formatPercent', () => {
    expect(formatMoney(210.6)).toBe('$211');
    expect(formatMoney(3370.4)).toBe('$3,370');
    expect(formatMoney(2.5)).toBe('$2.50');
    expect(formatGiB(14.90123)).toBe('14.9 GiB');
    expect(formatGiB(1.863)).toBe('1.86 GiB');
    expect(formatGiB(224.3)).toBe('224 GiB');
    expect(formatPercent(0.4123)).toBe('41.2%');
    expect(formatPercent(0.4, 0)).toBe('40%');
  });
});
