import { describe, expect, it } from 'vitest';
import { formatQuoteDelta, quoteComparison } from './quote';
describe('quote display semantics', () => {
  it('does not fabricate a change from a fallback close', () => {
    expect(quoteComparison(null, 6.68)).toBeNull();
    expect(quoteComparison(6.68, null)).toBeNull();
    expect(quoteComparison(NaN, 6.68)).toBeNull();
    expect(quoteComparison(6.68, 0)).toBeNull();
  });
  it('renders exact and display-rounded zero as neutral without a signed zero', () => {
    expect(quoteComparison(6.68, 6.68)).toEqual({ delta: 0, percent: 0, tone: 'neutral' });
    expect(quoteComparison(6.67999, 6.68)).toEqual({ delta: 0, percent: 0, tone: 'neutral' });
    expect(formatQuoteDelta(-.001)).toBe('0.00');
    expect(formatQuoteDelta(.001)).toBe('0.00');
    expect(formatQuoteDelta(-.02)).toBe('-0.02');
    expect(formatQuoteDelta(.02)).toBe('+0.02');
  });
  it('preserves the sign and unrounded calculation of real differences', () => {
    expect(quoteComparison(7, 6)?.percent).toBeCloseTo(100 / 6);
    expect(quoteComparison(7, 6)?.tone).toBe('positive');
    expect(quoteComparison(6, 7)?.tone).toBe('negative');
  });
});
