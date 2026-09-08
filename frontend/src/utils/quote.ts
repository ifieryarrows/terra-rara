/** A fallback close is a displayed value, never its own market-change observation. */
export function quoteComparison(live: number | null, close: number | null) {
  if (live == null || close == null || !Number.isFinite(live) || !Number.isFinite(close) || close <= 0) return null;
  const delta = live - close;
  const percent = delta / close * 100;
  // Neutral when both displayed changes round to zero; avoid a green/red signed zero.
  const flat = Math.abs(delta) < .005 && Math.abs(percent) < .005;
  return { delta: flat ? 0 : delta, percent: flat ? 0 : percent, tone: flat ? 'neutral' : delta > 0 ? 'positive' : 'negative' };
}

export function formatQuoteDelta(value: number) {
  const rounded = value.toFixed(2);
  if (Number(rounded) === 0) return '0.00';
  return value > 0 ? `+${rounded}` : rounded;
}
