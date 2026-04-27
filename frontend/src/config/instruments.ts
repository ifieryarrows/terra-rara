export const COPPER_INSTRUMENT = {
  canonicalSymbol: 'HG=F',
  displayName: 'COMEX Copper Futures',
  shortLabel: 'HG=F',
  tradingViewSymbol: 'COMEX:HG1!',
  tradingViewExchange: 'COMEX',
  tradingViewTicker: 'HG1!',
  tradingViewLabel: 'COMEX HG1!',
  yahooStreamSymbol: 'HG=F',
  quoteDelayLabel: 'Delayed futures quote',
} as const;

export const DEFAULT_COPPER_SYMBOL = COPPER_INSTRUMENT.canonicalSymbol;
