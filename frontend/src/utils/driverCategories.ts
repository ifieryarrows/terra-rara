/** Shared category tone/label maps for model driver chips (Overview + Model focus). */
export const driverCategoryTone: Record<string, string> = {
  Momentum:   'bg-copper-500/15 text-copper-300',
  Trend:      'bg-blue-500/15 text-blue-300',
  Volatility: 'bg-amber-500/15 text-amber-300',
  Sentiment:  'bg-violet-500/15 text-violet-300',
  Macro:      'bg-emerald-500/15 text-emerald-300',
  Sector:     'bg-rose-500/15 text-rose-300',
  Embedding:  'bg-slate-500/15 text-slate-300',
  Other:      'bg-white/5 text-gray-400',
};

export const driverCategoryLabel: Record<string, string> = {
  Momentum: 'Momentum signal',
  Trend: 'Trend signal',
  Volatility: 'Price volatility',
  Sentiment: 'News sentiment',
  Macro: 'Macro market',
  Sector: 'Related market',
  Embedding: 'News pattern',
  Other: 'Other factor',
};
