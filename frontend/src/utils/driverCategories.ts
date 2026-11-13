/** Shared category tone/label maps for model driver chips (Overview + Model focus). */
export const driverCategoryTone: Record<string, string> = {
  Momentum:   'bg-[#dda174]/15 text-[#dda174]',
  Trend:      'bg-[#8eadde]/15 text-[#8eadde]',
  Volatility: 'bg-[#dda174]/10 text-[#e8bfa0]',
  Sentiment:  'bg-[#8eadde]/10 text-[#b4c8ea]',
  Macro:      'bg-[#74c7a9]/15 text-[#74c7a9]',
  Sector:     'bg-[#db8296]/15 text-[#db8296]',
  Embedding:  'bg-[#8eadde]/10 text-[#a3afbf]',
  Other:      'bg-white/[0.04] text-[#a3afbf]',
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
