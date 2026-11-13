import type { NewsLabel } from '../../types';

export interface NewsSentimentVisual {
  chip: string;
  filterTone: string;
  drawerTag: string;
  glyph: string;
  text: string;
  glyphClass?: string;
}

/** Shared news sentiment palette and glyphs so summary counts, cards, and drawer stay aligned. */
export const NEWS_SENTIMENT_STYLES: Record<NewsLabel, NewsSentimentVisual> = {
  BULLISH: {
    chip: 'bg-[#74c7a9]/12 text-[#74c7a9]',
    filterTone: 'bg-[#74c7a9]/15 text-[#74c7a9]',
    drawerTag: 'text-[#74c7a9] bg-[#74c7a9]/15 border-[#74c7a9]/35',
    glyph: '▲',
    text: 'Bullish',
  },
  BEARISH: {
    chip: 'bg-[#db8296]/12 text-[#db8296]',
    filterTone: 'bg-[#db8296]/15 text-[#db8296]',
    drawerTag: 'text-[#db8296] bg-[#db8296]/15 border-[#db8296]/35',
    glyph: '▼',
    text: 'Bearish',
  },
  NEUTRAL: {
    chip: 'bg-white/[0.06] text-[#a3afbf]',
    filterTone: 'bg-white/[0.06] text-[#a3afbf]',
    drawerTag: 'text-[#a3afbf] bg-white/[0.06] border-[#a3afbf]/35',
    glyph: '■',
    text: 'Neutral',
    glyphClass: 'text-[10px]',
  },
};

/** Normalize arbitrary backend label strings to a supported news sentiment key. */
export function normaliseNewsLabel(raw: string | null | undefined): NewsLabel {
  const upper = (raw ?? '').toUpperCase();
  if (upper === 'BULLISH' || upper === 'BEARISH' || upper === 'NEUTRAL') {
    return upper as NewsLabel;
  }
  return 'NEUTRAL';
}
