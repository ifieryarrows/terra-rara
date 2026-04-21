import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { Newspaper, TrendingUp, TrendingDown, Minus, Globe } from 'lucide-react';
import type { NewsItem, NewsLabel } from '../../types';

interface NewsCardProps {
  item: NewsItem;
  onSelect: (item: NewsItem) => void;
  selected?: boolean;
}

const LABEL_STYLES: Record<NewsLabel, { chip: string; icon: typeof TrendingUp; text: string }> = {
  BULLISH: {
    chip: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/40',
    icon: TrendingUp,
    text: 'Bullish',
  },
  BEARISH: {
    chip: 'bg-rose-500/15 text-rose-300 border-rose-400/40',
    icon: TrendingDown,
    text: 'Bearish',
  },
  NEUTRAL: {
    chip: 'bg-amber-500/10 text-amber-200 border-amber-400/30',
    icon: Minus,
    text: 'Neutral',
  },
};

const CHANNEL_LABELS: Record<string, string> = {
  google_news: 'GN',
  newsapi: 'NA',
};

function normaliseLabel(raw: string | null | undefined): NewsLabel {
  const upper = (raw ?? '').toUpperCase();
  if (upper === 'BULLISH' || upper === 'BEARISH' || upper === 'NEUTRAL') {
    return upper as NewsLabel;
  }
  return 'NEUTRAL';
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return '';
  const ts = new Date(iso).getTime();
  if (Number.isNaN(ts)) return '';
  const diffMs = Date.now() - ts;
  const diffMin = Math.round(diffMs / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.round(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.round(diffH / 24);
  if (diffD < 7) return `${diffD}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function formatEventType(raw: string | null | undefined): string {
  if (!raw) return '';
  return raw
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export const NewsCard: React.FC<NewsCardProps> = ({ item, onSelect, selected = false }) => {
  const sentiment = item.sentiment;
  const label = normaliseLabel(sentiment?.label);
  const style = LABEL_STYLES[label];
  const Icon = style.icon;
  const relevancePct = Math.max(0, Math.min(100, Math.round(((sentiment?.relevance ?? 0) * 100))));
  const confidencePct = Math.max(0, Math.min(100, Math.round(((sentiment?.confidence ?? 0) * 100))));
  const channelCode = CHANNEL_LABELS[item.channel] ?? item.channel?.slice(0, 2).toUpperCase() ?? '';

  const handleSelect = () => onSelect(item);

  return (
    <motion.button
      type="button"
      onClick={handleSelect}
      className={clsx(
        'w-full text-left rounded-xl border p-3 transition-all duration-200',
        'bg-midnight/50 hover:bg-midnight/80',
        'border-white/5 hover:border-copper-400/40',
        'focus:outline-none focus:ring-2 focus:ring-copper-400/50',
        selected && 'border-copper-400/70 bg-midnight/80 shadow-lg',
      )}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5 text-[10px] font-mono text-gray-500">
        <div className="flex items-center gap-1.5 min-w-0">
          <Globe size={11} className="text-copper-400/80 shrink-0" />
          <span className="truncate text-gray-400" title={item.publisher ?? item.channel}>
            {item.publisher ?? 'Unknown publisher'}
          </span>
          {channelCode && (
            <span
              className="ml-1 px-1.5 py-0.5 rounded bg-white/5 text-[9px] tracking-wider text-gray-500 shrink-0"
              title={`Ingestion channel: ${item.channel}`}
            >
              {channelCode}
            </span>
          )}
        </div>
        <span className="shrink-0">{formatRelativeTime(item.published_at)}</span>
      </div>

      <div className="flex items-start gap-2 mb-2">
        <Newspaper size={14} className="text-copper-400/70 mt-0.5 shrink-0" />
        <p className="text-sm text-gray-100 leading-snug line-clamp-2">{item.title}</p>
      </div>

      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span
          className={clsx(
            'inline-flex items-center gap-1 text-[10px] font-mono tracking-wider uppercase',
            'px-2 py-0.5 rounded-full border',
            style.chip,
          )}
        >
          <Icon size={10} />
          {style.text}
        </span>

        {sentiment?.event_type && (
          <span
            className="text-[10px] font-mono text-gray-400 bg-white/5 px-2 py-0.5 rounded-full"
            title={sentiment.event_type}
          >
            {formatEventType(sentiment.event_type)}
          </span>
        )}

        <div className="flex items-center gap-3 ml-auto text-[10px] font-mono text-gray-500">
          <div className="flex items-center gap-1" title={`Relevance ${relevancePct}%`}>
            <span className="text-gray-600">R</span>
            <div className="w-10 h-1 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-copper-400/80"
                style={{ width: `${relevancePct}%` }}
              />
            </div>
          </div>
          <span title={`Confidence ${confidencePct}%`}>{confidencePct}%</span>
        </div>
      </div>
    </motion.button>
  );
};

export default NewsCard;
