import React from 'react';
import clsx from 'clsx';
import { Newspaper, Globe } from 'lucide-react';
import type { NewsItem } from '../../types';
import { NEWS_SENTIMENT_STYLES, normaliseNewsLabel } from './newsSentiment';

interface NewsCardProps {
  item: NewsItem;
  onSelect: (item: NewsItem) => void;
  selected?: boolean;
  duplicate?: boolean;
}

const CHANNEL_LABELS: Record<string, string> = {
  google_news: 'GN',
  newsapi: 'NA',
};

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
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatEventType(raw: string | null | undefined): string {
  if (!raw) return '';
  return raw
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export const NewsCard: React.FC<NewsCardProps> = React.memo(({ item, onSelect, selected = false, duplicate = false }) => {
  const sentiment = item.sentiment;
  const label = normaliseNewsLabel(sentiment?.label);
  const style = NEWS_SENTIMENT_STYLES[label];
  const relevancePct = Math.max(0, Math.min(100, Math.round(((sentiment?.relevance ?? 0) * 100))));
  const confidencePct = Math.max(0, Math.min(100, Math.round(((sentiment?.confidence ?? 0) * 100))));
  const channelCode = CHANNEL_LABELS[item.channel] ?? item.channel?.slice(0, 2).toUpperCase() ?? '';

  const handleSelect = () => onSelect(item);

  const reasoning = sentiment?.reasoning?.trim() || null;

  return (
    <button
      type="button"
      tabIndex={duplicate ? -1 : undefined}
      aria-hidden={duplicate || undefined}
      onClick={handleSelect}
      className={clsx(
        'cm-news-card',
        `cm-news-card--${label.toLowerCase()}`,
        'w-full text-left px-3 py-2.5 transition-colors duration-200',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-[#f6c8a7]/60',
        selected && 'cm-news-card--selected',
      )}
      data-cm-route-reveal="surface"
    >
      <div className="cm-news-card-meta flex items-center justify-between gap-1.5 mb-1 text-xs font-mono text-slate-400">
        <div className="flex items-center gap-1 min-w-0">
          <Globe size={10} className="text-copper-400/80 shrink-0" />
          <span className="truncate text-gray-400" title={item.publisher ?? item.channel}>
            {item.publisher ?? 'Unknown publisher'}
          </span>
          {channelCode && (
            <span
              className="cm-news-card-channel px-1 py-0.5 rounded bg-white/5 text-xs tracking-wider text-slate-400 shrink-0"
              title={`Ingestion channel: ${item.channel}`}
            >
              {channelCode}
            </span>
          )}
        </div>
        <span className="shrink-0 whitespace-nowrap">{formatRelativeTime(item.published_at)}</span>
      </div>

      <div className="cm-news-card-headline flex items-start gap-1.5 mb-1.5">
        <Newspaper size={13} className="text-copper-400/70 mt-0.5 shrink-0" aria-hidden="true" />
        <p className="cm-news-card-title text-[13px] text-gray-100 leading-snug line-clamp-2 break-words">{item.title}</p>
      </div>

      {reasoning && (
        <p
          className="cm-news-card-summary text-xs text-gray-400/90 leading-snug line-clamp-2 mb-1.5 pl-[18px] break-words"
          title={reasoning}
        >
          {reasoning}
        </p>
      )}

      <div className="cm-news-card-footer flex items-center gap-1.5 flex-wrap">
        <span
          className={clsx(
            'cm-news-card-sentiment inline-flex items-center gap-1 text-xs font-mono tracking-wider uppercase',
            'px-1.5 py-0.5 rounded-sm',
            style.chip,
          )}
        >
          <span className="text-[10px] leading-none" aria-hidden="true">{style.glyph}</span>
          {style.text}
        </span>

        {sentiment?.event_type && (
          <span
            className="cm-news-card-event text-xs font-mono text-gray-400 bg-white/[0.04] px-1.5 py-0.5 rounded-sm truncate max-w-[110px]"
            title={sentiment.event_type}
          >
            {formatEventType(sentiment.event_type)}
          </span>
        )}

        {sentiment?.scoring_mode === 'deterministic_fallback' && (
          <span
            className="text-xs font-mono text-[#dda174] bg-[#dda174]/10 px-1.5 py-0.5 rounded-sm"
            title={sentiment.fallback_reason || 'Model service unavailable'}
          >
            Local fallback
          </span>
        )}

        <div className="cm-news-card-metrics flex items-center gap-2 ml-auto text-xs font-mono text-slate-400 shrink-0">
          <div className="flex items-center gap-1" title={`Relevance ${relevancePct}%`}>
            <span className="text-slate-400">R</span>
            <div className="w-8 h-[3px] bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-copper-400/80"
                style={{ width: `${relevancePct}%` }}
              />
            </div>
          </div>
          <span title={`Confidence ${confidencePct}%`}>{confidencePct}%</span>
        </div>
      </div>
    </button>
  );
});

NewsCard.displayName = 'NewsCard';

export default NewsCard;
