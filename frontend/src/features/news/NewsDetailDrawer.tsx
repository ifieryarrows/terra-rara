import React, { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X, ExternalLink, Globe, Radio } from 'lucide-react';
import clsx from 'clsx';
import type { NewsItem, NewsLabel } from '../../types';
import { useNewsDetail } from '../../hooks/useNews';

interface NewsDetailDrawerProps {
  item: NewsItem | null;
  onClose: () => void;
}

const LABEL_STYLES: Record<NewsLabel, string> = {
  BULLISH: 'text-emerald-300 bg-emerald-500/15 border-emerald-400/40',
  BEARISH: 'text-rose-300 bg-rose-500/15 border-rose-400/40',
  NEUTRAL: 'text-amber-200 bg-amber-500/10 border-amber-400/30',
};

function normaliseLabel(raw: string | null | undefined): NewsLabel {
  const upper = (raw ?? '').toUpperCase();
  if (upper === 'BULLISH' || upper === 'BEARISH' || upper === 'NEUTRAL') {
    return upper as NewsLabel;
  }
  return 'NEUTRAL';
}

function formatEventType(raw: string | null | undefined): string {
  if (!raw) return '—';
  return raw
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="w-10 text-gray-400 uppercase tracking-wider">{label}</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div className={clsx('h-full', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right text-gray-300">{pct}%</span>
    </div>
  );
}

export const NewsDetailDrawer: React.FC<NewsDetailDrawerProps> = ({ item, onClose }) => {
  const processedId = item?.id ?? null;
  // Fetch a fresh copy when a card is opened — the feed row may be stale,
  // and the detail endpoint is the authoritative source for reasoning text.
  const { data: freshItem } = useNewsDetail(processedId);
  const displayed = freshItem ?? item;
  const panelRef = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose);
  const isOpen = Boolean(item);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);

  useEffect(() => {
    if (!isOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); closeRef.current(); }
      if (e.key !== 'Tab') return;
      const nodes = panelRef.current?.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex="0"]');
      if (!nodes?.length) { e.preventDefault(); return; }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (e.shiftKey && (document.activeElement === first || document.activeElement === panelRef.current)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {item && displayed && (
        <>
          <motion.div
            key="news-drawer-backdrop"
            className="fixed inset-0 z-[60] bg-black/60"
            aria-hidden="true"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="news-drawer-title"
            tabIndex={-1}
            key="news-drawer-panel"
            className="fixed top-0 right-0 bottom-0 z-[70] w-full sm:w-[480px] bg-cm-surface border-l border-cm-border shadow-2xl overflow-y-auto"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between px-5 py-3 bg-midnight/95 border-b border-white/10">
              <div id="news-drawer-title" className="flex items-center gap-2 text-sm uppercase tracking-widest text-copper-400 font-semibold">
                <Radio size={14} />
                News detail
              </div>
              <button
                type="button"
                onClick={onClose}
                className="min-h-11 min-w-11 grid place-items-center rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>

            <div className="p-5 space-y-5">
              <div className="flex items-center gap-2 flex-wrap text-xs font-mono text-gray-400">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/5">
                  <Globe size={10} className="text-copper-400/80" />
                  {displayed.publisher ?? 'Unknown publisher'}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-white/5 tracking-wider uppercase">
                  {displayed.channel}
                </span>
                {displayed.language && (
                  <span className="px-2 py-0.5 rounded-full bg-white/5 uppercase">
                    {displayed.language}
                  </span>
                )}
                {displayed.published_at && (
                  <span>
                    {new Date(displayed.published_at).toLocaleString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                )}
              </div>

              <h2 className="text-lg font-semibold text-white leading-snug">{displayed.title}</h2>

              {displayed.description && (
                <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {displayed.description}
                </p>
              )}

              {displayed.sentiment && (
                <div className="space-y-3 rounded-xl border border-white/5 bg-white/[0.03] p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-widest text-gray-400 font-semibold">
                      Sentiment
                    </span>
                    <span
                      className={clsx(
                        'text-xs font-mono tracking-wider uppercase px-2 py-0.5 rounded-full border',
                        LABEL_STYLES[normaliseLabel(displayed.sentiment.label)],
                      )}
                    >
                      {displayed.sentiment.label ?? 'NEUTRAL'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                    <Stat label="Final score" value={displayed.sentiment.final_score} signed />
                    <Stat label="LLM impact" value={displayed.sentiment.impact_score_llm} signed />
                    <Stat label="Confidence" value={displayed.sentiment.confidence} percent />
                    <Stat label="Relevance" value={displayed.sentiment.relevance} percent />
                  </div>

                  <div className="text-xs font-mono text-gray-400 flex items-center justify-between">
                    <span>Event type</span>
                    <span className="text-gray-200">{formatEventType(displayed.sentiment.event_type)}</span>
                  </div>

                  {displayed.sentiment.finbert && (
                    <div className="space-y-1.5 pt-2 border-t border-white/5">
                      <div className="text-xs uppercase tracking-widest text-slate-400 mb-1">
                        FinBERT probabilities
                      </div>
                      <ProbBar label="Pos" value={displayed.sentiment.finbert.pos} color="bg-emerald-400/80" />
                      <ProbBar label="Neu" value={displayed.sentiment.finbert.neu} color="bg-amber-400/70" />
                      <ProbBar label="Neg" value={displayed.sentiment.finbert.neg} color="bg-rose-400/80" />
                    </div>
                  )}

                  {displayed.sentiment.reasoning && (
                    <div className="pt-3 border-t border-white/5">
                      <div className="text-xs uppercase tracking-widest text-slate-400 mb-1">
                        LLM rationale
                      </div>
                      <p className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                        {displayed.sentiment.reasoning}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {displayed.url && (
                <a
                  href={displayed.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-sm text-copper-300 hover:text-copper-200 font-medium"
                >
                  Read full article
                  <ExternalLink size={14} />
                </a>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
};

function Stat({
  label,
  value,
  signed = false,
  percent = false,
}: {
  label: string;
  value: number | null | undefined;
  signed?: boolean;
  percent?: boolean;
}) {
  let display = '—';
  let tone = 'text-gray-200';
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (percent) {
      display = `${Math.round(value * 100)}%`;
    } else if (signed) {
      display = value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
      tone = value > 0 ? 'text-emerald-300' : value < 0 ? 'text-rose-300' : 'text-gray-200';
    } else {
      display = value.toFixed(3);
    }
  }
  return (
    <div className="rounded-lg bg-white/[0.02] px-2 py-1.5">
      <div className="text-xs uppercase tracking-widest text-slate-400">{label}</div>
      <div className={clsx('text-sm', tone)}>{display}</div>
    </div>
  );
}

export default NewsDetailDrawer;
