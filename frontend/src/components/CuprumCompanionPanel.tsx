import { useEffect, useMemo, useState, useRef } from 'react';
import clsx from 'clsx';
import {
  CommentaryResponse,
  CommentarySections,
  ExplainAttentionPoint,
  ExplainFeatureEntry,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';
import { driverCategoryLabel, driverCategoryTone } from '../utils/driverCategories';

interface Props {
  commentary: CommentaryResponse | null;
  explain: ExplainResponse | null;
  tftAnalysis?: TFTAnalysisResponse | null;
  loading?: boolean;
  error?: string;
}

type BriefingTab = 'synthesis' | 'tactical' | 'xai';
type SpriteExpression = 'idle-1' | 'idle-2' | 'blink' | 'talk' | 'alert';

function formatRecent(entry: ExplainFeatureEntry): string | null {
  const r = entry.recent;
  if (!r) return null;
  const basis = r.basis || '5 sessions';
  if (r.kind === 'pct' && r.change_pct != null && Number.isFinite(r.change_pct)) {
    const sign = r.change_pct >= 0 ? '+' : '';
    return `${sign}${r.change_pct.toFixed(2)}% over last ${basis}`;
  }
  if (!Number.isFinite(r.change)) return null;
  const sign = r.change >= 0 ? '+' : '';
  return `${sign}${r.change.toFixed(3)} over last ${basis}`;
}

function parseCommentaryFallback(rawText: string | null | undefined): CommentarySections {
  const text = (rawText || '').trim();
  if (!text) {
    return { synthesis: '', risks: [], opportunities: [], xai_brief: null, bias_warning: null };
  }

  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !/^this is not financial advice\.?$/i.test(l));

  const risks: string[] = [];
  const opportunities: string[] = [];
  const synthesisParts: string[] = [];
  let biasWarning: string | null = null;
  let section: 'none' | 'risks' | 'opportunities' | 'synthesis' | 'bias' = 'none';

  for (const line of lines) {
    if (/^(?:\*\*)?risks(?:\s*\([^)]*\))?(?:\*\*)?\s*:?$/i.test(line)) {
      section = 'risks';
      continue;
    }
    if (/^(?:\*\*)?opportunities(?:\s*\([^)]*\))?(?:\*\*)?\s*:?$/i.test(line)) {
      section = 'opportunities';
      continue;
    }
    if (/^(?:\*\*)?(?:synthesis|summary)(?:\*\*)?\s*:/i.test(line)) {
      section = 'synthesis';
      const rest = line.replace(/^(?:\*\*)?(?:synthesis|summary)(?:\*\*)?\s*:\s*/i, '').trim();
      if (rest) synthesisParts.push(rest);
      continue;
    }
    if (/^(?:\*\*)?(?:bias acknowledgment|bias warning|model status note)(?:\*\*)?\s*:/i.test(line)) {
      section = 'bias';
      const rest = line
        .replace(/^(?:\*\*)?(?:bias acknowledgment|bias warning|model status note)(?:\*\*)?\s*:\s*/i, '')
        .trim();
      if (rest) biasWarning = biasWarning ? `${biasWarning} ${rest}` : rest;
      continue;
    }
    if (/^(?:\*\*)?disclaimer(?:\*\*)?\s*:/i.test(line)) {
      section = 'none';
      continue;
    }

    const cleanedBullet = line.replace(/^(?:[•\-*]|\d+[.)])\s*/, '').replace(/`/g, '').trim();
    if (section === 'risks' && cleanedBullet && risks.length < 3) {
      risks.push(cleanedBullet);
    } else if (section === 'opportunities' && cleanedBullet && opportunities.length < 3) {
      opportunities.push(cleanedBullet);
    } else if (section === 'synthesis' && cleanedBullet) {
      synthesisParts.push(cleanedBullet);
    } else if (section === 'bias' && cleanedBullet) {
      biasWarning = biasWarning ? `${biasWarning} ${cleanedBullet}` : cleanedBullet;
    }
  }

  const synthesis =
    synthesisParts.join(' ').trim() ||
    lines.find((l) => !/^(?:risks|opportunities|disclaimer)/i.test(l)) ||
    text.slice(0, 320);

  return {
    synthesis,
    risks,
    opportunities,
    xai_brief: null,
    bias_warning: biasWarning,
  };
}

interface MoodConfig {
  title: string;
  subtitle: string;
  prefix: string;
  badgeTone: string;
  ringColor: string;
  coreColor: string;
}

const MOOD_META: Record<string, MoodConfig> = {
  molten_bull: {
    title: 'Molten Resonance',
    subtitle: 'Fiery crimson bull soul',
    prefix: 'cuprum-bull',
    badgeTone: 'text-rose-300 border-rose-400/30 bg-rose-500/10',
    ringColor: '#f43f5e',
    coreColor: '#fb923c',
  },
  patina_bear: {
    title: 'Patina Resonance',
    subtitle: 'Oxidized verdigris bear soul',
    prefix: 'cuprum-bear',
    badgeTone: 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10',
    ringColor: '#10b981',
    coreColor: '#2dd4bf',
  },
  tempered_neutral: {
    title: 'Harmonic Core',
    subtitle: 'Calibrated bronze resonance soul',
    prefix: 'cuprum-neutral',
    badgeTone: 'text-amber-300 border-amber-400/30 bg-amber-500/10',
    ringColor: '#e08b56',
    coreColor: '#fbbf24',
  },
};

export function CuprumCompanionPanel({
  commentary,
  explain,
  tftAnalysis,
  loading = false,
  error,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<BriefingTab>('synthesis');
  const [activeFeatureId, setActiveFeatureId] = useState<string | null>(null);
  const [selectedDateIdx, setSelectedDateIdx] = useState<number | null>(null);
  const [expression, setExpression] = useState<SpriteExpression>('idle-1');
  const [isAlerting, setIsAlerting] = useState(false);

  const blinkTimerRef = useRef<number | null>(null);
  const animIntervalRef = useRef<number | null>(null);

  const companion = commentary?.companion_state;
  const stance = (
    companion?.stance ||
    commentary?.ai_stance ||
    tftAnalysis?.direction ||
    'NEUTRAL'
  ).toUpperCase();

  const moodKey =
    companion?.mood && MOOD_META[companion.mood]
      ? companion.mood
      : stance === 'BULLISH'
        ? 'molten_bull'
        : stance === 'BEARISH'
          ? 'patina_bear'
          : 'tempered_neutral';
  const moodMeta = MOOD_META[moodKey] || MOOD_META.tempered_neutral;

  const energy =
    companion?.energy ||
    (tftAnalysis?.risk_level === 'HIGH'
      ? 'volatile'
      : tftAnalysis?.risk_level === 'MEDIUM'
        ? 'charged'
        : 'calm');

  const sections = useMemo<CommentarySections>(() => {
    const parsed = parseCommentaryFallback(commentary?.commentary);
    if (commentary?.sections) {
      return {
        synthesis: commentary.sections.synthesis || parsed.synthesis,
        risks: commentary.sections.risks?.length ? commentary.sections.risks : parsed.risks,
        opportunities: commentary.sections.opportunities?.length
          ? commentary.sections.opportunities
          : parsed.opportunities,
        xai_brief: commentary.sections.xai_brief ?? parsed.xai_brief,
        bias_warning: commentary.sections.bias_warning ?? parsed.bias_warning,
      };
    }
    return parsed;
  }, [commentary?.commentary, commentary?.sections]);

  const topEncoder = useMemo<ExplainFeatureEntry[]>(() => {
    const fromCompanion = companion?.xai_highlights?.top_signals;
    if (Array.isArray(fromCompanion) && fromCompanion.length > 0) {
      return fromCompanion.slice(0, 5);
    }
    return (explain?.variable_selection?.encoder ?? []).slice(0, 5);
  }, [companion?.xai_highlights?.top_signals, explain?.variable_selection?.encoder]);

  const memoryAnchors = useMemo<ExplainAttentionPoint[]>(() => {
    const fromCompanion = companion?.xai_highlights?.memory_anchors;
    if (Array.isArray(fromCompanion) && fromCompanion.length > 0) {
      return fromCompanion.slice(0, 3);
    }
    const topFromExplain = (explain?.attention?.top_days ?? []).slice(0, 3);
    if (topFromExplain.length > 0) return topFromExplain;
    const series = explain?.attention?.series ?? [];
    return [...series].sort((a, b) => (b.weight || 0) - (a.weight || 0)).slice(0, 3);
  }, [companion?.xai_highlights?.memory_anchors, explain?.attention?.top_days, explain?.attention?.series]);

  const attentionSeries = useMemo(
    () => explain?.attention?.series ?? [],
    [explain?.attention?.series],
  );
  const maxEncoderWeight = topEncoder[0]?.weight || 1;
  const maxAttentionWeight = useMemo(
    () => Math.max(...attentionSeries.map((p) => p.weight || 0), 0.0001),
    [attentionSeries],
  );
  const meanAttentionWeight =
    attentionSeries.length > 0 ? 1 / attentionSeries.length : 0.02;

  const activeAttentionIndex = useMemo(() => {
    if (
      selectedDateIdx != null &&
      selectedDateIdx >= 0 &&
      selectedDateIdx < attentionSeries.length
    ) {
      return selectedDateIdx;
    }
    if (memoryAnchors[0]?.date) {
      const idx = attentionSeries.findIndex((p) => p.date === memoryAnchors[0].date);
      if (idx >= 0) return idx;
    }
    return attentionSeries.length > 0 ? attentionSeries.length - 1 : -1;
  }, [selectedDateIdx, memoryAnchors, attentionSeries]);

  const activeAttentionPoint =
    activeAttentionIndex >= 0 ? attentionSeries[activeAttentionIndex] : memoryAnchors[0] ?? null;

  const horizonLabel =
    companion?.xai_highlights?.horizon_label ||
    explain?.horizon?.label ||
    '5-day cumulative log return (primary)';
  const referenceDate =
    companion?.xai_highlights?.reference_price_date ||
    explain?.reference_price_date ||
    tftAnalysis?.reference_price_date ||
    null;

  const xaiNarrative = useMemo(() => {
    if (sections.xai_brief) return sections.xai_brief;
    if (topEncoder.length === 0 && memoryAnchors.length === 0) return null;
    const topNames = topEncoder
      .slice(0, 2)
      .map((e) => `${e.label || e.feature} (${(e.weight * 100).toFixed(1)}%)`)
      .join(' and ');
    const topDate = memoryAnchors[0]?.date
      ? `${memoryAnchors[0].date} (${(memoryAnchors[0].weight * 100).toFixed(1)}% share)`
      : null;
    if (topNames && topDate) {
      return `For the 5-day outlook, my strongest encoder focus is on ${topNames}, while my historical lookback memory matched the ${topDate} session most closely.`;
    }
    if (topNames) {
      return `For the 5-day outlook, my strongest encoder focus is on ${topNames}.`;
    }
    return null;
  }, [sections.xai_brief, topEncoder, memoryAnchors]);

  const activeFeature = useMemo(
    () => topEncoder.find((e) => e.feature === activeFeatureId) ?? topEncoder[0] ?? null,
    [topEncoder, activeFeatureId],
  );

  const hasAnyContent = Boolean(
    commentary?.commentary ||
      sections.synthesis ||
      topEncoder.length > 0 ||
      memoryAnchors.length > 0,
  );

  // --- Dynamic Retro Pixel Animation State Machine ---
  useEffect(() => {
    if (isAlerting) return;

    // Continuous 2-frame fire particle floating animation
    animIntervalRef.current = window.setInterval(() => {
      setExpression((prev) => (prev === 'idle-2' ? 'idle-1' : 'idle-2'));
    }, 600);

    return () => {
      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
    };
  }, [isAlerting]);

  // Periodic Blink every 3.8s
  useEffect(() => {
    const triggerBlink = () => {
      if (!isAlerting) {
        setExpression('blink');
        setTimeout(() => {
          setExpression('idle-1');
        }, 160);
      }
      blinkTimerRef.current = window.setTimeout(triggerBlink, 3400 + Math.random() * 1200);
    };

    blinkTimerRef.current = window.setTimeout(triggerBlink, 2800);
    return () => {
      if (blinkTimerRef.current) clearTimeout(blinkTimerRef.current);
    };
  }, [isAlerting]);

  // Handle clicking the Pet Mascot (Interactive Summon / Dismiss)
  const handlePetClick = () => {
    setIsAlerting(true);
    setExpression('alert');
    setIsOpen((prev) => !prev);
    setTimeout(() => {
      setIsAlerting(false);
      setExpression('idle-1');
    }, 380);
  };

  const currentWebp = `/assets/cuprum/${moodMeta.prefix}-${expression}.webp`;
  const currentPng = `/assets/cuprum/${moodMeta.prefix}-${expression}.png`;

  return (
    <div
      className="w-full relative transition-colors"
      data-testid="cuprum-companion-stage"
    >
      {/* Ambient alchemical copper backdrop glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          opacity: 0.15,
          filter: 'blur(24px)',
          transition: 'all 700ms ease',
          background: `radial-gradient(circle at 18% 45%, ${moodMeta.ringColor} 0%, transparent 60%)`,
        }}
        aria-hidden="true"
      />

      <div className="relative z-10 flex flex-wrap items-start gap-4">
        {/* Left: Interactive Alchemical Companion (Living Pet) */}
        <div
          className="flex flex-col items-center shrink-0"
          style={{ flex: '0 0 auto', minWidth: 200 }}
        >
          {/* Pet Stance & Title Header */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono uppercase tracking-widest text-copper-300">
              CUPRUM · SOUL
            </span>
            <span
              className={clsx(
                'text-xs font-mono uppercase tracking-wider px-2 py-0.5 rounded-full border',
                moodMeta.badgeTone,
              )}
            >
              {stance}
            </span>
          </div>

          {/* Interactive Floating Mascot Platform */}
          <button
            type="button"
            onClick={handlePetClick}
            className="group relative flex flex-col items-center justify-center p-3 rounded-xl focus:outline-none"
            style={{ cursor: 'pointer', transition: 'transform 200ms ease' }}
            aria-label={`Cuprum companion in ${moodMeta.title} state. Click to ${isOpen ? 'close' : 'open'} tactical briefing.`}
            title={`Click to ${isOpen ? 'close' : 'open'} Cuprum tactical briefing`}
          >
            {/* Ethereal Alchemical Rune Halo */}
            <svg
              viewBox="0 0 200 200"
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ animation: 'spin 40s linear infinite', opacity: 0.6 }}
              aria-hidden="true"
            >
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke={moodMeta.ringColor}
                strokeOpacity="0.25"
                strokeWidth="1.2"
                strokeDasharray="6 8"
              />
              <circle
                cx="100"
                cy="100"
                r="78"
                fill="none"
                stroke={moodMeta.coreColor}
                strokeOpacity="0.3"
                strokeWidth="1.5"
                strokeDasharray="16 12"
              />
            </svg>

            {/* Living Pixel Character Sprite (Transparent) */}
            <div
              className="relative flex items-center justify-center"
              style={{
                width: 140,
                height: 140,
              }}
            >
              <picture>
                <source srcSet={currentWebp} type="image/webp" />
                <img
                  src={currentPng}
                  alt={`Cuprum — ${moodMeta.title}`}
                  className="w-full h-full select-none"
                  style={{
                    imageRendering: 'pixelated',
                    objectFit: 'contain',
                    transition: 'opacity 150ms ease',
                    filter: `drop-shadow(0 0 16px ${moodMeta.ringColor}55)`,
                  }}
                  loading="lazy"
                />
              </picture>
            </div>

            {/* Tactical summoning rune prompt */}
            <div className="mt-1 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 group-hover:border-copper-400/40 transition-colors">
              <span
                className="rounded-full"
                style={{
                  width: 6,
                  height: 6,
                  backgroundColor: moodMeta.ringColor,
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                }}
                aria-hidden="true"
              />
              <span className="text-xs font-mono text-slate-300 group-hover:text-copper-300">
                {isOpen ? 'Close console' : 'Open console'}
              </span>
            </div>
          </button>

          {/* Character Identity & Volatility Telemetry */}
          <div className="text-center space-y-1 mt-1">
            <p className="text-sm font-medium text-slate-100">{moodMeta.title}</p>
            <p className="text-xs text-slate-400">{moodMeta.subtitle}</p>
            <div className="pt-1 flex items-center justify-center gap-1.5 text-xs font-mono">
              <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300">
                {String(energy).toUpperCase()}
              </span>
              {commentary?.generation_mode && (
                <span
                  className={clsx(
                    'px-1.5 py-0.5 rounded-full border',
                    commentary.generation_mode === 'llm' || commentary.generation_mode === 'llm_repaired'
                      ? 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10'
                      : 'text-amber-300 border-amber-400/30 bg-amber-500/10',
                  )}
                >
                  {commentary.generation_mode === 'deterministic_fallback'
                    ? 'Local fallback'
                    : commentary.generation_mode === 'llm' || commentary.generation_mode === 'llm_repaired'
                      ? 'AI generated'
                      : 'Unavailable'}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Game-Like 3 Tactical Runes & Content Console */}
        <div className="flex-1 min-w-0 w-full">
          {/* The 3 Game Action Runes (Tabs) */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3 mb-3">
            <div
              className="flex flex-wrap items-center gap-2"
              role="tablist"
              aria-label="Cuprum tactical console"
            >
              {/* Tab 1: Synthesis */}
              <button
                type="button"
                role="tab"
                aria-selected={isOpen && activeTab === 'synthesis'}
                onClick={() => {
                  setIsOpen(true);
                  setActiveTab('synthesis');
                }}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono border transition-colors',
                  isOpen && activeTab === 'synthesis'
                    ? 'bg-copper-500/10 border-copper-400/70 text-copper-300 font-semibold'
                    : 'bg-white/5 border-white/10 text-slate-300 hover:border-copper-400/40',
                )}
                style={{ cursor: 'pointer' }}
              >
                <span>📜</span>
                <span>Synthesis</span>
              </button>

              {/* Tab 2: Tactical Drivers (Risks & Opportunities) */}
              <button
                type="button"
                role="tab"
                aria-selected={isOpen && activeTab === 'tactical'}
                onClick={() => {
                  setIsOpen(true);
                  setActiveTab('tactical');
                }}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono border transition-colors',
                  isOpen && activeTab === 'tactical'
                    ? 'bg-copper-500/10 border-copper-400/70 text-copper-300 font-semibold'
                    : 'bg-white/5 border-white/10 text-slate-300 hover:border-copper-400/40',
                )}
                style={{ cursor: 'pointer' }}
                aria-label={`Risks (${sections.risks.length}) & Opportunities (${sections.opportunities.length})`}
              >
                <span>⚔️</span>
                <span>
                  Tactical Signals ({sections.risks.length + sections.opportunities.length})
                </span>
              </button>

              {/* Tab 3: Neural Memory & XAI Focus */}
              <button
                type="button"
                role="tab"
                aria-selected={isOpen && activeTab === 'xai'}
                onClick={() => {
                  setIsOpen(true);
                  setActiveTab('xai');
                }}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-mono border transition-colors',
                  isOpen && activeTab === 'xai'
                    ? 'bg-copper-500/10 border-copper-400/70 text-copper-300 font-semibold'
                    : 'bg-white/5 border-white/10 text-slate-300 hover:border-copper-400/40',
                )}
                style={{ cursor: 'pointer' }}
              >
                <span>🔮</span>
                <span>XAI Focus &amp; Memory</span>
              </button>
            </div>

            {commentary?.generated_at && (
              <span className="text-xs text-slate-400 font-mono">
                {new Date(commentary.generated_at).toLocaleTimeString()}
              </span>
            )}
          </div>

          {/* Console Body or Collapsed Teaser */}
          {!isOpen ? (
            <div
              onClick={() => setIsOpen(true)}
              className="group rounded-xl border border-white/5 bg-white/5 hover:border-copper-400/40 transition-colors"
              style={{ cursor: 'pointer', padding: '1rem' }}
            >
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-copper-300 mb-1">
                <span>CUPRUM DIALOGUE // MINIMIZED</span>
                <span>· Click to expand</span>
              </div>
              <p className="text-sm text-slate-300 line-clamp-2">
                &ldquo;{sections.synthesis || companion?.headline_quote || commentary?.commentary}&rdquo;
              </p>
            </div>
          ) : !hasAnyContent ? (
            <p className="text-slate-400 text-xs font-mono py-8 text-center" role="status">
              {loading
                ? 'Cuprum is synchronizing neural commentary and XAI signals…'
                : error || commentary?.error
                  ? 'AI commentary is temporarily unavailable.'
                  : 'Model focus will appear after the next model refresh.'}
            </p>
          ) : activeTab === 'synthesis' ? (
            /* TAB 1: EXECUTIVE SYNTHESIS */
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-copper-300">
                  <span>CUPRUM EXECUTIVE THESIS</span>
                  {referenceDate && <span className="text-slate-400">· Ref {referenceDate}</span>}
                </div>
                <div
                  className="text-sm text-slate-100 font-light leading-relaxed whitespace-pre-wrap"
                  style={{ borderLeft: '2px solid rgba(200, 122, 72, 0.5)', paddingLeft: '0.75rem' }}
                >
                  {sections.synthesis || companion?.headline_quote || commentary?.commentary}
                </div>
              </div>

              {xaiNarrative && (
                <div className="rounded-lg border border-copper-400/20 bg-copper-500/10 p-3 space-y-1.5">
                  <div className="flex items-center justify-between gap-2 text-xs font-mono uppercase tracking-wider text-copper-300">
                    <span>WHY I LEAN THIS WAY (5D TFT-ASRO XAI)</span>
                    <button
                      type="button"
                      onClick={() => setActiveTab('xai')}
                      className="hover:text-white"
                    >
                      Inspect signals &amp; dates →
                    </button>
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed">{xaiNarrative}</p>
                </div>
              )}

              {/* Quick XAI summary telemetry */}
              {(topEncoder.length > 0 || memoryAnchors.length > 0) && (
                <div className="flex flex-wrap gap-3 pt-1">
                  {topEncoder[0] && (
                    <button
                      type="button"
                      onClick={() => setActiveTab('xai')}
                      className="text-left rounded-lg border border-white/10 bg-white/5 hover:border-copper-400/40 p-3 transition-colors"
                      style={{ flex: '1 1 240px' }}
                    >
                      <span className="block text-xs font-mono uppercase tracking-wider text-slate-400">
                        Top 5D Signal Focus (VSN)
                      </span>
                      <span className="block text-xs font-medium text-slate-100 truncate mt-0.5">
                        {topEncoder[0].label || topEncoder[0].feature}{' '}
                        <span className="font-mono text-copper-300">
                          ({(topEncoder[0].weight * 100).toFixed(1)}%)
                        </span>
                      </span>
                      {formatRecent(topEncoder[0]) && (
                        <span className="block text-xs font-mono text-slate-400 mt-0.5">
                          {formatRecent(topEncoder[0])}
                        </span>
                      )}
                    </button>
                  )}
                  {memoryAnchors[0] && (
                    <button
                      type="button"
                      onClick={() => setActiveTab('xai')}
                      className="text-left rounded-lg border border-white/10 bg-white/5 hover:border-copper-400/40 p-3 transition-colors"
                      style={{ flex: '1 1 240px' }}
                    >
                      <span className="block text-xs font-mono uppercase tracking-wider text-slate-400">
                        #1 Referenced Past Session
                      </span>
                      <span className="block text-xs font-mono font-semibold text-slate-100 mt-0.5">
                        {memoryAnchors[0].date ?? '—'}{' '}
                        <span className="text-copper-300">
                          ({(memoryAnchors[0].weight * 100).toFixed(1)}% attention)
                        </span>
                      </span>
                      <span className="block text-xs text-slate-400 mt-0.5">
                        Closest historical regime match in 50D memory
                      </span>
                    </button>
                  )}
                </div>
              )}
            </div>
          ) : activeTab === 'tactical' ? (
            /* TAB 2: TACTICAL DRIVERS (RISKS & OPPORTUNITIES) */
            <div className="space-y-4">
              <div className="flex flex-wrap gap-4">
                {/* Upside Opportunities / Tailwinds */}
                <div className="space-y-2.5" style={{ flex: '1 1 260px' }}>
                  <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-emerald-300">
                    <span>◆ UPSIDE TAILWINDS ({sections.opportunities.length})</span>
                  </div>
                  {sections.opportunities.length > 0 ? (
                    <ul className="space-y-2">
                      {sections.opportunities.map((opp, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-2 text-xs text-slate-200 leading-relaxed"
                          style={{ borderLeft: '1px solid rgba(16, 185, 129, 0.3)', paddingLeft: '0.5rem' }}
                        >
                          <span className="text-emerald-400 font-mono text-xs">▸</span>
                          <span>{opp}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-400">No specific upside catalysts identified.</p>
                  )}
                </div>

                {/* Downside Risks / Headwinds */}
                <div className="space-y-2.5" style={{ flex: '1 1 260px' }}>
                  <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-rose-300">
                    <span>◆ DOWNSIDE HEADWINDS ({sections.risks.length})</span>
                  </div>
                  {sections.risks.length > 0 ? (
                    <ul className="space-y-2">
                      {sections.risks.map((risk, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-2 text-xs text-slate-200 leading-relaxed"
                          style={{ borderLeft: '1px solid rgba(244, 63, 94, 0.3)', paddingLeft: '0.5rem' }}
                        >
                          <span className="text-rose-400 font-mono text-xs">▸</span>
                          <span>{risk}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-400">No specific downside risks identified.</p>
                  )}
                </div>
              </div>

              {/* Model Bias Telemetry Note */}
              {sections.bias_warning && (
                <div className="pt-2 border-t border-white/5 flex items-start gap-2 text-xs text-amber-200 font-mono">
                  <span className="text-amber-400 font-bold shrink-0">BIAS NOTE:</span>
                  <span>{sections.bias_warning}</span>
                </div>
              )}
            </div>
          ) : (
            /* TAB 3: NEURAL FOCUS & MEMORY (XAI) */
            <div className="space-y-4" data-testid="cuprum-xai-tab">
              {/* Part 1: Top VSN Encoder Inputs */}
              {topEncoder.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-mono uppercase tracking-widest text-copper-300">
                      1. WHICH VARIABLES I WATCHED (5D VSN FOCUS)
                    </p>
                    <span className="text-xs font-mono text-slate-400">{horizonLabel}</span>
                  </div>
                  <div className="space-y-2">
                    {topEncoder.map((entry, idx) => {
                      const category = entry.category || 'Other';
                      const tone = driverCategoryTone[category] || driverCategoryTone.Other;
                      const categoryText = driverCategoryLabel[category] || driverCategoryLabel.Other;
                      const label = entry.label || entry.description || entry.feature;
                      const recentText = formatRecent(entry);
                      const pct = Math.max(0, Math.min(1, entry.weight / maxEncoderWeight)) * 100;
                      const isSelected = activeFeature?.feature === entry.feature;
                      return (
                        <button
                          key={entry.feature}
                          type="button"
                          onClick={() => setActiveFeatureId(entry.feature)}
                          onMouseEnter={() => setActiveFeatureId(entry.feature)}
                          className={clsx(
                            'w-full text-left rounded-lg border px-3 py-2 transition-colors cursor-pointer',
                            isSelected
                              ? 'border-copper-400/40 bg-copper-500/10'
                              : 'border-white/5 bg-white/5 hover:border-copper-400/40',
                          )}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="text-xs font-mono text-copper-300">#{idx + 1}</span>
                              <span
                                className={`text-xs px-1.5 py-0.5 rounded ${tone} font-medium shrink-0`}
                              >
                                {categoryText}
                              </span>
                              <span className="text-xs text-slate-100 truncate">{label}</span>
                            </div>
                            <span className="text-xs font-mono text-copper-300 shrink-0">
                              {(entry.weight * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-copper-500 to-copper-400"
                              style={{ width: `${pct.toFixed(1)}%` }}
                            />
                          </div>
                          {recentText && (
                            <p className="mt-1 text-xs font-mono text-slate-400">{recentText}</p>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Part 2: Historical Lookback Memory (50D Attention) */}
              {(memoryAnchors.length > 0 || attentionSeries.length > 0) && (
                <div
                  className="space-y-2 border-t border-white/10 pt-3"
                  data-testid="model-focus-attention"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs font-mono uppercase tracking-widest text-copper-300">
                      2. WHICH PAST TRADING DAYS MATCHED TODAY (50D MEMORY)
                    </p>
                    <span className="text-xs font-mono text-slate-400">
                      Equal day baseline = {(meanAttentionWeight * 100).toFixed(1)}%/day
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Cuprum scans the last {attentionSeries.length || 50} trading sessions and assigns
                    higher attention share to past days whose market regime or shock best anchors
                    the next step.
                  </p>

                  <div className="flex flex-wrap items-center gap-2">
                    {memoryAnchors.map((peak, rank) => {
                      const idxInSeries = attentionSeries.findIndex((p) => p.date === peak.date);
                      const isSelected =
                        activeAttentionPoint?.date != null &&
                        activeAttentionPoint.date === peak.date;
                      const vsAvg =
                        meanAttentionWeight > 0 ? peak.weight / meanAttentionWeight : 1;
                      return (
                        <button
                          key={peak.date ?? rank}
                          type="button"
                          onClick={() => {
                            if (idxInSeries >= 0) setSelectedDateIdx(idxInSeries);
                          }}
                          className={clsx(
                            'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono border transition-colors cursor-pointer',
                            isSelected
                              ? 'bg-copper-500/10 border-copper-400/70 text-copper-300 font-semibold'
                              : 'bg-white/5 border-white/10 text-slate-300 hover:border-copper-400/40',
                          )}
                        >
                          <span className="text-copper-400 font-semibold">#{rank + 1}</span>
                          <span>{peak.date}</span>
                          <span className="text-slate-400">
                            {(peak.weight * 100).toFixed(1)}% ({vsAvg.toFixed(2)}× normal)
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {activeAttentionPoint && (
                    <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                      <div>
                        <span className="text-slate-400">Selected session: </span>
                        <span className="text-slate-100 font-semibold">
                          {activeAttentionPoint.date ?? '—'}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-copper-300">
                          Share: {(activeAttentionPoint.weight * 100).toFixed(1)}%
                        </span>
                        <span className="text-slate-300">
                          {(
                            (meanAttentionWeight > 0
                              ? activeAttentionPoint.weight / meanAttentionWeight
                              : 1)
                          ).toFixed(2)}
                          × normal day weight (max {(maxAttentionWeight * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Non-causal XAI & Financial Disclaimer Footer */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 pt-3 border-t border-white/5 mt-3">
            <span>
              {companion?.xai_highlights?.notes ||
                explain?.notes ||
                'VSN and attention weights show raw, pre-calibration model focus (not causal) · Not financial advice.'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
