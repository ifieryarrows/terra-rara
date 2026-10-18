import { useMemo, useState } from 'react';
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

type BriefingTab = 'synthesis' | 'risks' | 'opportunities' | 'xai';

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

    const cleanedBullet = line.replace(/^(?:[•\-*]|\d+[.)])\s*/, '').trim();
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

const MOOD_META: Record<
  string,
  {
    title: string;
    subtitle: string;
    portraitWebp: string;
    portraitPng: string;
    badgeTone: string;
    ringColor: string;
    coreColor: string;
  }
> = {
  molten_bull: {
    title: 'Molten Resonance',
    subtitle: 'Fiery crimson bull soul',
    portraitWebp: '/assets/cuprum/cuprum-molten-bull.webp',
    portraitPng: '/assets/cuprum/cuprum-molten-bull.png',
    badgeTone: 'text-rose-300 border-rose-400/30 bg-rose-500/10',
    ringColor: '#f43f5e',
    coreColor: '#fb923c',
  },
  patina_bear: {
    title: 'Patina Resonance',
    subtitle: 'Oxidized verdigris bear soul',
    portraitWebp: '/assets/cuprum/cuprum-patina-bear.webp',
    portraitPng: '/assets/cuprum/cuprum-patina-bear.png',
    badgeTone: 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10',
    ringColor: '#10b981',
    coreColor: '#2dd4bf',
  },
  tempered_neutral: {
    title: 'Harmonic Core',
    subtitle: 'Calibrated bronze resonance soul',
    portraitWebp: '/assets/cuprum/cuprum-tempered-neutral.webp',
    portraitPng: '/assets/cuprum/cuprum-tempered-neutral.png',
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
  const [activeTab, setActiveTab] = useState<BriefingTab>('synthesis');
  const [activeFeatureId, setActiveFeatureId] = useState<string | null>(null);
  const [selectedDateIdx, setSelectedDateIdx] = useState<number | null>(null);
  const [pulseCount, setPulseCount] = useState(0);

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

  return (
    <div
      className={clsx(
        'rounded-xl border p-4 transition-colors',
        moodKey === 'molten_bull'
          ? 'border-rose-500/30 bg-slate-900/20'
          : moodKey === 'patina_bear'
            ? 'border-emerald-500/30 bg-slate-900/20'
            : 'border-copper-400/20 bg-slate-900/20',
      )}
      data-testid="cuprum-companion-stage"
    >
      <div className="flex flex-wrap gap-4">
        {/* Left column: Living Cuprum Pet Avatar */}
        <div
          className="flex flex-col items-center justify-between rounded-xl border border-white/10 bg-slate-950/85 p-4 relative overflow-hidden"
          style={{ flex: '1 1 210px' }}
        >
          <div className="w-full flex items-center justify-between gap-2 z-10">
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

          {/* Interactive Avatar Crucible */}
          <button
            type="button"
            onClick={() => setPulseCount((c) => c + 1)}
            className="group relative mt-2 mb-2 rounded-full flex items-center justify-center"
            style={{ width: 152, height: 152 }}
            aria-label={`Cuprum companion in ${moodMeta.title} state. Click to pulse resonance.`}
          >
            {/* Reactive SVG Resonance Halo */}
            <svg
              viewBox="0 0 200 200"
              className="absolute inset-0 w-full h-full pointer-events-none"
              aria-hidden="true"
            >
              <circle
                cx="100"
                cy="100"
                r="92"
                fill="none"
                stroke={moodMeta.ringColor}
                strokeOpacity="0.24"
                strokeWidth="1.2"
                strokeDasharray="6 6"
              />
              <circle
                cx="100"
                cy="100"
                r="82"
                fill="none"
                stroke={moodMeta.coreColor}
                strokeOpacity="0.34"
                strokeWidth="1.5"
                strokeDasharray="18 10"
              />
              <circle
                cx="100"
                cy="100"
                r="72"
                fill="none"
                stroke={moodMeta.ringColor}
                strokeOpacity="0.2"
                strokeWidth="1"
              />
            </svg>

            <div
              key={pulseCount}
              className="relative rounded-full overflow-hidden border border-white/10 bg-slate-950 flex items-center justify-center"
              style={{ width: 120, height: 120 }}
            >
              <picture>
                <source srcSet={moodMeta.portraitWebp} type="image/webp" />
                <img
                  src={moodMeta.portraitPng}
                  alt={`Cuprum — ${moodMeta.title}`}
                  className="w-full h-full select-none"
                  style={{ imageRendering: 'pixelated', objectFit: 'cover' }}
                  loading="lazy"
                />
              </picture>
              <div
                className="pointer-events-none absolute inset-0 rounded-full"
                style={{
                  boxShadow: `inset 0 0 22px ${moodMeta.ringColor}33`,
                }}
              />
            </div>
          </button>

          {/* Pet Mood & Telemetry Strip */}
          <div className="w-full text-center space-y-1 z-10">
            <p className="text-sm font-medium text-slate-100">{moodMeta.title}</p>
            <p className="text-xs text-slate-400">{moodMeta.subtitle}</p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-1.5 text-xs font-mono">
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

        {/* Right column: Interactive Speech Bubble & XAI Briefing Console */}
        <div
          className="flex flex-col justify-between gap-3 min-w-0"
          style={{ flex: '2 1 320px' }}
        >
          {/* Interactive Topic Pills */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2.5">
            <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Cuprum XAI briefing topics">
              {(
                [
                  { id: 'synthesis', label: 'Synthesis' },
                  { id: 'risks', label: `Risks (${sections.risks.length})` },
                  { id: 'opportunities', label: `Opportunities (${sections.opportunities.length})` },
                  { id: 'xai', label: 'XAI Focus & Memory' },
                ] as const
              ).map((tab) => {
                const isSelected = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={isSelected}
                    onClick={() => setActiveTab(tab.id)}
                    className={clsx(
                      'text-xs font-mono px-2.5 py-1 rounded-lg border transition-colors',
                      isSelected
                        ? 'bg-copper-500/10 border-copper-400/70 text-copper-300 font-semibold'
                        : 'bg-white/5 border-white/10 text-slate-300 hover:border-copper-400/40',
                    )}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {commentary?.generated_at && (
              <span className="text-xs text-slate-400 font-mono">
                {new Date(commentary.generated_at).toLocaleTimeString()}
              </span>
            )}
          </div>

          {/* Speech Bubble Body */}
          <div className="cm-commentary flex-1 rounded-xl border border-white/10 bg-slate-950/85 p-4 text-sm text-gray-300 leading-relaxed">
            {!hasAnyContent ? (
              <p className="text-slate-400 text-xs font-mono py-8 text-center" role="status">
                {loading
                  ? 'Cuprum is synchronizing neural commentary and XAI signals…'
                  : error || commentary?.error
                    ? 'AI commentary is temporarily unavailable.'
                    : 'Model focus will appear after the next model refresh.'}
              </p>
            ) : activeTab === 'synthesis' ? (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-copper-300">
                    <span>CUPRUM EXECUTIVE SYNTHESIS</span>
                    {referenceDate && <span className="text-slate-400">· Ref {referenceDate}</span>}
                  </div>
                  <p className="text-sm text-slate-100 font-light leading-relaxed whitespace-pre-wrap">
                    {sections.synthesis || companion?.headline_quote || commentary?.commentary}
                  </p>
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

                {/* Quick XAI summary pills */}
                {(topEncoder.length > 0 || memoryAnchors.length > 0) && (
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    {topEncoder[0] && (
                      <button
                        type="button"
                        onClick={() => setActiveTab('xai')}
                        className="text-left rounded-lg border border-white/10 bg-white/5 hover:border-copper-400/40 px-3 py-2 transition-colors"
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
                        className="text-left rounded-lg border border-white/10 bg-white/5 hover:border-copper-400/40 px-3 py-2 transition-colors"
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
            ) : activeTab === 'risks' ? (
              <div className="space-y-3">
                <p className="text-xs font-mono uppercase tracking-widest text-rose-300">
                  DOWNSIDE RISKS &amp; HEADWINDS
                </p>
                {sections.risks.length > 0 ? (
                  <ul className="space-y-2">
                    {sections.risks.map((risk, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2 rounded-lg border border-rose-400/20 bg-rose-500/10 px-3 py-2 text-xs text-slate-200"
                      >
                        <span className="font-mono text-rose-300 font-semibold shrink-0">
                          0{idx + 1}
                        </span>
                        <span>{risk}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-300 whitespace-pre-wrap">
                    {commentary?.commentary || 'No specific risk bullets extracted.'}
                  </p>
                )}
                {sections.bias_warning && (
                  <div className="rounded-lg border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                    <strong className="font-mono uppercase tracking-wider">Bias note: </strong>
                    {sections.bias_warning}
                  </div>
                )}
              </div>
            ) : activeTab === 'opportunities' ? (
              <div className="space-y-3">
                <p className="text-xs font-mono uppercase tracking-widest text-emerald-300">
                  UPSIDE CATALYSTS &amp; TAILWINDS
                </p>
                {sections.opportunities.length > 0 ? (
                  <ul className="space-y-2">
                    {sections.opportunities.map((opp, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2 rounded-lg border border-emerald-400/20 bg-emerald-500/10 px-3 py-2 text-xs text-slate-200"
                      >
                        <span className="font-mono text-emerald-300 font-semibold shrink-0">
                          0{idx + 1}
                        </span>
                        <span>{opp}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-slate-300 whitespace-pre-wrap">
                    {commentary?.commentary || 'No specific opportunity bullets extracted.'}
                  </p>
                )}
              </div>
            ) : (
              /* XAI Focus & Historical Memory Tab */
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
                              'w-full text-left rounded-lg border px-3 py-2 transition-colors',
                              isSelected
                                ? 'border-copper-400/40 bg-copper-500/10'
                                : 'border-white/5 bg-white/5 hover:border-copper-400/40',
                            )}
                          >
                            <div className="flex items-center justify-between gap-2 mb-1">
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="text-xs font-mono text-copper-300">
                                  #{idx + 1}
                                </span>
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
                              <p className="mt-1 text-xs font-mono text-slate-400">
                                {recentText}
                              </p>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Part 2: Historical Lookback Memory (Plain-language Attention) */}
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
                      higher attention share to past days whose market regime or shock best anchors the
                      next step.
                    </p>

                    <div className="flex flex-wrap items-center gap-1.5">
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
                              'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono border transition-colors',
                              isSelected
                                ? 'bg-copper-500/10 border-copper-400/70 text-copper-300'
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
                      <div className="rounded-lg border border-white/10 bg-slate-950/85 px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
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
          </div>

          {/* Non-causal XAI & Financial Disclaimer Footer */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
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
