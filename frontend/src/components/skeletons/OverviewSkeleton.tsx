import { Activity, Brain, BarChart3, Crosshair, Cpu, Globe } from 'lucide-react';
import { BrandMark } from '../ui/BrandMark';
import { SkeletonBone } from './SkeletonBone';
import { COPPER_INSTRUMENT } from '../../config/instruments';

// Map skeleton matching OverviewPage MapSkeleton
const MapSkeleton = () => (
  <div className="cm-map-skeleton h-[400px] w-full flex items-center justify-center bg-midnight/50 rounded-xl" aria-hidden="true">
    <div className="flex flex-col items-center gap-3">
      <div className="cm-map-skeleton__visual" aria-hidden="true">
        <span className="cm-map-skeleton__orbit" />
        <span className="cm-map-skeleton__orbit cm-map-skeleton__orbit--reverse" />
        <Globe size={32} className="cm-map-skeleton__globe" />
      </div>
      <span className="text-slate-400 text-xs font-mono">Loading intelligence map...</span>
    </div>
  </div>
);

export function OverviewSkeleton() {
  return (
    <div
      className="cm-market-dashboard font-sans selection:bg-copper-500/30 cm-skeleton-view"
      data-cm-dashboard-ready="false"
      role="status"
      aria-busy="true"
      aria-live="polite"
      aria-label="Loading copper market overview"
    >
      <span className="sr-only">Opening your market view. Retrieving prices, market context and available forecasts.</span>
      <div className="cm-dashboard-page relative z-10 grid min-w-0 grid-cols-1">
        {/* Top Hero Strip Skeleton */}
        <header className="cm-dashboard-hero">
          <div className="cm-dashboard-hero-copy">
            <p className="cm-eyebrow cm-dashboard-hero-eyebrow" data-cm-route-reveal="copy">
              <BrandMark size={18} variant="small" />
              COPPER INTELLIGENCE / TERRA RARA
            </p>
            <h1 className="cm-dashboard-title" data-cm-route-reveal="copy">
              Copper<br /><span>market</span>
            </h1>
            <p className="cm-dashboard-hero-subtitle" data-cm-route-reveal="copy">
              COMEX futures · Price action, model outlook and market context.
            </p>
          </div>

          {/* Quote & Sentiment Strip Skeleton */}
          <div
            className="cm-dashboard-hero-data"
            data-cm-route-reveal="surface"
            aria-hidden="true"
          >
            <div className="cm-dashboard-instrument">
              <div>
                <p className="cm-dashboard-instrument-name">{COPPER_INSTRUMENT.displayName}</p>
                <span className="cm-dashboard-symbol">{COPPER_INSTRUMENT.canonicalSymbol}</span>
              </div>
              <span className="cm-dashboard-quote-status">
                <SkeletonBone className="w-2.5 h-2.5 rounded-full" />
                <SkeletonBone className="w-20 h-3" />
              </span>
            </div>

            <div className="cm-dashboard-price-row">
              <SkeletonBone className="w-40 h-12" rounded="md" />
              <span className="cm-dashboard-currency">USD</span>
              <SkeletonBone className="w-24 h-5" rounded="md" />
            </div>

            <div className="cm-dashboard-quote-meta">
              <SkeletonBone className="w-44 h-3" />
              <SkeletonBone className="w-32 h-3" />
            </div>

            <div className="cm-dashboard-sentiment">
              <div className="cm-dashboard-sentiment-copy">
                <span>NEWS SENTIMENT</span>
                <small>Headline tone</small>
              </div>
              <div className="cm-sentiment-badge">
                <SkeletonBone className="w-20 h-5" rounded="md" />
              </div>
              <span className="cm-dashboard-sentiment-score">
                <SkeletonBone className="w-14 h-5 inline-block" rounded="md" />
              </span>
            </div>
          </div>
        </header>

        {/* Overview Tools Skeleton */}
        <div className="cm-overview-tools" data-cm-route-reveal="surface" aria-hidden="true">
          <nav aria-label="Market overview sections">
            <a href="#price-forecast" tabIndex={-1}>Price action</a>
            <a href="#weekly-outlook" tabIndex={-1}>Weekly outlook</a>
            <a href="#market-drivers" tabIndex={-1}>Drivers</a>
            <a href="#news-intelligence" tabIndex={-1}>News flow</a>
            <a href="#market-map" tabIndex={-1}>Market map</a>
          </nav>
          <button type="button" disabled tabIndex={-1} className="cm-button cm-button--secondary cm-refresh opacity-60 flex items-center gap-2">
            <SkeletonBone className="w-4 h-4 rounded-full" />
            <SkeletonBone className="w-24 h-3.5" />
          </button>
        </div>

        {/* Content Grid */}
        <div className="cm-dashboard-content grid min-w-0 gap-4">
          <div className="cm-dashboard-primary-column grid grid-cols-12 gap-6" data-cm-route-reveal="surface">
            {/* Price action & Forecast chart container skeleton */}
            <section
              id="price-forecast"
              className="cm-panel cm-financial-panel cm-price-action-section"
              style={{ '--cm-panel-span': 12 } as any}
              aria-label="Price action skeleton"
            >
              <h2 className="cm-panel-title">
                <Activity size={18} aria-hidden="true" />
                Price action / {COPPER_INSTRUMENT.canonicalSymbol}
              </h2>
              <div className="cm-panel-body" aria-hidden="true">
                <div className="cm-price-chart">
                  {/* Toolbar with window tab bar */}
                  <div className="cm-chart-toolbar">
                    <fieldset className="cm-chart-window">
                      <legend>WINDOW</legend>
                      <div className="cm-chart-window-options">
                        <button type="button" aria-pressed="true" disabled>30<span>D</span></button>
                        <button type="button" aria-pressed="false" disabled>90<span>D</span></button>
                        <button type="button" aria-pressed="false" disabled>180<span>D</span></button>
                      </div>
                    </fieldset>
                  </div>
                  {/* Series Legend */}
                  <div className="cm-chart-legend" role="group" aria-label="Chart series">
                    <span><i className="cm-chart-key cm-chart-key--observed" aria-hidden="true" />Observed close</span>
                    <button type="button" aria-pressed="true" disabled><i className="cm-chart-key cm-chart-key--median" aria-hidden="true" />Forecast median</button>
                    <button type="button" aria-pressed="true" disabled><i className="cm-chart-key cm-chart-key--range" aria-hidden="true" />Q10–Q90 range</button>
                  </div>
                  {/* Chart plot container skeleton matching .cm-price-plot */}
                  <div className="cm-price-plot cm-chart-skeleton-plot relative overflow-hidden flex flex-col justify-between">
                    <div className="w-full h-px bg-white/5 my-auto" />
                    <div className="w-full h-px bg-white/5 my-auto" />
                    <div className="w-full h-px bg-white/5 my-auto" />
                    <div className="absolute inset-x-0 bottom-4 top-12 flex items-end px-4 opacity-30">
                      <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 500 150">
                        <path
                          d="M0,120 Q100,70 200,90 T350,50 T500,30 L500,150 L0,150 Z"
                          fill="url(#cm-skeleton-chart-grad)"
                        />
                        <defs>
                          <linearGradient id="cm-skeleton-chart-grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--cm-copper)" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="var(--cm-copper)" stopOpacity="0.0" />
                          </linearGradient>
                        </defs>
                      </svg>
                    </div>
                  </div>
                  <p className="cm-chart-note">Hover or use the chart’s arrow keys to inspect a date. On touch screens, the data table provides every value. Q10–Q90 is the model’s 80% interval; outcomes can fall outside it.</p>
                  <details className="cm-chart-table">
                    <summary tabIndex={-1}>View chart data</summary>
                  </details>
                </div>
              </div>
            </section>

            {/* Weekly Outlook skeleton */}
            <section
              id="weekly-outlook"
              className="cm-panel cm-financial-panel cm-weekly-forecast-panel"
              style={{ '--cm-panel-span': 6 } as any}
              aria-label="Weekly model outlook skeleton"
            >
              <h2 className="cm-panel-title">
                <Brain size={18} aria-hidden="true" />
                Weekly model outlook
              </h2>
              <div className="cm-panel-body space-y-4" aria-hidden="true">
                <SkeletonBone className="w-28 h-8" rounded="xl" />
                <div className="space-y-2">
                  <SkeletonBone className="w-36 h-3" />
                  <div className="flex items-baseline gap-2">
                    <SkeletonBone className="w-28 h-8" rounded="md" />
                    <SkeletonBone className="w-16 h-4" />
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <SkeletonBone className="w-24 h-3" />
                    <SkeletonBone className="w-24 h-3 ml-auto" />
                  </div>
                </div>
                <div className="cm-weekly-range">
                  <p className="text-xs text-slate-400 uppercase tracking-widest mb-1.5">
                    <SkeletonBone className="w-28 h-3" />
                  </p>
                  <div className="flex items-center justify-between">
                    <SkeletonBone className="w-12 h-4" />
                    <div className="flex-1 mx-3 h-1.5 rounded-full bg-white/5 relative overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-r from-rose-500/40 via-gray-500/20 to-emerald-500/40 rounded-full" />
                    </div>
                    <SkeletonBone className="w-12 h-4" />
                  </div>
                </div>
                <details className="cm-data-disclosure">
                  <summary tabIndex={-1}>T+1 diagnostics</summary>
                </details>
                <div className="border-t border-white/5 pt-3 space-y-1.5">
                  <SkeletonBone className="w-36 h-3" />
                  <SkeletonBone className="w-24 h-4" />
                  <SkeletonBone className="w-64 h-3" />
                </div>
              </div>
            </section>

            {/* Market Drivers skeleton */}
            <section
              id="market-drivers"
              className="cm-panel cm-financial-panel"
              style={{ '--cm-panel-span': 6 } as any}
              aria-label="Market drivers skeleton"
            >
              <h2 className="cm-panel-title">
                <BarChart3 size={18} aria-hidden="true" />
                Market drivers
              </h2>
              <div className="cm-panel-body space-y-4" aria-hidden="true">
                {[80, 65, 50, 38, 25].map((widthPct, idx) => (
                  <div key={idx} className="space-y-1.5">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <SkeletonBone className="w-16 h-4" rounded="md" />
                        <SkeletonBone className="w-36 h-4" />
                      </div>
                      <SkeletonBone className="w-10 h-3" />
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <SkeletonBone className="h-full" style={{ width: `${widthPct}%` }} rounded="full" />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Model Reliability skeleton */}
            <section
              id="model-reliability"
              className="cm-panel cm-financial-panel"
              style={{ '--cm-panel-span': 4 } as any}
              aria-label="Model reliability skeleton"
            >
              <h2 className="cm-panel-title">
                <Crosshair size={18} aria-hidden="true" />
                Model reliability
              </h2>
              <div className="cm-panel-body" aria-hidden="true">
                <div className="cm-reliability">
                  <p className="cm-chart-note">
                    Weekly direction and daily risk-adjusted performance describe different horizons.
                  </p>
                  <div className="cm-reliability-row">
                    <span>
                      <SkeletonBone className="w-36 h-3.5 mb-1.5" />
                      <small><SkeletonBone className="w-28 h-3" /></small>
                    </span>
                    <SkeletonBone className="w-12 h-5" rounded="md" />
                  </div>
                  <div className="cm-reliability-row">
                    <span>
                      <SkeletonBone className="w-24 h-3.5 mb-1.5" />
                      <small><SkeletonBone className="w-32 h-3" /></small>
                    </span>
                    <SkeletonBone className="w-12 h-5" rounded="md" />
                  </div>
                  <span className="cm-text-link">Review model metrics →</span>
                </div>
              </div>
            </section>

            {/* Neural Analysis skeleton */}
            <section
              id="neural-analysis"
              className="cm-panel cm-financial-panel"
              style={{ '--cm-panel-span': 8 } as any}
              aria-label="Neural analysis skeleton"
            >
              <h2 className="cm-panel-title">
                <Cpu size={18} aria-hidden="true" />
                Neural analysis
              </h2>
              <div className="cm-panel-body space-y-3" aria-hidden="true">
                <div className="flex justify-between items-center mb-3">
                  <SkeletonBone className="w-20 h-4" />
                  <SkeletonBone className="w-24 h-5" rounded="full" />
                </div>
                <div className="cm-commentary text-sm text-gray-300 leading-relaxed space-y-2">
                  <SkeletonBone className="w-full h-4" />
                  <SkeletonBone className="w-11/12 h-4" />
                  <SkeletonBone className="w-4/5 h-4" />
                  <SkeletonBone className="w-9/12 h-4" />
                </div>
              </div>
            </section>
          </div>

          {/* Aside News Intelligence skeleton */}
          <aside id="news-intelligence" className="cm-dashboard-news-section min-w-0" data-cm-route-reveal="surface" aria-hidden="true">
            <div className="cm-news-panel glass-panel">
              <div className="cm-news-header flex items-center justify-between px-3 sm:px-4 pt-4 pb-2.5 border-b border-white/5">
                <div className="cm-news-title-group">
                  <span className="cm-news-icon"><SkeletonBone className="w-4 h-4 rounded-full" /></span>
                  <div>
                    <h2>Copper news flow</h2>
                    <p>Headlines, sentiment &amp; sources</p>
                  </div>
                </div>
                <SkeletonBone className="w-8 h-8" rounded="md" />
              </div>
              <div className="cm-news-scroll">
                <div className="cm-news-summary px-3 sm:px-4 pt-2.5 pb-3 border-b border-white/5 space-y-2">
                  <div className="flex items-center gap-1.5 text-xs font-mono mb-2">
                    <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300">↑ —</span>
                    <span className="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300">↓ —</span>
                    <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-200">· —</span>
                    <span className="ml-auto text-slate-400">
                      <SkeletonBone className="w-24 h-3.5 inline-block" />
                    </span>
                  </div>
                  <div className="cm-news-controls">
                    <div className="cm-field flex-1">
                      <span className="text-xs text-slate-400">Search headlines</span>
                      <SkeletonBone className="w-full h-12" rounded="md" />
                    </div>
                    <SkeletonBone className="w-12 h-12 shrink-0" rounded="md" />
                  </div>
                </div>
                <div className="cm-news-feed-shell">
                  <div className="cm-news-feed">
                    <div className="cm-news-feed-track">
                      <div className="cm-news-feed-set">
                        {[1, 2, 3].map((n) => (
                          <div key={n} className="cm-news-card space-y-3">
                            <div className="flex items-center justify-between">
                              <SkeletonBone className="w-20 h-4" rounded="full" />
                              <SkeletonBone className="w-16 h-3.5" />
                            </div>
                            <div className="space-y-2 flex-1">
                              <SkeletonBone className="w-full h-4" />
                              <SkeletonBone className="w-4/5 h-4" />
                              <SkeletonBone className="w-3/5 h-3.5" />
                            </div>
                            <div className="flex items-center justify-between pt-2 border-t border-white/5">
                              <SkeletonBone className="w-16 h-4" rounded="md" />
                              <SkeletonBone className="w-20 h-3.5" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </div>

        {/* Bottom Market Map skeleton */}
        <div id="market-map" className="cm-dashboard-market-map min-w-0 w-full" data-cm-route-reveal="surface">
          <MapSkeleton />
        </div>
      </div>
    </div>
  );
}
