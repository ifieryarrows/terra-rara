import type { ReactNode } from 'react';
import { PageHeader } from '../ui/PageHeader';
import { SectionHeader } from '../ui/SectionHeader';
import { SkeletonBone } from './SkeletonBone';

export function ModelsSkeleton({ header }: { header?: ReactNode }) {
  const fallbackHeader = (
    <PageHeader
      eyebrow="02 / MODEL INTELLIGENCE"
      title="TFT-ASRO Model"
      description={
        <>
          Weekly strategy, daily diagnostics and the evidence behind each forecast.
          <span className="block mt-1">
            <SkeletonBone className="w-64 h-3.5 inline-block" />
          </span>
        </>
      }
      actions={<SkeletonBone className="w-28 h-7" rounded="full" />}
    />
  );

  return (
    <div
      className="space-y-6 cm-skeleton-view"
      role="status"
      aria-busy="true"
      aria-live="polite"
      aria-label="Loading model intelligence"
    >
      <span className="sr-only">Retrieving the available checkpoint and validation metrics.</span>
      {header ?? fallbackHeader}

      {/* Checkpoint Summary Card Skeleton */}
      <div
        className="rounded-lg border border-slate-800 bg-slate-900/20 p-4 space-y-2"
        aria-hidden="true"
      >
        <div className="flex items-center justify-between">
          <SkeletonBone className="w-28 h-3.5 uppercase" />
          <SkeletonBone className="w-20 h-5" rounded="full" />
        </div>
        <SkeletonBone className="w-3/4 h-3.5" />
        <SkeletonBone className="w-1/2 h-3.5" />
      </div>

      {/* Primary Horizon (5D) Section */}
      <section>
        <SectionHeader
          eyebrow="PRIMARY / 5 TRADING DAYS"
          title="Weekly strategy"
          description="Read the five-day cumulative forecast first. Weekly risk ratios use √52 annualization."
        />
        <div className="cm-metric-grid" aria-hidden="true">
          {[
            'Weekly Directional Accuracy',
            'Weekly Sharpe Ratio',
            'Weekly Sortino Ratio',
            'Weekly Magnitude Ratio',
            'Weekly Tail Capture',
            'Raw Magnitude Ratio',
            'Cap Clipping Rate',
            'Weekly PI80 Coverage',
          ].map((label, idx) => (
            <div key={idx} className="cm-metric">
              <p className="cm-metric-label">{label}</p>
              <p className="cm-metric-value">
                <SkeletonBone className="w-24 h-7 my-1" rounded="md" />
              </p>
              <p className="cm-metric-hint">
                <SkeletonBone className="w-32 h-3" />
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Single-Step Diagnostics (T+1) */}
      <details className="cm-data-disclosure cm-diagnostics">
        <summary>
          <span>Daily diagnostics <small>T+1 · single-step evidence</small></span>
          <span aria-hidden="true">+</span>
        </summary>
        <p className="cm-section-description">
          These metrics describe the daily path. Daily risk ratios use √252 annualization; evaluate them separately from the weekly strategy above.
        </p>
        <div className="cm-metric-grid" aria-hidden="true">
          {[
            'Daily Directional Accuracy',
            'Daily Sharpe Ratio',
            'Daily Sortino Ratio',
            'Variance Ratio',
            'MAE',
            'RMSE',
            'Daily Tail Capture',
            'Pred σ / Actual σ',
          ].map((label, idx) => (
            <div key={idx} className="cm-metric">
              <p className="cm-metric-label">{label}</p>
              <p className="cm-metric-value">
                <SkeletonBone className="w-20 h-7 my-1" rounded="md" />
              </p>
              <p className="cm-metric-hint">
                <SkeletonBone className="w-28 h-3" />
              </p>
            </div>
          ))}
        </div>
      </details>

      {/* Feature Importance / Variable Inputs Section Placeholder */}
      <section>
        <SectionHeader
          title="Model inputs"
          description="Relative feature importance from the published model. Importance describes model use, not a causal effect on price."
        />
        <div className="cm-panel space-y-4" aria-hidden="true">
          {[95, 78, 62, 45, 30].map((pct, idx) => (
            <div key={idx} className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2">
                  <SkeletonBone className="w-16 h-4" rounded="md" />
                  <SkeletonBone className="w-36 h-4" />
                </div>
                <SkeletonBone className="w-12 h-3.5 font-mono" />
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <SkeletonBone className="h-full" style={{ width: `${pct}%` }} rounded="full" />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Configuration Section Placeholder */}
      <details className="cm-data-disclosure">
        <summary>Training configuration</summary>
      </details>
    </div>
  );
}
