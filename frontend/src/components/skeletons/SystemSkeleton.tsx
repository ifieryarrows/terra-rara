import type { ReactNode } from 'react';
import { PageHeader } from '../ui/PageHeader';
import { SectionHeader } from '../ui/SectionHeader';
import { SkeletonBone } from './SkeletonBone';

export function SystemSkeleton({ header }: { header?: ReactNode }) {
  const fallbackHeader = (
    <PageHeader
      eyebrow="04 / AVAILABILITY & FRESHNESS"
      title="System Status"
      description="Infrastructure health, snapshot freshness and queue connectivity."
      actions={
        <>
          <SkeletonBone className="w-20 h-7" rounded="full" />
          <SkeletonBone className="w-24 h-7" rounded="md" />
        </>
      }
    />
  );

  return (
    <div
      className="space-y-6 cm-skeleton-view"
      role="status"
      aria-busy="true"
      aria-live="polite"
      aria-label="Loading system status"
    >
      <span className="sr-only">Checking system status. Retrieving service availability and data timestamps.</span>
      {header ?? fallbackHeader}

      {/* Health status grid */}
      <div className="cm-system-grid" aria-hidden="true">
        <section className="cm-panel">
          <SectionHeader
            title="Core services"
            description="Availability reported by the health service."
          />
          {[
            'Database',
            'Redis queue',
            'Pipeline lock',
            'Trained models on disk',
          ].map((label, idx) => (
            <div key={idx} className="cm-status-row">
              <span className="text-sm text-slate-400">{label}</span>
              <div className="flex items-center gap-2">
                <SkeletonBone className="w-2 h-2 rounded-full shrink-0" />
                <SkeletonBone className="w-16 h-4" />
              </div>
            </div>
          ))}
        </section>

        <section className="cm-panel">
          <SectionHeader
            title="Snapshot & data"
            description="Stored observations and the latest available snapshot."
          />
          {[
            'Latest snapshot age',
            'News articles',
            'Price bars',
            'Server timestamp',
          ].map((label, idx) => (
            <div key={idx} className="cm-status-row">
              <span className="text-sm text-slate-400">{label}</span>
              <div className="flex items-center gap-2">
                <SkeletonBone className="w-2 h-2 rounded-full shrink-0" />
                <SkeletonBone className="w-20 h-4" />
              </div>
            </div>
          ))}
        </section>
      </div>

      {/* Freshness table / section */}
      <section className="cm-panel" aria-hidden="true">
        <SectionHeader
          title="Data freshness"
          description="Compare the worker run, forecast creation and underlying market date separately. A recent run can still use an older market close."
        />
        {[
          'Pipeline run (worker) completed',
          'Pipeline status',
          'XGBoost snapshot generated',
          'TFT prediction persisted',
          'TFT baseline close date',
          'TFT model trained',
          'Latest PriceBar (HG.CMX)',
          'PriceBar staleness',
        ].map((label, idx) => (
          <div key={idx} className="cm-status-row">
            <span className="text-sm text-slate-400">{label}</span>
            <div className="flex items-center gap-2">
              <SkeletonBone className="w-2 h-2 rounded-full shrink-0" />
              <SkeletonBone className="w-28 h-4" />
            </div>
          </div>
        ))}
      </section>

      {/* Artifact Storage Section */}
      <section className="cm-panel" aria-hidden="true">
        <SectionHeader title="Model artifact storage" />
        <p className="text-xs text-slate-400 leading-relaxed">
          HF Hub is used <span className="text-slate-300">only as a model artifact
          store</span> — the weekly training workflow uploads the TFT checkpoint
          there, and the worker downloads it back on cold start. The daily pipeline
          does <span className="text-slate-300">not</span> write predictions or logs
          to HF; all prediction state lives in this database.
        </p>
      </section>
    </div>
  );
}
