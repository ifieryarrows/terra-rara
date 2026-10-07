import type { ReactNode } from 'react';
import { PageHeader } from '../ui/PageHeader';
import { SectionHeader } from '../ui/SectionHeader';
import { SkeletonBone } from './SkeletonBone';

export function ValidationSkeleton({ header }: { header?: ReactNode }) {
  const fallbackHeader = (
    <PageHeader
      eyebrow="03 / THE EVIDENCE"
      title="Walk-Forward Validation"
      description={
        <>
          Out-of-sample backtest results and baseline comparisons.
          <span className="block mt-1">
            <SkeletonBone className="w-56 h-3.5 inline-block" />
          </span>
        </>
      }
      actions={
        <>
          <SkeletonBone className="w-24 h-7" rounded="full" />
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
      aria-label="Loading walk-forward validation"
    >
      <span className="sr-only">Retrieving the available out-of-sample report.</span>
      {header ?? fallbackHeader}

      {/* Out of Sample Backtest Summary Metrics */}
      <section>
        <SectionHeader
          eyebrow="OUT-OF-SAMPLE EVIDENCE"
          title="Validation at a glance"
          description="Read aggregate results first, then compare the baseline and individual windows. Metrics retain the horizon and aggregation of the published report."
        />
        <div className="cm-metric-grid" aria-hidden="true">
          {[
            { label: 'Directional Accuracy', hasHint: false },
            { label: 'Sharpe Ratio', hasHint: false },
            { label: 'Variance Ratio', hasHint: false },
            { label: 'MAE', hasHint: true },
            { label: 'RMSE', hasHint: true },
          ].map((item, idx) => (
            <div key={idx} className="cm-metric">
              <p className="cm-metric-label">{item.label}</p>
              <p className="cm-metric-value">
                <SkeletonBone className="w-24 h-7 my-1" rounded="md" />
              </p>
              {item.hasHint && (
                <p className="cm-metric-hint">
                  <SkeletonBone className="w-32 h-3" />
                </p>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Against the Theta baseline / equity curve placeholder */}
      <section>
        <SectionHeader
          title="Against the Theta baseline"
          description="Compare the reported model and baseline values on the same metric. A missing value is shown as a dash."
        />
        <div className="cm-table-scroll" aria-hidden="true">
          <table className="cm-table">
            <caption>Reported TFT-ASRO and Theta baseline metrics.</caption>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">TFT-ASRO</th>
                <th scope="col">Theta</th>
                <th scope="col">Reading guide</th>
              </tr>
            </thead>
            <tbody>
              {['Direction accuracy', 'Sharpe ratio', 'MAE'].map((label, idx) => (
                <tr key={idx}>
                  <th scope="row">{label}</th>
                  <td><SkeletonBone className="w-16 h-4" /></td>
                  <td><SkeletonBone className="w-16 h-4" /></td>
                  <td><SkeletonBone className="w-24 h-3" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details className="cm-data-disclosure mt-4" aria-hidden="true">
          <summary tabIndex={-1}>Comparison report details</summary>
        </details>
      </section>

      {/* Window-by-window slice table skeleton */}
      <section>
        <SectionHeader
          title="Window-by-window results"
          description="Look for consistency across evaluation windows. The table scrolls horizontally on smaller screens."
        />
        <div className="cm-table-scroll" aria-hidden="true">
          <table className="cm-table">
            <caption>Out-of-sample results by validation window.</caption>
            <thead>
              <tr>
                <th scope="col">Window</th>
                <th scope="col">Direction accuracy</th>
                <th scope="col">Sharpe</th>
                <th scope="col">MAE</th>
                <th scope="col">RMSE</th>
                <th scope="col">Variance ratio</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2, 3, 4, 5].map((windowNum) => (
                <tr key={windowNum} className="border-t border-slate-800">
                  <th scope="row">{windowNum}</th>
                  <td className="px-3 py-1.5"><SkeletonBone className="w-16 h-4" /></td>
                  <td className="px-3 py-1.5"><SkeletonBone className="w-14 h-4" /></td>
                  <td className="px-3 py-1.5"><SkeletonBone className="w-16 h-4" /></td>
                  <td className="px-3 py-1.5"><SkeletonBone className="w-16 h-4" /></td>
                  <td className="px-3 py-1.5"><SkeletonBone className="w-14 h-4" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
