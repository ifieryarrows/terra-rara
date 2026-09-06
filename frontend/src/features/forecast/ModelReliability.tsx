import { Link } from 'react-router-dom';
import type { TFTModelMetadata } from '../../types';
import { finite } from './chart-data';

export function ModelReliability({ metrics, unavailable }: { metrics?: TFTModelMetadata['metrics']; unavailable: boolean }) {
  const weekly = metrics?.weekly_directional_accuracy;
  const dailySharpe = metrics?.sharpe_ratio;
  const threshold = metrics?.weekly_sample_count != null && metrics.weekly_sample_count < 80 ? .51 : .53;
  return <div className="cm-reliability">
    <p className="cm-chart-note">Weekly direction and daily risk-adjusted performance describe different horizons.</p>
    <div className="cm-reliability-row">
      <span>Weekly Direction Accuracy<small>5 trading days · gate ≥ {(threshold * 100).toFixed(0)}%</small></span>
      <strong className={!finite(weekly) ? '' : weekly >= threshold ? 'cm-tone-good' : 'cm-tone-bad'}>{finite(weekly) ? `${(weekly * 100).toFixed(1)}%` : '—'}</strong>
    </div>
    {!finite(weekly) && <p className="cm-chart-note">Weekly metric unavailable</p>}
    <div className="cm-reliability-row">
      <span>Daily Sharpe<small>T+1 diagnostic · annualized daily returns</small></span>
      <strong>{finite(dailySharpe) ? dailySharpe.toFixed(2) : '—'}</strong>
    </div>
    {unavailable && <p className="cm-chart-note">Model metrics could not be refreshed.</p>}
    <Link className="cm-text-link" to="/models">Review model metrics →</Link>
  </div>;
}
