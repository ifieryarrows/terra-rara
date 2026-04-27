import { useSystemStatus } from '../hooks/useQueries';
import { DEFAULT_COPPER_SYMBOL } from '../config/instruments';

const StatusDot = ({ tone }: { tone: 'good' | 'bad' | 'neutral' }) => (
  <span
    className={`inline-block w-2 h-2 rounded-full ${
      tone === 'good' ? 'bg-emerald-400' : tone === 'bad' ? 'bg-rose-400' : 'bg-amber-400'
    }`}
  />
);

const Row = ({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: React.ReactNode;
  tone?: 'good' | 'bad' | 'neutral';
}) => (
  <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-b-0">
    <span className="text-sm text-slate-400">{label}</span>
    <div className="flex items-center gap-2">
      <StatusDot tone={tone} />
      <span className="font-mono text-sm text-slate-100">{value}</span>
    </div>
  </div>
);

const fmtSeconds = (s?: number | null) => {
  if (s == null) return '—';
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
};

export const SystemPage = () => {
  const { data, isLoading, isError, error, refetch, isFetching } = useSystemStatus();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 border-2 border-slate-600 border-t-amber-400 rounded-full animate-spin" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-10 border border-rose-800/40 bg-rose-950/30 rounded-lg text-rose-200 text-sm">
        Unable to reach the API health endpoint.
        {error instanceof Error ? ` (${error.message})` : ''}
      </div>
    );
  }

  const d: any = data;
  const overallTone: 'good' | 'bad' | 'neutral' =
    d.status === 'healthy' ? 'good' : d.status === 'unhealthy' ? 'bad' : 'neutral';

  const redisTone: 'good' | 'bad' | 'neutral' =
    d.redis_ok === true ? 'good' : d.redis_ok === false ? 'bad' : 'neutral';

  const modelTone: 'good' | 'bad' | 'neutral' =
    typeof d.models_found === 'number' && d.models_found > 0 ? 'good' : 'bad';

  const snapshotAge = d.last_snapshot_age_seconds;
  const snapshotTone: 'good' | 'bad' | 'neutral' =
    snapshotAge == null
      ? 'neutral'
      : snapshotAge < 3600 * 24
      ? 'good'
      : snapshotAge < 3600 * 36
      ? 'neutral'
      : 'bad';

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">System Status</h2>
          <p className="text-sm text-slate-400 mt-1">
            Infrastructure health, snapshot freshness, and queue connectivity.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded text-xs font-bold tracking-wider uppercase ${
              overallTone === 'good'
                ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/30'
                : overallTone === 'bad'
                ? 'bg-rose-400/10 text-rose-400 border border-rose-400/30'
                : 'bg-amber-400/10 text-amber-400 border border-amber-400/30'
            }`}
          >
            {d.status}
          </span>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="text-slate-400 hover:text-white px-3 py-1 bg-slate-800 rounded border border-slate-700 text-xs disabled:opacity-50"
          >
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <section className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">Core Services</h3>
        <Row label="Database" value={d.db_type ?? '—'} tone="good" />
        <Row
          label="Redis queue"
          value={d.redis_ok === null || d.redis_ok === undefined ? 'unknown' : d.redis_ok ? 'ok' : 'down'}
          tone={redisTone}
        />
        <Row
          label="Pipeline lock"
          value={d.pipeline_locked ? 'locked (running)' : 'free'}
          tone={d.pipeline_locked ? 'neutral' : 'good'}
        />
        <Row
          label="Trained models on disk"
          value={d.models_found ?? 0}
          tone={modelTone}
        />
      </section>

      <section className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">Snapshot & Data</h3>
        <Row
          label="Latest snapshot age"
          value={fmtSeconds(snapshotAge)}
          tone={snapshotTone}
        />
        <Row label="News articles" value={d.news_count ?? '—'} />
        <Row label="Price bars" value={d.price_bars_count ?? '—'} />
        <Row
          label="Server timestamp"
          value={d.timestamp ? new Date(d.timestamp).toLocaleString() : '—'}
        />
      </section>

      <section className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">
          Pipeline & Data Freshness
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Each timestamp answers a different question. They are NOT interchangeable
          — e.g. a stale baseline close can still be produced by a fresh worker run
          if Yahoo is delayed.
        </p>
        <Row
          label="Pipeline run (worker) completed"
          value={
            d.last_pipeline_run_at
              ? new Date(d.last_pipeline_run_at).toLocaleString()
              : '—'
          }
          tone={
            d.last_pipeline_status === 'ok'
              ? 'good'
              : d.last_pipeline_status === 'stale' || d.last_pipeline_status === 'failed'
              ? 'bad'
              : 'neutral'
          }
        />
        <Row
          label="Pipeline status"
          value={d.last_pipeline_status ?? '—'}
          tone={
            d.last_pipeline_status === 'ok'
              ? 'good'
              : d.last_pipeline_status === 'stale' || d.last_pipeline_status === 'failed'
              ? 'bad'
              : 'neutral'
          }
        />
        <Row
          label="XGBoost snapshot generated"
          value={
            d.last_snapshot_generated_at
              ? new Date(d.last_snapshot_generated_at).toLocaleString()
              : '—'
          }
        />
        <Row
          label="TFT prediction persisted"
          value={
            d.last_tft_prediction_at
              ? new Date(d.last_tft_prediction_at).toLocaleString()
              : '—'
          }
          tone={d.last_tft_prediction_at ? 'good' : 'neutral'}
        />
        <Row
          label="TFT baseline close date"
          value={d.tft_reference_price_date ?? '—'}
        />
        <Row
          label="TFT model trained"
          value={
            d.tft_model_trained_at
              ? new Date(d.tft_model_trained_at).toLocaleString()
              : '—'
          }
        />
        <Row
          label={`Latest PriceBar (${DEFAULT_COPPER_SYMBOL})`}
          value={d.price_bar_latest_date ?? '—'}
          tone={
            typeof d.price_bar_staleness_days === 'number'
              ? d.price_bar_staleness_days <= 2
                ? 'good'
                : d.price_bar_staleness_days <= 4
                ? 'neutral'
                : 'bad'
              : 'neutral'
          }
        />
        <Row
          label="PriceBar staleness"
          value={
            typeof d.price_bar_staleness_days === 'number'
              ? `${d.price_bar_staleness_days} day${d.price_bar_staleness_days === 1 ? '' : 's'}`
              : '—'
          }
          tone={
            typeof d.price_bar_staleness_days === 'number'
              ? d.price_bar_staleness_days <= 2
                ? 'good'
                : d.price_bar_staleness_days <= 4
                ? 'neutral'
                : 'bad'
              : 'neutral'
          }
        />
      </section>

      <section className="bg-slate-900 border border-slate-800 rounded-lg p-4">
        <h3 className="text-xs uppercase tracking-widest text-slate-500 mb-3">
          HuggingFace Hub
        </h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          HF Hub is used <span className="text-slate-300">only as a model artifact
          store</span> — the weekly training workflow uploads the TFT checkpoint
          there, and the worker downloads it back on cold start. The daily pipeline
          does <span className="text-slate-300">not</span> write predictions or logs
          to HF; all prediction state lives in this database.
        </p>
      </section>
    </div>
  );
};
