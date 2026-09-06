import { useSystemStatus } from '../hooks/useQueries';
import { PageHeader } from '../components/ui/PageHeader';
import { ViewState } from '../components/ui/ViewState';
import { RefreshButton } from '../components/ui/RefreshButton';
import { SectionHeader } from '../components/ui/SectionHeader';
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
  <div className="cm-status-row">
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
  const { data, isLoading, isError, refetch, isFetching } = useSystemStatus();
  const header = <PageHeader eyebrow="04 / AVAILABILITY & FRESHNESS" title="System Status" description="Infrastructure health, snapshot freshness and queue connectivity." actions={<>{data?.status && <span className={`cm-filter-chip cm-tone-${data.status === 'healthy' ? 'good' : data.status === 'unhealthy' ? 'bad' : 'neutral'}`}>{data.status}</span>}<RefreshButton onClick={() => refetch()} busy={isFetching}/></>}/>;

  if (isLoading) {
    return <div className="space-y-6">{header}<ViewState kind="loading" title="Checking system status" description="Retrieving service availability and data timestamps."/></div>;
  }

  if (isError || !data) {
    return <div className="space-y-6">{header}<ViewState kind="error" title="System status is unavailable" description="The health service could not be reached. Refresh to check availability again."/></div>;
  }

  const d: any = data;
  const redisTone: 'good' | 'bad' | 'neutral' =
    d.redis_ok === true ? 'good' : d.redis_ok === false ? 'bad' : 'neutral';

  const modelTone: 'good' | 'bad' | 'neutral' =
    typeof d.models_found !== 'number' ? 'neutral' : d.models_found > 0 ? 'good' : 'bad';

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
      {header}

      <div className="cm-system-grid">
      <section className="cm-panel">
        <SectionHeader title="Core services" description="Availability reported by the health service."/>
        <Row label="Database" value={d.db_type ?? '—'} />
        <Row
          label="Redis queue"
          value={d.redis_ok === null || d.redis_ok === undefined ? 'unknown' : d.redis_ok ? 'ok' : 'down'}
          tone={redisTone}
        />
        <Row
          label="Pipeline lock"
          value={d.pipeline_locked == null ? 'unknown' : d.pipeline_locked ? 'locked (running)' : 'free'}
          tone={d.pipeline_locked === false ? 'good' : 'neutral'}
        />
        <Row
          label="Trained models on disk"
          value={d.models_found ?? '—'}
          tone={modelTone}
        />
      </section>

      <section className="cm-panel">
        <SectionHeader title="Snapshot & data" description="Stored observations and the latest available snapshot."/>
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
      </div>

      <section className="cm-panel">
        <SectionHeader title="Data freshness" description="Compare the worker run, forecast creation and underlying market date separately. A recent run can still use an older market close."/>
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

      <section className="cm-panel">
        <SectionHeader title="Model artifact storage"/>
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
};
