import { useId, type ReactNode } from 'react';
import { motion, useTransform, type MotionValue } from 'framer-motion';
import { linePath, previewMarkets, previewSeries } from './preview-data';

function ForecastRange({ progress }: { progress: MotionValue<number> }) {
  const opacity = useTransform(progress, [.35, .85], [0, 1]);
  return <motion.g style={{ opacity }}><RangeDrawing/></motion.g>;
}

function RangeDrawing() {
  return <><path d="M530,42 L585,20 L644,6 L710,-4 L760,-10 L760,130 L710,112 L644,93 L585,66 Z" fill="var(--cm-forecast)" opacity=".16"/><path d="M530,42 L585,43 L644,39 L710,48 L760,37" fill="none" stroke="var(--cm-forecast)" strokeWidth="2.6" strokeDasharray="5 6"/></>;
}

export function ForecastPreview({ progress }: { progress?: MotionValue<number> }) {
  const id = useId().replace(/:/g, '');
  return <figure className="cm-preview cm-forecast-preview">
    <div className="cm-preview-top"><span>COPPER / HG=F</span><span className="cm-preview-tag">Illustrative preview</span></div>
    <div className="cm-chart-heading"><div><span className="cm-data-caption">DAILY PATH / T+1 TO T+5</span><h3>Beyond the last close.</h3></div><span className="cm-chart-unit">INDEX · BASE 100</span></div>
    <svg viewBox="0 0 840 350" role="img" aria-labelledby={`${id}-title ${id}-desc`} className="cm-preview-chart">
      <title id={`${id}-title`}>Illustrative price path and forecast uncertainty</title>
      <desc id={`${id}-desc`}>A sample series rises from index 100 to 108. The blue range widens into the future to explain uncertainty. This is not a live quote or a model prediction.</desc>
      <defs><linearGradient id={`${id}-fill`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--cm-copper)" stopOpacity=".14"/><stop offset="100%" stopColor="var(--cm-copper)" stopOpacity="0"/></linearGradient></defs>
      {[110, 107, 104, 101, 98].map(value => { const y = 40 + (110 - value) / 12 * 250; return <g key={value}><line x1="24" x2="792" y1={y} y2={y} stroke="var(--cm-border)" strokeDasharray="2 7"/><text x="805" y={y + 4} fill="var(--cm-muted)" fontSize="12">{value}</text></g>; })}
      <g transform="translate(24 40)">
        <path d={`${linePath(previewSeries, 530, 250)} L530,270 L0,270 Z`} fill={`url(#${id}-fill)`}/>
        <motion.path d={linePath(previewSeries, 530, 250)} fill="none" stroke="var(--cm-copper)" strokeWidth="2.6" strokeLinejoin="round" style={progress ? { pathLength: progress } : undefined}/>
        {progress ? <ForecastRange progress={progress}/> : <RangeDrawing/>}
        <line x1="530" x2="530" y1="-16" y2="275" stroke="var(--cm-muted)" strokeDasharray="3 7"/>
        <circle cx="530" cy="42" r="5" fill="var(--cm-bg)" stroke="var(--cm-copper)" strokeWidth="2"/>
      </g>
      <text x="24" y="342" fill="var(--cm-muted)" fontSize="12">HISTORICAL CONTEXT</text><text x="571" y="342" fill="var(--cm-forecast)" fontSize="12">T+1 TO T+5</text>
    </svg>
    <div className="cm-horizon-summary"><div><span className="cm-data-caption">PRIMARY VIEW / 5D</span><strong>One cumulative outlook.</strong><p>Assess five sessions against the last close.</p></div><div><span className="cm-data-caption">UNCERTAINTY / Q10-Q90</span><strong>A range, not a promise.</strong><p>The daily path adds detail; T+1 is a separate diagnostic.</p></div></div><figcaption className="cm-preview-caption"><span><i className="cm-key cm-key--copper"/>History / last close</span><span><i className="cm-key cm-key--blue"/>Median + Q10-Q90</span><span>Sample data · not a forecast</span></figcaption>
  </figure>;
}

function AnimatedMarketCell({ index, progress, className, children }: { index: number; progress: MotionValue<number>; className: string; children: ReactNode }) {
  const start = index * .09;
  const x = useTransform(progress, [start, start + .5], [index % 2 ? 24 : -24, 0]);
  const y = useTransform(progress, [start, start + .5], [18 + index * 3, 0]);
  const opacity = useTransform(progress, [start, start + .5], [.25, 1]);
  return <motion.div className={className} style={{ x, y, opacity }}>{children}</motion.div>;
}

export function MarketPreview({ progress }: { progress?: MotionValue<number> }) {
  return <figure className="cm-preview">
    <div className="cm-preview-top"><span>MARKET CONTEXT</span><span className="cm-preview-tag">Illustrative preview</span></div>
    <div className="cm-chart-heading"><div><span className="cm-data-caption">From the metal to the market.</span><h3>See what moves together.</h3></div></div>
    <div className="cm-preview-map">{previewMarkets.map((market, index) => {
      const content = <><strong>{market.symbol}</strong><span>{market.change}</span><small>{market.name}</small></>;
      const className = `cm-preview-cell cm-preview-cell--${market.tone} cm-preview-cell--${market.size}`;
      return progress ? <AnimatedMarketCell key={market.symbol} index={index} progress={progress} className={className}>{content}</AnimatedMarketCell> : <div key={market.symbol} className={className}>{content}</div>;
    })}</div>
    <figcaption className="cm-preview-caption">Sample changes and tile sizes · the dashboard provides the full interactive market map.</figcaption>
  </figure>;
}

const newsSteps = [
  { title: 'What if mine supply tightens?', text: 'A hypothetical disruption raises a question: could inventories and demand offset the pressure?', label: 'RESEARCH SCENARIO' },
  { title: 'Potential upward pressure', text: 'Less supply could support prices. This illustrative sentiment depends on demand and available stocks.', label: 'SENTIMENT' },
  { title: 'Supply conditions / inventories / demand', text: 'Read these drivers alongside the commentary. A plausible explanation is not proof of causation.', label: 'DRIVERS' },
];

function NewsStep({ index, progress, children }: { index: number; progress: MotionValue<number>; children: ReactNode }) {
  const opacity = useTransform(progress, [index * .2, index * .2 + .5], [.65, 1]);
  const x = useTransform(progress, [index * .2, index * .2 + .5], [20, 0]);
  return <motion.li style={{ opacity, x }}>{children}</motion.li>;
}

export function NewsPreview({ progress }: { progress?: MotionValue<number> }) {
  return <figure className="cm-preview cm-news-preview">
    <div className="cm-preview-top"><span>NEWS INTELLIGENCE</span><span className="cm-preview-tag">Illustrative preview</span></div>
    <div className="cm-reader-source"><span>EXAMPLE SOURCE / RESEARCH SCENARIO</span><span>06 SEP 2026 · EXAMPLE DATE</span></div>
    <ol className="cm-news-flow">
      {newsSteps.map((step, index) => {
        const content = <><span className="cm-flow-number">0{index + 1}</span><div><h4>{step.title}</h4><p>{step.text}</p></div><span className="cm-flow-label">{step.label}</span></>;
        return progress ? <NewsStep key={step.label} index={index} progress={progress}>{content}</NewsStep> : <li key={step.label}>{content}</li>;
      })}
    </ol>
    <figcaption className="cm-preview-caption">Hypothetical scenario · not a published article or live sentiment. Inspect actual coverage in the dashboard.</figcaption>
  </figure>;
}

export function EvidencePreview() {
  return <figure className="cm-preview cm-evidence-preview">
    <div className="cm-preview-top"><span>VALIDATION / REPORT READER</span><span className="cm-preview-tag">Illustrative preview</span></div>
    <div className="cm-report-body"><span className="cm-data-caption">WALK-FORWARD / OUT OF SAMPLE</span><h3>Question the result.</h3><p>Compare the same horizon and evaluation period.</p>
      <div className="cm-report-period"><span>Primary horizon<strong>5 trading sessions</strong></span><span>Evaluation period<strong>Read from the report</strong></span></div>
      <dl className="cm-report-metrics"><div><dt>Weekly direction accuracy<small>How often the 5D direction was correct.</small></dt><dd>—</dd></div><div><dt>MAE / RMSE<small>Smaller errors are better. RMSE emphasizes larger misses.</small></dt><dd>—</dd></div><div><dt>Baseline comparison<small>Compare model and reference forecast on the same sample.</small></dt><dd>—</dd></div></dl>
      <p className="cm-report-empty">Values omitted in this preview. Missing results are not a zero score or a passed check.</p>
      <div className="cm-report-freshness"><span className="cm-data-caption">CHECK THE DATA DATE</span><p>Match the forecast’s reference close to the latest market close. Check snapshot age before interpreting the result.</p></div>
    </div><figcaption className="cm-preview-caption">Reading guide · no performance claims or live status.</figcaption>
  </figure>;
}
