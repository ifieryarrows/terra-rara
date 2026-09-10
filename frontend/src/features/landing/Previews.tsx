import { useId, type ReactNode } from 'react';
import { motion, useMotionValue, useTransform, type MotionValue } from 'framer-motion';
import { linePath, newsIntelligencePreview, previewMarkets, previewSeries } from './preview-data';

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

function NewsSequenceLayer({
  className,
  progress,
  range,
  children,
}: {
  className: string;
  progress: MotionValue<number>;
  range: [number, number, number, number];
  children: ReactNode;
}) {
  const opacity = useTransform(progress, range, [0, 1, 1, 0]);
  const y = useTransform(progress, range, [24, 0, 0, -18]);
  const scale = useTransform(progress, range, [.985, 1, 1, .99]);
  return <motion.div className={`cm-news-sequence-layer ${className}`} style={{ opacity, y, scale }}>{children}</motion.div>;
}

function NewsSignal({ progress }: { progress: MotionValue<number> }) {
  const pathLength = useTransform(progress, [.28, .54], [0, 1]);
  const signalOpacity = useTransform(progress, [.26, .34, .68, .76], [0, 1, 1, 0]);
  const markerX = useTransform(progress, [.34, .54], ['38%', '72%']);
  return <motion.div className="cm-news-tone-signal" style={{ opacity: signalOpacity }}>
    <svg viewBox="0 0 520 150" role="img" aria-label="Illustrative tone spectrum from negative to positive">
      <defs>
        <linearGradient id="cm-news-tone-spectrum" x1="0" x2="1">
          <stop offset="0" stopColor="var(--cm-bearish, #cb7683)" stopOpacity=".72"/>
          <stop offset=".5" stopColor="var(--cm-neutral, #d6a868)" stopOpacity=".55"/>
          <stop offset="1" stopColor="var(--cm-copper)" stopOpacity=".9"/>
        </linearGradient>
      </defs>
      <line x1="28" x2="492" y1="104" y2="104" stroke="url(#cm-news-tone-spectrum)" strokeWidth="3" strokeLinecap="round"/>
      <motion.path d="M28 104 C96 103 118 82 168 90 S245 119 294 82 S383 31 492 44" fill="none" stroke="var(--cm-copper)" strokeWidth="2.5" strokeLinecap="round" style={{ pathLength }}/>
      <motion.circle cx="0" cy="104" r="7" fill="var(--cm-copper)" style={{ cx: markerX }} />
      <line x1="28" x2="28" y1="98" y2="111" stroke="var(--cm-muted)"/><line x1="260" x2="260" y1="98" y2="111" stroke="var(--cm-muted)"/><line x1="492" x2="492" y1="98" y2="111" stroke="var(--cm-muted)"/>
      <text x="28" y="132" fill="var(--cm-muted)" fontSize="11">NEGATIVE</text><text x="260" y="132" fill="var(--cm-muted)" fontSize="11" textAnchor="middle">NEUTRAL</text><text x="492" y="132" fill="var(--cm-copper)" fontSize="11" textAnchor="end">POSITIVE</text>
    </svg>
    <div className="cm-news-tone-readout"><span>FINBERT TONE</span><strong>direction gains shape</strong></div>
  </motion.div>;
}

function NewsHeadlineLayer({ progress }: { progress: MotionValue<number> }) {
  const clipPath = useTransform(progress, [0, .16], ['inset(0 100% 0 0)', 'inset(0 0% 0 0)']);
  return <NewsSequenceLayer className="cm-news-headline-layer" progress={progress} range={[0, .08, .2, .3]}>
    <span className="cm-news-step-label">01 / SOURCE HEADLINE</span>
    <div className="cm-news-headline-copy">
      <span className="cm-news-route">{newsIntelligencePreview.symbol} / {newsIntelligencePreview.company}</span>
      <motion.h3 style={{ clipPath }}>{newsIntelligencePreview.headline}</motion.h3>
      <p>{newsIntelligencePreview.description}</p>
    </div>
  </NewsSequenceLayer>;
}

function NewsEntitiesLayer({ progress }: { progress: MotionValue<number> }) {
  const firstClip = useTransform(progress, [.15, .28], ['inset(0 100% 0 0)', 'inset(0 0% 0 0)']);
  const secondClip = useTransform(progress, [.2, .34], ['inset(0 100% 0 0)', 'inset(0 0% 0 0)']);
  return <NewsSequenceLayer className="cm-news-entities-layer" progress={progress} range={[.13, .22, .36, .46]}>
    <span className="cm-news-step-label">02 / SEMANTIC EMPHASIS</span>
    <p className="cm-news-entity-line"><motion.mark style={{ clipPath: firstClip }}>{newsIntelligencePreview.company}</motion.mark><span> / entity</span></p>
    <p className="cm-news-entity-line"><motion.mark style={{ clipPath: secondClip }}>{newsIntelligencePreview.symbol}</motion.mark><span> / ticker · market context</span></p>
    <p className="cm-news-entity-line cm-news-entity-line--event"><motion.mark style={{ clipPath: secondClip }}>copper supply</motion.mark><span> / {newsIntelligencePreview.eventType.replace('_', ' ')}</span></p>
  </NewsSequenceLayer>;
}

function NewsScoreLayer({ progress }: { progress: MotionValue<number> }) {
  const scoreScale = useTransform(progress, [.48, .62], [.72, 1]);
  return <NewsSequenceLayer className="cm-news-score-layer" progress={progress} range={[.44, .53, .7, .82]}>
    <span className="cm-news-step-label">04 / IMPACT SCORE</span>
    <div className="cm-news-score-layout">
      <div className="cm-news-score-orbit" aria-hidden="true"><motion.span style={{ scale: scoreScale }}><b>+{newsIntelligencePreview.impactScoreLlm.toFixed(2)}</b><small>LLM impact</small></motion.span></div>
      <dl className="cm-news-score-list">
        <div><dt>Label</dt><dd>{newsIntelligencePreview.label}</dd></div>
        <div><dt>Final score</dt><dd>+{newsIntelligencePreview.finalScore.toFixed(2)}</dd></div>
        <div><dt>Calibrated confidence</dt><dd>{Math.round(newsIntelligencePreview.confidence * 100)}%</dd></div>
        <div><dt>Relevance</dt><dd>{Math.round(newsIntelligencePreview.relevance * 100)}%</dd></div>
      </dl>
    </div>
    <p className="cm-news-score-event">{newsIntelligencePreview.eventType.replace('_', ' ')} · {newsIntelligencePreview.horizon}</p>
  </NewsSequenceLayer>;
}

function NewsInterpretationLayer({ progress }: { progress: MotionValue<number> }) {
  const lineClip = useTransform(progress, [.7, .83], ['inset(0 100% 0 0)', 'inset(0 0% 0 0)']);
  return <NewsSequenceLayer className="cm-news-interpretation-layer" progress={progress} range={[.68, .77, .93, 1]}>
    <span className="cm-news-step-label">05 / INTERPRETATION</span>
    <motion.blockquote style={{ clipPath: lineClip }}>“{newsIntelligencePreview.reasoning}”</motion.blockquote>
    <div className="cm-news-interpretation-meta"><span>LLM rationale · one-line article read</span><span>not a live score</span></div>
  </NewsSequenceLayer>;
}

export function NewsPreview({ progress }: { progress?: MotionValue<number> }) {
  const fallbackProgress = useMotionValue(1);
  const timeline = progress ?? fallbackProgress;
  const animated = Boolean(progress);
  return <figure className={`cm-preview cm-news-preview cm-news-intelligence-preview${animated ? '' : ' cm-news-preview--static'}`} data-news-sequence="market-to-intelligence">
    <div className="cm-preview-top"><span>NEWS INTELLIGENCE</span><span className="cm-preview-tag">Illustrative preview</span></div>
    <div className="cm-news-origin"><span className="cm-news-origin-symbol">{newsIntelligencePreview.symbol}</span><span>selected in market context</span><i aria-hidden="true"/><span>source → intelligence</span></div>
    <div className={`cm-news-demo-stage${animated ? '' : ' cm-news-demo-stage--static'}`}>
      <NewsHeadlineLayer progress={timeline}/>
      <NewsEntitiesLayer progress={timeline}/>
      <NewsSequenceLayer className="cm-news-tone-layer" progress={timeline} range={[.27, .36, .56, .67]}><span className="cm-news-step-label">03 / TONE SIGNAL</span><NewsSignal progress={timeline}/><div className="cm-news-finbert-snapshot"><span>POS {Math.round(newsIntelligencePreview.finbert.pos * 100)}%</span><span>NEU {Math.round(newsIntelligencePreview.finbert.neu * 100)}%</span><span>NEG {Math.round(newsIntelligencePreview.finbert.neg * 100)}%</span></div></NewsSequenceLayer>
      <NewsScoreLayer progress={timeline}/>
      <NewsInterpretationLayer progress={timeline}/>
    </div>
    <figcaption className="cm-preview-caption"><span><i className="cm-key cm-key--copper"/>tone → score → rationale</span><span>Production-shaped fields · deterministic input</span></figcaption>
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
