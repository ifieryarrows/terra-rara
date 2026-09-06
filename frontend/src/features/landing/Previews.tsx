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
    <div className="cm-chart-heading"><div><span className="cm-data-caption">A range of possibilities.</span><h3>One clearer perspective.</h3></div><span className="cm-chart-unit">INDEX · BASE 100</span></div>
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
      <text x="24" y="342" fill="var(--cm-muted)" fontSize="12">HISTORICAL CONTEXT</text><text x="571" y="342" fill="var(--cm-forecast)" fontSize="12">ILLUSTRATIVE RANGE</text>
    </svg>
    <figcaption className="cm-preview-caption"><span><i className="cm-key cm-key--copper"/>Sample price path</span><span><i className="cm-key cm-key--blue"/>Possible range</span><span>Sample data · not a forecast</span></figcaption>
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
  { title: 'Read the source', text: 'Publisher, timestamp and the original article stay in view.', label: 'NEWS' },
  { title: 'Inspect the sentiment', text: 'Review the label, score and reasoning behind each signal.', label: 'CONTEXT' },
  { title: 'Open the wider analysis', text: 'Compare the news picture with the available AI commentary.', label: 'ANALYSIS' },
];

function NewsStep({ index, progress, children }: { index: number; progress: MotionValue<number>; children: ReactNode }) {
  const opacity = useTransform(progress, [index * .2, index * .2 + .5], [.2, 1]);
  const x = useTransform(progress, [index * .2, index * .2 + .5], [20, 0]);
  return <motion.li style={{ opacity, x }}>{children}</motion.li>;
}

export function NewsPreview({ progress }: { progress?: MotionValue<number> }) {
  return <figure className="cm-preview">
    <div className="cm-preview-top"><span>NEWS INTELLIGENCE</span><span className="cm-preview-tag">Workflow preview</span></div>
    <div className="cm-chart-heading"><div><span className="cm-data-caption">Context behind the headline.</span><h3>Connect news to sentiment.</h3></div></div>
    <ol className="cm-news-flow">
      {newsSteps.map((step, index) => {
        const content = <><span className="cm-flow-number">0{index + 1}</span><div><h4>{step.title}</h4><p>{step.text}</p></div><span className="cm-flow-label">{step.label}</span></>;
        return progress ? <NewsStep key={step.label} index={index} progress={progress}>{content}</NewsStep> : <li key={step.label}>{content}</li>;
      })}
    </ol>
    <figcaption className="cm-preview-caption">A preview of the research workflow · no generated news or live sentiment claims.</figcaption>
  </figure>;
}
