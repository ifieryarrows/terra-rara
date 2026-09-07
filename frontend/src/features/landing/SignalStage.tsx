import { useId } from 'react';
import { motion, useTransform, type MotionValue } from 'framer-motion';
import { previewSeries } from './preview-data';
import { sceneProgress, sceneVisibility } from './story-timeline';

const historyPath = previewSeries.map((v, i) => `${i ? 'L' : 'M'}${40 + i / (previewSeries.length - 1) * 380},${280 - (v - 100) * 15}`).join(' ');
const sceneNames = ['Copper and its market context', 'From a source to a conditional interpretation', 'Observed prices and an illustrative future range'];

/** Same instrument marker remains mounted as the research question changes. */
export function SignalStage({ progress, active, staticScene }: { progress: MotionValue<number>; active: number; staticScene?: boolean }) {
  const id = useId().replace(/:/g, '');
  const market = useTransform(progress, p => staticScene ? Number(active === 0) : sceneVisibility(p, 0));
  const news = useTransform(progress, p => staticScene ? Number(active === 1) : sceneVisibility(p, 1));
  const forecast = useTransform(progress, p => staticScene ? Number(active === 2) : sceneVisibility(p, 2));
  const trace = useTransform(progress, p => staticScene ? 1 : sceneProgress(p, 0));
  const rangeWidth = useTransform(progress, p => 200 * (staticScene ? 1 : sceneProgress(p, 2)));
  const markerX = useTransform(progress, [0, .24, .36, .64, .76, 1], [120, 120, 120, 120, 420, 420]);
  const markerY = useTransform(progress, [0, .64, .76, 1], [170, 170, 160, 160]);
  return <div className="cm-signal-stage">
    <div className="cm-signal-stage-header"><span>Cu / 29 <span className="cm-stage-symbol">HG=F</span></span><span>Illustrative preview</span></div>
    <svg viewBox="0 0 640 360" className="cm-signal-diagram" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>{sceneNames[active]}</title>
      <desc id={`${id}-desc`}>{active === 0 ? 'Copper is shown alongside producer equities and macro context. Links are research categories, not measured correlations.' : active === 1 ? 'An example supply disruption is interpreted conditionally alongside inventories and demand. This is not a published article.' : 'A fixed sample history ends at the last observation. A median path and widening illustrative range extend five sessions. These are not model results.'}</desc>
      <defs><clipPath id={`${id}-range`}><motion.rect x="420" y="30" width={rangeWidth} height="290"/></clipPath></defs>
      <path d="M24 334H616 M24 26H616" stroke="var(--cm-border)"/>
      <motion.g style={{ opacity: market }}>
        <motion.path d="M120 170H200V84H262 M200 170V263H262" fill="none" stroke="var(--cm-copper)" strokeWidth="1.5" style={{ pathLength: trace }}/>
        <rect x="262" y="42" width="340" height="92" fill="#143b32"/>
        <rect x="262" y="217" width="340" height="92" fill="var(--cm-surface-raised)"/>
        <text x="282" y="76" className="cm-diagram-label">PRODUCER EQUITIES</text><text x="282" y="112" className="cm-diagram-value">FCX · BHP · RIO</text>
        <text x="282" y="253" className="cm-diagram-label">MACRO CONTEXT</text><text x="282" y="287" className="cm-diagram-value">Dollar · Energy · Demand</text>
        <text x="82" y="234" className="cm-diagram-label">COPPER</text>
      </motion.g>
      <motion.g style={{ opacity: news }}>
        <path d="M120 170H246V74H602 M246 170H602 M246 170V272H602" fill="none" stroke="var(--cm-copper)" strokeWidth="1.5"/>
        <rect x="267" y="48" width="230" height="36" fill="var(--cm-bg)"/>
        <rect x="267" y="144" width="260" height="36" fill="var(--cm-bg)"/>
        <rect x="267" y="246" width="300" height="36" fill="var(--cm-bg)"/>
        <text x="280" y="72" className="cm-diagram-value">Supply tightens?</text>
        <text x="280" y="168" className="cm-diagram-value">Inspect the source.</text>
        <text x="280" y="270" className="cm-diagram-value">Question the reading.</text>
        <text x="82" y="234" className="cm-diagram-label">CONTEXT</text>
      </motion.g>
      <motion.g style={{ opacity: forecast }}>
        {[90, 160, 230, 300].map(y => <path key={y} d={`M40 ${y}H612`} stroke="var(--cm-border)" strokeDasharray="2 6"/>)}
        <path d={historyPath} fill="none" stroke="var(--cm-copper)" strokeWidth="2"/>
        <g clipPath={`url(#${id}-range)`}><path d="M420 160L460 133L500 111L540 85L600 55L600 265L540 229L500 207L460 183Z" fill="var(--cm-forecast)" opacity=".18"/><path d="M420 160L460 157L500 163L540 153L600 158" stroke="var(--cm-forecast)" strokeWidth="2" strokeDasharray="4 5" fill="none"/></g>
        <path d="M420 48V307" stroke="var(--cm-muted)" strokeDasharray="3 7"/>
        <text x="40" y="324" className="cm-diagram-label">HISTORY</text><text x="420" y="40" className="cm-diagram-label" textAnchor="middle">LAST CLOSE</text><text x="600" y="324" textAnchor="end" className="cm-diagram-label">5 SESSIONS</text>
      </motion.g>
      <motion.g style={{ x: staticScene ? (active === 2 ? 420 : 120) : markerX, y: staticScene ? (active === 2 ? 160 : 170) : markerY }}>
        <circle r="28" fill="var(--cm-bg)" stroke="var(--cm-copper)"/><text y="7" textAnchor="middle" fill="var(--cm-copper)" fontSize="22">Cu</text>
      </motion.g>
    </svg>
  </div>;
}

export function SceneReading({ index }: { index: number }) {
  if (index === 0) return <div className="cm-scene-reading"><p className="cm-eyebrow">ONE INSTRUMENT. A WIDER VIEW.</p><h4>Follow the context.</h4><p>Start with copper, then inspect producer equities and the wider market.</p><details><summary>Why these connections?</summary><p>Producer equities and macro instruments provide research context. These links do not represent measured correlations, trading signals or causation.</p></details></div>;
  if (index === 1) return <div className="cm-scene-reading"><p className="cm-eyebrow">HYPOTHETICAL SUPPLY SCENARIO</p><h4>What if mine supply tightens?</h4><p><span className="cm-reading-key">Interpretation</span> Potential upward pressure, depending on demand and available inventories.</p><details><summary>Read the reasoning</summary><p>Less supply can support prices, but inventories or weaker demand may offset it. Inspect a real source before treating an explanation as evidence. This scenario is illustrative, not a published article or live sentiment.</p></details></div>;
  return <div className="cm-scene-reading"><p className="cm-eyebrow">THE FUTURE HAS A RANGE</p><h4>Beyond the last observation.</h4><p>Read the primary five-day outlook alongside the daily path. The range expresses uncertainty; it does not guarantee an outcome.</p><div className="cm-scene-legend"><span><i/>Observed history</span><span><i/>Median & illustrative range</span></div><p className="cm-scene-note">Sample data · not a forecast. T+1 diagnostics are separate from the primary 5D view.</p></div>;
}
