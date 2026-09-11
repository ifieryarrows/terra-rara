import { useId } from 'react';
import { motion, useTransform, type MotionValue } from 'framer-motion';
import { sceneOpacity, tracePath } from './narrative';
function WorldLayer({ progress, index, children, staticIndex }: {
    staticIndex?: number;
    progress: MotionValue<number>;
    index: number;
    children: React.ReactNode;
}) {
    const opacity = useTransform(progress, p => sceneOpacity(p, index));
    if (staticIndex !== undefined && staticIndex !== index)
        return null;
    return <motion.g style={{ opacity }}>{children}</motion.g>;
}
export function CopperWorld({ progress, staticIndex }: {
    progress: MotionValue<number>;
    staticIndex?: number;
}) {
    const id = useId().replace(/:/g, '');
    const d = useTransform(progress, tracePath);
    const opacity = useTransform(progress, [0, .12, .22, .8, 1], [.9, .6, .25, .35, .9]);
    const x = useTransform(progress, [0, .5, 1], ['10%', '-15%', '20%']);
    return <div className="cw-world" aria-hidden="true">
    <motion.div className="cw-atmosphere" style={{ opacity, x }}/>
    <svg className="cw-diagram" viewBox="0 0 640 500">
      <defs><linearGradient id={id}><stop stopColor="#915032"/><stop offset=".5" stopColor="#ffd5ad"/><stop offset="1" stopColor="#c97a4e"/></linearGradient></defs>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={0}><g fill="none" stroke={`url(#${id})`} strokeWidth=".8">{Array.from({ length: 23 }, (_, i) => <path key={i} d={tracePath(0)} transform={`translate(320 250) scale(${.4 + i * .034}) translate(-320 -250)`} opacity={.3 + i / 40}/>)}</g><text x="320" y="260" className="cw-cu" textAnchor="middle">Cu</text><text x="320" y="292" textAnchor="middle">29 / COPPER</text></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={1}><text x="320" y="155" textAnchor="middle">ONE CONTINUOUS THREAD</text><circle cx="320" cy="250" r="9" fill="var(--cm-copper)"/></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={2}>{['HG=F', 'FCX', 'BHP', 'RIO'].map((v, i) => <g key={v}><path d={`M${110 + i * 140} 250V${i % 2 ? 325 : 175}`} className="cw-fine"/><circle cx={110 + i * 140} cy={i % 2 ? 325 : 175} r="32" className="cw-node"/><text x={110 + i * 140} y={i % 2 ? 330 : 180} textAnchor="middle">{v}</text></g>)}<text x="60" y="410">RELATED INSTRUMENTS / CONCEPTUAL MAP</text></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={3}><path d="M70 150H570M70 350H570" className="cw-fine"/><text x="80" y="190">SOURCE</text><text x="80" y="225" className="cw-large">What if supply tightens?</text><text x="80" y="385">QUESTION → CONTEXT → INTERPRETATION</text><text x="80" y="415">ILLUSTRATIVE SCENARIO / NOT A LIVE ARTICLE</text></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={4}><path d="M370 230L580 115V265Z" fill="var(--cm-forecast)" opacity=".16"/><path d="M370 230L580 180" stroke="var(--cm-forecast)" strokeDasharray="5 7" fill="none"/><path d="M370 110V370M60 370H580" className="cw-fine"/><text x="60" y="410">HISTORY</text><text x="370" y="410">POSSIBLE RANGE</text><text x="60" y="445">ILLUSTRATIVE PATH / NOT A FORECAST</text></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={5}>{Array.from({ length: 13 }, (_, i) => <path key={i} d={`M${60 + i * 43.3} 330v${i % 3 === 0 ? 20 : 10}`} className="cw-fine"/>)}<text x="60" y="290" className="cw-large">A result must be inspectable.</text><text x="60" y="395">DATE / HORIZON / METHOD / LIMITS</text></WorldLayer>
      <WorldLayer staticIndex={staticIndex} progress={progress} index={6}><circle cx="320" cy="250" r="60" className="cw-node"/><text x="320" y="265" textAnchor="middle" className="cw-cu">Cu</text></WorldLayer>
      <motion.path style={{ d }} stroke={`url(#${id})`} strokeWidth="2" fill="none"/>
    </svg>
  </div>;
}
