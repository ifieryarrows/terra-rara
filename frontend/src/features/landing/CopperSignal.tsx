import { useId } from 'react';
import { motion, type MotionValue } from 'framer-motion';

const contours = Array.from({ length: 26 }, (_, layer) => {
  const radius = 58 + layer * 6.2;
  return Array.from({ length: 97 }, (_, point) => {
    const angle = point / 96 * Math.PI * 2;
    const ripple = Math.sin(angle * 3 + layer * .055) * (12 + layer * .38);
    const x = 270 + Math.cos(angle) * (radius + ripple);
    const y = 230 + Math.sin(angle) * (radius + ripple) * .78 + Math.cos(angle * 2) * 18;
    return `${point ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ') + ' Z';
});

export function CopperSignal({ progress }: { progress?: MotionValue<number> }) {
  const id = useId().replace(/:/g, '');
  return <figure className="cm-copper-signal" aria-label="Copper at the center of market context, news intelligence and quantitative forecasts">
    <div className="cm-signal-coordinate"><span>MARKET SIGNAL</span><span>THE METAL. THE SIGNAL.</span></div>
    <svg className="cm-signal-contours" viewBox="0 0 540 470" aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-copper`} x1="0" y1="0" x2="1" y2="1"><stop stopColor="#76422d"/><stop offset=".34" stopColor="#e6a47a"/><stop offset=".54" stopColor="#ffe1c0"/><stop offset=".74" stopColor="#a35e3c"/><stop offset="1" stopColor="#492f2a"/></linearGradient>
        <radialGradient id={`${id}-core`}><stop stopColor="#e6a47a" stopOpacity=".12"/><stop offset="1" stopColor="#e6a47a" stopOpacity="0"/></radialGradient>
      </defs>
      <circle cx="270" cy="230" r="218" fill={`url(#${id}-core)`}/>
      <g fill="none" stroke={`url(#${id}-copper)`} strokeWidth="1.25">{contours.map((d, i) => <path key={i} d={d} opacity={.5 + i / 52}/>)}</g>
      <motion.path d={contours[15]} fill="none" stroke="var(--cm-copper)" strokeWidth="2.5" style={progress ? { pathLength: progress } : undefined}/><motion.path d="M270 230C365 230 450 318 450 380S270 430 270 470" fill="none" stroke="var(--cm-copper)" strokeWidth="2" style={progress ? { pathLength: progress } : undefined}/>
      <path d="M18 230H112 M428 230H522 M270 14V72 M270 394V450" stroke="var(--cm-border)" strokeDasharray="2 5"/>
      <circle cx="270" cy="230" r="4" fill="var(--cm-copper)"/>
      <text x="270" y="214" textAnchor="middle" fill="var(--cm-text)" fontSize="60" fontWeight="300" letterSpacing="-4">Cu</text>
      <text x="270" y="262" textAnchor="middle" fill="var(--cm-copper)" fontSize="11" letterSpacing="3">COPPER</text>
    </svg>
    <figcaption className="cm-signal-origin"><span>01 — THE METAL</span><strong>A single point of departure.</strong><span>Follow copper into the wider market.</span></figcaption>
  </figure>;
}
