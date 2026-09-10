import { lazy, Suspense, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ArrowRight, ChartNoAxesCombined, Newspaper, ScanLine, ShieldCheck } from 'lucide-react';
import { motion, useMotionValue, useTransform, type MotionValue } from 'framer-motion';
import { Brand } from '../../components/ui/Brand';
import { Hero } from './Hero';
import { EvidencePreview } from './Previews';
import { ResearchStory } from './ResearchStory';
import { useExperiencePolicy } from './useExperiencePolicy';
import './landing.css';

const CinematicLanding = lazy(() => import('./CinematicLanding').then(module => ({ default: module.CinematicLanding })));

function EntryRevealLine({ progress, index, startAt = .06, step = .11, duration = .16, className = '', children }: { progress: MotionValue<number> | null; index: number; startAt?: number; step?: number; duration?: number; className?: string; children: ReactNode }) {
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const start = startAt + index * step;
  const maskX = useTransform(timeline, [start, start + duration], ['0%', '-105%']);
  const textOpacity = useTransform(timeline, [Math.max(0, start - .03), start + .055], [0, 1]);
  const textY = useTransform(timeline, [start, start + duration], [12, 0]);
  return <span className={`cm-enter-reveal-line ${className}`}>
    <motion.span className="cm-enter-reveal-text" style={progress ? { opacity: textOpacity, y: textY } : undefined}>{children}</motion.span>
    {progress ? <motion.span className="cm-enter-reveal-mask" style={{ x: maskX }} aria-hidden="true"/> : null}
  </span>;
}

function EntryCTAContent({ progress, animated, cinematic = false }: { progress: MotionValue<number> | null; animated: boolean; cinematic?: boolean }) {
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const lineStart = cinematic ? .84 : .06;
  const lineStep = cinematic ? .025 : .11;
  const lineDuration = cinematic ? .08 : .16;
  const bridgeScale = useTransform(timeline, cinematic ? [.80, .87] : [0, .22], [0, 1]);
  const actionOpacity = useTransform(timeline, cinematic ? [.93, .98] : [.53, .73], [0, 1]);
  const actionY = useTransform(timeline, cinematic ? [.93, .98] : [.53, .73], [18, 0]);
  const noteOpacity = useTransform(timeline, cinematic ? [.97, 1] : [.68, .84], [0, 1]);
  return <>
    {animated ? <div className="cm-enter-bridge" aria-hidden="true"><motion.span style={{ scaleX: bridgeScale }}/></div> : null}
    <p className="cm-eyebrow"><EntryRevealLine progress={progress} index={0} startAt={lineStart} step={lineStep} duration={lineDuration}>YOUR RESEARCH STARTS HERE</EntryRevealLine></p>
    <h2 id="enter-title"><EntryRevealLine progress={progress} index={1} startAt={lineStart} step={lineStep} duration={lineDuration}>From perspective</EntryRevealLine><EntryRevealLine progress={progress} index={2} startAt={lineStart} step={lineStep} duration={lineDuration}>to your next question.</EntryRevealLine></h2>
    <p className="cm-enter-lede"><EntryRevealLine progress={progress} index={3} startAt={lineStart} step={lineStep} duration={lineDuration}>Open the workspace and explore the market.</EntryRevealLine></p>
    <motion.div className="cm-enter-action" style={animated ? { opacity: actionOpacity, y: actionY } : undefined}><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowRight size={19} aria-hidden="true"/></Link></motion.div>
    <motion.span className="cm-enter-note" style={animated ? { opacity: noteOpacity } : undefined}>Forecasts are uncertain. Availability and freshness are shown in the workspace.</motion.span>
  </>;
}

function CinematicEntryCTA({ progress }: { progress: MotionValue<number> }) {
  const opacity = useTransform(progress, [.80, .88], [0, 1]);
  const y = useTransform(progress, [.80, .91], [28, 0]);
  const visibility = useTransform(progress, value => value >= .80 ? 'visible' : 'hidden');
  const pointerEvents = useTransform(progress, value => value >= .80 ? 'auto' : 'none');
  return <motion.section className="cm-enter cm-enter--cinematic cm-cinematic-entry" aria-labelledby="enter-title" style={{ opacity, y, visibility, pointerEvents }}><EntryCTAContent progress={progress} animated cinematic/></motion.section>;
}

function StaticEntryCTA() {
  return <section className="cm-enter" aria-labelledby="enter-title"><EntryCTAContent progress={null} animated={false}/></section>;
}

export function LandingPage() {
  const { enhanced, quality } = useExperiencePolicy();
  return <div className={`cm-landing cm-landing--quality-${quality}${enhanced ? ' cm-landing--cinematic' : ''}`}>
    <a className="cm-skip" href="#main-content">Skip to content</a>
    <main id="main-content" tabIndex={-1}>
      {enhanced ? <Suspense fallback={<Hero enhanced={false}/>}><CinematicLanding quality={quality === 'high' ? 'high' : 'balanced'} entry={progress => <CinematicEntryCTA progress={progress}/>}/></Suspense> : <>
        <Hero enhanced={false}/>
        <section id="research" className="cm-research-intro" aria-labelledby="research-title"><p className="cm-eyebrow">THE CONNECTED VIEW</p><h2 id="research-title">A price is a point.<br/><span>Intelligence is the connection.</span></h2><p>Move from market structure to evidence in one connected view.</p><div className="cm-capabilities">{[{label:'Market context',icon:ChartNoAxesCombined},{label:'News intelligence',icon:Newspaper},{label:'Forecast ranges',icon:ScanLine},{label:'Model validation',icon:ShieldCheck}].map(({label,icon:Icon},i)=><div key={label}><span className="cm-capability-index">0{i+1}</span><Icon size={22} strokeWidth={1.4} aria-hidden="true"/><span>{label}</span></div>)}</div></section>
        <ResearchStory enhanced={false}/>
        <section id="evidence" className="cm-evidence" aria-labelledby="evidence-title"><div><p className="cm-eyebrow">04 / THE EVIDENCE</p><h2 id="evidence-title">A signal should<br/>stand up to scrutiny.</h2><p>Check the model, horizon and data date behind the signal.</p></div><div><EvidencePreview/><div className="cm-evidence-links"><Link to="/models"><span><small>MODEL INTELLIGENCE</small><strong>Understand the model.</strong><p>Metrics and quality-gate status.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/validation"><span><small>WALK-FORWARD VALIDATION</small><strong>Examine the evidence.</strong><p>Out-of-sample results and comparisons.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/system"><span><small>SYSTEM STATUS</small><strong>Know how fresh it is.</strong><p>Freshness and availability.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link></div></div></section>
      </>}
      {!enhanced ? <StaticEntryCTA/> : null}
    </main>
    <footer className="cm-landing-footer"><Brand/><p>Market context. Quantitative perspective.</p><Link to="/dashboard">Go straight to the dashboard <ArrowUpRight size={14} aria-hidden="true"/></Link></footer>
  </div>;
}
