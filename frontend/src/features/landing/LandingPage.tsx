import { lazy, Suspense, type ReactNode } from 'react';
import { ArrowRight, ChartNoAxesCombined, Newspaper, ScanLine, ShieldCheck } from 'lucide-react';
import { motion, useMotionValue, useSpring, useTransform, type MotionValue } from 'framer-motion';
import { Hero } from './Hero';
import { BrandMark } from '../../components/ui/BrandMark';
import { EnterWorkspaceLink } from '../../components/ui/LogoTransition';
import { ResearchStory } from './ResearchStory';
import { useExperiencePolicy } from './useExperiencePolicy';
import { CINEMATIC_ENTRY_FADE_FROM, CINEMATIC_ENTRY_TRIGGER, getCinematicEntryOpacity } from './cinematic-entry';
import './landing.css';

const CinematicLanding = lazy(() => import('./CinematicLanding').then(module => ({ default: module.CinematicLanding })));
const TerraAtmosphere = lazy(() => import('../../components/ui/TerraAtmosphere').then(module => ({ default: module.TerraAtmosphere })));

function EntryRevealLine({ progress, index, startAt = .06, step = .11, duration = .16, className = '', children }: { progress: MotionValue<number> | null; index: number; startAt?: number; step?: number; duration?: number; className?: string; children: ReactNode }) {
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const start = startAt + index * step;
  const textOpacity = useTransform(timeline, [Math.max(0, start - .02), start + duration * .78], [0, 1]);
  const textY = useTransform(timeline, [start, start + duration], [14, 0]);
  return <span className={`cm-enter-reveal-line ${className}`}>
    <motion.span className="cm-enter-reveal-text" style={progress ? { opacity: textOpacity, y: textY } : undefined}>{children}</motion.span>
  </span>;
}

function EntryCTAContent({ progress, revealProgress, animated, cinematic = false }: { progress: MotionValue<number> | null; revealProgress?: MotionValue<number>; animated: boolean; cinematic?: boolean }) {
  const staticProgress = useMotionValue(1);
  const timeline = progress ?? staticProgress;
  const contentTimeline = revealProgress ?? timeline;
  const lineStart = cinematic ? .02 : .06;
  const lineStep = cinematic ? .09 : .11;
  const lineDuration = cinematic ? .20 : .16;
  const bridgeScale = useTransform(timeline, cinematic ? [CINEMATIC_ENTRY_FADE_FROM, .97] : [0, .22], [0, 1]);
  const actionOpacity = useTransform(contentTimeline, cinematic ? [.32, .56] : [.53, .73], [0, 1]);
  const actionY = useTransform(contentTimeline, cinematic ? [.32, .56] : [.53, .73], [18, 0]);
  const noteOpacity = useTransform(contentTimeline, cinematic ? [.48, .72] : [.68, .84], [0, 1]);
  const brandOpacity = useTransform(contentTimeline, [0, .14], [0, 1]);
  const brandY = useTransform(contentTimeline, [0, .14], [14, 0]);
  const lineProgress = animated ? contentTimeline : progress;
  return <>
    {animated && !cinematic ? <div className="cm-enter-bridge" aria-hidden="true"><motion.span style={{ scaleX: bridgeScale }}/></div> : null}
    <motion.div className="cm-enter-brand-mark" style={animated ? { opacity: brandOpacity, y: brandY } : undefined} aria-hidden="true"><BrandMark size={52} variant="primary"/></motion.div>
    <p className="cm-eyebrow"><EntryRevealLine progress={lineProgress} index={0} startAt={lineStart} step={lineStep} duration={lineDuration}>YOUR RESEARCH STARTS HERE</EntryRevealLine></p>
    <h2 id="enter-title"><EntryRevealLine progress={lineProgress} index={1} startAt={lineStart} step={lineStep} duration={lineDuration}>From perspective</EntryRevealLine><EntryRevealLine progress={lineProgress} index={2} startAt={lineStart} step={lineStep} duration={lineDuration}>to your next question.</EntryRevealLine></h2>
    <p className="cm-enter-lede"><EntryRevealLine progress={lineProgress} index={3} startAt={lineStart} step={lineStep} duration={lineDuration}>Open the workspace and explore the market.</EntryRevealLine></p>
    <motion.div className="cm-enter-action" style={animated ? { opacity: actionOpacity, y: actionY } : undefined}><EnterWorkspaceLink className="cm-button">Enter CopperMind <ArrowRight size={19} aria-hidden="true"/></EnterWorkspaceLink></motion.div>
    <motion.span className="cm-enter-note" style={animated ? { opacity: noteOpacity } : undefined}>Forecasts are uncertain. Availability and freshness are shown in the workspace.</motion.span>
  </>;
}

const CINEMATIC_CTA_SPRING = { stiffness: 105, damping: 21, mass: .9 };

function CinematicEntryCTA({ progress, revealProgress }: { progress: MotionValue<number>; revealProgress: MotionValue<number> }) {
  const targetOpacity = useTransform(progress, getCinematicEntryOpacity);
  const opacity = useSpring(targetOpacity, CINEMATIC_CTA_SPRING);
  const scrollY = useTransform(progress, [CINEMATIC_ENTRY_FADE_FROM, CINEMATIC_ENTRY_TRIGGER], [28, 0]);
  const revealY = useTransform(revealProgress, [0, 1], [12, 0]);
  const combinedY = useTransform([scrollY, revealY], ([sy, ry]: number[]) => sy + ry);
  const y = useSpring(combinedY, CINEMATIC_CTA_SPRING);
  const targetScale = useTransform(progress, [CINEMATIC_ENTRY_FADE_FROM, CINEMATIC_ENTRY_TRIGGER], [.968, 1]);
  const scale = useSpring(targetScale, CINEMATIC_CTA_SPRING);
  const visibility = useTransform(opacity, value => value > .004 ? 'visible' : 'hidden');
  const pointerEvents = useTransform(progress, value => value >= .905 ? 'auto' : 'none');
  return <motion.section className="cm-enter cm-enter--cinematic cm-cinematic-entry" aria-labelledby="enter-title" style={{ opacity, y, scale, visibility, pointerEvents }}><EntryCTAContent progress={progress} revealProgress={revealProgress} animated cinematic/></motion.section>;
}

function StaticEntryCTA() {
  return <section className="cm-enter" aria-labelledby="enter-title"><EntryCTAContent progress={null} animated={false}/></section>;
}

function StaticLanding() {
  return <>
    <Hero enhanced={false}/>
    <section id="research" className="cm-research-intro" aria-labelledby="research-title"><p className="cm-eyebrow">THE CONNECTED VIEW</p><h2 id="research-title">A price is a point.<br/><span>Intelligence is the connection.</span></h2><p>Move from market structure to evidence in one connected view.</p><div className="cm-capabilities">{[{label:'Market context',icon:ChartNoAxesCombined},{label:'News intelligence',icon:Newspaper},{label:'Forecast ranges',icon:ScanLine},{label:'Model validation',icon:ShieldCheck}].map(({label,icon:Icon},i)=><div key={label}><span className="cm-capability-index">0{i+1}</span><Icon size={22} strokeWidth={1.4} aria-hidden="true"/><span>{label}</span></div>)}</div></section>
    <ResearchStory enhanced={false}/>
    <StaticEntryCTA/>
  </>;
}

export function LandingPage() {
  const { enhanced, quality } = useExperiencePolicy();
  return <div className={`cm-landing cm-landing--quality-${quality}${enhanced ? ' cm-landing--cinematic' : ''}`}>
    {!enhanced ? <Suspense fallback={<div aria-hidden="true" style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none', background: '#080e17' }}/>}><TerraAtmosphere className="cm-terra-atmosphere--landing-static"/></Suspense> : null}
    <a className="cm-skip" href="#main-content">Skip to content</a>
    <main id="main-content" tabIndex={-1}>
      {enhanced ? <Suspense fallback={<StaticLanding/>}><CinematicLanding quality={quality === 'high' ? 'high' : 'balanced'} entry={(progress, revealProgress) => <CinematicEntryCTA progress={progress} revealProgress={revealProgress}/>}/></Suspense> : <StaticLanding/>}
    </main>
  </div>;
}
