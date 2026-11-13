import { useCallback, useEffect, useRef, type MouseEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform, type MotionValue } from 'framer-motion';
import { CopperSignal } from './CopperSignal';
import { BrandMark } from '../../components/ui/BrandMark';
import { EnterWorkspaceLink } from '../../components/ui/LogoTransition';
import { TerraAtmosphere } from '../../components/ui/TerraAtmosphere';
import { ForecastPreview, MarketPreview, NewsPreview } from './Previews';
import { ParticleWorld, type ParticleQuality } from './ParticleWorld';
import { CosmicHud } from './CosmicHud';
import { createCinematicSceneStepper } from './cinematic-scroll';
import { useCinematicEntryReveal } from './cinematic-entry';
import './cinematic.css';

const beats = [
  {
    id: 'market', number: '01', label: 'THE MARKET', title: <>Start with the<br/>bigger picture.</>,
    text: 'See copper in context before focusing on a single forecast.',
    tags: ['Market heatmap', 'Related instruments'], link: 'Explore market context', to: '/dashboard#market-map', center: .25,
  },
  {
    id: 'news', number: '02', label: 'THE CONTEXT', title: <>See the signal<br/>behind the story.</>,
    text: 'Watch a headline become tone, impact and rationale.',
    tags: ['Tone scoring', 'LLM rationale'], link: 'Read the intelligence', to: '/dashboard#news-intelligence', center: .5,
  },
  {
    id: 'forecast', number: '03', label: 'THE POSSIBILITIES', title: <>See the range.<br/>Keep the uncertainty.</>,
    text: 'See the forecast range alongside the last close.',
    tags: ['Deep-learning forecasts', 'Price & risk context'], link: 'Examine the forecasts', to: '/dashboard#price-forecast', center: .75,
  },
] as const;

const ANCHOR_PROGRESS: Record<string, number> = {
  '#market': .24,
  '#news': .43,
  '#forecast': .66,
};

type CinematicEntryRenderer = (progress: MotionValue<number>, revealProgress: MotionValue<number>) => ReactNode;

function BackgroundWord({ progress, word, range, reverse = false, finale = false }: { progress: MotionValue<number>; word: string; range: [number, number, number, number]; reverse?: boolean; finale?: boolean }) {
  const peakOpacity = finale ? .05 : .10;
  const holdOpacity = finale ? .035 : .10;
  const opacity = useTransform(progress, range, [0, peakOpacity, holdOpacity, 0]);
  const visibility = useTransform(opacity, value => value > .002 ? 'visible' : 'hidden');
  const x = useTransform(progress, range, reverse ? ['-10%', '0%', '6%', '14%'] : ['12%', '3%', '-4%', '-12%']);
  const scale = useTransform(progress, range, finale ? [.94, 1, 1.02, 1.05] : [.88, 1, 1.04, 1.12]);
  return <motion.span className={`cm-background-word${finale ? ' cm-background-word--finale' : ''}`} style={{ opacity, visibility, x, scale }}>{word}</motion.span>;
}

function SceneSurface({ progress, range, persist = false, className = '', children }: { progress: MotionValue<number>; range: [number, number, number, number]; persist?: boolean; className?: string; children: ReactNode }) {
  const opacity = useTransform(progress, range, persist ? [0, 1, 1, 1] : [0, 1, 1, 0]);
  const visibility = useTransform(opacity, value => value > .003 ? 'visible' : 'hidden');
  const x = useTransform(progress, range, persist ? [110, 0, -24, -24] : [110, 0, -24, -150]);
  const y = useTransform(progress, range, [34, 0, 0, -18]);
  const scale = useTransform(progress, range, [.95, 1, 1, .96]);
  return <motion.div className={`cm-cinematic-surface ${className}`.trim()} style={{ opacity, visibility, x, y, scale }} aria-hidden="true">{children}</motion.div>;
}

function DashboardComposition({ progress }: { progress: MotionValue<number> }) {
  const marketProgress = useTransform(progress, [.17, .30], [0, 1]);
  const newsProgress = useTransform(progress, [.39, .59], [0, 1]);
  // Let the forecast clear before p = 0.75 so Venus and the emerging Milky Way
  // have an unobstructed negative-space stage.
  const forecastProgress = useTransform(progress, [.60, .74], [0, 1]);
  return <div className="cm-dashboard-composition">
    <SceneSurface progress={progress} range={[.16, .22, .36, .39]}><MarketPreview progress={marketProgress}/></SceneSurface>
    <SceneSurface progress={progress} range={[.39, .42, .575, .59]}><NewsPreview progress={newsProgress}/></SceneSurface>
    <SceneSurface progress={progress} range={[.60, .625, .69, .74]} className="cm-cinematic-surface--forecast"><ForecastPreview progress={forecastProgress}/></SceneSurface>
  </div>;
}

function HeroCopy({ progress, onNavigateAnchor }: { progress: MotionValue<number>; onNavigateAnchor: (event: MouseEvent<HTMLAnchorElement>) => void }) {
  const opacity = useTransform(progress, [0, .08, .165], [1, 1, 0]);
  const y = useTransform(progress, [0, .165], [0, -40]);
  const visibility = useTransform(opacity, value => value > .003 ? 'visible' : 'hidden');
  const pointerEvents = useTransform(progress, value => value <= .14 ? 'auto' : 'none');
  return <article className="cm-cinematic-beat cm-cinematic-hero" aria-labelledby="hero-title">
    <motion.div className="cm-cinematic-copy-block" style={{ opacity, y, visibility, pointerEvents }}>
      <p className="cm-eyebrow cm-brand-eyebrow" data-cm-route-reveal="copy"><BrandMark size={18} variant="small"/>COPPER INTELLIGENCE / TERRA RARA</p>
      <h1 id="hero-title" data-cm-route-reveal="copy">Read the market.<br/><span>See the structure.</span></h1>
      <p className="cm-hero-description" data-cm-route-reveal="copy">Behind every copper price, a bigger picture. Market context, news intelligence, and 5-day quantile forecasts in one workspace.</p>
      <div className="cm-hero-actions" data-cm-route-reveal="surface"><EnterWorkspaceLink className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></EnterWorkspaceLink><a href="#market" onClick={onNavigateAnchor} className="cm-discover">Follow the signal <ArrowDown size={16} aria-hidden="true"/></a></div>
      <nav className="cm-hero-capabilities" aria-label="Explore the platform" data-cm-route-reveal="surface"><a href="#market" onClick={onNavigateAnchor}>Market</a><a href="#news" onClick={onNavigateAnchor}>News</a><a href="#forecast" onClick={onNavigateAnchor}>Forecasts</a><Link to="/validation">Validation</Link></nav>
    </motion.div>
  </article>;
}

function StoryCopy({ progress, beat }: { progress: MotionValue<number>; beat: typeof beats[number] }) {
  const shortNewsWindow = beat.id === 'news';
  const shortForecastWindow = beat.id === 'forecast';
  const start = shortNewsWindow ? .39 : shortForecastWindow ? .60 : .16;
  const reveal = shortNewsWindow ? .425 : shortForecastWindow ? .63 : .21;
  const hold = shortNewsWindow ? .57 : shortForecastWindow ? .695 : .34;
  const end = shortNewsWindow ? .595 : shortForecastWindow ? .745 : .385;
  const opacity = useTransform(progress, [start, reveal, hold, end], [0, 1, 1, 0]);
  const y = useTransform(progress, [start, reveal, hold, end], [36, 0, 0, -28]);
  const titleOpacity = useTransform(progress, [start, reveal], [0, 1]);
  const titleY = useTransform(progress, [start, reveal], [16, 0]);
  const detailOpacity = useTransform(progress, [start + .015, reveal + .02], [0, 1]);
  const detailY = useTransform(progress, [start + .015, reveal + .02], [18, 0]);
  const visibility = useTransform(opacity, value => value > .003 ? 'visible' : 'hidden');
  const pointerEvents = useTransform(opacity, value => value > .12 ? 'auto' : 'none');
  return <article id={beat.id} className={`cm-cinematic-beat cm-cinematic-beat--${beat.id}`} aria-labelledby={`${beat.id}-title`}>
    <motion.div className="cm-cinematic-copy-block" style={{ opacity, y, visibility, pointerEvents }}>
      <p className="cm-eyebrow"><span>{beat.number}</span> / {beat.label}</p>
      <motion.h2 id={`${beat.id}-title`} style={{ opacity: titleOpacity, y: titleY }}>{beat.title}</motion.h2>
      <motion.div style={{ opacity: detailOpacity, y: detailY }}>
        <p className="cm-story-description">{beat.text}</p>
        <ul className="cm-story-tags">{beat.tags.map(tag => <li key={tag}>{tag}</li>)}</ul>
        <Link to={beat.to} className="cm-story-link">{beat.link}<ArrowUpRight size={16} aria-hidden="true"/></Link>
      </motion.div>
    </motion.div>
  </article>;
}

function StickyWorld({ progress, releaseProgress, quality, onStepToProgress }: { progress: MotionValue<number>; releaseProgress: MotionValue<number>; quality: ParticleQuality; onStepToProgress: (targetProgress: number) => void }) {
  const signalOpacity = useTransform(progress, [0, .09, .18], [1, .82, 0]);
  const signalScale = useTransform(progress, [0, .14, .19], [1, 1.02, 1.06]);
  const signalY = useTransform(progress, [0, .19], [0, -18]);
  const signalPath = useTransform(progress, [0, .14], [.28, 1]);
  const signalVisibility = useTransform(signalOpacity, value => value > .003 ? 'visible' : 'hidden');
  return <div className="cm-cinematic-sticky">
    <TerraAtmosphere progress={progress}/>
    <div className="cm-background-typography" aria-hidden="true">
      <BackgroundWord progress={progress} word="SIGNAL" range={[0, .025, .13, .21]}/>
      <BackgroundWord progress={progress} word="MARKET" range={[.14, .22, .34, .43]} reverse/>
      <BackgroundWord progress={progress} word="CONTEXT" range={[.36, .42, .56, .59]}/>
      <BackgroundWord progress={progress} word="FORECAST" range={[.60, .64, .69, .745]} reverse/>
      <BackgroundWord progress={progress} word="OBSERVATORY" range={[.86, .90, .95, .995]} finale/>
    </div>
    <ParticleWorld progress={progress} releaseProgress={entryRevealProgressOrFallback(releaseProgress)} quality={quality}/>
    <motion.div className="cm-cinematic-symbol" style={{ opacity: signalOpacity, visibility: signalVisibility, scale: signalScale, y: signalY }} aria-hidden="true"><div data-cm-route-reveal="surface" className="cm-cinematic-symbol-reveal"><CopperSignal progress={signalPath}/></div></motion.div>
    <DashboardComposition progress={progress}/>
    <CosmicHud progress={progress} onStepToProgress={onStepToProgress}/>
    <div className="cm-cinematic-progress" aria-hidden="true"><motion.span style={{ scaleX: progress }}/></div>
  </div>;
}

function entryRevealProgressOrFallback(value: MotionValue<number>) {
  return value;
}

export function CinematicLanding({ quality = 'high', entry }: { quality?: ParticleQuality; entry?: CinematicEntryRenderer }) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
  const entryRevealProgress = useCinematicEntryReveal(scrollYProgress);
  const scrollToSceneProgress = useCallback((targetProgress: number) => {
    const scene = ref.current;
    if (!scene) return;
    const rect = scene.getBoundingClientRect();
    const top = rect.top + window.scrollY;
    const length = Math.max(0, rect.height - window.innerHeight);
    window.scrollTo({
      top: Math.round(top + length * targetProgress),
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    });
  }, []);
  const handleNavigateAnchor = (event: MouseEvent<HTMLAnchorElement>) => {
    const hash = event.currentTarget.getAttribute('href') || '';
    const targetProgress = ANCHOR_PROGRESS[hash];
    if (targetProgress === undefined) return;
    event.preventDefault();
    scrollToSceneProgress(targetProgress);
  };
  useEffect(() => {
    const scene = ref.current;
    if (!scene) return;
    let frame = 0;
    let x = 72;
    let y = 46;
    const flush = () => {
      frame = 0;
      scene.style.setProperty('--pointer-x', `${x}%`);
      scene.style.setProperty('--pointer-y', `${y}%`);
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(flush); };
    const onPointerMove = (event: PointerEvent) => {
      x = event.clientX / window.innerWidth * 100;
      y = event.clientY / window.innerHeight * 100;
      schedule();
    };
    const onPointerLeave = () => { x = 72; y = 46; schedule(); };
    const sceneStepper = createCinematicSceneStepper({
      // Progress mapping matches useScroll(offset: start start → end end).
      getGeometry: () => {
        const rect = scene.getBoundingClientRect();
        return { top: rect.top + window.scrollY, length: Math.max(0, rect.height - window.innerHeight) };
      },
    });
    const onWheel = (event: WheelEvent) => { sceneStepper.handleWheel(event); };
    const onKeyDown = (event: KeyboardEvent) => { sceneStepper.handleKeyDown(event); };
    scene.addEventListener('pointermove', onPointerMove, { passive: true });
    scene.addEventListener('pointerleave', onPointerLeave, { passive: true });
    scene.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('keydown', onKeyDown);
    return () => {
      scene.removeEventListener('pointermove', onPointerMove);
      scene.removeEventListener('pointerleave', onPointerLeave);
      scene.removeEventListener('wheel', onWheel);
      window.removeEventListener('keydown', onKeyDown);
      sceneStepper.dispose();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);
  return <section ref={ref} id="research" className="cm-cinematic cm-story cm-story--enhanced" aria-label="A connected copper research journey">
    <StickyWorld progress={scrollYProgress} releaseProgress={entryRevealProgress} quality={quality} onStepToProgress={scrollToSceneProgress}/>
    <div className="cm-cinematic-copy">
      <HeroCopy progress={scrollYProgress} onNavigateAnchor={handleNavigateAnchor}/>
      {beats.map(beat => <StoryCopy key={beat.id} progress={scrollYProgress} beat={beat}/>) }
    </div>
    {entry ? <div className="cm-cinematic-entry-layer">{entry(scrollYProgress, entryRevealProgress)}</div> : null}
  </section>;
}
