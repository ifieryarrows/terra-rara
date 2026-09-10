import { useEffect, useRef, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform, type MotionValue } from 'framer-motion';
import { CopperSignal } from './CopperSignal';
import { EvidencePreview, ForecastPreview, MarketPreview, NewsPreview } from './Previews';
import { ParticleWorld, type ParticleQuality } from './ParticleWorld';
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
  {
    id: 'evidence', number: '04', label: 'THE EVIDENCE', title: <>Make the signal<br/>answer questions.</>,
    text: 'Check the model, horizon and data date behind the signal.',
    tags: ['Walk-forward validation', 'Freshness & model status'], link: 'Examine the evidence', to: '/validation', center: 1,
  },
] as const;

function BackgroundWord({ progress, word, range, reverse = false }: { progress: MotionValue<number>; word: string; range: [number, number, number, number]; reverse?: boolean }) {
  const opacity = useTransform(progress, range, [0, .12, .12, 0]);
  const visibility = useTransform(progress, value => value >= range[0] && value <= range[3] ? 'visible' : 'hidden');
  const x = useTransform(progress, range, reverse ? ['-10%', '0%', '6%', '14%'] : ['12%', '3%', '-4%', '-12%']);
  const scale = useTransform(progress, range, [.88, 1, 1.04, 1.12]);
  const clipPath = useTransform(progress, [range[0], range[1]], ['inset(0 100% 0 0)', 'inset(0 0% 0 0)']);
  return <motion.span className="cm-background-word" style={{ opacity, visibility, x, scale, clipPath }}>{word}</motion.span>;
}

function Atmosphere({ progress }: { progress: MotionValue<number> }) {
  const backgroundColor = useTransform(progress, [0, .25, .5, .75, 1], ['#080e17', '#0b1218', '#0a101a', '#090f1b', '#0b1119']);
  const copperX = useTransform(progress, [0, .5, 1], ['8%', '42%', '74%']);
  const blueX = useTransform(progress, [0, .5, 1], ['96%', '72%', '38%']);
  const bandX = useTransform(progress, [0, 1], ['-16%', '16%']);
  return <motion.div className="cm-global-atmosphere" style={{ backgroundColor }} aria-hidden="true">
    <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--copper" style={{ x: copperX }}/>
    <motion.div className="cm-atmosphere-glow cm-atmosphere-glow--blue" style={{ x: blueX }}/>
    <motion.div className="cm-atmosphere-band" style={{ x: bandX }}/>
    <div className="cm-atmosphere-pointer"/>
    <div className="cm-atmosphere-grain"/>
  </motion.div>;
}

function SceneSurface({ progress, range, persist = false, children }: { progress: MotionValue<number>; range: [number, number, number, number]; persist?: boolean; children: ReactNode }) {
  const opacity = useTransform(progress, range, persist ? [0, 1, 1, 1] : [0, 1, 1, 0]);
  const visibility = useTransform(progress, value => value >= range[0] && (persist || value <= range[3]) ? 'visible' : 'hidden');
  const x = useTransform(progress, range, persist ? [110, 0, -24, -24] : [110, 0, -24, -150]);
  const y = useTransform(progress, range, [34, 0, 0, -18]);
  const scale = useTransform(progress, range, [.94, 1, 1, .96]);
  const clipPath = useTransform(progress, range, persist ? ['inset(0 0 0 100%)', 'inset(0 0 0 0%)', 'inset(0 0 0 0%)', 'inset(0 0 0 0%)'] : ['inset(0 0 0 100%)', 'inset(0 0 0 0%)', 'inset(0 0 0 0%)', 'inset(0 100% 0 0)']);
  return <motion.div className="cm-cinematic-surface" style={{ opacity, visibility, x, y, scale, clipPath }} aria-hidden="true">{children}</motion.div>;
}

function DashboardComposition({ progress }: { progress: MotionValue<number> }) {
  const marketProgress = useTransform(progress, [.17, .39], [0, 1]);
  const newsProgress = useTransform(progress, [.36, .59], [0, 1]);
  const forecastProgress = useTransform(progress, [.60, .84], [0, 1]);
  return <div className="cm-dashboard-composition">
    <SceneSurface progress={progress} range={[.16, .22, .36, .43]}><MarketPreview progress={marketProgress}/></SceneSurface>
    <SceneSurface progress={progress} range={[.36, .42, .575, .59]}><NewsPreview progress={newsProgress}/></SceneSurface>
    <SceneSurface progress={progress} range={[.60, .625, .77, .84]}><ForecastPreview progress={forecastProgress}/></SceneSurface>
    <SceneSurface progress={progress} range={[.77, .84, .985, 1]} persist><EvidencePreview/></SceneSurface>
  </div>;
}

function HeroCopy({ progress }: { progress: MotionValue<number> }) {
  const opacity = useTransform(progress, [0, .1, .2], [1, 1, 0]);
  const y = useTransform(progress, [0, .2], [0, -72]);
  const blur = useTransform(progress, [.1, .2], ['blur(0px)', 'blur(10px)']);
  return <article className="cm-cinematic-beat cm-cinematic-hero" aria-labelledby="hero-title">
    <motion.div className="cm-cinematic-copy-block" style={{ opacity, y, filter: blur }}>
      <p className="cm-eyebrow"><span className="cm-eyebrow-line"/>COPPER INTELLIGENCE / TERRA RARA</p>
      <h1 id="hero-title">Read the market.<br/><span>See the structure.</span></h1>
      <p className="cm-hero-description">Behind every copper price, a bigger picture.</p>
      <p className="cm-hero-detail">Market, news, forecasts and evidence in one workspace.</p>
      <div className="cm-hero-actions"><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></Link><a href="#market" className="cm-discover">Follow the signal <ArrowDown size={16} aria-hidden="true"/></a></div>
      <nav className="cm-hero-capabilities" aria-label="Explore the platform"><a href="#market">Market</a><a href="#news">News</a><a href="#forecast">Forecasts</a><a href="#evidence">Validation</a></nav>
      <p className="cm-hero-caption">Built around copper. Designed for perspective.</p>
    </motion.div>
  </article>;
}

function StoryCopy({ progress, beat }: { progress: MotionValue<number>; beat: typeof beats[number] }) {
  const shortNewsWindow = beat.id === 'news';
  const shortForecastWindow = beat.id === 'forecast';
  const start = beat.center === 1 ? .82 : shortNewsWindow ? .39 : shortForecastWindow ? .60 : Math.max(0, beat.center - .18);
  const reveal = beat.center === 1 ? .9 : shortNewsWindow ? .43 : shortForecastWindow ? .64 : Math.max(.02, beat.center - .1);
  const hold = beat.center === 1 ? 1 : shortNewsWindow ? .47 : shortForecastWindow ? .78 : beat.center + .07;
  const end = beat.center === 1 ? 1 : shortNewsWindow ? .535 : shortForecastWindow ? .84 : beat.center + .18;
  const opacity = useTransform(progress, beat.center === 1 ? [start, reveal, 1] : [start, reveal, hold, end], beat.center === 1 ? [0, 1, 1] : [0, 1, 1, 0]);
  const y = useTransform(progress, beat.center === 1 ? [start, reveal, 1] : [start, reveal, hold, end], beat.center === 1 ? [64, 0, 0] : [64, 0, 0, -48]);
  const titleClip = useTransform(progress, [start, reveal], ['inset(0 0 100% 0)', 'inset(0 0 0% 0)']);
  const detailOpacity = useTransform(progress, [start + .025, reveal + .045], [0, 1]);
  const detailY = useTransform(progress, [start + .025, reveal + .045], [26, 0]);
  const blur = useTransform(progress, [start, reveal, end], ['blur(9px)', 'blur(0px)', 'blur(5px)']);
  return <article id={beat.id} className="cm-cinematic-beat" aria-labelledby={`${beat.id}-title`}>
    <motion.div className="cm-cinematic-copy-block" style={{ opacity, y, filter: blur }}>
      <p className="cm-eyebrow"><span>{beat.number}</span> / {beat.label}</p>
      <motion.h2 id={`${beat.id}-title`} style={{ clipPath: titleClip }}>{beat.title}</motion.h2>
      <motion.div style={{ opacity: detailOpacity, y: detailY }}>
        <p className="cm-story-description">{beat.text}</p>
        <ul className="cm-story-tags">{beat.tags.map(tag => <li key={tag}>{tag}</li>)}</ul>
        <Link to={beat.to} className="cm-story-link">{beat.link}<ArrowUpRight size={16} aria-hidden="true"/></Link>
      </motion.div>
    </motion.div>
  </article>;
}

function StickyWorld({ progress, quality }: { progress: MotionValue<number>; quality: ParticleQuality }) {
  const signalOpacity = useTransform(progress, [0, .11, .22], [1, .75, 0]);
  const signalScale = useTransform(progress, [0, .16, .23], [1, 1.03, 1.16]);
  const signalRotate = useTransform(progress, [0, .23], [0, -5]);
  const signalBlur = useTransform(progress, [.12, .23], ['blur(0px)', 'blur(12px)']);
  const signalPath = useTransform(progress, [0, .14], [.66, 1]);
  const signalVisibility = useTransform(progress, value => value <= .23 ? 'visible' : 'hidden');
  return <div className="cm-cinematic-sticky">
    <Atmosphere progress={progress}/>
    <div className="cm-background-typography" aria-hidden="true">
      <BackgroundWord progress={progress} word="SIGNAL" range={[0, .025, .13, .21]}/>
      <BackgroundWord progress={progress} word="MARKET" range={[.14, .22, .34, .43]} reverse/>
      <BackgroundWord progress={progress} word="CONTEXT" range={[.36, .42, .56, .59]}/>
      <BackgroundWord progress={progress} word="FORECAST" range={[.60, .64, .77, .84]} reverse/>
      <BackgroundWord progress={progress} word="EVIDENCE" range={[.78, .86, .99, 1]}/>
    </div>
    <ParticleWorld progress={progress} quality={quality}/>
    <motion.div className="cm-cinematic-symbol" style={{ opacity: signalOpacity, visibility: signalVisibility, scale: signalScale, rotate: signalRotate, filter: signalBlur }} aria-hidden="true"><CopperSignal progress={signalPath}/></motion.div>
    <DashboardComposition progress={progress}/>
    <div className="cm-cinematic-progress" aria-hidden="true"><motion.span style={{ scaleX: progress }}/></div>
  </div>;
}

export function CinematicLanding({ quality = 'high' }: { quality?: ParticleQuality }) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
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
    scene.addEventListener('pointermove', onPointerMove, { passive: true });
    scene.addEventListener('pointerleave', onPointerLeave, { passive: true });
    return () => {
      scene.removeEventListener('pointermove', onPointerMove);
      scene.removeEventListener('pointerleave', onPointerLeave);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);
  return <section ref={ref} id="research" className="cm-cinematic cm-story cm-story--enhanced" aria-label="A connected copper research journey">
    <StickyWorld progress={scrollYProgress} quality={quality}/>
    <div className="cm-cinematic-copy">
      <HeroCopy progress={scrollYProgress}/>
      {beats.map(beat => <StoryCopy key={beat.id} progress={scrollYProgress} beat={beat}/>) }
    </div>
  </section>;
}
