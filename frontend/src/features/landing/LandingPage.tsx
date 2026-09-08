import { lazy, Suspense, useEffect, useRef, useState, type MouseEvent } from 'react';
import { Link } from 'react-router-dom';
import { motion, useMotionValue, useMotionValueEvent, useScroll, useTransform } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import { Brand } from '../../components/ui/Brand';
import { CopperWorld } from './CopperWorld';
import { scenes, sceneIndex } from './narrative';
import { useExperiencePolicy } from './useExperiencePolicy';
import './landing.css';
import { SpatialHeading } from './SpatialHeading';
const EvidenceReading = lazy(() => import('./EvidenceReading').then(m => ({ default: m.EvidenceReading })));
function EvidenceSlot() {
    const ref = useRef<HTMLDivElement>(null);
    const [ready, setReady] = useState(false);
    useEffect(() => { if (!('IntersectionObserver' in window))
        return; const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) {
        setReady(true);
        observer.disconnect();
    } }, { rootMargin: '300px' }); if (ref.current)
        observer.observe(ref.current); return () => observer.disconnect(); }, []);
    return <div ref={ref} className="cw-evidence-slot">{ready ? <Suspense fallback={<p>Checking for published validation…</p>}><EvidenceReading /></Suspense> : <p>Inspect published results and availability in <Link to="/validation">Validation</Link>.</p>}</div>;
}
function StaticWorld({ index }: {
    index: number;
}) { const p = useMotionValue(index / 6); return <div className="cw-static-world"><CopperWorld progress={p} staticIndex={index}/></div>; }
export function LandingPage() {
    const { enhanced, reducedMotion } = useExperiencePolicy();
    const ref = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
    const [active, setActive] = useState(0);
    const last = useRef(0);
    useMotionValueEvent(scrollYProgress, 'change', p => { const next = sceneIndex(p); if (last.current !== next) {
        last.current = next;
        setActive(next);
    } });
    const progress = useTransform(scrollYProgress, p => `${p * 100}%`);
    const cancelNavigation = useRef<() => void>(() => { });
    useEffect(() => () => cancelNavigation.current(), []);
    const navigate = (event: MouseEvent<HTMLAnchorElement>, id: string) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
            return;
        const target = document.getElementById(id);
        if (!target)
            return;
        event.preventDefault();
        cancelNavigation.current();
        history.pushState(history.state, '', `#${id}`);
        let timer = 0;
        const cleanup = () => { window.clearTimeout(timer); window.removeEventListener('scrollend', finish); window.removeEventListener('wheel', cancel); window.removeEventListener('touchstart', cancel); window.removeEventListener('keydown', cancel); };
        const finish = () => { cleanup(); target.focus({ preventScroll: true }); };
        const cancel = () => { cleanup(); window.scrollTo({ top: window.scrollY, behavior: 'instant' }); };
        cancelNavigation.current = cleanup;
        const immediate = reducedMotion;
        target.scrollIntoView({ behavior: immediate ? 'instant' : 'smooth', block: 'start' });
        if (immediate)
            finish();
        else {
            window.addEventListener('scrollend', finish, { once: true });
            window.addEventListener('wheel', cancel, { passive: true, once: true });
            window.addEventListener('touchstart', cancel, { passive: true, once: true });
            window.addEventListener('keydown', cancel, { once: true });
            timer = window.setTimeout(finish, 1200);
        }
    };
    return <div className={`cm-landing cw-landing ${enhanced ? 'cw-enhanced' : 'cw-static'}`}>
 <a className="cm-skip" href="#main-content">Skip to content</a><header className="cm-landing-nav"><Brand /><nav aria-label="Introduction"><a href="#research" onClick={e => navigate(e, 'research')}>The platform</a><Link to="/dashboard" className="cm-button cm-button--secondary">Open dashboard <ArrowUpRight size={16} aria-hidden="true"/></Link></nav></header>
 <main id="main-content" tabIndex={-1}><div ref={ref} className="cw-narrative">
 {enhanced && <div className="cw-stage"><CopperWorld progress={scrollYProgress}/><div className="cw-stage-index" aria-hidden="true">Cu / 29 <span>{scenes[active].label}</span></div></div>}
 {scenes.map((scene, index) => <section id={scene.id} key={scene.id} tabIndex={-1} className={`cw-chapter cw-chapter--${scene.id}`} aria-labelledby={`${scene.id}-title`}>
 <div className="cw-spatial-word" aria-hidden="true">{scene.word}</div><div className="cw-copy"><p className="cm-eyebrow">{scene.label}</p><SpatialHeading progress={scrollYProgress} index={index} title={scene.title} id={`${scene.id}-title`} enhanced={enhanced}/><p className="cw-description">{scene.text}</p>{index === 5 && <EvidenceSlot />}
 {scene.to.startsWith('#') ? <a className="cw-action" href={scene.to} onClick={e => navigate(e, scene.to.slice(1))}>{scene.action}<ArrowUpRight size={18} aria-hidden="true"/></a> : <Link className={index === 0 || index === 6 ? 'cm-button' : 'cw-action'} to={scene.to}>{scene.action}<ArrowUpRight size={18} aria-hidden="true"/></Link>}
 {index === 0 && <nav className="cw-bridges" aria-label="Explore the platform">{[2, 3, 4, 5].map(i => <a key={i} href={`#${scenes[i].id}`} onClick={e => navigate(e, scenes[i].id)}>{['Market', 'News', 'Forecasts', 'Validation'][i - 2]}</a>)}</nav>}{index === 6 && <p className="cw-note">Forecasts are uncertain. Check availability and freshness in the workspace.</p>}</div>{!enhanced && <StaticWorld index={index}/>}</section>)}
 </div>{enhanced && <nav className="cw-chapter-nav" aria-label="Research chapters">{[2, 3, 4, 5].map(i => <a key={i} aria-current={active === i ? 'step' : undefined} href={`#${scenes[i].id}`} onClick={e => navigate(e, scenes[i].id)}>{scenes[i].id}</a>)}<motion.span className="cw-progress" style={{ width: progress }}/></nav>}</main>
 <footer className="cm-landing-footer"><Brand /><p>Market context. Quantitative perspective.</p><Link to="/dashboard">Go straight to the dashboard</Link></footer></div>;
}
