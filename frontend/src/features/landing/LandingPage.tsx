import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowUpRight, ArrowRight, ChartNoAxesCombined, Newspaper, ScanLine, ShieldCheck } from 'lucide-react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Brand } from '../../components/ui/Brand';
import { ForecastPreview } from './Previews';
import { ResearchStory } from './ResearchStory';
import { useExperiencePolicy } from './useExperiencePolicy';
import './landing.css';

function EnhancedHeroChart() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] });
  const path = useTransform(scrollYProgress, [0, .5], [.2, 1]);
  const y = useTransform(scrollYProgress, [0, 1], [22, -22]);
  const rotateX = useTransform(scrollYProgress, [0, .5, 1], [7, 0, -2]);
  const scale = useTransform(scrollYProgress, [0, .5, 1], [.96, 1, 1]);
  return <div ref={ref} className="cm-hero-chart-wrap"><motion.div style={{ y, rotateX, scale, transformPerspective: 1200 }}><ForecastPreview progress={path} hero/></motion.div></div>;
}

export function LandingPage() {
  const { enhanced } = useExperiencePolicy();
  return <div className="cm-landing">
    <a className="cm-skip" href="#main-content">Skip to content</a>
    <header className="cm-landing-nav"><div className="cm-landing-nav-inner"><Brand/><nav aria-label="Introduction"><a href="#research" className="cm-nav-text">The platform</a><Link to="/validation" className="cm-nav-text">Validation</Link><Link to="/dashboard" className="cm-button cm-button--secondary">Open dashboard <ArrowUpRight size={16} aria-hidden="true"/></Link></nav></div></header>
    <main id="main-content" tabIndex={-1}>
      <section className="cm-hero" aria-labelledby="hero-title">
        <div className="cm-hero-heading"><p className="cm-eyebrow">COPPER INTELLIGENCE / TERRA RARA</p><h1 id="hero-title">Read the market.<br/><span>See the structure.</span></h1><div className="cm-hero-bottom"><p>Copper, context and quantitative thinking.<br/>One workspace for a more informed view.</p><div className="cm-hero-actions"><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></Link><a href="#research" className="cm-discover">Discover the platform <ArrowDown size={16} aria-hidden="true"/></a></div></div></div>
        {enhanced ? <EnhancedHeroChart/> : <div className="cm-hero-chart-wrap"><ForecastPreview hero/></div>}
        <div className="cm-hero-index"><span>MARKET INTELLIGENCE, IN CONTEXT.</span><span>01 — 04</span></div>
      </section>
      <section id="research" className="cm-research-intro" aria-labelledby="research-title"><p className="cm-eyebrow">THE CONNECTED VIEW</p><h2 id="research-title">A price is a point.<br/><span>Intelligence is the connection.</span></h2><p>Move from what the market is doing to what may be driving it. Explore the evidence, compare the signals and keep the uncertainty visible.</p><div className="cm-capabilities">{[{label:'Market context',icon:ChartNoAxesCombined},{label:'News intelligence',icon:Newspaper},{label:'Forecast ranges',icon:ScanLine},{label:'Model validation',icon:ShieldCheck}].map(({label,icon:Icon},i)=><div key={label}><span className="cm-capability-index">0{i+1}</span><Icon size={22} strokeWidth={1.4} aria-hidden="true"/><span>{label}</span></div>)}</div></section>
      <ResearchStory enhanced={enhanced}/>
      <section className="cm-evidence" aria-labelledby="evidence-title"><div><p className="cm-eyebrow">04 / THE EVIDENCE</p><h2 id="evidence-title">A signal should<br/>stand up to scrutiny.</h2><p>Keep model validation, forecast horizons and data freshness close to every decision. CopperMind exposes the context needed to question a forecast.</p></div><div className="cm-evidence-links"><Link to="/models"><span><small>MODEL INTELLIGENCE</small><strong>Understand the model.</strong><p>Inspect available metrics, checkpoint metadata and quality-gate results.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/validation"><span><small>WALK-FORWARD VALIDATION</small><strong>Examine the evidence.</strong><p>Review available out-of-sample reports and baseline comparisons.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/system"><span><small>SYSTEM STATUS</small><strong>Know how fresh it is.</strong><p>Check snapshot age, model availability and system health.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link></div></section>
      <section className="cm-enter" aria-labelledby="enter-title"><p className="cm-eyebrow">YOUR RESEARCH STARTS HERE</p><h2 id="enter-title">From perspective<br/>to your next question.</h2><p>Open the workspace. Explore the market in context.</p><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowRight size={19} aria-hidden="true"/></Link><span className="cm-enter-note">Forecasts are uncertain. Availability and freshness are shown in the workspace.</span></section>
    </main>
    <footer className="cm-landing-footer"><Brand/><p>Market context. Quantitative perspective.</p><Link to="/dashboard">Go straight to the dashboard <ArrowUpRight size={14} aria-hidden="true"/></Link></footer>
  </div>;
}
