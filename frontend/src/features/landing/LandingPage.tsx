import { lazy, Suspense } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ArrowRight, ChartNoAxesCombined, Newspaper, ScanLine, ShieldCheck } from 'lucide-react';
import { Brand } from '../../components/ui/Brand';
import { Hero } from './Hero';
import { EvidencePreview } from './Previews';
import { ResearchStory } from './ResearchStory';
import { useExperiencePolicy } from './useExperiencePolicy';
import './landing.css';

const CinematicLanding = lazy(() => import('./CinematicLanding').then(module => ({ default: module.CinematicLanding })));

export function LandingPage() {
  const { enhanced, quality } = useExperiencePolicy();
  return <div className={`cm-landing cm-landing--quality-${quality}${enhanced ? ' cm-landing--cinematic' : ''}`}>
    <a className="cm-skip" href="#main-content">Skip to content</a>
    <main id="main-content" tabIndex={-1}>
      {enhanced ? <Suspense fallback={<Hero enhanced={false}/>}><CinematicLanding quality={quality === 'high' ? 'high' : 'balanced'}/></Suspense> : <>
        <Hero enhanced={false}/>
        <section id="research" className="cm-research-intro" aria-labelledby="research-title"><p className="cm-eyebrow">THE CONNECTED VIEW</p><h2 id="research-title">A price is a point.<br/><span>Intelligence is the connection.</span></h2><p>Move from what the market is doing to what may be driving it. Explore the evidence, compare the signals and keep the uncertainty visible.</p><div className="cm-capabilities">{[{label:'Market context',icon:ChartNoAxesCombined},{label:'News intelligence',icon:Newspaper},{label:'Forecast ranges',icon:ScanLine},{label:'Model validation',icon:ShieldCheck}].map(({label,icon:Icon},i)=><div key={label}><span className="cm-capability-index">0{i+1}</span><Icon size={22} strokeWidth={1.4} aria-hidden="true"/><span>{label}</span></div>)}</div></section>
        <ResearchStory enhanced={false}/>
        <section id="evidence" className="cm-evidence" aria-labelledby="evidence-title"><div><p className="cm-eyebrow">04 / THE EVIDENCE</p><h2 id="evidence-title">A signal should<br/>stand up to scrutiny.</h2><p>Keep model validation, forecast horizons and data freshness close to every decision. CopperMind exposes the context needed to question a forecast.</p></div><div><EvidencePreview/><div className="cm-evidence-links"><Link to="/models"><span><small>MODEL INTELLIGENCE</small><strong>Understand the model.</strong><p>Inspect available metrics, checkpoint metadata and quality-gate results.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/validation"><span><small>WALK-FORWARD VALIDATION</small><strong>Examine the evidence.</strong><p>Review available out-of-sample reports and baseline comparisons.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link><Link to="/system"><span><small>SYSTEM STATUS</small><strong>Know how fresh it is.</strong><p>Check snapshot age, model availability and system health.</p></span><ArrowUpRight size={24} aria-hidden="true"/></Link></div></div></section>
      </>}
      <section className="cm-enter" aria-labelledby="enter-title"><p className="cm-eyebrow">YOUR RESEARCH STARTS HERE</p><h2 id="enter-title">From perspective<br/>to your next question.</h2><p>Open the workspace. Explore the market in context.</p><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowRight size={19} aria-hidden="true"/></Link><span className="cm-enter-note">Forecasts are uncertain. Availability and freshness are shown in the workspace.</span></section>
    </main>
    <footer className="cm-landing-footer"><Brand/><p>Market context. Quantitative perspective.</p><Link to="/dashboard">Go straight to the dashboard <ArrowUpRight size={14} aria-hidden="true"/></Link></footer>
  </div>;
}
