import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { CopperSignal } from './CopperSignal';

function EnhancedSignal() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] });
  const y = useTransform(scrollYProgress, [0, 1], [0, 64]);
  const rotate = useTransform(scrollYProgress, [0, 1], [0, -5]);
  const path = useTransform(scrollYProgress, [0, .55], [.25, 1]);
  return <div ref={ref} className="cm-hero-visual"><motion.div style={{ y, rotate }}><CopperSignal progress={path}/></motion.div></div>;
}

export function Hero({ enhanced }: { enhanced: boolean }) {
  return <section className="cm-hero" aria-labelledby="hero-title">
    <div className="cm-hero-grid">
      <div className="cm-hero-heading">
        <p className="cm-eyebrow"><span className="cm-eyebrow-line"/>COPPER INTELLIGENCE / TERRA RARA</p>
        <h1 id="hero-title">Read the market.<br/><span>Follow the signal.</span></h1>
        <p className="cm-hero-description">Behind every copper price, a bigger picture.</p>
        <p className="cm-hero-detail">Connect market moves, news intelligence, forecast ranges and model validation in one research workspace.</p>
        <div className="cm-hero-actions"><Link to="/dashboard" className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></Link><a href="#research" className="cm-discover">Explore the connections <ArrowDown size={16} aria-hidden="true"/></a></div>
        <nav className="cm-hero-capabilities" aria-label="Explore the platform"><a href="#market">Market</a><a href="#news">News</a><a href="#forecast">Forecasts</a><a href="#evidence">Validation</a></nav>
      </div>
      {enhanced ? <EnhancedSignal/> : <div className="cm-hero-visual"><CopperSignal/></div>}
    </div>
    <div className="cm-hero-index"><span>ONE METAL. A CONNECTED MARKET.</span><a href="#research">FOLLOW THE COPPER SIGNAL <ArrowDown size={14} aria-hidden="true"/></a></div>
  </section>;
}
