import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { CopperSignal } from './CopperSignal';
import { BrandMark } from '../../components/ui/BrandMark';
import { EnterWorkspaceLink } from '../../components/ui/LogoTransition';

function EnhancedSignal() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] });
  const y = useTransform(scrollYProgress, [0, 1], [0, 40]);
  const rotate = useTransform(scrollYProgress, [0, 1], [0, -3]);
  const path = useTransform(scrollYProgress, [0, .55], [.7, 1]);
  return <div ref={ref} className="cm-hero-visual" data-cm-route-reveal="surface"><motion.div style={{ y, rotate }}><CopperSignal progress={path}/></motion.div></div>;
}

export function Hero({ enhanced }: { enhanced: boolean }) {
  return <section className="cm-hero" aria-labelledby="hero-title">
    <div className="cm-hero-grid">
      <div className="cm-hero-heading">
        <p className="cm-eyebrow cm-brand-eyebrow" data-cm-route-reveal="copy"><BrandMark size={18} variant="small"/>COPPER INTELLIGENCE / TERRA RARA</p>
        <h1 id="hero-title" data-cm-route-reveal="copy">Read the market.<br/><span>See the structure.</span></h1>
        <p className="cm-hero-description" data-cm-route-reveal="copy">Behind every copper price, a bigger picture.</p>
        <p className="cm-hero-detail" data-cm-route-reveal="copy">Market, news, forecasts and evidence in one workspace.</p>
        <div className="cm-hero-actions" data-cm-route-reveal="surface"><EnterWorkspaceLink className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></EnterWorkspaceLink><a href="#research" className="cm-discover">Explore the connections <ArrowDown size={16} aria-hidden="true"/></a></div>
        <nav className="cm-hero-capabilities" aria-label="Explore the platform" data-cm-route-reveal="surface"><a href="#market">Market</a><a href="#news">News</a><a href="#forecast">Forecasts</a><Link to="/validation">Validation</Link></nav>
        <p className="cm-hero-caption" data-cm-route-reveal="copy">Built around copper. Designed for perspective.</p>
      </div>
      {enhanced ? <EnhancedSignal/> : <div className="cm-hero-visual" data-cm-route-reveal="surface"><CopperSignal/></div>}
    </div>
    <div className="cm-hero-index" data-cm-route-reveal="surface"><span>ONE METAL. A CONNECTED MARKET.</span><a href="#research">SCROLL TO CONNECT THE DOTS <ArrowDown size={14} aria-hidden="true"/></a></div>
  </section>;
}
