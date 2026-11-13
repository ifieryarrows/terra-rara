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
  const y = useTransform(scrollYProgress, [0, 1], [0, 32]);
  const path = useTransform(scrollYProgress, [0, .55], [.28, 1]);
  return <div ref={ref} className="cm-hero-visual" data-cm-route-reveal="surface"><motion.div style={{ y }}><CopperSignal progress={path}/></motion.div></div>;
}

export function Hero({ enhanced }: { enhanced: boolean }) {
  return <section className="cm-hero" aria-labelledby="hero-title">
    <div className="cm-hero-grid">
      <div className="cm-hero-heading">
        <p className="cm-eyebrow cm-brand-eyebrow" data-cm-route-reveal="copy"><BrandMark size={18} variant="small"/>COPPER INTELLIGENCE / TERRA RARA</p>
        <h1 id="hero-title" data-cm-route-reveal="copy">Read the market.<br/><span>See the structure.</span></h1>
        <p className="cm-hero-description" data-cm-route-reveal="copy">Behind every copper price, a bigger picture. Market context, news intelligence, and 5-day quantile forecasts in one workspace.</p>
        <div className="cm-hero-actions" data-cm-route-reveal="surface"><EnterWorkspaceLink className="cm-button">Enter CopperMind <ArrowUpRight size={17} aria-hidden="true"/></EnterWorkspaceLink><a href="#research" className="cm-discover">Follow the signal <ArrowDown size={16} aria-hidden="true"/></a></div>
        <nav className="cm-hero-capabilities" aria-label="Explore the platform" data-cm-route-reveal="surface"><a href="#market">Market</a><a href="#news">News</a><a href="#forecast">Forecasts</a><Link to="/validation">Validation</Link></nav>
      </div>
      {enhanced ? <EnhancedSignal/> : <div className="cm-hero-visual" data-cm-route-reveal="surface"><CopperSignal/></div>}
    </div>
  </section>;
}
