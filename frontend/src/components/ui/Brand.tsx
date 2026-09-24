import { Link } from 'react-router-dom';
import { BrandMark } from './BrandMark';

export function Brand() {
  return (
    <Link to="/" className="cm-brand" aria-label="CopperMind Terra Rara home">
      <BrandMark size={38} variant="on-dark" className="cm-brand-mark"/>
      <span className="cm-brand-wordmark" aria-hidden="true">
        <span className="cm-brand-name">COPPERMIND</span>
        <span className="cm-brand-sub">TERRA RARA</span>
      </span>
    </Link>
  );
}
