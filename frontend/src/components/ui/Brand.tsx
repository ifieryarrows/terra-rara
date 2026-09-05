import { Link } from 'react-router-dom';

export function Brand() {
  return (
    <Link to="/" className="cm-brand" aria-label="CopperMind Terra Rara home">
      <span className="cm-brand-mark" aria-hidden="true">Cu</span>
      <span><span className="cm-brand-name"><strong>Copper</strong>Mind</span><span className="cm-brand-sub">TERRA RARA</span></span>
    </Link>
  );
}
