import { renderToString } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom';
import { LandingPage } from './features/landing/LandingPage';

export function renderLanding() {
  return renderToString(<StaticRouter location="/"><LandingPage/></StaticRouter>);
}
