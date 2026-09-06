import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { motion, useScroll, useTransform, type MotionValue } from 'framer-motion';
import { MarketPreview, NewsPreview, ForecastPreview } from './Previews';

const chapters = [
  { id: 'market', number: '01', label: 'THE MARKET', title: 'Start with the bigger picture.', text: 'Explore the market heatmap, related instruments and sector context. Find the relationships around copper before focusing on a single forecast.', tags: ['Market heatmap', 'Related instruments'], link: 'Explore market context', to: '/dashboard#market-map', Preview: MarketPreview },
  { id: 'news', number: '02', label: 'THE CONTEXT', title: 'Understand the forces behind the price.', text: 'Bring source news, sentiment and AI commentary into the same research flow. Inspect the reasoning, then return to the market with more context.', tags: ['News intelligence', 'Sentiment & commentary'], link: 'Read the intelligence', to: '/dashboard#news-intelligence', Preview: NewsPreview },
  { id: 'forecast', number: '03', label: 'THE POSSIBILITIES', title: 'See the range. Keep the uncertainty.', text: 'Study quantitative forecasts alongside historical prices. Keep the primary weekly view distinct from T+1 diagnostics, and examine the available uncertainty intervals.', tags: ['Deep-learning forecasts', 'Price & risk context'], link: 'Examine the forecasts', to: '/dashboard#price-forecast', Preview: ForecastPreview },
];

function StoryLayer({ progress, index }: { progress: MotionValue<number>; index: number }) {
  // Three viewport-height chapters: reading centers align at 0, .5 and 1.
  const stops = index === 0 ? [0, .18, .32, 1] : index === 1 ? [.18, .32, .68, .82] : [0, .68, .82, 1];
  const opacity = useTransform(progress, stops, index === 0 ? [1, 1, 0, 0] : index === 1 ? [0, 1, 1, 0] : [0, 0, 1, 1]);
  const local = useTransform(progress, index === 0 ? [0, .17] : index === 1 ? [.22, .5] : [.72, .97], [0, 1]);
  const y = useTransform(local, [0, 1], [18, 0]);
  const Preview = chapters[index].Preview;
  return <motion.div className="cm-story-layer" style={{ opacity, y }}><Preview progress={local}/></motion.div>;
}

function EnhancedStory() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
  return <section ref={ref} className="cm-story cm-story--enhanced" aria-labelledby="story-heading">
    <div className="cm-story-copy"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map(chapter => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><ChapterCopy chapter={chapter}/></article>)}</div>
    <div className="cm-story-stage"><div className="cm-story-stage-inner"><nav className="cm-story-nav" aria-label="Research chapters">{chapters.map(chapter => <a key={chapter.id} href={`#${chapter.id}`}><span>{chapter.number}</span>{chapter.label.replace('THE ', '')}</a>)}</nav><div className="cm-story-visuals" aria-hidden="true">{chapters.map((chapter, i) => <StoryLayer progress={scrollYProgress} index={i} key={chapter.id}/>)}</div><div className="cm-story-track" aria-hidden="true"><motion.span style={{ scaleX: scrollYProgress }}/></div><span className="cm-story-footnote">Illustrative previews / one connected research workflow</span></div></div>
  </section>;
}

function ChapterCopy({ chapter }: { chapter: typeof chapters[number] }) {
  return <><p className="cm-eyebrow"><span>{chapter.number}</span> / {chapter.label}</p><h3>{chapter.title}</h3><p className="cm-story-description">{chapter.text}</p><ul className="cm-story-tags">{chapter.tags.map(tag => <li key={tag}>{tag}</li>)}</ul><Link to={chapter.to} className="cm-story-link">{chapter.link}<ArrowUpRight size={16} aria-hidden="true"/></Link></>;
}

export function ResearchStory({ enhanced }: { enhanced: boolean }) {
  if (enhanced) return <EnhancedStory/>;
  return <section className="cm-story cm-story--static" aria-labelledby="story-heading"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map(chapter => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><div><ChapterCopy chapter={chapter}/></div><chapter.Preview/></article>)}</section>;
}
