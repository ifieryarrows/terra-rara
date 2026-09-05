import { useRef, type ReactNode } from 'react';
import { motion, useScroll, useTransform, type MotionValue } from 'framer-motion';
import { MarketPreview, NewsPreview, ForecastPreview } from './Previews';

const chapters = [
  { id: 'market', number: '01', label: 'THE MARKET', title: 'Start with the bigger picture.', text: 'Explore the market heatmap, related instruments and sector context. Find the relationships around copper before focusing on a single forecast.', tags: ['Market heatmap', 'Related instruments'], preview: <MarketPreview /> },
  { id: 'news', number: '02', label: 'THE CONTEXT', title: 'Understand the forces behind the price.', text: 'Bring source news, sentiment and AI commentary into the same research flow. Inspect the reasoning, then return to the market with more context.', tags: ['News intelligence', 'Sentiment & commentary'], preview: <NewsPreview /> },
  { id: 'forecast', number: '03', label: 'THE POSSIBILITIES', title: 'See the range. Keep the uncertainty.', text: 'Study quantitative forecasts alongside historical prices. Keep the primary weekly view distinct from T+1 diagnostics, and examine the available uncertainty intervals.', tags: ['Deep-learning forecasts', 'Price & risk context'], preview: <ForecastPreview /> },
];

function StoryLayer({ progress, index, children }: { progress: MotionValue<number>; index: number; children: ReactNode }) {
  const stops = index === 0 ? [0, .26, .35, 1] : index === 1 ? [.26, .35, .60, .70] : [0, .60, .70, 1];
  const opacity = useTransform(progress, stops, index === 0 ? [1, 1, 0, 0] : index === 1 ? [0, 1, 1, 0] : [0, 0, 1, 1]);
  const y = useTransform(progress, [Math.max(0, index * .33 - .08), index * .33 + .05], [index ? 32 : 0, 0]);
  return <motion.div className="cm-story-layer" style={{ opacity, y }}>{children}</motion.div>;
}

function EnhancedStory() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
  return <section ref={ref} className="cm-story cm-story--enhanced" aria-labelledby="story-heading">
    <div className="cm-story-copy"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map(chapter => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><ChapterCopy chapter={chapter}/></article>)}</div>
    <div className="cm-story-stage" aria-hidden="true"><div className="cm-story-stage-inner">{chapters.map((chapter, i) => <StoryLayer progress={scrollYProgress} index={i} key={chapter.id}>{chapter.preview}</StoryLayer>)}<div className="cm-story-track"><motion.span style={{ scaleX: scrollYProgress }}/></div><span className="cm-story-footnote">A connected research workflow / scroll to explore</span></div></div>
  </section>;
}

function ChapterCopy({ chapter }: { chapter: typeof chapters[number] }) {
  return <><p className="cm-eyebrow"><span>{chapter.number}</span> / {chapter.label}</p><h3>{chapter.title}</h3><p className="cm-story-description">{chapter.text}</p><ul className="cm-story-tags">{chapter.tags.map(tag => <li key={tag}>{tag}</li>)}</ul></>;
}

export function ResearchStory({ enhanced }: { enhanced: boolean }) {
  if (enhanced) return <EnhancedStory/>;
  return <section className="cm-story cm-story--static" aria-labelledby="story-heading"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map(chapter => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><div><ChapterCopy chapter={chapter}/></div>{chapter.preview}</article>)}</section>;
}
