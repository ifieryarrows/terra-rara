import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { motion, useMotionValue, useMotionValueEvent, useScroll, useTransform, type MotionValue } from 'framer-motion';
import { SignalStage, SceneReading } from './SignalStage';
import { chapterAt, sceneVisibility, storyPhases } from './story-timeline';

const chapters = [
  { id: 'market', title: 'A price is only the beginning.', text: 'Explore the market heatmap and the instruments around copper. Build the context before focusing on a forecast.', link: 'Explore market context', to: '/dashboard#market-map' },
  { id: 'news', title: 'Find the story. Then question it.', text: 'Bring source news, sentiment and AI commentary into the same research flow. Separate what happened from how it is interpreted.', link: 'Read the intelligence', to: '/dashboard#news-intelligence' },
  { id: 'forecast', title: 'Keep the uncertainty in view.', text: 'Compare the five-day outlook with historical prices. Inspect the daily path and available intervals without treating a prediction as a promise.', link: 'Examine the forecasts', to: '/dashboard#price-forecast' },
];

function ReadingLayer({ progress, index, active }: { progress: MotionValue<number>; index: number; active: number }) {
  const opacity = useTransform(progress, p => sceneVisibility(p, index));
  // Only the current reading is mounted: no invisible focus targets or lingering disclosures.
  return <motion.div className="cm-reading-layer" style={{ opacity }} hidden={active !== index}>{active === index && <SceneReading index={index}/>}</motion.div>;
}

function EnhancedStory() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] });
  const [active, setActive] = useState(0);
  // A disclosure must remain mounted while keyboard users interact with it.
  const focusWithin = useRef(false);
  const updateChapter = (p: number) => { if (!focusWithin.current) setActive(chapterAt(p)); };
  useMotionValueEvent(scrollYProgress, 'change', updateChapter);
  const readingProgress = useMotionValue(0);
  useMotionValueEvent(scrollYProgress, 'change', p => { if (!focusWithin.current) readingProgress.set(p); });
  return <section ref={ref} className="cm-story cm-story--enhanced cm-signal-story" aria-labelledby="story-heading">
    <div className="cm-story-copy"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map((chapter, index) => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><ChapterCopy index={index}/></article>)}</div>
    <div className="cm-story-stage"><div className="cm-story-stage-inner">
      <nav className="cm-story-nav" aria-label="Research chapters">{storyPhases.map((chapter, index) => <a key={chapter.id} href={`#${chapter.id}`} aria-current={active === index ? 'step' : undefined}><span>0{index + 1}</span>{chapter.label}</a>)}</nav>
      <SignalStage progress={readingProgress} active={active}/>
      <div className="cm-scene-readings" onFocus={() => { focusWithin.current = true; readingProgress.set([.12, .5, .9][active]); }} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) { focusWithin.current = false; readingProgress.set(scrollYProgress.get()); setActive(chapterAt(scrollYProgress.get())); } }}>
        {chapters.map((chapter, index) => <ReadingLayer key={chapter.id} progress={readingProgress} index={index} active={active}/>)}
      </div>
      <div className="cm-story-track" aria-hidden="true"><motion.span style={{ scaleX: scrollYProgress }}/></div>
    </div></div>
  </section>;
}

function ChapterCopy({ index }: { index: number }) {
  const chapter = chapters[index];
  return <><p className="cm-eyebrow">0{index + 1} / {storyPhases[index].label.toUpperCase()}</p><h3>{chapter.title}</h3><p className="cm-story-description">{chapter.text}</p><Link to={chapter.to} className="cm-story-link">{chapter.link}<ArrowUpRight size={16} aria-hidden="true"/></Link></>;
}

function StaticStory() {
  const progress = useMotionValue(1);
  return <section className="cm-story cm-story--static cm-signal-story" aria-labelledby="story-heading"><h2 id="story-heading" className="sr-only">A connected research workflow</h2>{chapters.map((chapter, index) => <article id={chapter.id} className="cm-story-chapter" key={chapter.id}><div><ChapterCopy index={index}/></div><div><SignalStage progress={progress} active={index} staticScene/><SceneReading index={index}/></div></article>)}</section>;
}

export function ResearchStory({ enhanced }: { enhanced: boolean }) {
  return enhanced ? <EnhancedStory/> : <StaticStory/>;
}
