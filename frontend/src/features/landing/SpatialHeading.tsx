import { motion, useTransform, type MotionValue } from 'framer-motion';
import { clamp } from './narrative';
export function SpatialHeading({ progress, index, title, id, enhanced }: {
    progress: MotionValue<number>;
    index: number;
    title: string;
    id: string;
    enhanced: boolean;
}) {
    const reveal = useTransform(progress, p => enhanced && index > 0 ? clamp((p * 6 - index + .8) / .5) : 1);
    const clipPath = useTransform(reveal, q => `inset(0 ${(1 - q) * 100}% 0 0)`);
    const x = useTransform(reveal, q => `${(1 - q) * -6}%`);
    const Heading = index === 0 ? 'h1' : 'h2';
    return <Heading id={id}><span className="sr-only">{title}</span><motion.span aria-hidden="true" className="cw-heading-reveal" style={{ clipPath, x }}>{title}</motion.span></Heading>;
}
