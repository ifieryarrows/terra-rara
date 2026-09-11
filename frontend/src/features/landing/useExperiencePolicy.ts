import { useEffect, useState } from 'react';
import { useReducedMotion } from 'framer-motion';
export function useExperiencePolicy() {
    const reduce = useReducedMotion();
    const [desktop, setDesktop] = useState(false);
    useEffect(() => { const media = window.matchMedia('(min-width: 75em) and (min-height: 50em)'); const hints = navigator as Navigator & {
        connection?: EventTarget & {
            saveData?: boolean;
        };
    }; const update = () => setDesktop(media.matches && !hints.connection?.saveData); update(); media.addEventListener('change', update); hints.connection?.addEventListener('change', update); return () => { media.removeEventListener('change', update); hints.connection?.removeEventListener('change', update); }; }, []);
    return { enhanced: desktop && reduce === false, reducedMotion: reduce !== false };
}
