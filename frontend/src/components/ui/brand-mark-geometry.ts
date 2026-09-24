import geometry from './brand-mark-geometry.json';

/** Runtime aliases for the shared geometry used by the SVG mark and native cursor. */
export const TERRA_RARA_MARK_VIEWBOX = geometry.viewBox;
export const TERRA_RARA_MARK_STAR_PATH = geometry.starPath;

/** Optical-size correction: a wider star stays legible after the orbit is removed. */
export const TERRA_RARA_MARK_COMPACT_PATH = geometry.compactStarPath;
