import { useEffect, useRef } from 'react';
import type { MotionValue } from 'framer-motion';
import { getCinematicParticleProgress } from './cinematic-entry';

export type ParticleQuality = 'high' | 'balanced';

const MAX_PARTICLES = 640;
const PARTICLE_RESPONSE_MS = 220;
const SCROLL_IMPULSE_DAMPING = 0.965;
const SCROLL_IMPULSE_GAIN = 0.48;
const QUALITY = {
  high: { count: MAX_PARTICLES, dpr: 1.5, pointSize: 8.8, canvasAlpha: .94, largeRadius: 1.55, radius: 0.88 },
  balanced: { count: 420, dpr: 1, pointSize: 8.2, canvasAlpha: .90, largeRadius: 1.42, radius: 0.80 },
} as const;

function noise(index: number, salt: number) {
  const value = Math.sin((index + 1) * (12.9898 + salt * 17.17)) * 43758.5453;
  return value - Math.floor(value);
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function quintic(edge0: number, edge1: number, value: number) {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function makeDrift(count: number) {
  const drift = new Float32Array(count * 4);
  for (let index = 0; index < count; index += 1) {
    const offset = index * 4;
    drift[offset] = noise(index, 13) * Math.PI * 2;
    drift[offset + 1] = noise(index, 14) * Math.PI * 2;
    const direction = noise(index, 17) > .5 ? 1 : -1;
    drift[offset + 2] = (0.00072 + noise(index, 15) * 0.00086) * direction;
    drift[offset + 3] = 0.0011 + noise(index, 16) * 0.0024;
  }
  return drift;
}

function makeShape(pointFor: (index: number) => [number, number]) {
  const points = new Float32Array(MAX_PARTICLES * 2);
  for (let index = 0; index < MAX_PARTICLES; index += 1) {
    const [x, y] = pointFor(index);
    points[index * 2] = x;
    points[index * 2 + 1] = y;
  }
  return points;
}

// 1. Calm Hero Opening (p = 0.00 -> 0.16): sparse deep-space stardust framing the 2D Venus emblem
const heroCalmField = makeShape(index => {
  const angle = noise(index, 50) * Math.PI * 2;
  const r = 0.16 + Math.pow(noise(index, 51), 0.72) * 0.44;
  return [
    0.54 + Math.cos(angle) * r * 0.92,
    0.50 + Math.sin(angle) * r * 0.72,
  ];
});

const dispersedField = makeShape(index => {
  const right = noise(index, 2) > .48;
  return [right ? .79 + noise(index, 3) * .19 : .015 + noise(index, 4) * .21, .08 + noise(index, 5) * .84];
});

const marketCells = [
  { x: .5, y: .21, w: .21, h: .43 },
  { x: .72, y: .21, w: .24, h: .2 },
  { x: .72, y: .43, w: .115, h: .21 },
  { x: .845, y: .43, w: .115, h: .21 },
  { x: .5, y: .66, w: .27, h: .14 },
  { x: .78, y: .66, w: .18, h: .14 },
];
const marketField = makeShape(index => {
  const cell = marketCells[index % marketCells.length];
  const edge = noise(index, 6);
  if (edge < .5) return [cell.x + noise(index, 7) * cell.w, cell.y + (edge < .25 ? 0 : cell.h)];
  return [cell.x + (edge < .75 ? 0 : cell.w), cell.y + noise(index, 8) * cell.h];
});

const networkNodes = [
  [.52, .28], [.5, .7], [.66, .18], [.64, .78], [.79, .34], [.8, .67], [.93, .49],
] as const;
const intelligenceNetwork = makeShape(index => {
  const from = networkNodes[index % (networkNodes.length - 1)];
  const to = index % 3 === 0 ? networkNodes[networkNodes.length - 1] : networkNodes[(index + 2) % networkNodes.length];
  const amount = noise(index, 9);
  const bend = Math.sin(amount * Math.PI) * (noise(index, 10) - .5) * .08;
  return [from[0] + (to[0] - from[0]) * amount, from[1] + (to[1] - from[1]) * amount + bend];
});

const forecastPath = makeShape(index => {
  const amount = index / (MAX_PARTICLES - 1);
  const uncertainty = Math.max(0, amount - .62) * (noise(index, 11) - .5) * .34;
  return [.44 + amount * .53, .66 - amount * .32 + Math.sin(amount * Math.PI * 4.2) * .04 + uncertainty];
});

const venusGlobe2D = makeShape(index => {
  const centerX = 0.34;
  const centerY = 0.44;
  const sphereCutoff = Math.floor(MAX_PARTICLES * 0.72);
  if (index < sphereCutoff) {
    const u = (index + 0.5) / sphereCutoff;
    const ySphere = 1 - 2 * u;
    const rRing = Math.sqrt(Math.max(0, 1 - ySphere * ySphere));
    const phi = index * 2.399963229728653;
    const xSphere = Math.cos(phi) * rRing;
    return [
      centerX + xSphere * 0.145,
      centerY + ySphere * 0.195,
    ];
  }
  const ringT = (index - sphereCutoff) / Math.max(1, MAX_PARTICLES - sphereCutoff);
  const angle = ringT * Math.PI * 2 + (noise(index, 18) - 0.5) * 0.08;
  const rx = 0.25 + (noise(index, 19) - 0.5) * 0.025;
  const ry = 0.068 + (noise(index, 22) - 0.5) * 0.012;
  const tilt = -0.28;
  const ux = Math.cos(angle) * rx;
  const uy = Math.sin(angle) * ry;
  return [
    centerX + ux * Math.cos(tilt) - uy * Math.sin(tilt),
    centerY + ux * Math.sin(tilt) + uy * Math.cos(tilt),
  ];
});

const milkyWay2D = makeShape(index => {
  const angle = index * 2.399963229728653;
  const r = 0.04 + Math.pow(noise(index, 20), 0.7) * 0.36;
  return [0.50 + Math.cos(angle) * r, 0.48 + Math.sin(angle) * r * 0.72];
});

const sourceKeyframes = [
  { at: 0, points: heroCalmField },
  { at: .16, points: heroCalmField },
  { at: .27, points: dispersedField },
  { at: .39, points: marketField },
  { at: .56, points: intelligenceNetwork },
  { at: .70, points: forecastPath },
  { at: .80, points: venusGlobe2D },
  { at: .86, points: milkyWay2D },
  { at: 1, points: milkyWay2D },
];

function sampleShape(source: Float32Array, count: number) {
  if (count === MAX_PARTICLES) return source;
  const sampled = new Float32Array(count * 2);
  const ratio = (MAX_PARTICLES - 1) / (count - 1);
  for (let index = 0; index < count; index += 1) {
    const sourceIndex = Math.round(index * ratio);
    const baseOffset = sourceIndex * 2;
    const jitter = count > MAX_PARTICLES ? (noise(index, 33) - 0.5) * 0.0035 : 0;
    const jitterY = count > MAX_PARTICLES ? (noise(index, 34) - 0.5) * 0.0035 : 0;
    sampled[index * 2] = source[baseOffset] + jitter;
    sampled[index * 2 + 1] = source[baseOffset + 1] + jitterY;
  }
  return sampled;
}

const keyframesCache = new Map<number, { at: number; points: Float32Array }[]>();
function keyframesFor(count: number) {
  let cached = keyframesCache.get(count);
  if (!cached) {
    cached = sourceKeyframes.map(keyframe => ({
      at: keyframe.at,
      points: sampleShape(keyframe.points, count),
    }));
    keyframesCache.set(count, cached);
  }
  return cached;
}

function segmentFor(value: number) {
  let index = sourceKeyframes.findIndex(keyframe => keyframe.at >= value);
  if (index <= 0) index = 1;
  return index;
}

const MW_BAR_ANGLE = 0.58; // ~33 deg barred-spiral bar orientation
const MW_BAR_HALF_LEN = 0.23;

/**
 * Synthesizes 3D celestial states per particle:
 * 1. `galaxy`: Barred Spiral Milky Way Galaxy (warm amber-copper central bar, cool blue-silver arms,
 *    delicate rose-magenta H-II nebula knots, and dark inner interstellar dust lanes — ordered by index
 *    strata for clean depth layering in WebGL2).
 * 2. `heroField`: Calm, spacious deep-space starfield for the Hero opening (no chaotic scatter behind 2D Venus logo).
 * 3. `milkyway`: Diagonal Milky Way Star Cloud + Dark Molecular Dust Rift (Stage 01 -> 02).
 * 4. `venus`: 3D Planet Venus Sphere + orbital ring (Stage 03 approach -> center flyby -> smooth dissolution).
 */
function makeCelestialStates3D(count: number) {
  let seed = 0x54455252;
  const rand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const gaussian = () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const galaxy = new Float32Array(count * 3);
  const heroField = new Float32Array(count * 3);
  const milkyway = new Float32Array(count * 3);
  const venus = new Float32Array(count * 3);
  const warp = new Float32Array(count * 3);
  const meta = new Float32Array(count * 4); // x: size, y: temp, z: dust, w: role (0: warp, 1: story/outerDust, 2: venusSphere, 4: mwPersist)
  const nebula = new Float32Array(count * 2); // x: roseMagentaKnotWeight [0..1], y: heroStarAlpha [0..1]

  const storyCount = Math.floor((count + 4) / 5);
  const nonStoryTotal = count - storyCount;
  const venusSphereCount = Math.floor(nonStoryTotal * 0.68);
  const mwPersistCount = Math.floor(nonStoryTotal * 0.16);
  let nonStoryIdx = 0;

  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const cosBar = Math.cos(MW_BAR_ANGLE);
  const sinBar = Math.sin(MW_BAR_ANGLE);

  // Pre-seed 14 delicate rose-magenta H-II star-forming nebula knots sparsely along spiral arm ridges
  const armOffsets = [0, Math.PI, Math.PI * 0.46, Math.PI * 1.46];
  const knots: Array<{ x: number; z: number; spread: number }> = [];
  for (let k = 0; k < 14; k += 1) {
    const armIdx = k % 4;
    const s = 0.20 + (k / 14) * 0.68;
    const radius = MW_BAR_HALF_LEN * 0.96 + s * 0.88;
    const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
    const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + (rand() - 0.5) * 0.06;
    knots.push({
      x: Math.cos(theta) * radius,
      z: Math.sin(theta) * radius,
      spread: 0.012 + rand() * 0.016,
    });
  }

  for (let i = 0; i < count; i += 1) {
    const gOffset = i * 3;
    const mOffset = i * 4;
    const nOffset = i * 2;
    const frac = i / count;

    // --- A. Barred Spiral Milky Way Galaxy (layered by index fraction for crisp depth) ---
    if (frac < 0.08) {
      // Cool blue halo + subtle distant companion galaxy cluster at upper-left
      if (frac < 0.014) {
        const cx = -0.94 + gaussian() * 0.052;
        const cz = -0.82 + gaussian() * 0.028;
        const cy = 0.04 + gaussian() * 0.018;
        galaxy[gOffset] = cx * cosBar - cz * sinBar;
        galaxy[gOffset + 1] = cy;
        galaxy[gOffset + 2] = cx * sinBar + cz * cosBar;
        meta[mOffset] = 0.72 + 0.45 * rand();
        meta[mOffset + 1] = 0.72 + 0.22 * rand();
        meta[mOffset + 2] = 0.04;
      } else {
        const rHalo = 0.35 + 1.38 * Math.pow(rand(), 0.7);
        const cosT = rand() * 2.0 - 1.0;
        const sinT = Math.sqrt(Math.max(0.0, 1.0 - cosT * cosT));
        const phiH = rand() * Math.PI * 2.0;
        galaxy[gOffset] = rHalo * sinT * Math.cos(phiH);
        galaxy[gOffset + 1] = rHalo * cosT * 0.32;
        galaxy[gOffset + 2] = rHalo * sinT * Math.sin(phiH);
        meta[mOffset] = 0.34 + 0.38 * rand();
        meta[mOffset + 1] = 0.06 + 0.24 * rand();
        meta[mOffset + 2] = 0.04;
      }
      nebula[nOffset] = 0.0;
    } else if (frac < 0.42) {
      // Broad outer spiral disc & feathered secondary spurs (#8eadde cool galactic silver-blue wash)
      const armIdx = i % 4;
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.88);
      const radius = MW_BAR_HALF_LEN * 0.78 + s * 1.12;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      const spread = gaussian() * (0.12 + 0.14 * s) * (isMajor ? 1.05 : 1.45);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      galaxy[gOffset] = Math.cos(theta) * radius;
      galaxy[gOffset + 1] = gaussian() * (0.018 + 0.025 * s);
      galaxy[gOffset + 2] = Math.sin(theta) * radius;

      const rNorm = Math.min(1.0, radius / 1.35);
      const innerWarm = Math.max(0, 1.0 - Math.max(0, rNorm - 0.16) / 0.22);
      meta[mOffset] = 0.56 + 0.48 * rand();
      meta[mOffset + 1] = Math.min(0.46, 0.05 + 0.23 * rand() + innerWarm * 0.22);
      meta[mOffset + 2] = 0.06 + 0.10 * rand();
      nebula[nOffset] = 0.0;
    } else if (frac < 0.53) {
      // Dark Interstellar Dust Lanes carving the inner concave edges of spiral arms & bar shoulders
      const armIdx = i % 4;
      const s = Math.pow(rand(), 0.92);
      const radius = MW_BAR_HALF_LEN * 0.66 + s * 0.98;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      // Inner concave side sits slightly behind/inside the bright arm ridge
      const dustOffset = -0.13 - Math.abs(gaussian()) * 0.09;
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + dustOffset;
      galaxy[gOffset] = Math.cos(theta) * radius;
      galaxy[gOffset + 1] = gaussian() * 0.012;
      galaxy[gOffset + 2] = Math.sin(theta) * radius;

      meta[mOffset] = 1.05 + 0.65 * rand();
      meta[mOffset + 1] = 0.14;
      meta[mOffset + 2] = 0.78 + 0.20 * rand();
      nebula[nOffset] = 0.0;
    } else if (frac < 0.855) {
      // Luminous Primary Bar-Tip Spiral Arms & crisp #8eadde silver-blue OB star ridges
      const armIdx = (i % 10 < 7) ? (i % 2) : (2 + (i % 2));
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.85);
      const radius = MW_BAR_HALF_LEN * 0.86 + s * 1.04;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      const spread = gaussian() * (0.042 + 0.068 * s) * (isMajor ? 1.0 : 1.28);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      galaxy[gOffset] = Math.cos(theta) * radius;
      galaxy[gOffset + 1] = gaussian() * (0.012 + 0.018 * s);
      galaxy[gOffset + 2] = Math.sin(theta) * radius;

      const barRootWarm = Math.max(0, 1.0 - Math.max(0, radius - MW_BAR_HALF_LEN * 0.85) / 0.22);
      meta[mOffset] = (isMajor ? 0.66 : 0.54) * (0.74 + 0.52 * rand());
      meta[mOffset + 1] = Math.min(0.68, 0.04 + 0.28 * rand() + barRootWarm * 0.36);
      meta[mOffset + 2] = 0.0;
      nebula[nOffset] = 0.0;
    } else if (frac < 0.89) {
      // Sparse, delicate Rose-Magenta (#db8296) H-II Star-Forming Nebula Knots (subtle arm accents only)
      const knot = knots[i % knots.length];
      galaxy[gOffset] = knot.x + gaussian() * knot.spread;
      galaxy[gOffset + 1] = gaussian() * 0.010;
      galaxy[gOffset + 2] = knot.z + gaussian() * knot.spread;

      meta[mOffset] = 0.58 + 0.42 * rand();
      meta[mOffset + 1] = 0.24 + 0.16 * rand();
      meta[mOffset + 2] = 0.0;
      nebula[nOffset] = 0.48 + 0.32 * rand();
    } else {
      // Luminous Warm Amber-Copper Central Bar & Galactic Nucleus (#dda174 -> #fff3e2)
      const isCore = frac >= 0.955;
      if (isCore) {
        const r = Math.sqrt(-Math.log(Math.max(1e-5, 1.0 - 0.99 * rand()))) * 0.068;
        const theta = rand() * Math.PI * 2.0;
        const bx = r * Math.cos(theta) * 1.45;
        const bz = r * Math.sin(theta) * 0.78;
        galaxy[gOffset] = bx * cosBar - bz * sinBar;
        galaxy[gOffset + 1] = gaussian() * 0.026;
        galaxy[gOffset + 2] = bx * sinBar + bz * cosBar;
        meta[mOffset] = 0.72 + 0.48 * rand();
        meta[mOffset + 1] = 0.94 + 0.06 * rand();
      } else {
        const sign = rand() < 0.5 ? -1 : 1;
        const u = sign * Math.pow(rand(), 1.25) * MW_BAR_HALF_LEN * 1.12;
        const widthEnv = Math.max(0.28, 1.0 - Math.abs(u / (MW_BAR_HALF_LEN * 1.25)) * 0.55);
        const w = gaussian() * 0.048 * widthEnv;
        galaxy[gOffset] = u * cosBar - w * sinBar;
        galaxy[gOffset + 1] = gaussian() * 0.024 * widthEnv;
        galaxy[gOffset + 2] = u * sinBar + w * cosBar;
        meta[mOffset] = 0.58 + 0.48 * rand();
        meta[mOffset + 1] = 0.74 + 0.22 * rand();
      }
      meta[mOffset + 2] = 0.02 * rand();
      nebula[nOffset] = 0.0;
    }

    // --- B. Calm Deep-Space Hero Field (never clutters the 2D Venus emblem) ---
    const hAngle = rand() * Math.PI * 2.0;
    const hRadius = 0.22 + Math.pow(rand(), 0.58) * 1.55;
    heroField[gOffset] = Math.cos(hAngle) * hRadius;
    heroField[gOffset + 1] = gaussian() * 0.22;
    heroField[gOffset + 2] = Math.sin(hAngle) * hRadius;
    nebula[nOffset + 1] = rand() < 0.35 ? 0.55 + 0.45 * rand() : 0.24 + 0.32 * rand();

    // --- C. Diagonal Milky Way Star Cloud & Dust Rift State (for p = 0.18 -> 0.62) ---
    const mwSlot = rand();
    if (mwSlot < 0.018) {
      const uP = rand();
      const vP = rand();
      const thetaP = uP * 2.0 * Math.PI;
      const phiP = Math.acos(2.0 * vP - 1.0);
      const rPlanet = 0.022 * Math.cbrt(rand());
      milkyway[gOffset] = -0.08 + rPlanet * Math.sin(phiP) * Math.cos(thetaP);
      milkyway[gOffset + 1] = 0.02 + rPlanet * Math.cos(phiP);
      milkyway[gOffset + 2] = 0.18 + rPlanet * Math.sin(phiP) * Math.sin(thetaP);
    } else if (mwSlot < 0.78) {
      const spineS = (rand() - 0.5) * 2.2;
      const side = rand() < 0.52 ? -1.0 : 1.0;
      const riftGap = 0.035 + Math.abs(gaussian()) * 0.14;
      const perp = side * riftGap + Math.sin(spineS * 3.2) * 0.035;
      const tiltMW = 1.18;
      const cosM = Math.cos(tiltMW);
      const sinM = Math.sin(tiltMW);
      milkyway[gOffset] = 0.12 + spineS * cosM - perp * sinM;
      milkyway[gOffset + 1] = spineS * sinM + perp * cosM;
      milkyway[gOffset + 2] = gaussian() * 0.18;
    } else {
      milkyway[gOffset] = (rand() - 0.5) * 2.4;
      milkyway[gOffset + 1] = (rand() - 0.5) * 1.8;
      milkyway[gOffset + 2] = -0.35 + (rand() - 0.5) * 0.6;
    }

    // --- D. 3D Planet Venus Sphere, Orbital Ring, Outer Atmospheric Motes & Persistent MW ---
    if (i % 5 === 0) {
      const uDust = rand() * 2.0 - 1.0;
      const rDustXY = Math.sqrt(Math.max(0.0, 1.0 - uDust * uDust));
      const phiDust = rand() * Math.PI * 2.0;
      const dustR = 0.24 + Math.pow(rand(), 0.72) * 0.17;
      venus[gOffset] = Math.cos(phiDust) * rDustXY * dustR;
      venus[gOffset + 1] = uDust * dustR * 0.86;
      venus[gOffset + 2] = Math.sin(phiDust) * rDustXY * dustR;
      meta[mOffset + 3] = 1.0;
    } else {
      const slot = nonStoryIdx;
      nonStoryIdx += 1;
      if (slot < venusSphereCount) {
        const yNorm = 1.0 - (slot / Math.max(1, venusSphereCount - 1)) * 2.0;
        const radiusAtY = Math.sqrt(Math.max(0.0, 1.0 - yNorm * yNorm));
        const thetaV = goldenAngle * slot;
        const shellR = 0.21 * (0.988 + 0.022 * (slot % 3) * 0.5);
        venus[gOffset] = Math.cos(thetaV) * radiusAtY * shellR;
        venus[gOffset + 1] = yNorm * shellR;
        venus[gOffset + 2] = Math.sin(thetaV) * radiusAtY * shellR;
        meta[mOffset + 3] = 2.0;
      } else if (slot < venusSphereCount + mwPersistCount) {
        venus[gOffset] = (rand() - 0.5) * 2.2;
        venus[gOffset + 1] = (rand() - 0.5) * 1.6;
        venus[gOffset + 2] = -0.45 - rand() * 0.6;
        meta[mOffset + 3] = 4.0;
      } else {
        venus[gOffset] = (rand() - 0.5) * 2.2;
        venus[gOffset + 1] = (rand() - 0.5) * 1.6;
        venus[gOffset + 2] = -0.45 - rand() * 0.6;
        meta[mOffset + 3] = 0.0;
      }
    }

    // --- E. 3D Warp Flight Starfield ---
    const warpTheta = rand() * Math.PI * 2.0;
    const warpRadius = 0.08 + Math.pow(rand(), 0.62) * 2.75;
    warp[gOffset] = warpRadius * Math.cos(warpTheta);
    warp[gOffset + 1] = warpRadius * Math.sin(warpTheta) * 0.72;
    warp[gOffset + 2] = 0.15 + rand() * 4.65;
  }

  return { galaxy, heroField, milkyway, venus, warp, meta, nebula };
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Unable to create particle shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'Unknown particle shader error';
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function releaseWebglContext(canvas: HTMLCanvasElement) {
  canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context')?.loseContext();
}

type Renderer = {
  draw: (progress: number, pointerX: number, pointerY: number, time: number, scrollImpulse: number, travel?: number, warpSpeed?: number) => void;
  resize: (width: number, height: number, ratio: number) => void;
  dispose: () => void;
  name: 'webgl2' | 'canvas2d';
};

function createWebglRenderer(canvas: HTMLCanvasElement, quality: ParticleQuality): Renderer | null {
  const config = QUALITY[quality];
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    powerPreference: quality === 'high' ? 'high-performance' : 'low-power',
  });
  if (!gl) return null;
  const compilationStarted = performance.now();

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, `#version 300 es
    precision highp float;

    in vec3 a_galaxy;
    in vec3 a_hero;
    in vec3 a_milkyway;
    in vec3 a_venus;
    in vec3 a_warp;
    in vec4 a_meta; // x: size, y: temp, z: dust, w: role (0: warp, 1: story/outerDust, 2: venusSphere, 3: venusRing, 4: mwPersist)
    in vec2 a_nebula; // x: roseMagentaKnotWeight, y: heroStarWeight
    in vec4 a_drift;
    in vec2 a_from;
    in vec2 a_to;
    in float a_blue;

    uniform float u_amount;
    uniform float u_progress;
    uniform float u_time;
    uniform float u_scroll_impulse;
    uniform float u_pixel_ratio;
    uniform float u_aspect;
    uniform vec2 u_pointer;
    uniform float u_point_size_base;
    uniform vec2 u_venus_center;
    uniform float u_travel;
    uniform float u_warp_speed;

    out vec3 v_color;
    out float v_alpha;
    out float v_core_glow;
    out vec2 v_streak_dir;
    out float v_streak_len;

    float quintic(float edge0, float edge1, float value) {
      float t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
      return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    vec3 getBarredGalaxyColor(float temp, float dust, float nebulaKnot) {
      vec3 col;
      if (temp > 0.72) {
        float t = (temp - 0.72) / 0.28;
        col = mix(vec3(0.867, 0.631, 0.455), vec3(1.0, 0.953, 0.886), t);
      } else if (temp > 0.40) {
        float t = (temp - 0.40) / 0.32;
        vec3 warmCream = vec3(0.945, 0.878, 0.816);
        col = t < 0.45
          ? mix(vec3(0.557, 0.678, 0.871), warmCream, t / 0.45)
          : mix(warmCream, vec3(0.867, 0.631, 0.455), (t - 0.45) / 0.55);
      } else {
        col = mix(vec3(0.455, 0.608, 0.863), vec3(0.757, 0.843, 0.969), temp / 0.40);
      }
      // Subtle rose-magenta H-II nebula knots along arm ridges
      col = mix(col, vec3(0.859, 0.510, 0.588), nebulaKnot * 0.54);
      // Soft interstellar dust lanes without heavy dark bokeh circles
      col = mix(col, vec3(0.075, 0.088, 0.132), dust * 0.54);
      return col;
    }

    vec3 projectCamera(vec3 pos, float yaw, float pitch, float roll, float camDist, float fov, float aspect, vec2 center) {
      float cosY = cos(yaw); float sinY = sin(yaw);
      float x1 = pos.x * cosY - pos.z * sinY;
      float z1 = pos.x * sinY + pos.z * cosY;
      float y1 = pos.y;

      float cosP = cos(pitch); float sinP = sin(pitch);
      float y2 = y1 * cosP - z1 * sinP;
      float z2 = y1 * sinP + z1 * cosP;
      float x2 = x1;

      float cosR = cos(roll); float sinR = sin(roll);
      float x3 = x2 * cosR - y2 * sinR;
      float y3 = x2 * sinR + y2 * cosR;
      float z3 = z2;

      float zView = max(0.32, z3 + camDist);
      float projX = (x3 / zView) * fov;
      float projY = (y3 / zView) * fov;
      return vec3(center + vec2(projX / aspect, projY), zView);
    }

    // Tilt galaxy plane in pitch FIRST so 3D yaw banks the inclined spiral wings in depth
    vec3 projectGalaxy3D(vec3 pos, float pitch, float yaw, float roll, float camDist, float fov, float aspect, vec2 center) {
      float cosP = cos(pitch); float sinP = sin(pitch);
      float x1 = pos.x;
      float y1 = pos.y * cosP - pos.z * sinP;
      float z1 = pos.y * sinP + pos.z * cosP;

      float cosY = cos(yaw); float sinY = sin(yaw);
      float x2 = x1 * cosY - z1 * sinY;
      float z2 = x1 * sinY + z1 * cosY;
      float y2 = y1;

      float cosR = cos(roll); float sinR = sin(roll);
      float x3 = x2 * cosR - y2 * sinR;
      float y3 = x2 * sinR + y2 * cosR;
      float z3 = z2;

      float zView = max(0.32, z3 + camDist);
      float projX = (x3 / zView) * fov;
      float projY = (y3 / zView) * fov;
      return vec3(center + vec2(projX / aspect, projY), zView);
    }

    void main() {
      float p = u_progress;
      float role = a_meta.w;
      float isStory = (role > 0.5 && role < 1.5) ? 1.0 : 0.0;
      float isVenusSphere = (role > 1.5 && role < 2.5) ? 1.0 : 0.0;
      float isVenusObj = (role > 1.5 && role < 2.5) ? 1.0 : 0.0;
      float isMwPersist = (role > 3.5) ? 1.0 : 0.0;

      // Cosmic stage blend weights:
      // 0.00 -> 0.16: Calm deep-space stars behind Hero + 2D Venus emblem
      // 0.16 -> 0.62: Diagonal Milky Way dust rift + warp star flight + Story 01/02
      // 0.52 -> 0.85: Planet Venus approach -> 0.72 AU -> center flyby -> smooth spiral dissolution
      // 0.80 -> 1.00: Interactive 3D-Perspective Milky Way Barred Spiral Galaxy
      float heroWeight = 1.0 - smoothstep(0.10, 0.22, p);
      float mwRampIn = smoothstep(0.14, 0.26, p);
      float mwPersistFade = 1.0 - smoothstep(0.54, 0.72, p);
      float venusBodyMwFade = 1.0 - smoothstep(0.45, 0.52, p);

      float venusAppear = smoothstep(0.51, 0.60, p);
      float ringAppear = smoothstep(0.56, 0.66, p);
      float venusDissolve = smoothstep(0.79, 0.86, p);
      float venusFadeOut = 1.0 - venusDissolve;
      float venusActive = step(0.52, p) * (1.0 - step(0.87, p));
      float venusBeaconPhase = smoothstep(0.20, 0.29, p) * (1.0 - smoothstep(0.51, 0.59, p));
      float isVenusBeaconPt = isVenusSphere * step(0.86, a_meta.y);

      float finaleMwAppear = smoothstep(0.80, 0.865, p);

      vec2 ptr = (u_pointer.x < -1.0) ? vec2(0.5) : u_pointer;
      vec2 ptrDelta = ptr - vec2(0.5);
      float yaw = ptrDelta.x * 0.14;
      float pitchOffset = ptrDelta.y * 0.10;
      float fov = 1.32;

      // 1A. Hero calm field & Mid-story Diagonal Milky Way dust rift
      float heroRot = u_time * 0.000035;
      vec3 heroPos = vec3(
        a_hero.x * cos(heroRot) - a_hero.z * sin(heroRot),
        a_hero.y,
        a_hero.x * sin(heroRot) + a_hero.z * cos(heroRot)
      );
      vec3 mwPos = mix(heroPos, a_milkyway, mwRampIn);
      float mwCamPitch = mix(0.26, 0.12, smoothstep(0.16, 0.34, p)) + pitchOffset;
      float mwCamRoll = -0.14 * (1.0 - smoothstep(0.16, 0.38, p));
      float mwCamDist = mix(1.42, 1.12, smoothstep(0.0, 0.25, p));
      vec2 mwStageCenter = mix(vec2(0.60, 0.48), vec2(0.54, 0.49), smoothstep(0.12, 0.30, p));
      vec3 mwProj = projectCamera(mwPos, yaw, mwCamPitch, mwCamRoll, mwCamDist, fov, u_aspect, mwStageCenter);
      vec2 mwScreen = mwProj.xy;
      float mwZView = mwProj.z;

      // 1B. Planet Venus retrograde axial rotation, 1/z approach & spiral dissolution
      float venusSpin = -u_time * 0.00028 + p * 1.8;
      float cosV = cos(venusSpin);
      float sinV = sin(venusSpin);
      vec3 venusRot = vec3(
        a_venus.x * cosV - a_venus.z * sinV,
        a_venus.y,
        a_venus.x * sinV + a_venus.z * cosV
      );
      float vTilt = 0.22;
      vec3 venusPos = vec3(
        venusRot.x,
        venusRot.y * cos(vTilt) - venusRot.z * sin(vTilt),
        venusRot.y * sin(vTilt) + venusRot.z * cos(vTilt)
      );

      // Smooth radial swirl dispersal as Venus dissolves (0.79 -> 0.86)
      if (venusDissolve > 0.001) {
        float swirl = venusDissolve * (1.45 + a_meta.x * 0.9);
        float cSw = cos(swirl);
        float sSw = sin(swirl);
        float expand = 1.0 + venusDissolve * (2.6 + a_drift.w * 420.0);
        venusPos = vec3(
          (venusPos.x * cSw - venusPos.z * sSw) * expand,
          venusPos.y * (1.0 - venusDissolve * 0.35),
          (venusPos.x * sSw + venusPos.z * cSw) * expand
        );
      }

      float approachT = clamp((p - 0.52) / (0.74 - 0.52), 0.0, 1.0);
      float venusDist = pow(24.0, 1.0 - approachT);
      float resolvedScale = clamp(1.0 / venusDist, 0.0416, 1.0);
      float starToGlobe = smoothstep(0.50, 0.58, p);
      float perspectiveScale = mix(0.009, resolvedScale, starToGlobe);
      float aspectScale = min(1.0, u_aspect) * (u_aspect < 1.0 ? 0.84 : 1.0);
      vec3 scaledVenus = venusPos * perspectiveScale * aspectScale;

      float transferT = quintic(0.74, 0.81, p);
      float venusCamPitch = 0.08 + pitchOffset;
      float venusCamDist = mix(1.05, 0.84, transferT);
      vec2 flightStart = (u_venus_center.x < 0.45) ? vec2(0.40, 0.46) : vec2(0.50, 0.30);
      vec2 corridorCenter = mix(flightStart, u_venus_center, smoothstep(0.52, 0.68, p));
      vec2 venusCenter = mix(corridorCenter, vec2(0.50, 0.48), transferT);

      vec3 venusCamInput = vec3(scaledVenus.x, scaledVenus.y, -scaledVenus.z);
      vec3 venusProj = projectCamera(venusCamInput, yaw, venusCamPitch, 0.0, venusCamDist, fov, u_aspect, venusCenter);
      vec2 venusScreen = venusProj.xy;
      float venusZView = venusProj.z;

      // 1C. Finale Barred Spiral Milky Way (3D interactive perspective tilt with pointer)
      float rGal = length(a_galaxy.xz);
      float omega = 0.000105 / sqrt(rGal * rGal + 0.06);
      float galAngle = u_time * omega;
      float cosG = cos(galAngle);
      float sinG = sin(galAngle);
      vec3 galRotPos = vec3(
        a_galaxy.x * cosG - a_galaxy.z * sinG,
        a_galaxy.y,
        a_galaxy.x * sinG + a_galaxy.z * cosG
      );
      float finaleExpand = mix(0.68, 1.0, finaleMwAppear);
      vec3 finalePos = galRotPos * finaleExpand * (u_aspect < 1.0 ? 0.68 : 0.88);
      // True 3D perspective inclination & dramatic mouse gimbal tilt (X pitch -> Y 3D bank -> Z roll):
      // pointer Y pitches from nearly edge-on (~16 deg) to open face (~73 deg); pointer X banks left/right ~+/-33 deg in 3D depth!
      float finalePitch = clamp(0.76 - ptrDelta.y * 0.96, 0.28, 1.28);
      float finaleYaw = ptrDelta.x * 1.16;
      float finaleRoll = -0.16 + ptrDelta.x * 0.24;
      float finaleCamDist = 1.50;
      vec2 finaleCenter = vec2(0.50, 0.48);
      vec3 finaleProj = projectGalaxy3D(finalePos, finalePitch, finaleYaw, finaleRoll, finaleCamDist, 1.32, u_aspect, finaleCenter);
      vec2 finaleScreen = finaleProj.xy;
      float finaleZView = finaleProj.z;

      // 3D Forward Spaceflight Warp Star Streaming:
      float zSpan = 4.65;
      float zNear = 0.15;
      float zRel = mod(a_warp.z - u_travel, zSpan);
      if (zRel < 0.0) zRel += zSpan;
      float zStar = zRel + zNear;

      vec2 warpCenter = mix(mwStageCenter, venusCenter, smoothstep(0.52, 0.74, p));
      warpCenter = mix(warpCenter, finaleCenter, finaleMwAppear);
      float fovAspect = fov / u_aspect;
      vec2 projWarp = vec2((a_warp.x / zStar) * fovAspect, (a_warp.y / zStar) * fov);
      vec2 warpScreen = warpCenter + projWarp;
      vec2 starPos = mix(warpScreen, mwScreen, heroWeight);

      float streakDeltaZ = (u_warp_speed * 0.42 + 0.018) * (1.0 - heroWeight * 0.85) * (1.0 - finaleMwAppear * 0.85);
      float zTail = zStar + streakDeltaZ;
      vec2 projTail = vec2((a_warp.x / zTail) * fovAspect, (a_warp.y / zTail) * fov);
      vec2 warpTail = warpCenter + projTail;
      vec2 deltaV = warpScreen - warpTail;
      float deltaLen = length(deltaV);
      vec2 sDir = normalize(deltaV + vec2(0.00001));
      float sLen = clamp(deltaLen * 42.0 * (0.35 + u_warp_speed * 0.65), 0.0, 1.0) * (1.0 - heroWeight) * (1.0 - finaleMwAppear);

      // Select active coordinate space per phase:
      float useFinaleSpace = smoothstep(0.81, 0.865, p);
      float useBeaconSpace = isVenusBeaconPt * step(0.20, p) * (1.0 - step(0.54, p));
      float useVenusSpace = max(venusActive * (isVenusObj + isStory), useBeaconSpace) * (1.0 - useFinaleSpace);
      float useMwSpace = (1.0 - useVenusSpace) * (1.0 - useFinaleSpace) * (isVenusObj + isMwPersist + isStory);
      vec2 activeCosmicPos = starPos;
      float activeZView = mix(zStar, mwZView, heroWeight);

      if (useFinaleSpace > 0.01) {
        vec2 preFinalePos = mix(starPos, venusScreen, venusActive * (isVenusObj + isStory));
        if (useVenusSpace < 0.5 && isMwPersist > 0.5) preFinalePos = mwScreen;
        activeCosmicPos = mix(preFinalePos, finaleScreen, useFinaleSpace);
        activeZView = mix(activeZView, finaleZView, useFinaleSpace);
        v_streak_dir = sDir;
        v_streak_len = sLen * (1.0 - useFinaleSpace);
      } else if (useVenusSpace > 0.5) {
        activeCosmicPos = venusScreen;
        activeZView = venusZView;
        v_streak_dir = vec2(0.0);
        v_streak_len = 0.0;
      } else if (useMwSpace > 0.5) {
        activeCosmicPos = mwScreen;
        activeZView = mwZView;
        v_streak_dir = vec2(0.0);
        v_streak_len = 0.0;
      } else {
        v_streak_dir = sDir;
        v_streak_len = sLen;
      }

      // Story morphing for the evenly interleaved 20% constellation particles during beats 01..03 (p = 0.18 -> 0.78)
      float storyActive = isStory * smoothstep(0.18, 0.26, p) * (1.0 - smoothstep(0.70, 0.78, p));
      vec2 finalPos = activeCosmicPos;

      if (storyActive > 0.001) {
        float t = u_amount * u_amount * (3.0 - 2.0 * u_amount);
        vec2 storyPos = mix(a_from, a_to, t);

        vec2 delta = storyPos - ptr;
        vec2 metric = delta * vec2(u_aspect, 1.0);
        float push = (1.0 - smoothstep(0.0, 0.095, length(metric))) * 0.018;
        storyPos += normalize(delta + vec2(0.00001)) * push;

        float driftPhaseX = a_drift.x + u_time * a_drift.z;
        float driftPhaseY = a_drift.y + u_time * a_drift.z * 0.73;
        storyPos.x += sin(driftPhaseX) * a_drift.w;
        storyPos.y += cos(driftPhaseY) * a_drift.w * 0.68;

        float scatterBlend = smoothstep(0.20, 0.38, p);
        float directionPhase = a_drift.x * 1.7 + a_drift.y * 0.45;
        vec2 impulseDir = normalize(vec2(cos(directionPhase), sin(a_drift.y * 1.3 + a_drift.x)));
        float individualImpulse = 0.35 + fract(sin(a_drift.x * 12.9898 + a_drift.y * 78.233) * 43758.5453) * 0.65;
        storyPos += impulseDir * u_scroll_impulse * scatterBlend * individualImpulse;

        finalPos = mix(activeCosmicPos, storyPos, storyActive);
        v_streak_len *= (1.0 - storyActive);
      }

      gl_Position = vec4(finalPos.x * 2.0 - 1.0, 1.0 - finalPos.y * 2.0, 0.0, 1.0);

      // Point size scaling with 3D perspective depth
      float pScale = 1.0 / max(0.32, activeZView);
      float spaceSize = a_meta.x * (u_point_size_base * 0.62) * u_pixel_ratio * clamp(pScale, 0.42, 2.35);
      float warpSize = spaceSize * (1.0 + v_streak_len * 3.0);
      float storySize = (a_meta.x > 0.85 ? u_point_size_base * 1.14 : u_point_size_base * 0.90) * u_pixel_ratio;

      // Color synthesis:
      // A. Hero calm stars & mid-story Diagonal Milky Way
      vec3 heroCol = mix(vec3(0.557, 0.678, 0.871), vec3(0.867, 0.631, 0.455), a_meta.y);
      vec3 mwColor = mix(vec3(0.88, 0.78, 0.66), vec3(1.0, 0.90, 0.74), a_meta.y);
      if (a_meta.z > 0.45) {
        mwColor = mix(mwColor, vec3(0.22, 0.16, 0.14), smoothstep(0.45, 0.82, a_meta.z) * 0.65);
      } else {
        mwColor = mix(mwColor, vec3(0.36, 0.24, 0.19), a_meta.z * 0.48);
      }
      vec3 cosmicColor = mix(heroCol, mwColor, mwRampIn);

      // B. 3D Planet Venus Shading + Distant Morning Star Pinpoint Flash
      vec3 vNorm = normalize(venusPos + vec3(0.0, 0.0, 0.0001));
      vec3 sunDir = normalize(vec3(-0.62, -0.38, 0.68));
      float ndotl = dot(vNorm, sunDir);
      float dayFactor = smoothstep(-0.24, 0.62, ndotl);
      float bandPattern = 0.5 + 0.5 * sin(a_venus.y * 51.0 + sin(a_venus.x * 16.0) * 1.45);
      vec3 venusDay = mix(vec3(0.64, 0.37, 0.24), vec3(0.867, 0.631, 0.455), smoothstep(0.0, 0.55, bandPattern));
      venusDay = mix(venusDay, vec3(1.0, 0.86, 0.68), smoothstep(0.55, 1.0, bandPattern));
      vec3 venusNight = vec3(0.07, 0.035, 0.024);
      float viewDot = abs(vNorm.z);
      float fresnel = pow(1.0 - clamp(viewDot, 0.0, 1.0), 2.5);
      vec3 venusCol = mix(venusNight, venusDay, dayFactor) + vec3(0.95, 0.72, 0.46) * fresnel * (0.35 + 0.45 * dayFactor);
      float starCoreBlend = venusBeaconPhase * isVenusBeaconPt * (1.0 - smoothstep(0.50, 0.58, p));
      venusCol = mix(venusCol, vec3(1.0, 0.97, 0.90), starCoreBlend);
      if (isStory > 0.5) {
        venusCol = mix(vec3(0.867, 0.631, 0.455), vec3(1.0, 0.86, 0.68), a_meta.y);
      }
      cosmicColor = mix(cosmicColor, venusCol, useVenusSpace);

      // C. Finale Barred Spiral Milky Way Color (warm copper bar + blue-silver arms + rose nebula + subtle dust)
      vec3 finaleColor = getBarredGalaxyColor(a_meta.y, a_meta.z, a_nebula.x);
      cosmicColor = mix(cosmicColor, finaleColor, useFinaleSpace);

      // D. Story Constellation Color
      float blueMod = a_blue * clamp((p - 0.58) * 1.4, 0.0, 0.48);
      vec3 storyColor = mix(vec3(0.867, 0.631, 0.455), vec3(0.557, 0.678, 0.871), blueMod);
      v_color = mix(cosmicColor, storyColor, storyActive);

      // Core glow: soft nucleus radiance for Finale Milky Way, Venus limb & blazing distant Morning Star beacon
      float ctaCoreSoften = 1.0 - 0.24 * smoothstep(0.88, 0.96, p);
      float finaleCoreGlow = smoothstep(0.24, 0.0, rGal) * smoothstep(0.74, 1.0, a_meta.y) * finaleMwAppear * 0.92 * ctaCoreSoften;
      float nebulaGlow = a_nebula.x * finaleMwAppear * 0.22;
      float beaconFactor = (1.0 - smoothstep(0.0, 0.42, approachT)) * venusAppear * venusFadeOut;
      float venusLimbGlow = (fresnel * dayFactor * 0.90 + beaconFactor * 0.60) * venusAppear * venusFadeOut * isVenusSphere;
      float distantStarFlare = starCoreBlend * 1.75;
      v_core_glow = finaleCoreGlow + nebulaGlow + venusLimbGlow + distantStarFlare;

      // Point size
      float approachSizeScale = mix(0.72, 1.05, approachT) * (1.0 + 0.08 * transferT);
      float venusSizeBoost = (0.42 + 0.38 * dayFactor) * isVenusSphere;
      float beaconSpikeSize = starCoreBlend * (u_point_size_base * 1.65) * u_pixel_ratio;
      float venusPointSize = spaceSize * approachSizeScale * (1.0 + venusSizeBoost) + beaconFactor * 1.6 * isVenusSphere + beaconSpikeSize;
      float finaleSizeBoost = (a_meta.z > 0.5 ? 1.04 : (a_nebula.x > 0.35 ? 1.08 : 0.94));
      float finalePointSize = spaceSize * finaleSizeBoost * clamp(1.32 / finaleZView, 0.62, 1.50);

      float preFinaleSize = (useVenusSpace > 0.5) ? venusPointSize : ((useMwSpace > 0.5) ? spaceSize : warpSize);
      float activeSize = mix(preFinaleSize, finalePointSize, useFinaleSpace);
      gl_PointSize = clamp(mix(activeSize, storySize, storyActive), 1.1, 36.0);

      // Alpha orchestration
      float heroAlpha = a_nebula.y * 0.52;
      float densityAtten = mix(0.68, 0.95, smoothstep(0.0, 0.18, rGal));
      float mwBaseAlpha = densityAtten * (a_meta.z > 0.45 ? 0.24 : (1.0 - a_meta.z * 0.28));
      float venusSphereAlpha = mix(0.14, 0.98, smoothstep(-0.32, 0.22, vNorm.z)) * venusAppear * venusFadeOut;
      float beaconAlpha = starCoreBlend * 0.96;
      float outerDustAlpha = 0.24 * smoothstep(0.70, 0.79, p) * venusFadeOut;

      // Finale Milky Way alpha (light dust shading without oversized opaque black spots)
      float finaleAlpha = (a_meta.z > 0.5)
        ? 0.20 * a_meta.z
        : (a_nebula.x > 0.35 ? 0.62 : mix(0.92, 0.76, smoothstep(0.0, 0.35, rGal)));
      finaleAlpha *= finaleMwAppear * (1.0 - 0.14 * smoothstep(0.88, 0.96, p));

      float baseCosmicAlpha = mwBaseAlpha;
      if (useVenusSpace > 0.5) {
        baseCosmicAlpha = isVenusSphere > 0.5 ? max(venusSphereAlpha, beaconAlpha) : outerDustAlpha;
      } else if (isVenusObj > 0.5) {
        baseCosmicAlpha = mwBaseAlpha * venusBodyMwFade;
      } else if (isMwPersist > 0.5) {
        baseCosmicAlpha = mwBaseAlpha * mwPersistFade;
      } else {
        baseCosmicAlpha = mix(0.62, heroAlpha, heroWeight);
      }

      baseCosmicAlpha = mix(baseCosmicAlpha, finaleAlpha, useFinaleSpace);
      v_alpha = mix(baseCosmicAlpha, 0.94, storyActive);
    }`);

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;

    in vec3 v_color;
    in float v_alpha;
    in float v_core_glow;
    in vec2 v_streak_dir;
    in float v_streak_len;

    out vec4 fragColor;

    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord) * 2.0;
      if (dist > 1.0 && v_streak_len < 0.05) discard;

      float core = exp(-dist * dist * 20.5);
      float corona = exp(-dist * 5.2);
      float shapeCirc = core * 0.86 + corona * 0.14;

      float spikeMask = smoothstep(1.05, 1.55, v_core_glow);
      float spikeH = exp(-abs(coord.y) * 34.0) * exp(-abs(coord.x) * 4.8);
      float spikeV = exp(-abs(coord.x) * 34.0) * exp(-abs(coord.y) * 4.8);
      float starSpike = (spikeH + spikeV) * spikeMask * 0.92;

      vec2 dir = vec2(v_streak_dir.x, -v_streak_dir.y);
      float along = dot(coord, dir);
      float perp = length(coord - along * dir);
      float alongNorm = along / 0.48;
      float alongFade = clamp(1.0 - abs(alongNorm), 0.0, 1.0);
      alongFade = alongFade * alongFade * (3.0 - 2.0 * alongFade);
      float perpWidth = mix(0.16, 0.065, v_streak_len);
      float perpFade = exp(-perp * perp / (perpWidth * perpWidth));
      float shapeStreak = alongFade * perpFade * 1.25;

      float shape = mix(shapeCirc, shapeStreak, v_streak_len);
      shape += v_core_glow * exp(-dist * 4.4) * 0.24 + starSpike;

      float finalAlpha = shape * v_alpha;
      if (finalAlpha < 0.005) discard;
      fragColor = vec4(v_color * finalAlpha, finalAlpha);
    }`);

  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || 'Unable to link particle shader');
  }
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  canvas.dataset.shaderCompileMs = (performance.now() - compilationStarted).toFixed(3);

  const webglCount = quality === 'high' ? 14000 : 4800;
  canvas.dataset.galaxyCount = String(webglCount);

  const { galaxy, heroField, milkyway, venus, warp, meta, nebula } = makeCelestialStates3D(webglCount);
  const frames = keyframesFor(webglCount);
  const drift = makeDrift(webglCount);
  const blue = new Float32Array(webglCount);
  for (let index = 0; index < webglCount; index += 1) {
    blue[index] = index % 17 === 6 ? 1 : 0;
  }

  const buffers: WebGLBuffer[] = [];
  const attribute = (name: string, size: number, data: Float32Array, usage: number) => {
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error(`Unable to create ${name} particle buffer`);
    buffers.push(buffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    const location = gl.getAttribLocation(program, name);
    if (location !== -1) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    }
    return buffer;
  };

  attribute('a_galaxy', 3, galaxy, gl.STATIC_DRAW);
  attribute('a_hero', 3, heroField, gl.STATIC_DRAW);
  attribute('a_milkyway', 3, milkyway, gl.STATIC_DRAW);
  attribute('a_venus', 3, venus, gl.STATIC_DRAW);
  attribute('a_warp', 3, warp, gl.STATIC_DRAW);
  attribute('a_meta', 4, meta, gl.STATIC_DRAW);
  attribute('a_nebula', 2, nebula, gl.STATIC_DRAW);
  attribute('a_drift', 4, drift, gl.STATIC_DRAW);
  attribute('a_blue', 1, blue, gl.STATIC_DRAW);
  const fromBuffer = attribute('a_from', 2, frames[0].points, gl.DYNAMIC_DRAW);
  const toBuffer = attribute('a_to', 2, frames[1].points, gl.DYNAMIC_DRAW);

  const amountLocation = gl.getUniformLocation(program, 'u_amount');
  const progressLocation = gl.getUniformLocation(program, 'u_progress');
  const timeLocation = gl.getUniformLocation(program, 'u_time');
  const scrollImpulseLocation = gl.getUniformLocation(program, 'u_scroll_impulse');
  const ratioLocation = gl.getUniformLocation(program, 'u_pixel_ratio');
  const aspectLocation = gl.getUniformLocation(program, 'u_aspect');
  const pointerLocation = gl.getUniformLocation(program, 'u_pointer');
  const pointSizeBaseLocation = gl.getUniformLocation(program, 'u_point_size_base');
  const venusCenterLocation = gl.getUniformLocation(program, 'u_venus_center');
  const travelLocation = gl.getUniformLocation(program, 'u_travel');
  const warpSpeedLocation = gl.getUniformLocation(program, 'u_warp_speed');

  let currentSegment = -1;
  let width = 1;
  let height = 1;
  let ratio = 1;
  gl.useProgram(program);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return {
    name: 'webgl2',
    resize(nextWidth, nextHeight, nextRatio) {
      width = nextWidth;
      height = nextHeight;
      ratio = nextRatio;
      gl.viewport(0, 0, canvas.width, canvas.height);
    },
    draw(value, pointerX, pointerY, time, scrollImpulse, travel, warpSpeed) {
      const segment = segmentFor(value);
      const previous = frames[segment - 1];
      const next = frames[segment];
      if (segment !== currentSegment) {
        gl.bindBuffer(gl.ARRAY_BUFFER, fromBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, previous.points);
        gl.bindBuffer(gl.ARRAY_BUFFER, toBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, next.points);
        currentSegment = segment;
      }
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(amountLocation, (value - previous.at) / Math.max(.001, next.at - previous.at));
      gl.uniform1f(progressLocation, value);
      gl.uniform1f(timeLocation, time);
      gl.uniform1f(scrollImpulseLocation, scrollImpulse);
      gl.uniform1f(ratioLocation, ratio);
      gl.uniform1f(aspectLocation, width / Math.max(1, height));
      gl.uniform2f(pointerLocation, pointerX, pointerY);
      gl.uniform1f(pointSizeBaseLocation, config.pointSize);
      const targetVenusX = width >= 1024 ? 0.34 : 0.50;
      const targetVenusY = width >= 1024 ? 0.44 : 0.28;
      gl.uniform2f(venusCenterLocation, targetVenusX, targetVenusY);
      gl.uniform1f(travelLocation, travel ?? 0);
      gl.uniform1f(warpSpeedLocation, warpSpeed ?? 0);
      gl.drawArrays(gl.POINTS, 0, webglCount);
    },
    dispose() {
      for (const buffer of buffers) gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}

function getVenusTrajectory(progress: number, width: number, height: number) {
  const isDesktop = width >= 1024;
  const corridorX = isDesktop ? 0.34 : 0.50;
  const corridorY = isDesktop ? 0.44 : 0.28;
  const startX = isDesktop ? 0.40 : 0.50;
  const startY = isDesktop ? 0.46 : 0.30;
  const flightAlignment = smoothstep(0.52, 0.68, progress);
  const alignedX = startX + (corridorX - startX) * flightAlignment;
  const alignedY = startY + (corridorY - startY) * flightAlignment;

  const transferT = quintic(0.74, 0.81, progress);
  const centerX = alignedX + (0.50 - alignedX) * transferT;
  const centerY = alignedY + (0.48 - alignedY) * transferT;

  const approachT = Math.min(1, Math.max(0, (progress - 0.52) / (0.74 - 0.52)));
  const venusDist = Math.pow(24.0, 1.0 - approachT);
  const perspectiveScale = Math.min(1.0, Math.max(0.0416, 1.0 / venusDist));
  const orbitCamDist = 1.05 + (0.84 - 1.05) * transferT;

  const aspect = width / Math.max(1, height);
  const aspectScale = Math.min(1.0, aspect) * (aspect < 1.0 ? 0.84 : 1.0);

  return {
    isDesktop,
    centerX,
    centerY,
    approachT,
    perspectiveScale,
    transferT,
    orbitCamDist,
    aspectScale,
  };
}

function projectCamera3D(
  px: number,
  py: number,
  pz: number,
  cosY: number,
  sinY: number,
  cosP: number,
  sinP: number,
  cosR: number,
  sinR: number,
  camDist: number,
  fov: number,
  aspect: number,
  centerX: number,
  centerY: number,
  width: number,
  height: number,
): [number, number, number] {
  const x1 = px * cosY - pz * sinY;
  const z1 = px * sinY + pz * cosY;
  const y2 = py * cosP - z1 * sinP;
  const z2 = py * sinP + z1 * cosP;
  const x3 = x1 * cosR - y2 * sinR;
  const y3 = x1 * sinR + y2 * cosR;
  const zView = Math.max(0.32, z2 + camDist);
  const sx = (centerX + (x3 / zView) * fov / aspect) * width;
  const sy = (centerY + (y3 / zView) * fov) * height;
  return [sx, sy, zView];
}

function projectGalaxyFinale3D(
  px: number,
  py: number,
  pz: number,
  cosP: number,
  sinP: number,
  cosY: number,
  sinY: number,
  cosR: number,
  sinR: number,
  camDist: number,
  fov: number,
  aspect: number,
  centerX: number,
  centerY: number,
  width: number,
  height: number,
): [number, number, number] {
  const x1 = px;
  const y1 = py * cosP - pz * sinP;
  const z1 = py * sinP + pz * cosP;
  const x2 = x1 * cosY - z1 * sinY;
  const z2 = x1 * sinY + z1 * cosY;
  const y2 = y1;
  const x3 = x2 * cosR - y2 * sinR;
  const y3 = x2 * sinR + y2 * cosR;
  const zView = Math.max(0.32, z2 + camDist);
  const sx = (centerX + (x3 / zView) * fov / aspect) * width;
  const sy = (centerY + (y3 / zView) * fov) * height;
  return [sx, sy, zView];
}

/**
 * 3D Perspective Modeler for Planet Venus (approaches at 0.52 -> 0.72 AU corridor ->
 * brief center flyby -> smoothly dissolves and disperses by p = 0.86 as Milky Way opens).
 */
function drawVenus(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  progress: number,
  time: number,
) {
  if (progress < 0.18 || progress >= 0.865) return;

  const {
    isDesktop,
    centerX,
    centerY,
    perspectiveScale,
    orbitCamDist,
  } = getVenusTrajectory(progress, width, height);

  const vx = centerX * width;
  const vy = centerY * height;
  const hasRadial = typeof context.createRadialGradient === 'function';
  if (!hasRadial) return;

  // Phase A (0.18 -> 0.58): Ultra-bright distant Morning Star pinpoint beacon before planetary resolution
  const starAlpha = smoothstep(0.18, 0.28, progress) * (1 - smoothstep(0.51, 0.60, progress));
  if (starAlpha > 0.01) {
    const twinkle = 0.90 + 0.10 * Math.sin(time * 0.0042 + progress * 12.0);
    const sA = starAlpha * twinkle;
    const haloR = (isDesktop ? 26 : 20) * (0.75 + 0.45 * smoothstep(0.24, 0.52, progress));
    const starGlow = context.createRadialGradient(vx, vy, 0, vx, vy, haloR);
    starGlow.addColorStop(0, `rgba(255, 252, 244, ${sA * 0.98})`);
    starGlow.addColorStop(0.14, `rgba(255, 232, 198, ${sA * 0.72})`);
    starGlow.addColorStop(0.45, `rgba(221, 161, 116, ${sA * 0.24})`);
    starGlow.addColorStop(1, 'rgba(6, 10, 17, 0)');
    context.fillStyle = starGlow;
    context.beginPath();
    context.arc(vx, vy, haloR, 0, Math.PI * 2);
    context.fill();

    // Razor-crisp 4-point diffraction spikes
    const spikeLen = haloR * 1.15;
    context.save();
    context.strokeStyle = `rgba(255, 246, 232, ${sA * 0.78})`;
    context.lineWidth = 1.0;
    context.beginPath();
    context.moveTo(vx - spikeLen, vy);
    context.lineTo(vx + spikeLen, vy);
    context.moveTo(vx, vy - spikeLen);
    context.lineTo(vx, vy + spikeLen);
    context.stroke();
    context.fillStyle = `rgba(255, 255, 255, ${sA})`;
    context.beginPath();
    context.arc(vx, vy, 1.8, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  if (progress < 0.50) return;

  const venusFadeOut = 1 - smoothstep(0.79, 0.86, progress);
  const venusAlpha = smoothstep(0.51, 0.60, progress) * venusFadeOut;
  if (venusAlpha <= 0.005) return;

  const dissolveExpand = 1 + smoothstep(0.79, 0.86, progress) * 0.45;
  const baseRadius = Math.min(width, height) * (isDesktop ? 0.20 : 0.17);
  const vr = Math.max(4.0, baseRadius * perspectiveScale * (1.05 / orbitCamDist) * dissolveExpand);

  const hasLinear = typeof context.createLinearGradient === 'function';
  const hasEllipse = typeof context.ellipse === 'function';

  const resolveT = smoothstep(10, 24, vr);
  const beaconWeight = 1 - resolveT;
  const globeWeight = resolveT;

  if (beaconWeight > 0.01) {
    const bAlpha = venusAlpha * beaconWeight;
    const beaconAureole = context.createRadialGradient(vx, vy, 0, vx, vy, vr * 4.2);
    beaconAureole.addColorStop(0, `rgba(255, 224, 182, ${bAlpha * 0.65})`);
    beaconAureole.addColorStop(0.35, `rgba(221, 161, 116, ${bAlpha * 0.35})`);
    beaconAureole.addColorStop(1, 'rgba(6, 10, 17, 0)');
    context.fillStyle = beaconAureole;
    context.beginPath();
    context.arc(vx, vy, vr * 4.2, 0, Math.PI * 2);
    context.fill();

    const beaconCore = context.createRadialGradient(vx - vr * 0.3, vy - vr * 0.25, 0, vx, vy, vr);
    beaconCore.addColorStop(0, `rgba(255, 248, 238, ${bAlpha * 0.98})`);
    beaconCore.addColorStop(0.42, `rgba(221, 161, 116, ${bAlpha * 0.94})`);
    beaconCore.addColorStop(0.82, `rgba(163, 94, 60, ${bAlpha * 0.88})`);
    beaconCore.addColorStop(1, `rgba(32, 18, 14, ${bAlpha * 0.85})`);
    context.fillStyle = beaconCore;
    context.beginPath();
    context.arc(vx, vy, vr, 0, Math.PI * 2);
    context.fill();
  }

  if (globeWeight <= 0.01) return;

  const gAlpha = venusAlpha * globeWeight;
  const aureoleG = context.createRadialGradient(vx - vr * 0.25, vy - vr * 0.20, vr * 0.45, vx, vy, vr * 1.75);
  aureoleG.addColorStop(0, `rgba(255, 224, 184, ${gAlpha * 0.36})`);
  aureoleG.addColorStop(0.38, `rgba(221, 161, 116, ${gAlpha * 0.18})`);
  aureoleG.addColorStop(1, 'rgba(6, 10, 17, 0)');
  context.fillStyle = aureoleG;
  context.beginPath();
  context.arc(vx, vy, vr * 1.75, 0, Math.PI * 2);
  context.fill();

  const globeG = context.createRadialGradient(
    vx - vr * 0.36, vy - vr * 0.28, vr * 0.05,
    vx + vr * 0.22, vy + vr * 0.16, vr * 1.25,
  );
  globeG.addColorStop(0, `rgba(255, 244, 226, ${gAlpha * 0.99})`);
  globeG.addColorStop(0.24, `rgba(244, 206, 168, ${gAlpha * 0.98})`);
  globeG.addColorStop(0.50, `rgba(221, 161, 116, ${gAlpha * 0.96})`);
  globeG.addColorStop(0.76, `rgba(148, 88, 58, ${gAlpha * 0.95})`);
  globeG.addColorStop(1, `rgba(16, 10, 8, ${gAlpha * 0.99})`);
  context.save();
  context.beginPath();
  context.arc(vx, vy, vr, 0, Math.PI * 2);
  context.clip();
  context.fillStyle = globeG;
  context.fillRect(vx - vr, vy - vr, vr * 2, vr * 2);

  if (hasLinear && hasEllipse) {
    const bandRotation = -time * 0.00024 + progress * 1.4;
    const bandCount = 6;
    for (let b = 0; b < bandCount; b += 1) {
      const latNorm = (b + 0.5) / bandCount - 0.5;
      const by = vy + latNorm * vr * 1.48;
      const bh = vr * (0.22 + Math.cos(latNorm * Math.PI) * 0.08);
      const bandG = context.createLinearGradient(vx, by - bh, vx, by + bh);
      const bandColor = b % 2 === 0 ? '255, 228, 192' : '178, 102, 68';
      const bandAlpha = (b % 3 === 0 ? 0.30 : 0.20) * gAlpha;
      bandG.addColorStop(0, `rgba(${bandColor}, 0)`);
      bandG.addColorStop(0.5, `rgba(${bandColor}, ${bandAlpha})`);
      bandG.addColorStop(1, `rgba(${bandColor}, 0)`);
      context.fillStyle = bandG;
      context.beginPath();
      const bandTilt = -0.06 + Math.sin(bandRotation + b * 0.8) * 0.02;
      context.ellipse(vx, by, vr * 1.18, bh, bandTilt, 0, Math.PI * 2);
      context.fill();
    }
  }

  const termG = context.createLinearGradient(vx - vr * 0.40, vy - vr * 0.30, vx + vr * 0.72, vy + vr * 0.58);
  termG.addColorStop(0, 'rgba(6, 10, 17, 0)');
  termG.addColorStop(0.48, `rgba(6, 10, 17, ${gAlpha * 0.18})`);
  termG.addColorStop(0.78, `rgba(6, 10, 17, ${gAlpha * 0.88})`);
  termG.addColorStop(1, `rgba(6, 10, 17, ${gAlpha * 0.98})`);
  context.fillStyle = termG;
  context.fillRect(vx - vr, vy - vr, vr * 2, vr * 2);
  context.restore();

  context.save();
  context.strokeStyle = `rgba(255, 236, 204, ${gAlpha * 0.86})`;
  context.lineWidth = Math.max(1.4, 2.6 * (vr / baseRadius));
  context.beginPath();
  context.arc(vx, vy, vr - 0.6, Math.PI * 0.70, Math.PI * 1.78);
  context.stroke();
  context.restore();
}

function createCanvasRenderer(canvas: HTMLCanvasElement, quality: ParticleQuality): Renderer | null {
  const context = canvas.getContext('2d');
  if (!context) return null;
  const config = QUALITY[quality];
  const frames = keyframesFor(config.count);
  const drift = makeDrift(config.count);
  const celestialCount = 3600;
  const { galaxy, heroField, milkyway, venus, meta, nebula } = makeCelestialStates3D(celestialCount);
  const warpStarCount = 260;
  const warpStars = Array.from({ length: warpStarCount }, (_, i) => ({
    x: (noise(i, 80) - 0.5) * 2.2,
    y: (noise(i, 81) - 0.5) * 1.5,
    z: noise(i, 82) * 3.5 + 0.2,
    baseAlpha: 0.32 + noise(i, 83) * 0.62,
    colorRgb: (i % 5 === 0) ? '255, 240, 218' : (i % 7 === 0) ? '142, 173, 222' : '221, 161, 116',
  }));
  let width = 1;
  let height = 1;
  return {
    name: 'canvas2d',
    resize(nextWidth, nextHeight, ratio) {
      width = nextWidth;
      height = nextHeight;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    },
    draw(value, pointerX, pointerY, time, scrollImpulse, travel, warpSpeed) {
      const segment = segmentFor(value);
      const previous = frames[segment - 1];
      const next = frames[segment];
      const raw = (value - previous.at) / Math.max(.001, next.at - previous.at);
      const amount = raw * raw * (3 - 2 * raw);
      const earlyTide = smoothstep(.04, .16, value) * (1 - smoothstep(.18, .3, value));
      context.clearRect(0, 0, width, height);

      const hasRadial = typeof context.createRadialGradient === 'function';
      const hasEllipse = typeof context.ellipse === 'function';

      const heroWeight = 1 - smoothstep(0.10, 0.22, value);
      const mwStageAlpha = smoothstep(0.16, 0.28, value) * (1 - smoothstep(0.54, 0.72, value));
      const mwRampIn = smoothstep(0.14, 0.26, value);
      const mwPersistFade = 1 - smoothstep(0.54, 0.72, value);
      const venusBodyMwFade = 1 - smoothstep(0.45, 0.52, value);

      const venusFadeOut = 1 - smoothstep(0.79, 0.86, value);
      const isVenusPhase = value >= 0.52 && value < 0.87;
      const venusAppear = smoothstep(0.52, 0.60, value) * venusFadeOut;
      const outerDustAppear = smoothstep(0.70, 0.79, value) * venusFadeOut;
      const venusDissolve = smoothstep(0.79, 0.86, value);

      const finaleMwAppear = smoothstep(0.80, 0.865, value);
      const ctaCoreSoften = 1 - 0.22 * smoothstep(0.88, 0.96, value);

      const venusTraj = getVenusTrajectory(value, width, height);

      const aspect = width / Math.max(1, height);
      const fov = 1.32;
      const ptrX = pointerX < -1 ? 0.5 : pointerX;
      const ptrY = pointerY < -1 ? 0.5 : pointerY;
      const ptrDX = ptrX - 0.5;
      const ptrDY = ptrY - 0.5;
      const yaw = ptrDX * 0.14;
      const pitchOffset = ptrDY * 0.10;
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);

      // 0. 3D Warp Starfield & Streaming
      const activeWarpSpeed = warpSpeed ?? (Math.abs(scrollImpulse) * 2.8 + 0.14);
      const isWarping = (warpSpeed !== undefined ? warpSpeed > 0.05 : Math.abs(scrollImpulse) > 0.06) && finaleMwAppear < 0.5;
      const flightZ = travel ?? (value * 18.0 + (time * 0.00038));
      const mwStageT = smoothstep(0.12, 0.30, value);
      const mwCenterX = 0.60 + (0.54 - 0.60) * mwStageT;
      const mwCenterY = 0.48 + (0.49 - 0.48) * mwStageT;
      const warpFocusBlend = smoothstep(0.52, 0.74, value);
      let focusX = mwCenterX + (venusTraj.centerX - mwCenterX) * warpFocusBlend;
      let focusY = mwCenterY + (venusTraj.centerY - mwCenterY) * warpFocusBlend;
      focusX = (focusX + (0.50 - focusX) * finaleMwAppear) * width;
      focusY = (focusY + (0.48 - focusY) * finaleMwAppear) * height;

      context.save();
      for (let s = 0; s < warpStarCount; s += 1) {
        const star = warpStars[s];
        const sz = ((star.z - flightZ) % 3.4 + 3.4) % 3.4 + 0.12;
        const pScale = 1.0 / sz;
        const sx = focusX + star.x * width * 0.52 * pScale;
        const sy = focusY + star.y * height * 0.52 * pScale;
        if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) continue;

        const starAlpha = Math.min(1.0, (1.0 - sz / 3.6)) * star.baseAlpha * (1 - heroWeight * 0.48) * (1 - finaleMwAppear * 0.45);
        if (isWarping && starAlpha > 0.10) {
          const angle = Math.atan2(sy - focusY, sx - focusX);
          const streakLen = Math.min(58, (activeWarpSpeed / (sz * sz)) * 15 + 3);
          context.strokeStyle = `rgba(${star.colorRgb}, ${starAlpha * 0.86})`;
          context.lineWidth = Math.min(2.4, 0.8 * pScale);
          context.beginPath();
          context.moveTo(sx - Math.cos(angle) * streakLen, sy - Math.sin(angle) * streakLen);
          context.lineTo(sx, sy);
          context.stroke();
        } else {
          const ptR = Math.min(2.2, 0.9 * pScale);
          context.fillStyle = `rgba(${star.colorRgb}, ${starAlpha})`;
          context.beginPath();
          context.arc(sx, sy, ptR, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.restore();

      // 1. Mid-story Diagonal Milky Way Star Cloud & Dust Rift [0.16, 0.72]
      if (mwStageAlpha > 0.01 && hasRadial) {
        const mwSpine = [
          { x: 0.71 * width, y: 0.14 * height, r: 0.24 * width, a: 0.10 },
          { x: 0.65 * width, y: 0.32 * height, r: 0.27 * width, a: 0.12 },
          { x: 0.59 * width, y: 0.50 * height, r: 0.28 * width, a: 0.13 },
          { x: 0.54 * width, y: 0.68 * height, r: 0.25 * width, a: 0.11 },
          { x: 0.49 * width, y: 0.86 * height, r: 0.22 * width, a: 0.08 },
        ];
        for (const node of mwSpine) {
          const g = context.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r);
          g.addColorStop(0, `rgba(221, 161, 116, ${mwStageAlpha * node.a})`);
          g.addColorStop(0.48, `rgba(191, 126, 79, ${mwStageAlpha * node.a * 0.40})`);
          g.addColorStop(1, 'rgba(6, 10, 17, 0)');
          context.fillStyle = g;
          context.beginPath();
          context.arc(node.x, node.y, node.r, 0, Math.PI * 2);
          context.fill();
        }
      }

      // 2. Planet Venus Modeler (approaches 0.52 -> dissolves cleanly by 0.86)
      drawVenus(context, width, height, value, time);

      // 3. Finale Milky Way 3D Perspective Camera & Soft Galactic Disc/Bar Glow
      const finalePitch = Math.max(0.28, Math.min(1.28, 0.76 - ptrDY * 0.96));
      const finaleYaw = ptrDX * 1.16;
      const finaleRoll = -0.16 + ptrDX * 0.24;
      const finaleCamDist = 1.50;
      const finaleFov = 1.32;
      const cosFY = Math.cos(finaleYaw);
      const sinFY = Math.sin(finaleYaw);
      const cosFP = Math.cos(finalePitch);
      const sinFP = Math.sin(finalePitch);
      const cosFR = Math.cos(finaleRoll);
      const sinFR = Math.sin(finaleRoll);
      const finaleScale = (0.68 + 0.32 * finaleMwAppear) * (aspect < 1.0 ? 0.68 : 0.88);

      if (finaleMwAppear > 0.01 && hasRadial && hasEllipse) {
        const [bx1, by1] = projectGalaxyFinale3D(
          -MW_BAR_HALF_LEN * Math.cos(MW_BAR_ANGLE) * finaleScale,
          0,
          -MW_BAR_HALF_LEN * Math.sin(MW_BAR_ANGLE) * finaleScale,
          cosFP, sinFP, cosFY, sinFY, cosFR, sinFR, finaleCamDist, finaleFov, aspect, 0.50, 0.48, width, height,
        );
        const [bx2, by2] = projectGalaxyFinale3D(
          MW_BAR_HALF_LEN * Math.cos(MW_BAR_ANGLE) * finaleScale,
          0,
          MW_BAR_HALF_LEN * Math.sin(MW_BAR_ANGLE) * finaleScale,
          cosFP, sinFP, cosFY, sinFY, cosFR, sinFR, finaleCamDist, finaleFov, aspect, 0.50, 0.48, width, height,
        );
        const cx = (bx1 + bx2) * 0.5;
        const cy = (by1 + by2) * 0.5;
        const barLenPx = Math.max(40, Math.hypot(bx2 - bx1, by2 - by1));
        const barScreenAngle = Math.atan2(by2 - by1, bx2 - bx1);
        const discRadius = Math.min(width, height) * 0.44 * finaleScale;
        const tiltAspect = Math.max(0.22, Math.abs(Math.sin(finalePitch)));

        // Outer cool galactic silver-blue spiral halo tilted in 3D perspective
        context.save();
        context.translate(cx, cy);
        context.rotate(finaleRoll);
        context.scale(1, tiltAspect);
        const haloG = context.createRadialGradient(0, 0, discRadius * 0.14, 0, 0, discRadius);
        haloG.addColorStop(0, `rgba(221, 161, 116, ${finaleMwAppear * 0.20 * ctaCoreSoften})`);
        haloG.addColorStop(0.32, `rgba(142, 173, 222, ${finaleMwAppear * 0.24 * ctaCoreSoften})`);
        haloG.addColorStop(0.72, `rgba(116, 155, 220, ${finaleMwAppear * 0.10 * ctaCoreSoften})`);
        haloG.addColorStop(1, 'rgba(6, 10, 17, 0)');
        context.fillStyle = haloG;
        context.beginPath();
        context.arc(0, 0, discRadius, 0, Math.PI * 2);
        context.fill();
        context.restore();

        // Warm luminous copper-amber central bar aligned with 3D projected bar vector
        context.save();
        context.translate(cx, cy);
        context.rotate(barScreenAngle);
        context.scale(1.35, 0.52);
        const barR = barLenPx * 0.72;
        const barG = context.createRadialGradient(0, 0, 0, 0, 0, barR);
        barG.addColorStop(0, `rgba(255, 243, 226, ${finaleMwAppear * 0.54 * ctaCoreSoften})`);
        barG.addColorStop(0.35, `rgba(221, 161, 116, ${finaleMwAppear * 0.36 * ctaCoreSoften})`);
        barG.addColorStop(1, 'rgba(6, 10, 17, 0)');
        context.fillStyle = barG;
        context.beginPath();
        context.arc(0, 0, barR, 0, Math.PI * 2);
        context.fill();
        context.restore();
      }

      // 4. 3D-projected particles
      const mwPitch = 0.26 + (0.12 - 0.26) * smoothstep(0.16, 0.34, value) + pitchOffset;
      const mwRoll = -0.14 * (1 - smoothstep(0.16, 0.38, value));
      const mwDist = 1.42 + (1.12 - 1.42) * smoothstep(0.0, 0.25, value);
      const cosMwP = Math.cos(mwPitch);
      const sinMwP = Math.sin(mwPitch);
      const cosMwR = Math.cos(mwRoll);
      const sinMwR = Math.sin(mwRoll);

      const vPitch = 0.08 + pitchOffset;
      const cosVP = Math.cos(vPitch);
      const sinVP = Math.sin(vPitch);
      const venusSpin = -time * 0.00028 + value * 1.8;
      const cosV = Math.cos(venusSpin);
      const sinV = Math.sin(venusSpin);
      const cosVTilt = Math.cos(0.22);
      const sinVTilt = Math.sin(0.22);
      const cVScale = venusTraj.perspectiveScale * venusTraj.aspectScale;

      const mwBuckets: [number, number, number][][] = [[], [], [], []];
      const venusBuckets: [number, number, number][][] = [[], [], []];
      const venusDustPts: [number, number, number][] = [];
      // Finale buckets: 0: core white-cream, 1: copper bar, 2: blue-silver arms, 3: rose-magenta nebula, 4: dark dust
      const finaleBuckets: [number, number, number][][] = [[], [], [], [], []];

      for (let i = 0; i < celestialCount; i += 1) {
        const gOff = i * 3;
        const mOff = i * 4;
        const nOff = i * 2;
        const sizeMeta = meta[mOff];
        const tempMeta = meta[mOff + 1];
        const dustMeta = meta[mOff + 2];
        const roleMeta = meta[mOff + 3];
        const nebulaWeight = nebula[nOff];

        // Finale Barred Spiral Milky Way rendering (0.80 -> 1.00)
        if (finaleMwAppear > 0.01) {
          const gx = galaxy[gOff];
          const gy = galaxy[gOff + 1];
          const gz = galaxy[gOff + 2];
          const rGal = Math.hypot(gx, gz);
          const omega = 0.000105 / Math.sqrt(rGal * rGal + 0.06);
          const rotAngle = time * omega;
          const cRot = Math.cos(rotAngle);
          const sRot = Math.sin(rotAngle);
          const rx = (gx * cRot - gz * sRot) * finaleScale;
          const ry = gy * finaleScale;
          const rz = (gx * sRot + gz * cRot) * finaleScale;

          const [sx, sy, zView] = projectGalaxyFinale3D(
            rx, ry, rz,
            cosFP, sinFP, cosFY, sinFY, cosFR, sinFR,
            finaleCamDist, finaleFov, aspect, 0.50, 0.48, width, height,
          );
          if (sx >= -12 && sx <= width + 12 && sy >= -12 && sy <= height + 12) {
            const pRadius = Math.max(0.5, Math.min(2.4, sizeMeta * (1.18 / zView)));
            if (dustMeta > 0.50) {
              finaleBuckets[4].push([sx, sy, pRadius * 0.95]);
            } else if (nebulaWeight > 0.35) {
              finaleBuckets[3].push([sx, sy, pRadius * 1.15]);
            } else if (tempMeta > 0.80) {
              finaleBuckets[0].push([sx, sy, pRadius * 1.12]);
            } else if (tempMeta > 0.52) {
              finaleBuckets[1].push([sx, sy, pRadius]);
            } else {
              finaleBuckets[2].push([sx, sy, pRadius]);
            }
          }
        }

        // Venus phase (0.52 -> 0.86)
        const isVenusRole = roleMeta > 1.5 && roleMeta < 2.5;
        const isOuterDust = roleMeta > 0.5 && roleMeta < 1.5;

        if (isVenusPhase && venusFadeOut > 0.01 && (isVenusRole || (isOuterDust && outerDustAppear > 0.02))) {
          const vx0 = venus[gOff];
          const vy0 = venus[gOff + 1];
          const vz0 = venus[gOff + 2];
          const vrx = vx0 * cosV - vz0 * sinV;
          const vrz = vx0 * sinV + vz0 * cosV;
          let venX = vrx * cVScale;
          const venY = (vy0 * cosVTilt - vrz * sinVTilt) * cVScale;
          let venZ = (vy0 * sinVTilt + vrz * cosVTilt) * cVScale;

          if (venusDissolve > 0.001) {
            const swirl = venusDissolve * (1.45 + sizeMeta * 0.9);
            const cSw = Math.cos(swirl);
            const sSw = Math.sin(swirl);
            const expand = 1.0 + venusDissolve * 2.5;
            const nx = (venX * cSw - venZ * sSw) * expand;
            const nz = (venX * sSw + venZ * cSw) * expand;
            venX = nx;
            venZ = nz;
          }

          if (roleMeta > 1.5 && roleMeta < 2.5 && venZ < -0.04 * cVScale && venusDissolve < 0.2) continue;

          const [sx, sy, zView] = projectCamera3D(
            venX, venY, -venZ,
            cosY, sinY, cosVP, sinVP, 1, 0,
            venusTraj.orbitCamDist, fov, aspect, venusTraj.centerX, venusTraj.centerY, width, height,
          );
          if (sx < -8 || sx > width + 8 || sy < -8 || sy > height + 8) continue;

          const pRadius = Math.max(0.6, Math.min(2.4, sizeMeta * (1.25 / zView)));
          if (isOuterDust) {
            venusDustPts.push([sx, sy, pRadius * 0.9]);
          } else if (tempMeta > 0.72) {
            venusBuckets[0].push([sx, sy, pRadius]);
          } else if (tempMeta < 0.35) {
            venusBuckets[2].push([sx, sy, pRadius]);
          } else {
            venusBuckets[1].push([sx, sy, pRadius]);
          }
          continue;
        }

        // Hero calm field & Mid-story Diagonal Milky Way
        const activeMwAlpha = isVenusRole ? venusBodyMwFade : mwPersistFade;
        if (heroWeight <= 0.02 && activeMwAlpha <= 0.02) continue;

        const heroRot = time * 0.000035;
        const cHR = Math.cos(heroRot);
        const sHR = Math.sin(heroRot);
        const hx = heroField[gOff] * cHR - heroField[gOff + 2] * sHR;
        const hy = heroField[gOff + 1];
        const hz = heroField[gOff] * sHR + heroField[gOff + 2] * cHR;

        const mwx = milkyway[gOff];
        const mwy = milkyway[gOff + 1];
        const mwz = milkyway[gOff + 2];
        const wx = hx + (mwx - hx) * mwRampIn;
        const wy = hy + (mwy - hy) * mwRampIn;
        const wz = hz + (mwz - hz) * mwRampIn;

        const [sx, sy, zView] = projectCamera3D(
          wx, wy, wz,
          cosY, sinY, cosMwP, sinMwP, cosMwR, sinMwR,
          mwDist, fov, aspect, mwCenterX, mwCenterY, width, height,
        );
        if (sx < -8 || sx > width + 8 || sy < -8 || sy > height + 8) continue;

        const fadeFactor = Math.max(heroWeight * 0.55, activeMwAlpha);
        const pRadius = Math.max(0.45, Math.min(1.65, sizeMeta * (1.02 / zView) * Math.sqrt(fadeFactor)));
        if (dustMeta > 0.45 && activeMwAlpha > 0.2) {
          mwBuckets[3].push([sx, sy, pRadius * 0.85]);
        } else if (tempMeta > 0.76) {
          mwBuckets[0].push([sx, sy, pRadius]);
        } else if (tempMeta < 0.38) {
          mwBuckets[2].push([sx, sy, pRadius]);
        } else {
          mwBuckets[1].push([sx, sy, pRadius]);
        }
      }

      // Draw Hero & Mid-story Milky Way buckets
      const mwAlphaScale = Math.max(heroWeight * 0.50, mwPersistFade) * config.canvasAlpha;
      if (mwAlphaScale > 0.01) {
        const mwStyles = [
          `rgba(255, 243, 226, ${mwAlphaScale * 0.88})`,
          `rgba(221, 161, 116, ${mwAlphaScale * 0.78})`,
          `rgba(142, 173, 222, ${mwAlphaScale * 0.76})`,
          `rgba(8, 12, 20, ${0.18 * mwPersistFade})`,
        ];
        for (let b = 0; b < 4; b += 1) {
          const pts = mwBuckets[b];
          if (pts.length === 0) continue;
          context.fillStyle = mwStyles[b];
          context.beginPath();
          for (let k = 0; k < pts.length; k += 1) {
            const [px, py, pr] = pts[k];
            context.moveTo(px + pr, py);
            context.arc(px, py, pr, 0, Math.PI * 2);
          }
          context.fill();
        }
      }

      // Draw Venus particles (dissolving by 0.86)
      if (isVenusPhase && venusAppear > 0.01) {
        const vAlpha = venusAppear * config.canvasAlpha;
        const venusStyles = [
          `rgba(255, 243, 226, ${vAlpha * 0.72})`,
          `rgba(221, 161, 116, ${vAlpha * 0.64})`,
          `rgba(142, 173, 222, ${vAlpha * 0.56})`,
        ];
        for (let b = 0; b < 3; b += 1) {
          const pts = venusBuckets[b];
          if (pts.length === 0) continue;
          context.fillStyle = venusStyles[b];
          context.beginPath();
          for (let k = 0; k < pts.length; k += 1) {
            const [px, py, pr] = pts[k];
            context.moveTo(px + pr, py);
            context.arc(px, py, pr, 0, Math.PI * 2);
          }
          context.fill();
        }
        if (venusDustPts.length > 0 && outerDustAppear > 0.01) {
          context.fillStyle = `rgba(255, 224, 184, ${outerDustAppear * config.canvasAlpha * 0.28})`;
          context.beginPath();
          for (let k = 0; k < venusDustPts.length; k += 1) {
            const [px, py, pr] = venusDustPts[k];
            context.moveTo(px + pr, py);
            context.arc(px, py, pr, 0, Math.PI * 2);
          }
          context.fill();
        }
      }

      // Draw Finale Barred Spiral Milky Way buckets in back-to-front depth order:
      // blue-silver outer arms -> dark dust lanes -> delicate rose-magenta knots -> copper bar -> bright core
      if (finaleMwAppear > 0.01) {
        const fAlpha = finaleMwAppear * config.canvasAlpha * (1 - 0.12 * smoothstep(0.88, 0.96, value));
        const orderAndStyle: Array<[number, string]> = [
          [2, `rgba(142, 173, 222, ${fAlpha * 0.90})`],
          [4, `rgba(4, 7, 14, ${finaleMwAppear * 0.22})`],
          [3, `rgba(219, 130, 150, ${fAlpha * 0.54})`],
          [1, `rgba(221, 161, 116, ${fAlpha * 0.90})`],
          [0, `rgba(255, 243, 226, ${fAlpha * 0.96 * ctaCoreSoften})`],
        ];
        for (const [bucketIdx, style] of orderAndStyle) {
          const pts = finaleBuckets[bucketIdx];
          if (pts.length === 0) continue;
          context.fillStyle = style;
          context.beginPath();
          for (let k = 0; k < pts.length; k += 1) {
            const [px, py, pr] = pts[k];
            context.moveTo(px + pr, py);
            context.arc(px, py, pr, 0, Math.PI * 2);
          }
          context.fill();
        }
      }

      // 5. Foreground keyframe constellation particles (`config.count`), releasing by p = 0.78
      const storyLayerAlpha = (1 - smoothstep(0.70, 0.78, value)) * config.canvasAlpha;
      if (storyLayerAlpha <= 0.01) return;
      context.fillStyle = `rgba(221, 161, 116, ${storyLayerAlpha})`;
      context.beginPath();

      const reach = Math.min(width, height) * .095;
      const scatterBlend = Math.max(0, Math.min(1, (value - .2) / .18));
      for (let index = 0; index < config.count; index += 1) {
        const offset = index * 2;
        let x = previous.points[offset] + (next.points[offset] - previous.points[offset]) * amount;
        let y = previous.points[offset + 1] + (next.points[offset + 1] - previous.points[offset + 1]) * amount;
        const cTilt = smoothstep(.02, .09, value) * (1 - smoothstep(.17, .24, value));
        const cAngle = .58 * cTilt;
        const cLocalX = x - .60;
        const cLocalY = y - .48;
        const cDepth = cLocalY * Math.sin(cAngle);
        const cPerspective = 1 / Math.max(.78, 1 + cDepth * 1.4);
        x = (.60 + (cLocalX + cDepth * .18) * cPerspective) * width;
        y = (.48 + cLocalY * Math.cos(cAngle) * cPerspective + .018 * cTilt) * height;
        const dx = x - pointerX * width;
        const dy = y - pointerY * height;
        const squared = dx * dx + dy * dy;
        if (squared > 0 && squared < reach * reach) {
          const distance = Math.sqrt(squared);
          const force = (1 - distance / reach) * 18;
          x += dx / distance * force;
          y += dy / distance * force;
        }
        const radius = (index % 11 === 0 ? config.largeRadius : config.radius) * (1 + (cPerspective - 1) * cTilt);
        const driftOffset = index * 4;
        const earlyTideX = Math.sin(drift[driftOffset] * 1.8 + drift[driftOffset + 1] * .65 + time * .0009 + value * 10) * earlyTide * .012 * width;
        const earlyTideY = Math.cos(drift[driftOffset + 1] * 1.2 + drift[driftOffset] * .45 + time * .0007 + value * 7) * earlyTide * .0035 * height;
        const driftPhaseX = drift[driftOffset] + time * drift[driftOffset + 2];
        const driftPhaseY = drift[driftOffset + 1] + time * drift[driftOffset + 2] * .73;
        const driftX = Math.sin(driftPhaseX) * drift[driftOffset + 3] * width;
        const driftY = Math.cos(driftPhaseY) * drift[driftOffset + 3] * .68 * height;
        const directionPhase = drift[driftOffset] * 1.7 + drift[driftOffset + 1] * .45;
        const impulseSeed = Math.sin(drift[driftOffset] * 12.9898 + drift[driftOffset + 1] * 78.233) * 43758.5453;
        const individualImpulse = .35 + (impulseSeed - Math.floor(impulseSeed)) * .65;
        const impulseX = Math.cos(directionPhase) * scrollImpulse * scatterBlend * individualImpulse * width;
        const impulseY = Math.sin(drift[driftOffset + 1] * 1.3 + drift[driftOffset]) * scrollImpulse * scatterBlend * individualImpulse * height;
        context.moveTo(x + radius + earlyTideX + driftX + impulseX, y + earlyTideY + driftY + impulseY);
        context.arc(x + earlyTideX + driftX + impulseX, y + earlyTideY + driftY + impulseY, radius, 0, Math.PI * 2);
      }
      context.fill();
    },
    dispose() {},
  };
}

export function ParticleWorld({ progress, releaseProgress, quality = 'high' }: { progress: MotionValue<number>; releaseProgress?: MotionValue<number>; quality?: ParticleQuality }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const fallbackCanvas = fallbackCanvasRef.current;
    if (!canvas || !fallbackCanvas) return;
    let activeQuality = quality;
    let activeCanvas = canvas;
    let renderer: Renderer | null = null;
    try {
      renderer = createWebglRenderer(canvas, quality);
    } catch (error) {
      releaseWebglContext(canvas);
      console.warn('[ParticleWorld] WebGL renderer unavailable; using the Canvas2D fallback.', error);
    }
    if (!renderer) {
      releaseWebglContext(canvas);
      activeQuality = 'balanced';
      activeCanvas = fallbackCanvas;
      renderer = createCanvasRenderer(activeCanvas, activeQuality);
    }
    if (!renderer) return;
    const config = QUALITY[activeQuality];
    canvas.style.display = renderer.name === 'webgl2' ? 'block' : 'none';
    fallbackCanvas.style.display = renderer.name === 'canvas2d' ? 'block' : 'none';
    activeCanvas.dataset.renderer = renderer.name;
    activeCanvas.dataset.particleCount = String(config.count);
    activeCanvas.dataset.quality = activeQuality;
    let frame = 0;
    let visible = true;
    let width = 1;
    let height = 1;
    let targetPointerX = 0.5;
    let targetPointerY = 0.5;
    let smoothPointerX = 0.5;
    let smoothPointerY = 0.5;
    let pointerActive = false;
    let elapsed = 0;
    let previousTimestamp = performance.now();
    let targetProgress = progress.get();
    let particleProgress = targetProgress;
    let previousParticleProgress = particleProgress;
    let scrollImpulse = 0;
    let travelDistance = 0;
    let warpVelocity = 0;

    const draw = (timestamp: number) => {
      frame = 0;
      if (!visible || document.hidden) return;
      const deltaTime = Math.min(50, Math.max(0, timestamp - previousTimestamp));
      previousTimestamp = timestamp;
      const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches ?? false;
      if (!prefersReducedMotion) {
        elapsed += Math.min(32, deltaTime);
      }

      const scrollProgress = Math.max(0, Math.min(1, progress.get()));
      const release = releaseProgress?.get() ?? 0;
      targetProgress = getCinematicParticleProgress(scrollProgress, release);
      const response = prefersReducedMotion ? 1 : 1 - Math.exp(-deltaTime / PARTICLE_RESPONSE_MS);
      particleProgress += (targetProgress - particleProgress) * response;
      if (Math.abs(targetProgress - particleProgress) < .0001) particleProgress = targetProgress;
      const delta = prefersReducedMotion ? 0 : Math.max(-.08, Math.min(.08, particleProgress - previousParticleProgress));
      scrollImpulse = prefersReducedMotion
        ? 0
        : scrollImpulse * Math.pow(SCROLL_IMPULSE_DAMPING, deltaTime / 16.667) + delta * SCROLL_IMPULSE_GAIN;

      const goalPtrX = prefersReducedMotion || !pointerActive ? 0.5 : targetPointerX;
      const goalPtrY = prefersReducedMotion || !pointerActive ? 0.5 : targetPointerY;
      const ptrLerp = prefersReducedMotion ? 1 : 1 - Math.exp(-deltaTime / 95);
      smoothPointerX += (goalPtrX - smoothPointerX) * ptrLerp;
      smoothPointerY += (goalPtrY - smoothPointerY) * ptrLerp;

      const progressDelta = Math.abs(particleProgress - previousParticleProgress);
      const cruiseSpeed = prefersReducedMotion ? 0 : 0.00038;
      const scrollSpeed = prefersReducedMotion ? 0 : (progressDelta / Math.max(1, deltaTime)) * 1.9 + Math.abs(scrollImpulse) * 0.92;
      const targetWarp = prefersReducedMotion ? 0 : Math.min(1.0, scrollSpeed * 26.0);
      warpVelocity += (targetWarp - warpVelocity) * (prefersReducedMotion ? 1 : 1 - Math.exp(-deltaTime / 120));
      if (!prefersReducedMotion) {
        travelDistance += (cruiseSpeed * deltaTime + (particleProgress - previousParticleProgress) * 4.2 + scrollImpulse * 0.018);
        travelDistance = ((travelDistance % 4650) + 4650) % 4650;
      }

      previousParticleProgress = particleProgress;
      renderer.draw(particleProgress, smoothPointerX, smoothPointerY, elapsed, scrollImpulse, travelDistance, warpVelocity);
      if (!prefersReducedMotion || Math.abs(targetProgress - particleProgress) >= .0001) {
        schedule();
      }
    };
    const schedule = () => { if (!frame && visible && !document.hidden) frame = window.requestAnimationFrame(draw); };
    const resize = () => {
      const bounds = activeCanvas.getBoundingClientRect();
      const nextWidth = Math.max(1, bounds.width);
      const nextHeight = Math.max(1, bounds.height);
      const isMobileViewport = window.matchMedia('(max-width: 1023px)').matches;
      if (width > 1 && Math.abs(nextWidth - width) < 2 && (Math.abs(nextHeight - height) < 2 || isMobileViewport)) return;
      const ratio = Math.min(window.devicePixelRatio || 1, config.dpr);
      width = Math.max(1, bounds.width); height = Math.max(1, bounds.height);
      activeCanvas.width = Math.max(1, Math.round(width * ratio)); activeCanvas.height = Math.max(1, Math.round(height * ratio));
      renderer.resize(width, height, ratio); schedule();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerType === 'touch') return;
      targetPointerX = Math.max(0, Math.min(1, event.clientX / Math.max(1, width)));
      targetPointerY = Math.max(0, Math.min(1, event.clientY / Math.max(1, height)));
      pointerActive = true;
      schedule();
    };
    const onPointerLeave = () => {
      pointerActive = false;
      schedule();
    };
    const onVisibility = () => schedule();
    const resizeObserver = new ResizeObserver(resize);
    const intersectionObserver = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? false; if (visible) schedule(); });
    const unsubscribe = progress.on('change', () => {
      schedule();
    });
    const unsubscribeRelease = releaseProgress?.on('change', () => {
      schedule();
    });
    resizeObserver.observe(activeCanvas);
    intersectionObserver.observe(activeCanvas);
    resize();
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.documentElement.addEventListener('mouseleave', onPointerLeave, { passive: true });
    window.addEventListener('blur', onPointerLeave);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      unsubscribe();
      unsubscribeRelease?.();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      window.removeEventListener('pointermove', onPointerMove);
      document.documentElement.removeEventListener('mouseleave', onPointerLeave);
      window.removeEventListener('blur', onPointerLeave);
      document.removeEventListener('visibilitychange', onVisibility);
      if (frame) window.cancelAnimationFrame(frame);
      renderer.dispose();
    };
  }, [progress, releaseProgress, quality]);

  return (
    <>
      <canvas ref={canvasRef} className="cm-particle-world" aria-hidden="true" />
      <canvas ref={fallbackCanvasRef} className="cm-particle-world cm-particle-world--fallback" aria-hidden="true" />
    </>
  );
}
