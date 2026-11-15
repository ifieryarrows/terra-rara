import { useEffect, useRef } from 'react';

type TransitionPhase = 'scatter' | 'assemble' | 'hold' | 'release' | 'reveal';

type LiveGalaxyState = {
  progress: number;
  pointerX: number;
  pointerY: number;
  elapsed: number;
};

type GalaxyParticlesData = {
  count: number;
  galaxy: Float32Array;
  logo: Float32Array;
  galCol: Float32Array;
  logoCol: Float32Array;
  meta: Float32Array; // size, temp, stagger, radius
  alpha: Float32Array; // galAlpha, logoAlpha, isDust, twinklePhase
};

const LOGO_SIZE = 224;
const MW_BAR_HALF_LEN = 0.24;
const MW_BAR_ANGLE = 0.44;
const FOV = 1.32;
const CAM_DIST_BASE = 1.5;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function clamp01(value: number) {
  return clamp(value, 0, 1);
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = clamp01((value - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function mix(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function cubicPoint(
  p0: readonly [number, number],
  p1: readonly [number, number],
  p2: readonly [number, number],
  p3: readonly [number, number],
  t: number,
): [number, number] {
  const u = 1 - t;
  const tt = t * t;
  const uu = u * u;
  const uuu = uu * u;
  const ttt = tt * t;
  return [
    uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0],
    uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1],
  ];
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  const n = Number.parseInt(clean, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const COPPER_PALETTE = [hexToRgb('#f7dcb8'), hexToRgb('#dda174'), hexToRgb('#b26b3e')];
const SWOOSH_PALETTE = [hexToRgb('#edf4fe'), hexToRgb('#bdd2f1'), hexToRgb('#8eadde'), hexToRgb('#7c9dcf')];

function getBarredGalaxyColor(temp: number, dust: number, nebulaWeight: number): [number, number, number] {
  const blueArm: [number, number, number] = [0.557, 0.678, 0.871];
  const brightOB: [number, number, number] = [0.855, 0.915, 0.996];
  const warmBar: [number, number, number] = [0.867, 0.631, 0.455];
  const brightNucleus: [number, number, number] = [1.0, 0.953, 0.886];
  const h2Rose: [number, number, number] = [0.859, 0.510, 0.588];

  let r = mix(blueArm[0], brightOB[0], smoothstep(0.18, 0.56, temp) * 0.65);
  let g = mix(blueArm[1], brightOB[1], smoothstep(0.18, 0.56, temp) * 0.65);
  let b = mix(blueArm[2], brightOB[2], smoothstep(0.18, 0.56, temp) * 0.65);

  const barW = smoothstep(0.34, 0.78, temp);
  r = mix(r, warmBar[0], barW);
  g = mix(g, warmBar[1], barW);
  b = mix(b, warmBar[2], barW);

  const nucW = smoothstep(0.76, 1.0, temp);
  r = mix(r, brightNucleus[0], nucW);
  g = mix(g, brightNucleus[1], nucW);
  b = mix(b, brightNucleus[2], nucW);

  r = mix(r, h2Rose[0], nebulaWeight * 0.46);
  g = mix(g, h2Rose[1], nebulaWeight * 0.46);
  b = mix(b, h2Rose[2], nebulaWeight * 0.46);

  const dW = dust * 0.54;
  return [mix(r, 0.075, dW), mix(g, 0.088, dW), mix(b, 0.132, dW)];
}

/**
 * Samples BrandMark ring (copper) and orbital swoosh (silver-blue) normalized in [-0.5, +0.5].
 */
function sampleBrandMarkTarget(
  role: 'ring' | 'swoosh',
  u: number,
  jitterA: number,
  jitterB: number,
): { x: number; y: number; col: [number, number, number]; alpha: number } {
  if (role === 'ring') {
    let angle: number;
    let baseRadius: number;
    if (u < 0.58) {
      const t = u / 0.58;
      angle = 2.634 + t * (Math.PI * 2 - 0.060 - 2.634);
      baseRadius = 14.8;
    } else if (u < 0.86) {
      const t = (u - 0.58) / 0.28;
      angle = 0.392 + t * (2.122 - 0.392);
      baseRadius = 14.8;
    } else {
      const t = (u - 0.86) / 0.14;
      angle = 2.66 + t * (Math.PI * 2 - 0.03 - 2.66);
      baseRadius = 12.1;
    }
    const radius = baseRadius + (jitterA - 0.5) * 1.1;
    const svgX = 24 + Math.cos(angle) * radius;
    const svgY = 24 + Math.sin(angle) * radius;
    const colIdx = jitterB < 0.28 ? 0 : jitterB < 0.74 ? 1 : 2;
    return {
      x: (svgX - 24) / 48,
      y: (svgY - 24) / 48,
      col: COPPER_PALETTE[colIdx],
      alpha: 0.72 + jitterB * 0.26,
    };
  }

  let ptTop: [number, number];
  let ptBot: [number, number];
  if (u < 0.72) {
    const t = 0.03 + (u / 0.72) * 0.94;
    ptTop = cubicPoint([2.20, 30.80], [17.03, 33.75], [30.56, 31.83], [45.80, 19.80], t);
    ptBot = cubicPoint([45.80, 19.80], [32.40, 32.44], [16.26, 38.90], [2.20, 30.80], 1 - t);
  } else {
    const t = 0.05 + ((u - 0.72) / 0.28) * 0.90;
    ptTop = cubicPoint([7.40, 33.80], [18.96, 37.63], [30.40, 35.05], [42.60, 24.40], t);
    ptBot = cubicPoint([42.60, 24.40], [30.91, 35.10], [18.72, 39.11], [7.40, 33.80], 1 - t);
  }
  const svgX = mix(ptTop[0], ptBot[0], jitterA);
  const svgY = mix(ptTop[1], ptBot[1], jitterA);
  const colIdx = jitterB < 0.28 ? 0 : jitterB < 0.72 ? 1 : jitterB < 0.9 ? 2 : 3;
  return {
    x: (svgX - 24) / 48,
    y: (svgY - 24) / 48,
    col: SWOOSH_PALETTE[colIdx],
    alpha: 0.74 + jitterB * 0.24,
  };
}

function generateTransitionGalaxy(count: number): GalaxyParticlesData {
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
  const logo = new Float32Array(count * 2);
  const galCol = new Float32Array(count * 3);
  const logoCol = new Float32Array(count * 3);
  const meta = new Float32Array(count * 4);
  const alpha = new Float32Array(count * 4);

  const cosBar = Math.cos(MW_BAR_ANGLE);
  const sinBar = Math.sin(MW_BAR_ANGLE);
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
    const gOff = i * 3;
    const lOff = i * 2;
    const mOff = i * 4;
    const frac = i / count;

    let gx: number;
    let gy: number;
    let gz: number;
    let size: number;
    let temp: number;
    let dust = 0;
    let nebulaWeight = 0;

    if (frac < 0.08) {
      if (frac < 0.014) {
        const cx = -0.94 + gaussian() * 0.052;
        const cz = -0.82 + gaussian() * 0.028;
        gx = cx * cosBar - cz * sinBar;
        gy = 0.04 + gaussian() * 0.018;
        gz = cx * sinBar + cz * cosBar;
        size = 0.72 + 0.45 * rand();
        temp = 0.72 + 0.22 * rand();
      } else {
        const rHalo = 0.35 + 1.38 * Math.pow(rand(), 0.7);
        const cosT = rand() * 2.0 - 1.0;
        const sinT = Math.sqrt(Math.max(0.0, 1.0 - cosT * cosT));
        const phiH = rand() * Math.PI * 2.0;
        gx = rHalo * sinT * Math.cos(phiH);
        gy = rHalo * cosT * 0.32;
        gz = rHalo * sinT * Math.sin(phiH);
        size = 0.42 + 0.38 * rand();
        temp = 0.06 + 0.24 * rand();
      }
      dust = 0.04;
    } else if (frac < 0.42) {
      const armIdx = i % 4;
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.88);
      const radius = MW_BAR_HALF_LEN * 0.78 + s * 1.12;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      const spread = gaussian() * (0.12 + 0.14 * s) * (isMajor ? 1.05 : 1.45);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * (0.018 + 0.025 * s);
      gz = Math.sin(theta) * radius;
      const rNorm = Math.min(1.0, radius / 1.35);
      const innerWarm = Math.max(0, 1.0 - Math.max(0, rNorm - 0.16) / 0.22);
      size = 0.56 + 0.48 * rand();
      temp = Math.min(0.46, 0.05 + 0.23 * rand() + innerWarm * 0.22);
      dust = 0.06 + 0.10 * rand();
    } else if (frac < 0.53) {
      const armIdx = i % 4;
      const s = Math.pow(rand(), 0.92);
      const radius = MW_BAR_HALF_LEN * 0.66 + s * 0.98;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      const dustOffset = -0.13 - Math.abs(gaussian()) * 0.09;
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + dustOffset;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * 0.012;
      gz = Math.sin(theta) * radius;
      size = 0.95 + 0.55 * rand();
      temp = 0.14;
      dust = 0.78 + 0.20 * rand();
    } else if (frac < 0.855) {
      const armIdx = (i % 10 < 7) ? (i % 2) : (2 + (i % 2));
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.85);
      const radius = MW_BAR_HALF_LEN * 0.86 + s * 1.04;
      const twist = 2.62 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.7) * 3.15);
      const spread = gaussian() * (0.042 + 0.068 * s) * (isMajor ? 1.0 : 1.28);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * (0.012 + 0.018 * s);
      gz = Math.sin(theta) * radius;
      const barRootWarm = Math.max(0, 1.0 - Math.max(0, radius - MW_BAR_HALF_LEN * 0.85) / 0.22);
      size = (isMajor ? 0.68 : 0.56) * (0.76 + 0.52 * rand());
      temp = Math.min(0.68, 0.04 + 0.28 * rand() + barRootWarm * 0.36);
    } else if (frac < 0.89) {
      const knot = knots[i % knots.length];
      gx = knot.x + gaussian() * knot.spread;
      gy = gaussian() * 0.010;
      gz = knot.z + gaussian() * knot.spread;
      size = 0.60 + 0.42 * rand();
      temp = 0.24 + 0.16 * rand();
      nebulaWeight = 0.48 + 0.32 * rand();
    } else {
      const isCore = frac >= 0.955;
      if (isCore) {
        const r = Math.sqrt(-Math.log(Math.max(1e-5, 1.0 - 0.99 * rand()))) * 0.068;
        const theta = rand() * Math.PI * 2.0;
        const bx = r * Math.cos(theta) * 1.45;
        const bz = r * Math.sin(theta) * 0.78;
        gx = bx * cosBar - bz * sinBar;
        gy = gaussian() * 0.026;
        gz = bx * sinBar + bz * cosBar;
        size = 0.74 + 0.46 * rand();
        temp = 0.94 + 0.06 * rand();
      } else {
        const sign = rand() < 0.5 ? -1 : 1;
        const u = sign * Math.pow(rand(), 1.25) * MW_BAR_HALF_LEN * 1.12;
        const widthEnv = Math.max(0.28, 1.0 - Math.abs(u / (MW_BAR_HALF_LEN * 1.25)) * 0.55);
        const w = gaussian() * 0.048 * widthEnv;
        gx = u * cosBar - w * sinBar;
        gy = gaussian() * 0.020;
        gz = u * sinBar + w * cosBar;
        const coreNear = Math.max(0, 1.0 - Math.abs(u) / (MW_BAR_HALF_LEN * 0.9));
        size = 0.62 + 0.44 * rand();
        temp = 0.68 + 0.26 * coreNear + 0.06 * rand();
      }
    }

    galaxy[gOff] = gx;
    galaxy[gOff + 1] = gz;
    galaxy[gOff + 2] = gy;

    const rGal = Math.hypot(gx, gz);
    const gColor = getBarredGalaxyColor(temp, dust, nebulaWeight);
    galCol[gOff] = gColor[0];
    galCol[gOff + 1] = gColor[1];
    galCol[gOff + 2] = gColor[2];

    // Semantic pairing: warm galactic bar & inner root stars forge the Copper Ring;
    // silver-blue spiral arm & halo stars forge the Silver-Blue Orbital Swoosh.
    const role: 'ring' | 'swoosh' = temp >= 0.42 ? 'ring' : 'swoosh';
    const target = sampleBrandMarkTarget(role, rand(), rand(), rand());
    logo[lOff] = target.x;
    logo[lOff + 1] = target.y;
    logoCol[gOff] = target.col[0];
    logoCol[gOff + 1] = target.col[1];
    logoCol[gOff + 2] = target.col[2];

    const stagger = Math.min(0.95, rGal * 0.42 + rand() * 0.35);
    meta[mOff] = size;
    meta[mOff + 1] = temp;
    meta[mOff + 2] = stagger;
    meta[mOff + 3] = rGal;

    const galAlpha = dust > 0.5
      ? 0.20 * dust
      : (nebulaWeight > 0.35 ? 0.62 : mix(0.88, 0.72, smoothstep(0.0, 0.35, rGal)));
    alpha[mOff] = galAlpha;
    alpha[mOff + 1] = target.alpha;
    alpha[mOff + 2] = dust > 0.5 ? 1 : 0;
    alpha[mOff + 3] = rand() * Math.PI * 2;
  }

  return { count, galaxy, logo, galCol, logoCol, meta, alpha };
}

function queryLiveGalaxyState(): LiveGalaxyState | null {
  const detail: { state?: LiveGalaxyState } = {};
  window.dispatchEvent(new CustomEvent('cm:galaxy-state', { detail }));
  return detail.state ?? null;
}

function createWebglTransitionRenderer(canvas: HTMLCanvasElement, data: GalaxyParticlesData) {
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    powerPreference: 'high-performance',
  });
  if (!gl) return null;

  const compile = (type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };

  const vert = compile(gl.VERTEX_SHADER, `#version 300 es
    precision highp float;

    in vec3 a_galaxy;
    in vec2 a_logo;
    in vec3 a_gal_col;
    in vec3 a_logo_col;
    in vec4 a_meta;  // x: size, y: temp, z: stagger, w: radius
    in vec4 a_alpha; // x: galAlpha, y: logoAlpha, z: isDust, w: phase

    uniform float u_mode;         // 0: inbound, 1: hold, 2: outbound
    uniform float u_progress;     // 0..1
    uniform float u_time;         // live elapsed ms
    uniform float u_aspect;       // width / height
    uniform float u_pixel_ratio;
    uniform float u_logo_scale;   // BrandMark width mapped to camera world
    uniform float u_fade_in;      // 1 when materializing over non-galaxy bg
    uniform vec3 u_init_tilt;     // (tiltX, yaw, roll) of starting galaxy

    out vec3 v_color;
    out float v_alpha;
    out float v_glow;

    vec3 rotateAndProject(vec3 pos, float tiltX, float yaw, float roll, float camDist, float aspect, vec2 center) {
      float cosP = cos(tiltX); float sinP = sin(tiltX);
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
      float fov = 1.32;
      vec2 projected = center + vec2((x3 / zView) * fov / aspect, (y3 / zView) * fov);
      return vec3(projected, zView);
    }

    void main() {
      float rGal = a_meta.w;
      float omega = 0.000105 / sqrt(rGal * rGal + 0.06);
      float baseSpin = u_time * omega;
      float cG = cos(baseSpin); float sG = sin(baseSpin);
      float gx = a_galaxy.x * cG - a_galaxy.y * sG;
      float gy = -(a_galaxy.x * sG + a_galaxy.y * cG);
      float galAspectScale = (u_aspect < 1.0) ? 0.68 : 0.88;
      vec3 startPos = vec3(gx, gy, a_galaxy.z) * galAspectScale;
      vec3 endPos = vec3(a_logo * u_logo_scale, 0.0);

      vec3 worldPos = endPos;
      float tiltX = 0.0;
      float yaw = 0.0;
      float roll = 0.0;
      float camDist = 1.50;
      vec2 center = vec2(0.50, 0.50);
      vec3 col = a_logo_col;
      float alphaOut = a_alpha.y;
      float sizeScale = 1.0;
      float glow = 0.14;

      if (u_mode < 0.5) {
        // Inbound: 3D galaxy spirals & rotates with perspective into BrandMark
        float p = clamp(u_progress, 0.0, 1.0);
        float gp = p * p * (3.0 - 2.0 * p);
        float stagger = a_meta.z * 0.24;
        float lp = clamp((p - stagger) / (1.0 - stagger), 0.0, 1.0);
        float eased = lp * lp * (3.0 - 2.0 * lp);

        tiltX = mix(u_init_tilt.x, 0.0, gp) + sin(gp * 3.14159265) * 0.22;
        yaw = mix(u_init_tilt.y, 0.0, gp) + sin(gp * 3.14159265) * 0.54;
        roll = mix(u_init_tilt.z, 0.0, gp);
        camDist = 1.50 - 0.14 * sin(gp * 3.14159265);
        center = mix(vec2(0.50, 0.48), vec2(0.50, 0.50), gp);

        vec3 basePos = mix(startPos, endPos, eased);
        float spinAngle = eased * 6.283185307 + sin(eased * 3.14159265) * (0.95 / (rGal + 0.34));
        float cS = cos(spinAngle); float sS = sin(spinAngle);
        worldPos = vec3(
          basePos.x * cS - basePos.y * sS,
          basePos.x * sS + basePos.y * cS,
          basePos.z + sin(eased * 3.14159265) * 0.12 * (a_meta.y - 0.5)
        );

        col = mix(a_gal_col, a_logo_col, smoothstep(0.32, 0.92, eased));
        float fadeInMask = u_fade_in > 0.5 ? smoothstep(0.0, 0.22, p) : 1.0;
        alphaOut = mix(a_alpha.x, a_alpha.y, smoothstep(0.35, 0.9, eased)) * fadeInMask;
        sizeScale = mix(0.95, 0.72, eased);
        glow = mix(smoothstep(0.24, 0.0, rGal) * 0.75, 0.18, eased);
      } else if (u_mode > 1.5) {
        // Outbound (release): BrandMark unfurls in opposite 3D perspective spiral
        float p = clamp(u_progress, 0.0, 1.0);
        float gp = 1.0 - pow(1.0 - p, 2.15);
        float stagger = a_meta.z * 0.18;
        float lp = clamp((p - stagger) / (1.0 - stagger), 0.0, 1.0);
        float eased = 1.0 - pow(1.0 - lp, 2.35);

        tiltX = mix(0.0, -0.82, gp);
        yaw = mix(0.0, -0.48, gp) - sin(gp * 3.14159265) * 0.22;
        roll = mix(0.0, -0.18, gp);
        camDist = mix(1.50, 1.66, gp);
        center = mix(vec2(0.50, 0.50), vec2(0.50, 0.48), gp);

        vec3 basePos = mix(endPos, startPos * 1.18, eased);
        float spinAngle = eased * 1.95 + sin(eased * 2.5) * (0.80 / (rGal + 0.34));
        float cS = cos(spinAngle); float sS = sin(spinAngle);
        worldPos = vec3(
          basePos.x * cS - basePos.y * sS,
          basePos.x * sS + basePos.y * cS,
          basePos.z + sin(eased * 3.14159265) * 0.13 * (0.5 - a_meta.y)
        );

        col = mix(a_logo_col, a_gal_col, smoothstep(0.08, 0.65, eased));
        float burst = 1.0 + 0.16 * sin(min(1.0, lp * 2.0) * 3.14159265);
        alphaOut = mix(a_alpha.y, a_alpha.x, smoothstep(0.1, 0.65, eased)) * burst * (1.0 - smoothstep(0.54, 1.0, lp));
        sizeScale = mix(0.74, 1.12, eased);
        glow = mix(0.22, smoothstep(0.26, 0.0, rGal) * 0.7, eased) * (1.0 - smoothstep(0.54, 1.0, lp));
      }

      vec3 proj = rotateAndProject(worldPos, tiltX, yaw, roll, camDist, u_aspect, center);
      gl_Position = vec4(proj.x * 2.0 - 1.0, 1.0 - proj.y * 2.0, 0.0, 1.0);

      float pScale = clamp(1.32 / proj.z, 0.58, 1.85);
      float basePt = max(1.0, a_meta.x * 4.8 * u_pixel_ratio * pScale * sizeScale);
      gl_PointSize = clamp(basePt, 1.2, 24.0);

      v_color = col;
      v_alpha = clamp(alphaOut, 0.0, 1.0) * (a_alpha.z > 0.5 ? 0.35 : 1.0);
      v_glow = glow;
    }`);

  const frag = compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    in vec3 v_color;
    in float v_alpha;
    in float v_glow;
    out vec4 fragColor;

    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      float dist = length(uv);
      if (dist > 0.5) discard;
      float core = smoothstep(0.5, 0.06, dist);
      float halo = exp(-dist * dist * 13.0) * (0.28 + v_glow * 0.45);
      float finalAlpha = clamp((core * 0.78 + halo) * v_alpha, 0.0, 1.0);
      fragColor = vec4(v_color * finalAlpha, finalAlpha);
    }`);

  if (!vert || !frag) {
    if (vert) gl.deleteShader(vert);
    if (frag) gl.deleteShader(frag);
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return null;
  }
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }

  const buffers: WebGLBuffer[] = [];
  const bindAttr = (name: string, size: number, arr: Float32Array) => {
    const buf = gl.createBuffer();
    if (!buf) return;
    buffers.push(buf);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(program, name);
    if (loc !== -1) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }
  };

  bindAttr('a_galaxy', 3, data.galaxy);
  bindAttr('a_logo', 2, data.logo);
  bindAttr('a_gal_col', 3, data.galCol);
  bindAttr('a_logo_col', 3, data.logoCol);
  bindAttr('a_meta', 4, data.meta);
  bindAttr('a_alpha', 4, data.alpha);

  const uMode = gl.getUniformLocation(program, 'u_mode');
  const uProgress = gl.getUniformLocation(program, 'u_progress');
  const uTime = gl.getUniformLocation(program, 'u_time');
  const uAspect = gl.getUniformLocation(program, 'u_aspect');
  const uRatio = gl.getUniformLocation(program, 'u_pixel_ratio');
  const uLogoScale = gl.getUniformLocation(program, 'u_logo_scale');
  const uFadeIn = gl.getUniformLocation(program, 'u_fade_in');
  const uInitTilt = gl.getUniformLocation(program, 'u_init_tilt');

  gl.useProgram(program);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return {
    draw(
      width: number,
      height: number,
      pixelRatio: number,
      mode: number,
      progress: number,
      time: number,
      fadeIn: boolean,
      initTilt: readonly [number, number, number],
    ) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const logoPx = Math.min(LOGO_SIZE, width * 0.7, height * 0.46);
      const logoWorldScale = (logoPx / Math.max(1, height)) * (CAM_DIST_BASE / FOV);
      gl.uniform1f(uMode, mode);
      gl.uniform1f(uProgress, progress);
      gl.uniform1f(uTime, time);
      gl.uniform1f(uAspect, width / Math.max(1, height));
      gl.uniform1f(uRatio, pixelRatio);
      gl.uniform1f(uLogoScale, logoWorldScale);
      gl.uniform1f(uFadeIn, fadeIn ? 1 : 0);
      gl.uniform3f(uInitTilt, initTilt[0], initTilt[1], initTilt[2]);
      gl.drawArrays(gl.POINTS, 0, data.count);
    },
    dispose() {
      for (const b of buffers) gl.deleteBuffer(b);
      gl.deleteProgram(program);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}

function projectCpu3D(
  x: number,
  y: number,
  z: number,
  tiltX: number,
  yaw: number,
  roll: number,
  camDist: number,
  aspect: number,
  centerX: number,
  centerY: number,
): [number, number, number] {
  const cosP = Math.cos(tiltX);
  const sinP = Math.sin(tiltX);
  const y1 = y * cosP - z * sinP;
  const z1 = y * sinP + z * cosP;
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const x2 = x * cosY - z1 * sinY;
  const z2 = x * sinY + z1 * cosY;
  const cosR = Math.cos(roll);
  const sinR = Math.sin(roll);
  const x3 = x2 * cosR - y1 * sinR;
  const y3 = x2 * sinR + y1 * cosR;
  const zView = Math.max(0.32, z2 + camDist);
  return [centerX + (x3 / zView) * FOV / aspect, centerY + (y3 / zView) * FOV, zView];
}

export default function GalaxyTransitionField({
  phase,
  reducedMotion,
  scatterMs,
  assembleMs,
  releaseMs,
}: {
  phase: TransitionPhase;
  reducedMotion: boolean;
  scatterMs: number;
  assembleMs: number;
  releaseMs: number;
}) {
  const webglCanvasRef = useRef<HTMLCanvasElement>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef<TransitionPhase>(phase);
  const inboundStartedAtRef = useRef<number | null>(null);
  const releaseStartedAtRef = useRef<number | null>(null);
  const scheduleDrawRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    phaseRef.current = phase;
    const now = performance.now();
    if (phase === 'scatter' && inboundStartedAtRef.current === null) {
      inboundStartedAtRef.current = now;
    } else if (phase === 'assemble' && inboundStartedAtRef.current === null) {
      inboundStartedAtRef.current = now - scatterMs;
    } else if (phase === 'release') {
      releaseStartedAtRef.current = now;
    }
    scheduleDrawRef.current?.();
  }, [phase, scatterMs]);

  useEffect(() => {
    if (reducedMotion || (typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent))) {
      return;
    }

    const webglCanvas = webglCanvasRef.current;
    const fallbackCanvas = fallbackCanvasRef.current;
    if (!webglCanvas || !fallbackCanvas) return;

    const live = queryLiveGalaxyState();
    const hasActiveMilkyWay = Boolean(live && live.progress >= 0.78);
    const ptrDeltaX = live ? live.pointerX - 0.5 : 0;
    const ptrDeltaY = live ? live.pointerY - 0.5 : 0;
    const initPitch = clamp(0.76 - ptrDeltaY * 0.96, 0.28, 1.28);
    const initTilt: readonly [number, number, number] = [
      initPitch - Math.PI / 2,
      hasActiveMilkyWay ? ptrDeltaX * 1.16 : -0.14,
      hasActiveMilkyWay ? -0.16 + ptrDeltaX * 0.24 : -0.16,
    ];
    const baseElapsed = live ? live.elapsed : 0;

    const webglData = generateTransitionGalaxy(4800);
    let glRenderer = createWebglTransitionRenderer(webglCanvas, webglData);
    let fallbackData: GalaxyParticlesData | null = null;
    let ctx2d: CanvasRenderingContext2D | null = null;

    if (!glRenderer) {
      fallbackData = generateTransitionGalaxy(1400);
      ctx2d = fallbackCanvas.getContext('2d', { alpha: true });
    }
    if (!glRenderer && !ctx2d) return;

    webglCanvas.style.display = glRenderer ? 'block' : 'none';
    fallbackCanvas.style.display = glRenderer ? 'none' : 'block';

    let width = window.innerWidth;
    let height = window.innerHeight;
    let pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    let frame = 0;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      const target = glRenderer ? webglCanvas : fallbackCanvas;
      target.width = Math.max(1, Math.round(width * pixelRatio));
      target.height = Math.max(1, Math.round(height * pixelRatio));
      if (ctx2d) {
        ctx2d.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      }
    };

    const drawCpu = (
      currentPhase: TransitionPhase,
      mode: number,
      progress: number,
      time: number,
      fadeIn: boolean,
    ) => {
      if (!ctx2d || !fallbackData) return;
      ctx2d.clearRect(0, 0, width, height);
      if (currentPhase === 'reveal') return;
      const aspect = width / Math.max(1, height);
      const galAspectScale = aspect < 1 ? 0.68 : 0.88;
      const logoPx = Math.min(LOGO_SIZE, width * 0.7, height * 0.46);
      const logoWorldScale = (logoPx / Math.max(1, height)) * (CAM_DIST_BASE / FOV);

      // Batch into copper and cool-blue paths for fast Canvas2D rendering
      const copperPts: Array<[number, number, number]> = [];
      const bluePts: Array<[number, number, number]> = [];
      const fadeInMask = fadeIn ? smoothstep(0, 0.22, progress) : 1;
      const releaseFade = mode > 1.5 ? 1 - smoothstep(0.54, 1, progress) : 1;
      const gp = mode < 0.5
        ? progress * progress * (3 - 2 * progress)
        : mode > 1.5
          ? 1 - Math.pow(1 - progress, 2.15)
          : 1;

      const tiltX = mode < 0.5
        ? mix(initTilt[0], 0, gp) + Math.sin(gp * Math.PI) * 0.22
        : mode > 1.5
          ? mix(0, -0.82, gp)
          : 0;
      const yaw = mode < 0.5
        ? mix(initTilt[1], 0, gp) + Math.sin(gp * Math.PI) * 0.54
        : mode > 1.5
          ? mix(0, -0.48, gp) - Math.sin(gp * Math.PI) * 0.22
          : 0;
      const roll = mode < 0.5
        ? mix(initTilt[2], 0, gp)
        : mode > 1.5
          ? mix(0, -0.18, gp)
          : 0;
      const camDist = mode < 0.5
        ? 1.5 - 0.14 * Math.sin(gp * Math.PI)
        : mode > 1.5
          ? mix(1.5, 1.66, gp)
          : 1.5;
      const centerX = 0.5;
      const centerY = mode < 0.5 ? mix(0.48, 0.5, gp) : mode > 1.5 ? mix(0.5, 0.48, gp) : 0.5;

      for (let i = 0; i < fallbackData.count; i += 1) {
        const gOff = i * 3;
        const lOff = i * 2;
        const mOff = i * 4;
        const gx0 = fallbackData.galaxy[gOff];
        const gz0 = fallbackData.galaxy[gOff + 1];
        const gy0 = fallbackData.galaxy[gOff + 2];
        const rGal = fallbackData.meta[mOff + 3];
        const omega = 0.000105 / Math.sqrt(rGal * rGal + 0.06);
        const spin = time * omega;
        const cG = Math.cos(spin);
        const sG = Math.sin(spin);
        const sx = (gx0 * cG - gz0 * sG) * galAspectScale;
        const sy = -(gx0 * sG + gz0 * cG) * galAspectScale;
        const sz = gy0 * galAspectScale;
        const ex = fallbackData.logo[lOff] * logoWorldScale;
        const ey = fallbackData.logo[lOff + 1] * logoWorldScale;

        let wx = ex;
        let wy = ey;
        let wz = 0;
        if (mode < 0.5) {
          const stagger = fallbackData.meta[mOff + 2] * 0.24;
          const lp = clamp01((progress - stagger) / (1 - stagger));
          const eased = lp * lp * (3 - 2 * lp);
          const bx = mix(sx, ex, eased);
          const by = mix(sy, ey, eased);
          const spinAng = eased * Math.PI * 2 + Math.sin(eased * Math.PI) * (0.95 / (rGal + 0.34));
          const cS = Math.cos(spinAng);
          const sS = Math.sin(spinAng);
          wx = bx * cS - by * sS;
          wy = bx * sS + by * cS;
          wz = sz * (1 - eased) + Math.sin(eased * Math.PI) * 0.12 * (fallbackData.meta[mOff + 1] - 0.5);
        } else if (mode > 1.5) {
          const stagger = fallbackData.meta[mOff + 2] * 0.18;
          const lp = clamp01((progress - stagger) / (1 - stagger));
          const eased = 1 - Math.pow(1 - lp, 2.35);
          const bx = mix(ex, sx * 1.18, eased);
          const by = mix(ey, sy * 1.18, eased);
          const spinAng = eased * 1.95 + Math.sin(eased * 2.5) * (0.8 / (rGal + 0.34));
          const cS = Math.cos(spinAng);
          const sS = Math.sin(spinAng);
          wx = bx * cS - by * sS;
          wy = bx * sS + by * cS;
          wz = sz * eased + Math.sin(eased * Math.PI) * 0.13 * (0.5 - fallbackData.meta[mOff + 1]);
        }

        const [pxNorm, pyNorm, zView] = projectCpu3D(wx, wy, wz, tiltX, yaw, roll, camDist, aspect, centerX, centerY);
        const pr = Math.max(0.75, fallbackData.meta[mOff] * 1.05 * clamp(1.32 / zView, 0.58, 1.8));
        if (fallbackData.meta[mOff + 1] >= 0.42) {
          copperPts.push([pxNorm * width, pyNorm * height, pr]);
        } else {
          bluePts.push([pxNorm * width, pyNorm * height, pr]);
        }
      }

      const alphaMask = clamp01(fadeInMask * releaseFade * 0.9);
      ctx2d.globalAlpha = alphaMask;
      const drawBatch = (pts: Array<[number, number, number]>, color: string) => {
        ctx2d.fillStyle = color;
        ctx2d.beginPath();
        for (const [px, py, pr] of pts) {
          ctx2d.moveTo(px + pr, py);
          ctx2d.arc(px, py, pr, 0, Math.PI * 2);
        }
        ctx2d.fill();
      };
      drawBatch(bluePts, '#9eb9e3');
      drawBatch(copperPts, '#e5ad82');
      ctx2d.globalAlpha = 1;
    };

    const mountTime = performance.now();
    if (inboundStartedAtRef.current === null && (phaseRef.current === 'scatter' || phaseRef.current === 'assemble')) {
      inboundStartedAtRef.current = phaseRef.current === 'assemble' ? mountTime - scatterMs : mountTime;
    }

    const totalInboundMs = scatterMs + assembleMs;
    let signaledReady = false;

    const drawFrame = (now: number) => {
      frame = 0;
      const currentPhase = phaseRef.current;
      if (currentPhase === 'reveal') {
        if (glRenderer) {
          glRenderer.draw(width, height, pixelRatio, 2, 1, baseElapsed, false, initTilt);
        } else {
          ctx2d?.clearRect(0, 0, width, height);
        }
        return;
      }

      let mode = 1;
      let progress = 1;
      if (currentPhase === 'scatter' || currentPhase === 'assemble') {
        mode = 0;
        const originNow = inboundStartedAtRef.current ?? now;
        progress = clamp01((now - originNow) / totalInboundMs);
      } else if (currentPhase === 'release') {
        mode = 2;
        const originNow = releaseStartedAtRef.current ?? now;
        progress = clamp01((now - originNow) / releaseMs);
      }

      const elapsedNow = baseElapsed + (now - mountTime);
      if (glRenderer) {
        glRenderer.draw(
          width,
          height,
          pixelRatio,
          mode,
          progress,
          elapsedNow,
          !hasActiveMilkyWay,
          initTilt,
        );
      } else {
        drawCpu(currentPhase, mode, progress, elapsedNow, !hasActiveMilkyWay);
      }

      if (!signaledReady) {
        signaledReady = true;
        document.documentElement.dataset.cmGalaxyActive = 'true';
      }

      if (currentPhase === 'scatter' || currentPhase === 'assemble' || currentPhase === 'release') {
        schedule();
      }
    };

    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(drawFrame);
    };
    scheduleDrawRef.current = schedule;

    resize();
    window.addEventListener('resize', resize, { passive: true });
    schedule();

    return () => {
      scheduleDrawRef.current = null;
      window.removeEventListener('resize', resize);
      if (frame) window.cancelAnimationFrame(frame);
      if (glRenderer) {
        glRenderer.dispose();
        glRenderer = null;
      }
      delete document.documentElement.dataset.cmGalaxyActive;
    };
  }, [reducedMotion, scatterMs, assembleMs, releaseMs]);

  if (reducedMotion) return null;

  return (
    <>
      <canvas ref={webglCanvasRef} className="cm-logo-transition__particles" aria-hidden="true" />
      <canvas ref={fallbackCanvasRef} className="cm-logo-transition__particles" aria-hidden="true" />
    </>
  );
}
