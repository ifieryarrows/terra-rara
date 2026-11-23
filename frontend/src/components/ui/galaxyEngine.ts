export type TransitionPhase = 'scatter' | 'assemble' | 'hold' | 'release' | 'reveal';

export type LiveGalaxyState = {
  progress: number;
  pointerX: number;
  pointerY: number;
  elapsed: number;
  fromWorkspace?: boolean;
  tilt?: readonly [number, number, number];
};

export type GalaxyParticlesData = {
  count: number;
  galaxy: Float32Array;
  logo: Float32Array;
  galCol: Float32Array;
  logoCol: Float32Array;
  meta: Float32Array; // size, temp, stagger, radius
  alpha: Float32Array; // galAlpha, logoAlpha, isDust, twinklePhase
};

export type GalaxyDrawOptions = {
  width: number;
  height: number;
  pixelRatio: number;
  mode: number;
  progress: number;
  time: number;
  fadeIn: boolean;
  initTilt: readonly [number, number, number];
  persistEnd: boolean;
  endAlpha: number;
  fromWorkspace: boolean;
};

export const LOGO_SIZE = 224;
export const WORKSPACE_GALAXY_ALPHA = 0.54;
export const WORKSPACE_GALAXY_TILT: readonly [number, number, number] = [-0.94, -0.34, -0.22];
export const WORKSPACE_GALAXY_CENTER: readonly [number, number] = [0.54, 0.46];
export const WORKSPACE_GALAXY_CAM_DIST = 1.88;
export const WORKSPACE_GALAXY_SCALE = 0.96;
export const WORKSPACE_SPIN_RATE = 0.14;

const MW_BAR_HALF_LEN = 0.23;
const MW_BAR_ANGLE = 0.44;
const FOV = 1.32;
const CAM_DIST_BASE = 1.5;
const RELEASE_FINAL_SPIN = 2.0943951; // 2*pi/3 uniform rotation at release end (zero differential distortion)
const GALAXY_PATTERN_OMEGA = 0.000055;

let sharedClockOffsetMs = 0;

export function syncSharedGalaxyClock(elapsedMs: number, nowMs = performance.now()) {
  sharedClockOffsetMs = elapsedMs - nowMs;
}

export function getSharedGalaxyTime(nowMs = performance.now()): number {
  return Math.max(0, nowMs + sharedClockOffsetMs);
}

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function clamp01(value: number) {
  return clamp(value, 0, 1);
}

export function smoothstep(edge0: number, edge1: number, value: number) {
  const t = clamp01((value - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function mix(a: number, b: number, t: number) {
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

export function generateTransitionGalaxy(count: number): GalaxyParticlesData {
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

    if (frac < 0.045) {
      if (frac < 0.012) {
        const cx = -0.82 + gaussian() * 0.042;
        const cz = -0.68 + gaussian() * 0.024;
        gx = cx * cosBar - cz * sinBar;
        gy = 0.03 + gaussian() * 0.014;
        gz = cx * sinBar + cz * cosBar;
        size = 0.58 + 0.36 * rand();
        temp = 0.76 + 0.20 * rand();
      } else {
        const rHalo = 0.28 + 0.88 * Math.pow(rand(), 0.85);
        const cosT = rand() * 2.0 - 1.0;
        const sinT = Math.sqrt(Math.max(0.0, 1.0 - cosT * cosT));
        const phiH = rand() * Math.PI * 2.0;
        gx = rHalo * sinT * Math.cos(phiH);
        gy = rHalo * cosT * 0.18;
        gz = rHalo * sinT * Math.sin(phiH);
        size = 0.34 + 0.28 * rand();
        temp = 0.06 + 0.22 * rand();
      }
      dust = 0.03;
    } else if (frac < 0.34) {
      const armIdx = i % 4;
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.90);
      const radius = MW_BAR_HALF_LEN * 0.76 + s * 0.98;
      const twist = 3.05 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.68) * 3.15);
      const spread = gaussian() * (0.065 + 0.085 * s) * (isMajor ? 1.0 : 1.32);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * (0.012 + 0.018 * s);
      gz = Math.sin(theta) * radius;
      const rNorm = Math.min(1.0, radius / 1.25);
      const innerWarm = Math.max(0, 1.0 - Math.max(0, rNorm - 0.16) / 0.22);
      size = 0.44 + 0.38 * rand();
      temp = Math.min(0.48, 0.05 + 0.24 * rand() + innerWarm * 0.24);
      dust = 0.05 + 0.08 * rand();
    } else if (frac < 0.44) {
      const armIdx = i % 4;
      const s = Math.pow(rand(), 0.92);
      const radius = MW_BAR_HALF_LEN * 0.68 + s * 0.90;
      const twist = 3.05 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.68) * 3.15);
      const dustOffset = -0.10 - Math.abs(gaussian()) * 0.065;
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + dustOffset;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * 0.009;
      gz = Math.sin(theta) * radius;
      size = 0.72 + 0.42 * rand();
      temp = 0.14;
      dust = 0.80 + 0.18 * rand();
    } else if (frac < 0.83) {
      const armIdx = (i % 10 < 7) ? (i % 2) : (2 + (i % 2));
      const isMajor = armIdx < 2;
      const s = Math.pow(rand(), 0.84);
      const radius = MW_BAR_HALF_LEN * 0.84 + s * 0.96;
      const twist = 3.05 * Math.log(1.0 + Math.max(0, radius - MW_BAR_HALF_LEN * 0.68) * 3.15);
      const spread = gaussian() * (0.026 + 0.044 * s) * (isMajor ? 1.0 : 1.24);
      const theta = MW_BAR_ANGLE + armOffsets[armIdx] + twist + spread;
      gx = Math.cos(theta) * radius;
      gy = gaussian() * (0.009 + 0.014 * s);
      gz = Math.sin(theta) * radius;
      const barRootWarm = Math.max(0, 1.0 - Math.max(0, radius - MW_BAR_HALF_LEN * 0.84) / 0.22);
      size = (isMajor ? 0.56 : 0.46) * (0.72 + 0.48 * rand());
      temp = Math.min(0.70, 0.04 + 0.28 * rand() + barRootWarm * 0.38);
    } else if (frac < 0.88) {
      const knot = knots[i % knots.length];
      gx = knot.x + gaussian() * knot.spread;
      gy = gaussian() * 0.008;
      gz = knot.z + gaussian() * knot.spread;
      size = 0.48 + 0.34 * rand();
      temp = 0.24 + 0.16 * rand();
      nebulaWeight = 0.52 + 0.34 * rand();
    } else {
      const isCore = frac >= 0.945;
      if (isCore) {
        const r = Math.sqrt(-Math.log(Math.max(1e-5, 1.0 - 0.99 * rand()))) * 0.064;
        const theta = rand() * Math.PI * 2.0;
        const bx = r * Math.cos(theta) * 1.42;
        const bz = r * Math.sin(theta) * 0.78;
        gx = bx * cosBar - bz * sinBar;
        gy = gaussian() * 0.020;
        gz = bx * sinBar + bz * cosBar;
        size = 0.62 + 0.40 * rand();
        temp = 0.94 + 0.06 * rand();
      } else {
        const sign = rand() < 0.5 ? -1 : 1;
        const u = sign * Math.pow(rand(), 1.25) * MW_BAR_HALF_LEN * 1.12;
        const widthEnv = Math.max(0.28, 1.0 - Math.abs(u / (MW_BAR_HALF_LEN * 1.25)) * 0.55);
        const w = gaussian() * 0.042 * widthEnv;
        gx = u * cosBar - w * sinBar;
        gy = gaussian() * 0.016;
        gz = u * sinBar + w * cosBar;
        const coreNear = Math.max(0, 1.0 - Math.abs(u) / (MW_BAR_HALF_LEN * 0.9));
        size = 0.52 + 0.38 * rand();
        temp = 0.70 + 0.24 * coreNear + 0.06 * rand();
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

export function createWebglTransitionRenderer(canvas: HTMLCanvasElement, data: GalaxyParticlesData) {
  let gl: WebGL2RenderingContext | null = null;
  try {
    gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: true,
      powerPreference: 'high-performance',
    });
  } catch {
    return null;
  }
  if (!gl || typeof gl.createShader !== 'function') return null;

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

    uniform float u_mode;           // 0: inbound, 1: hold, 2: outbound
    uniform float u_progress;       // 0..1
    uniform float u_time;           // live elapsed ms
    uniform float u_aspect;         // width / height
    uniform float u_pixel_ratio;
    uniform float u_logo_scale;     // BrandMark width mapped to camera world
    uniform float u_fade_in;        // 1 when materializing over non-galaxy bg
    uniform vec3 u_init_tilt;       // (tiltX, yaw, roll) of starting galaxy
    uniform float u_persist_end;    // 1 to keep end-state galaxy visible instead of fading out
    uniform float u_end_alpha;      // resting alpha multiplier when persisted
    uniform float u_from_workspace; // 1 when picking up directly from WorkspaceGalaxy

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
      // Density-wave pattern rotation preserves crisp spiral arms over time instead of winding-catastrophe shear
      float baseSpin = u_time * 0.000055;
      float cG = cos(baseSpin); float sG = sin(baseSpin);
      float gx = a_galaxy.x * cG - a_galaxy.y * sG;
      float gy = a_galaxy.x * sG + a_galaxy.y * cG;
      float galAspectScale = (u_aspect < 1.0) ? 0.66 : 0.86;
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
        float restSway = (1.0 - smoothstep(0.0, 0.35, p)) * u_from_workspace;

        tiltX = mix(u_init_tilt.x, 0.0, gp) + sin(gp * 3.14159265) * 0.22 + sin(u_time * 0.00028) * 0.045 * restSway;
        yaw = mix(u_init_tilt.y, 0.0, gp) + sin(gp * 3.14159265) * 0.54 + cos(u_time * 0.00022) * 0.055 * restSway;
        roll = mix(u_init_tilt.z, 0.0, gp) + sin(u_time * 0.00019) * 0.025 * restSway;
        camDist = mix(mix(1.50, 1.88, u_from_workspace), 1.50, gp) - 0.14 * sin(gp * 3.14159265);
        center = mix(mix(vec2(0.50, 0.48), vec2(0.54, 0.46), u_from_workspace), vec2(0.50, 0.50), gp);

        float galScale = mix(1.0, 0.96, u_from_workspace);
        float initSwirl = u_from_workspace * 2.0943951;
        float cW = cos(initSwirl); float sW = sin(initSwirl);
        vec3 workspaceStart = vec3(
          startPos.x * galScale * cW - startPos.y * galScale * sW,
          startPos.x * galScale * sW + startPos.y * galScale * cW,
          startPos.z * galScale
        );

        vec3 basePos = mix(workspaceStart, endPos, eased);
        float spinAngle = eased * 6.283185307 + sin(eased * 3.14159265) * (0.85 / (rGal + 0.36));
        float cS = cos(spinAngle); float sS = sin(spinAngle);
        worldPos = vec3(
          basePos.x * cS - basePos.y * sS,
          basePos.x * sS + basePos.y * cS,
          basePos.z + sin(eased * 3.14159265) * 0.12 * (a_meta.y - 0.5)
        );

        col = mix(a_gal_col, a_logo_col, smoothstep(0.32, 0.92, eased));
        float fadeInMask = u_fade_in > 0.5 ? smoothstep(0.0, 0.22, p) : 1.0;
        float wsAlphaScale = mix(1.0, mix(u_end_alpha * 0.72, 1.0, smoothstep(0.0, 0.35, eased)), u_from_workspace);
        alphaOut = mix(a_alpha.x, a_alpha.y, smoothstep(0.35, 0.9, eased)) * fadeInMask * wsAlphaScale;
        sizeScale = mix(mix(0.84, 0.76, u_from_workspace), 0.72, eased);
        glow = mix(smoothstep(0.28, 0.0, rGal) * mix(0.75, 0.65 * u_end_alpha, u_from_workspace), 0.18, eased);
      } else if (u_mode > 1.5) {
        // Outbound (release): BrandMark unfurls into an intact 3D barred spiral galaxy (zero terminal shear)
        float p = clamp(u_progress, 0.0, 1.0);
        float gp = 1.0 - pow(1.0 - p, 2.15);
        float stagger = a_meta.z * 0.18;
        float lp = clamp((p - stagger) / (1.0 - stagger), 0.0, 1.0);
        float eased = 1.0 - pow(1.0 - lp, 2.35);
        float restSway = smoothstep(0.75, 1.0, p) * u_persist_end;

        tiltX = mix(0.0, -0.94, gp) + sin(u_time * 0.00028) * 0.045 * restSway;
        yaw = mix(0.0, -0.34, gp) - sin(gp * 3.14159265) * 0.18 + cos(u_time * 0.00022) * 0.055 * restSway;
        roll = mix(0.0, -0.22, gp) + sin(u_time * 0.00019) * 0.025 * restSway;
        camDist = mix(1.50, 1.88, gp);
        center = mix(vec2(0.50, 0.50), vec2(0.54, 0.46), gp);

        vec3 basePos = mix(endPos, startPos * 0.96, eased);
        float spinAngle = eased * 2.0943951 + sin(eased * 3.14159265) * (0.68 / (rGal + 0.38));
        float cS = cos(spinAngle); float sS = sin(spinAngle);
        worldPos = vec3(
          basePos.x * cS - basePos.y * sS,
          basePos.x * sS + basePos.y * cS,
          basePos.z + sin(eased * 3.14159265) * 0.11 * (0.5 - a_meta.y)
        );

        col = mix(a_logo_col, a_gal_col, smoothstep(0.08, 0.65, eased));
        float burst = 1.0 + 0.10 * sin(min(1.0, lp * 2.0) * 3.14159265);
        float fadeTail = smoothstep(0.54, 1.0, lp);
        float endFade = mix(1.0 - fadeTail, mix(1.0, u_end_alpha * 0.72, fadeTail), u_persist_end);
        alphaOut = mix(a_alpha.y, a_alpha.x, smoothstep(0.1, 0.65, eased)) * burst * endFade;
        sizeScale = mix(0.72, 0.76, eased);
        glow = mix(0.20, smoothstep(0.30, 0.0, rGal) * 0.65, eased) * endFade;
      }

      vec3 proj = rotateAndProject(worldPos, tiltX, yaw, roll, camDist, u_aspect, center);
      gl_Position = vec4(proj.x * 2.0 - 1.0, 1.0 - proj.y * 2.0, 0.0, 1.0);

      float pScale = clamp(1.32 / proj.z, 0.52, 1.50);
      float basePt = max(1.0, a_meta.x * 3.2 * u_pixel_ratio * pScale * sizeScale);
      gl_PointSize = clamp(basePt, 1.0, 9.5);

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

  const discVert = compile(gl.VERTEX_SHADER, `#version 300 es
    precision highp float;
    in vec2 a_quad;
    out vec2 v_screen;
    void main() {
      v_screen = vec2(a_quad.x * 0.5 + 0.5, 0.5 - a_quad.y * 0.5);
      gl_Position = vec4(a_quad, 0.0, 1.0);
    }`);

  const discFrag = compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    in vec2 v_screen;
    uniform float u_mode;
    uniform float u_progress;
    uniform float u_time;
    uniform float u_aspect;
    uniform vec3 u_init_tilt;
    uniform float u_persist_end;
    uniform float u_end_alpha;
    uniform float u_from_workspace;
    out vec4 fragColor;

    float hash21(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    float vnoise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
      );
    }

    float fbm(vec2 p) {
      float v = 0.0;
      float a = 0.52;
      mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
      for (int i = 0; i < 3; i++) {
        v += a * vnoise(p);
        p = rot * p * 2.15 + vec2(3.7, 1.9);
        a *= 0.5;
      }
      return v;
    }

    void main() {
      float discVis = 0.0;
      float tiltX = -0.94;
      float yaw = -0.34;
      float roll = -0.22;
      float camDist = 1.88;
      vec2 center = vec2(0.54, 0.46);
      float p = clamp(u_progress, 0.0, 1.0);

      if (u_mode > 1.5) {
        float gp = 1.0 - pow(1.0 - p, 2.15);
        float restSway = smoothstep(0.75, 1.0, p) * u_persist_end;
        tiltX = mix(0.0, -0.94, gp) + sin(u_time * 0.00028) * 0.045 * restSway;
        yaw = mix(0.0, -0.34, gp) - sin(gp * 3.14159265) * 0.18 + cos(u_time * 0.00022) * 0.055 * restSway;
        roll = mix(0.0, -0.22, gp) + sin(u_time * 0.00019) * 0.025 * restSway;
        camDist = mix(1.50, 1.88, gp);
        center = mix(vec2(0.50, 0.50), vec2(0.54, 0.46), gp);
        float appear = smoothstep(0.26, 0.92, p);
        float endFade = mix(1.0 - smoothstep(0.62, 1.0, p), mix(1.0, u_end_alpha, smoothstep(0.54, 1.0, p)), u_persist_end);
        discVis = appear * endFade;
      } else if (u_mode < 0.5 && u_from_workspace > 0.5) {
        float gp = p * p * (3.0 - 2.0 * p);
        float restSway = 1.0 - smoothstep(0.0, 0.35, p);
        tiltX = mix(u_init_tilt.x, 0.0, gp) + sin(gp * 3.14159265) * 0.22 + sin(u_time * 0.00028) * 0.045 * restSway;
        yaw = mix(u_init_tilt.y, 0.0, gp) + sin(gp * 3.14159265) * 0.54 + cos(u_time * 0.00022) * 0.055 * restSway;
        roll = mix(u_init_tilt.z, 0.0, gp) + sin(u_time * 0.00019) * 0.025 * restSway;
        camDist = mix(1.88, 1.50, gp) - 0.14 * sin(gp * 3.14159265);
        center = mix(vec2(0.54, 0.46), vec2(0.50, 0.50), gp);
        discVis = (1.0 - smoothstep(0.0, 0.42, p)) * u_end_alpha;
      }

      if (discVis < 0.004) discard;

      // Ray-plane intersection onto z_world = 0 in rotated galaxy space
      float fov = 1.32;
      vec3 ray = vec3((v_screen.x - center.x) * u_aspect / fov, (v_screen.y - center.y) / fov, 1.0);
      vec3 cam = vec3(0.0, 0.0, -camDist);

      // Inverse rotation (roll -> yaw -> tiltX)
      float cr = cos(-roll); float sr = sin(-roll);
      vec3 r1 = vec3(ray.x * cr - ray.y * sr, ray.x * sr + ray.y * cr, ray.z);
      vec3 c1 = vec3(cam.x * cr - cam.y * sr, cam.x * sr + cam.y * cr, cam.z);

      float cy = cos(-yaw); float sy = sin(-yaw);
      vec3 r2 = vec3(r1.x * cy - r1.z * sy, r1.y, r1.x * sy + r1.z * cy);
      vec3 c2 = vec3(c1.x * cy - c1.z * sy, c1.y, c1.x * sy + c1.z * cy);

      float cp = cos(-tiltX); float sp = sin(-tiltX);
      vec3 r3 = vec3(r2.x, r2.y * cp - r2.z * sp, r2.y * sp + r2.z * cp);
      vec3 c3 = vec3(c2.x, c2.y * cp - c2.z * sp, c2.y * sp + c2.z * cp);

      if (abs(r3.z) < 0.0001) discard;
      float tHit = -c3.z / r3.z;
      if (tHit <= 0.0) discard;
      vec2 hit = c3.xy + r3.xy * tHit;

      float galAspectScale = (u_aspect < 1.0) ? 0.66 : 0.86;
      vec2 pGal = hit / (galAspectScale * 0.96);
      float r = length(pGal);
      if (r > 1.38) discard;

      // Undo galaxy pattern spin + release final spin so nebula matches star arms 1:1
      float totalSpin = u_time * 0.000055 + 2.0943951;
      float cs = cos(-totalSpin); float ss = sin(-totalSpin);
      vec2 gp2 = vec2(pGal.x * cs - pGal.y * ss, pGal.x * ss + pGal.y * cs);

      // Barred spiral geometry
      float cb = cos(-0.44); float sb = sin(-0.44);
      vec2 barP = vec2(gp2.x * cb - gp2.y * sb, gp2.x * sb + gp2.y * cb);
      float coreGlow = exp(-dot(barP * vec2(1.15, 1.55), barP * vec2(1.15, 1.55)) * 22.0);
      float barGlow = exp(-dot(barP * vec2(2.1, 7.2), barP * vec2(2.1, 7.2)) * 1.45) * smoothstep(0.36, 0.08, r);

      float angle = atan(gp2.y, gp2.x);
      float logSpiral = 3.05 * log(1.0 + max(0.0, r - 0.155) * 3.15);
      float phase = angle - 0.44 - logSpiral;

      float n1 = fbm(gp2 * 4.8 + vec2(1.2, -0.8));
      float n2 = fbm(gp2 * 10.5 - vec2(2.4, 1.5));
      float filament = mix(0.55, 1.25, n1 * 0.65 + n2 * 0.35);

      float majorArm = pow(max(0.0, 0.5 + 0.5 * cos(phase * 2.0 + (n1 - 0.5) * 0.55)), 3.1);
      float minorArm = pow(max(0.0, 0.5 + 0.5 * cos((phase - 1.445) * 2.0 + (n2 - 0.5) * 0.45)), 3.8) * 0.52;
      float armEnv = smoothstep(0.11, 0.28, r) * exp(-max(0.0, r - 0.28) * 1.85);
      float arms = (majorArm + minorArm) * armEnv * filament;

      // Subtle inner dust lanes carving contrast along the inner edge of major arms
      float dustPhase = pow(max(0.0, 0.5 + 0.5 * cos((phase + 0.28) * 2.0)), 4.5);
      float dustCut = 1.0 - 0.55 * dustPhase * smoothstep(0.16, 0.34, r) * smoothstep(1.10, 0.40, r) * (0.6 + 0.4 * n2);

      // Soft outer stellar halo disc
      float discBase = exp(-r * r * 3.2) * 0.16 * (0.75 + 0.25 * n1);

      vec3 colCore = vec3(1.0, 0.945, 0.865);
      vec3 colCopper = vec3(0.867, 0.631, 0.455);
      vec3 colBlue = vec3(0.557, 0.686, 0.875);
      vec3 colRose = vec3(0.859, 0.510, 0.588);

      float radialMix = smoothstep(0.18, 0.72, r);
      vec3 armCol = mix(colCopper, colBlue, radialMix);
      float h2Knot = smoothstep(0.68, 0.90, n2) * majorArm * armEnv;
      armCol = mix(armCol, colRose, clamp(h2Knot * 0.75, 0.0, 0.55));

      vec3 totalCol = colCore * coreGlow * 0.95
        + colCopper * barGlow * 0.62
        + armCol * arms * 0.72
        + mix(colCopper, colBlue, 0.65) * discBase;

      float rawAlpha = (coreGlow * 0.78 + barGlow * 0.48 + arms * 0.58 + discBase) * dustCut * smoothstep(1.36, 0.85, r);
      float outAlpha = clamp(rawAlpha * discVis * 0.95, 0.0, 0.88);
      if (outAlpha < 0.003) discard;
      fragColor = vec4(totalCol * outAlpha, outAlpha);
    }`);

  if (!vert || !frag || !discVert || !discFrag) {
    if (vert) gl.deleteShader(vert);
    if (frag) gl.deleteShader(frag);
    if (discVert) gl.deleteShader(discVert);
    if (discFrag) gl.deleteShader(discFrag);
    return null;
  }

  const linkProgram = (vs: WebGLShader, fs: WebGLShader) => {
    const p = gl.createProgram();
    if (!p) return null;
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      gl.deleteProgram(p);
      return null;
    }
    return p;
  };

  const program = linkProgram(vert, frag);
  const discProgram = linkProgram(discVert, discFrag);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  gl.deleteShader(discVert);
  gl.deleteShader(discFrag);
  if (!program || !discProgram) {
    if (program) gl.deleteProgram(program);
    if (discProgram) gl.deleteProgram(discProgram);
    return null;
  }

  const buffers: WebGLBuffer[] = [];
  const starsVao = gl.createVertexArray();
  const discVao = gl.createVertexArray();
  gl.bindVertexArray(starsVao);

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

  gl.bindVertexArray(discVao);
  const quadBuf = gl.createBuffer();
  if (quadBuf) {
    buffers.push(quadBuf);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const qLoc = gl.getAttribLocation(discProgram, 'a_quad');
    if (qLoc !== -1) {
      gl.enableVertexAttribArray(qLoc);
      gl.vertexAttribPointer(qLoc, 2, gl.FLOAT, false, 0, 0);
    }
  }
  gl.bindVertexArray(null);

  const dMode = gl.getUniformLocation(discProgram, 'u_mode');
  const dProgress = gl.getUniformLocation(discProgram, 'u_progress');
  const dTime = gl.getUniformLocation(discProgram, 'u_time');
  const dAspect = gl.getUniformLocation(discProgram, 'u_aspect');
  const dInitTilt = gl.getUniformLocation(discProgram, 'u_init_tilt');
  const dPersistEnd = gl.getUniformLocation(discProgram, 'u_persist_end');
  const dEndAlpha = gl.getUniformLocation(discProgram, 'u_end_alpha');
  const dFromWorkspace = gl.getUniformLocation(discProgram, 'u_from_workspace');

  const uMode = gl.getUniformLocation(program, 'u_mode');
  const uProgress = gl.getUniformLocation(program, 'u_progress');
  const uTime = gl.getUniformLocation(program, 'u_time');
  const uAspect = gl.getUniformLocation(program, 'u_aspect');
  const uRatio = gl.getUniformLocation(program, 'u_pixel_ratio');
  const uLogoScale = gl.getUniformLocation(program, 'u_logo_scale');
  const uFadeIn = gl.getUniformLocation(program, 'u_fade_in');
  const uInitTilt = gl.getUniformLocation(program, 'u_init_tilt');
  const uPersistEnd = gl.getUniformLocation(program, 'u_persist_end');
  const uEndAlpha = gl.getUniformLocation(program, 'u_end_alpha');
  const uFromWorkspace = gl.getUniformLocation(program, 'u_from_workspace');

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return {
    draw({
      width,
      height,
      pixelRatio,
      mode,
      progress,
      time,
      fadeIn,
      initTilt,
      persistEnd,
      endAlpha,
      fromWorkspace,
    }: GalaxyDrawOptions) {
      const aspect = width / Math.max(1, height);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Pass 1: volumetric barred-spiral nebula disc behind the stars
      if (mode > 1.5 || (mode < 0.5 && fromWorkspace)) {
        gl.useProgram(discProgram);
        gl.bindVertexArray(discVao);
        gl.uniform1f(dMode, mode);
        gl.uniform1f(dProgress, progress);
        gl.uniform1f(dTime, time);
        gl.uniform1f(dAspect, aspect);
        gl.uniform3f(dInitTilt, initTilt[0], initTilt[1], initTilt[2]);
        gl.uniform1f(dPersistEnd, persistEnd ? 1 : 0);
        gl.uniform1f(dEndAlpha, endAlpha);
        gl.uniform1f(dFromWorkspace, fromWorkspace ? 1 : 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }

      // Pass 2: embedded fine stellar points & BrandMark morph stars
      gl.useProgram(program);
      gl.bindVertexArray(starsVao);
      const logoPx = Math.min(LOGO_SIZE, width * 0.7, height * 0.46);
      const logoWorldScale = (logoPx / Math.max(1, height)) * (CAM_DIST_BASE / FOV);
      gl.uniform1f(uMode, mode);
      gl.uniform1f(uProgress, progress);
      gl.uniform1f(uTime, time);
      gl.uniform1f(uAspect, aspect);
      gl.uniform1f(uRatio, pixelRatio);
      gl.uniform1f(uLogoScale, logoWorldScale);
      gl.uniform1f(uFadeIn, fadeIn ? 1 : 0);
      gl.uniform3f(uInitTilt, initTilt[0], initTilt[1], initTilt[2]);
      gl.uniform1f(uPersistEnd, persistEnd ? 1 : 0);
      gl.uniform1f(uEndAlpha, endAlpha);
      gl.uniform1f(uFromWorkspace, fromWorkspace ? 1 : 0);
      gl.drawArrays(gl.POINTS, 0, data.count);
      gl.bindVertexArray(null);
    },
    dispose() {
      for (const b of buffers) gl.deleteBuffer(b);
      if (starsVao) gl.deleteVertexArray(starsVao);
      if (discVao) gl.deleteVertexArray(discVao);
      gl.deleteProgram(program);
      gl.deleteProgram(discProgram);
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

export function drawGalaxyCpu2D(
  ctx2d: CanvasRenderingContext2D,
  fallbackData: GalaxyParticlesData,
  options: Omit<GalaxyDrawOptions, 'pixelRatio'>,
) {
  const { width, height, mode, progress, time, fadeIn, initTilt, persistEnd, endAlpha, fromWorkspace } = options;
  ctx2d.clearRect(0, 0, width, height);
  const aspect = width / Math.max(1, height);
  const galAspectScale = aspect < 1 ? 0.66 : 0.86;
  const logoPx = Math.min(LOGO_SIZE, width * 0.7, height * 0.46);
  const logoWorldScale = (logoPx / Math.max(1, height)) * (CAM_DIST_BASE / FOV);

  const corePts: Array<[number, number, number]> = [];
  const copperPts: Array<[number, number, number]> = [];
  const bluePts: Array<[number, number, number]> = [];
  const fadeInMask = fadeIn ? smoothstep(0, 0.22, progress) : 1;
  const fadeTail = smoothstep(0.54, 1, progress);
  const releaseFade = mode > 1.5
    ? (persistEnd ? mix(1, endAlpha, fadeTail) : 1 - fadeTail)
    : mode < 0.5 && fromWorkspace
      ? mix(endAlpha, 1, smoothstep(0, 0.35, progress))
      : 1;
  const gp = mode < 0.5
    ? progress * progress * (3 - 2 * progress)
    : mode > 1.5
      ? 1 - Math.pow(1 - progress, 2.15)
      : 1;

  const restSway = mode > 1.5
    ? smoothstep(0.75, 1, progress) * (persistEnd ? 1 : 0)
    : mode < 0.5 && fromWorkspace
      ? 1 - smoothstep(0, 0.35, progress)
      : 0;
  const tiltX = (mode < 0.5
    ? mix(initTilt[0], 0, gp) + Math.sin(gp * Math.PI) * 0.22
    : mode > 1.5
      ? mix(0, WORKSPACE_GALAXY_TILT[0], gp)
      : 0) + Math.sin(time * 0.00028) * 0.045 * restSway;
  const yaw = (mode < 0.5
    ? mix(initTilt[1], 0, gp) + Math.sin(gp * Math.PI) * 0.54
    : mode > 1.5
      ? mix(0, WORKSPACE_GALAXY_TILT[1], gp) - Math.sin(gp * Math.PI) * 0.18
      : 0) + Math.cos(time * 0.00022) * 0.055 * restSway;
  const roll = (mode < 0.5
    ? mix(initTilt[2], 0, gp)
    : mode > 1.5
      ? mix(0, WORKSPACE_GALAXY_TILT[2], gp)
      : 0) + Math.sin(time * 0.00019) * 0.025 * restSway;
  const camDist = mode < 0.5
    ? mix(fromWorkspace ? WORKSPACE_GALAXY_CAM_DIST : CAM_DIST_BASE, CAM_DIST_BASE, gp) - 0.14 * Math.sin(gp * Math.PI)
    : mode > 1.5
      ? mix(CAM_DIST_BASE, WORKSPACE_GALAXY_CAM_DIST, gp)
      : CAM_DIST_BASE;
  const centerX = mode < 0.5
    ? mix(fromWorkspace ? WORKSPACE_GALAXY_CENTER[0] : 0.5, 0.5, gp)
    : mode > 1.5
      ? mix(0.5, WORKSPACE_GALAXY_CENTER[0], gp)
      : 0.5;
  const centerY = mode < 0.5
    ? mix(fromWorkspace ? WORKSPACE_GALAXY_CENTER[1] : 0.48, 0.5, gp)
    : mode > 1.5
      ? mix(0.5, WORKSPACE_GALAXY_CENTER[1], gp)
      : 0.5;
  const galScale = fromWorkspace ? WORKSPACE_GALAXY_SCALE : 1;
  const spin = time * GALAXY_PATTERN_OMEGA;
  const cG = Math.cos(spin);
  const sG = Math.sin(spin);

  for (let i = 0; i < fallbackData.count; i += 1) {
    const gOff = i * 3;
    const lOff = i * 2;
    const mOff = i * 4;
    const gx0 = fallbackData.galaxy[gOff];
    const gz0 = fallbackData.galaxy[gOff + 1];
    const gy0 = fallbackData.galaxy[gOff + 2];
    const rGal = fallbackData.meta[mOff + 3];
    const sx = (gx0 * cG - gz0 * sG) * galAspectScale;
    const sy = (gx0 * sG + gz0 * cG) * galAspectScale;
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
      let wsx = sx * galScale;
      let wsy = sy * galScale;
      const wsz = sz * galScale;
      if (fromWorkspace) {
        const cW = Math.cos(RELEASE_FINAL_SPIN);
        const sW = Math.sin(RELEASE_FINAL_SPIN);
        const rx = wsx * cW - wsy * sW;
        const ry = wsx * sW + wsy * cW;
        wsx = rx;
        wsy = ry;
      }
      const bx = mix(wsx, ex, eased);
      const by = mix(wsy, ey, eased);
      const spinAng = eased * Math.PI * 2 + Math.sin(eased * Math.PI) * (0.85 / (rGal + 0.36));
      const cS = Math.cos(spinAng);
      const sS = Math.sin(spinAng);
      wx = bx * cS - by * sS;
      wy = bx * sS + by * cS;
      wz = wsz * (1 - eased) + Math.sin(eased * Math.PI) * 0.12 * (fallbackData.meta[mOff + 1] - 0.5);
    } else if (mode > 1.5) {
      const stagger = fallbackData.meta[mOff + 2] * 0.18;
      const lp = clamp01((progress - stagger) / (1 - stagger));
      const eased = 1 - Math.pow(1 - lp, 2.35);
      const bx = mix(ex, sx * WORKSPACE_GALAXY_SCALE, eased);
      const by = mix(ey, sy * WORKSPACE_GALAXY_SCALE, eased);
      const spinAng = eased * RELEASE_FINAL_SPIN + Math.sin(eased * Math.PI) * (0.68 / (rGal + 0.38));
      const cS = Math.cos(spinAng);
      const sS = Math.sin(spinAng);
      wx = bx * cS - by * sS;
      wy = bx * sS + by * cS;
      wz = sz * eased + Math.sin(eased * Math.PI) * 0.11 * (0.5 - fallbackData.meta[mOff + 1]);
    }

    const [pxNorm, pyNorm, zView] = projectCpu3D(wx, wy, wz, tiltX, yaw, roll, camDist, aspect, centerX, centerY);
    const pr = Math.max(0.65, fallbackData.meta[mOff] * 0.92 * clamp(1.32 / zView, 0.54, 1.65));
    const temp = fallbackData.meta[mOff + 1];
    if (temp >= 0.78) {
      corePts.push([pxNorm * width, pyNorm * height, pr * 1.15]);
    } else if (temp >= 0.42) {
      copperPts.push([pxNorm * width, pyNorm * height, pr]);
    } else {
      bluePts.push([pxNorm * width, pyNorm * height, pr]);
    }
  }

  const alphaMask = clamp01(fadeInMask * releaseFade * 0.9);
  ctx2d.globalAlpha = alphaMask;

  if (mode > 1.5 && alphaMask > 0.02 && typeof ctx2d.createRadialGradient === 'function') {
    const gx = centerX * width;
    const gy = centerY * height;
    const glowR = Math.min(width, height) * 0.22 * gp;
    if (glowR > 4) {
      const coreGlow = ctx2d.createRadialGradient(gx, gy, 0, gx, gy, glowR);
      coreGlow.addColorStop(0, 'rgba(255, 240, 218, 0.38)');
      coreGlow.addColorStop(0.32, 'rgba(221, 161, 116, 0.16)');
      coreGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx2d.fillStyle = coreGlow;
      ctx2d.fillRect(gx - glowR, gy - glowR, glowR * 2, glowR * 2);
    }
  }

  const drawBatch = (pts: Array<[number, number, number]>, color: string) => {
    if (pts.length === 0) return;
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
  drawBatch(corePts, '#fff3e2');
  ctx2d.globalAlpha = 1;
}
