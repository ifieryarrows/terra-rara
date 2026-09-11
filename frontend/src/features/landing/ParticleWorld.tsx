import { useEffect, useRef } from 'react';
import type { MotionValue } from 'framer-motion';

export type ParticleQuality = 'high' | 'balanced';

const MAX_PARTICLES = 640;
const PARTICLE_RESPONSE_MS = 150;
const SCROLL_IMPULSE_DAMPING = 0.965;
const SCROLL_IMPULSE_GAIN = 0.48;
const QUALITY = {
  high: { count: MAX_PARTICLES, dpr: 1.5, pointSize: 13, canvasAlpha: .92, largeRadius: 2.35, radius: 1.3 },
  balanced: { count: 420, dpr: 1, pointSize: 12, canvasAlpha: .88, largeRadius: 2.15, radius: 1.18 },
} as const;

const EVIDENCE_RING_PARTICLES = Math.round(MAX_PARTICLES * .62);
const EVIDENCE_TRACE_PARTICLES = MAX_PARTICLES - EVIDENCE_RING_PARTICLES;

function noise(index: number, salt: number) {
  const value = Math.sin((index + 1) * (12.9898 + salt * 17.17)) * 43758.5453;
  return value - Math.floor(value);
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
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

const copperForm = makeShape(index => {
  const angle = Math.PI * .23 + (index / (MAX_PARTICLES - 1)) * Math.PI * 1.54;
  const radius = .205 + (noise(index, 1) - .5) * .055;
  return [.72 + Math.cos(angle) * radius, .49 + Math.sin(angle) * radius * 1.08];
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

const evidenceMark = makeShape(index => {
  if (index < EVIDENCE_RING_PARTICLES) {
    const angle = index / EVIDENCE_RING_PARTICLES * Math.PI * 2;
    const radius = .205 + (noise(index, 12) - .5) * .025;
    return [.75 + Math.cos(angle) * radius, .49 + Math.sin(angle) * radius];
  }
  const amount = (index - EVIDENCE_RING_PARTICLES) / Math.max(1, EVIDENCE_TRACE_PARTICLES - 1);
  if (amount < .43) {
    const local = amount / .43;
    return [.65 + local * .075, .5 + local * .075];
  }
  const local = (amount - .43) / .57;
  return [.725 + local * .15, .575 - local * .21];
});

// A compact, slightly faceted bar gives the final CTA a tangible destination
// before the field releases back into the open research space.
const copperIngot = makeShape(index => {
  const horizontal = noise(index, 18);
  const vertical = noise(index, 19);
  const lane = index % 8;
  if (lane < 5) {
    // Front face: a shallow trapezoid rather than a flat rectangle.
    const halfWidth = .135 + vertical * .035;
    return [.735 + (horizontal - .5) * halfWidth * 2, .445 + vertical * .17];
  }
  if (lane < 7) {
    // Top face: the offset makes the silhouette read as a small 3D ingot.
    return [.595 + horizontal * .28 + vertical * .025, .378 + vertical * .07];
  }
  // Lower lip catches a soft line of copper as the form settles.
  return [.57 + horizontal * .33, .605 + vertical * .018];
});

const researchSpreadField = makeShape(index => {
  const side = index % 2 === 0 ? -1 : 1;
  const spread = .08 + noise(index, 20) * .39;
  return [.5 + side * spread, .1 + noise(index, 21) * .8];
});

const sourceKeyframes = [
  { at: 0, points: copperForm }, { at: .16, points: copperForm },
  { at: .27, points: dispersedField }, { at: .39, points: marketField },
  { at: .56, points: intelligenceNetwork }, { at: .70, points: forecastPath },
  // The forecast surface has cleared by this point; the evidence mark and
  // ingot therefore resolve in the intentional negative-space tail.
  { at: .74, points: evidenceMark }, { at: .84, points: copperIngot },
  { at: .93, points: copperIngot }, { at: 1, points: researchSpreadField },
];

function sampleShape(source: Float32Array, count: number) {
  if (count === MAX_PARTICLES) return source;
  const sampled = new Float32Array(count * 2);
  for (let index = 0; index < count; index += 1) {
    const sourceIndex = Math.round(index * (MAX_PARTICLES - 1) / (count - 1));
    sampled[index * 2] = source[sourceIndex * 2];
    sampled[index * 2 + 1] = source[sourceIndex * 2 + 1];
  }
  return sampled;
}

function keyframesFor(count: number) {
  const cache = new Map<Float32Array, Float32Array>();
  return sourceKeyframes.map(keyframe => {
    let points = cache.get(keyframe.points);
    if (!points) {
      points = sampleShape(keyframe.points, count);
      cache.set(keyframe.points, points);
    }
    return { at: keyframe.at, points };
  });
}

function segmentFor(value: number) {
  let index = sourceKeyframes.findIndex(keyframe => keyframe.at >= value);
  if (index <= 0) index = 1;
  return index;
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
  draw: (progress: number, pointerX: number, pointerY: number, time: number, scrollImpulse: number) => void;
  resize: (width: number, height: number, ratio: number) => void;
  dispose: () => void;
  name: 'webgl2' | 'canvas2d';
};

function createWebglRenderer(canvas: HTMLCanvasElement, quality: ParticleQuality): Renderer | null {
  const config = QUALITY[quality];
  const gl = canvas.getContext('webgl2', { alpha: true, antialias: false, depth: false, stencil: false, premultipliedAlpha: true, powerPreference: quality === 'high' ? 'high-performance' : 'low-power' });
  if (!gl) return null;
  const compilationStarted = performance.now();
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, `#version 300 es
    precision highp float;
    in vec2 a_from; in vec2 a_to; in float a_size; in float a_blue; in vec4 a_drift;
    uniform float u_amount; uniform float u_progress; uniform float u_time; uniform float u_scroll_impulse; uniform float u_pixel_ratio; uniform float u_aspect;
    uniform vec2 u_pointer; out float v_blue;
    void main() {
      float t = u_amount * u_amount * (3.0 - 2.0 * u_amount);
      vec2 position = mix(a_from, a_to, t);
      vec2 delta = position - u_pointer;
      vec2 metric = delta * vec2(u_aspect, 1.0);
      float push = (1.0 - smoothstep(0.0, 0.095, length(metric))) * 0.018;
      position += normalize(delta + vec2(0.00001)) * push;
      float cTilt = smoothstep(0.02, 0.09, u_progress) * (1.0 - smoothstep(0.17, 0.24, u_progress));
      float cAngle = 0.58 * cTilt;
      vec2 cLocal = position - vec2(0.72, 0.49);
      float cDepth = cLocal.y * sin(cAngle);
      float cPerspective = 1.0 / max(0.78, 1.0 + cDepth * 1.4);
      cLocal.y *= cos(cAngle) * cPerspective;
      cLocal.x = (cLocal.x + cDepth * 0.18) * cPerspective;
      position = vec2(0.72, 0.49) + cLocal + vec2(0.0, 0.018 * cTilt);
      float earlyTide = smoothstep(0.04, 0.16, u_progress) * (1.0 - smoothstep(0.18, 0.3, u_progress));
      position.x += sin(a_drift.x * 1.8 + a_drift.y * 0.65 + u_time * 0.0009 + u_progress * 10.0) * earlyTide * 0.012;
      position.y += cos(a_drift.y * 1.2 + a_drift.x * 0.45 + u_time * 0.0007 + u_progress * 7.0) * earlyTide * 0.0035;
      float driftPhaseX = a_drift.x + u_time * a_drift.z;
      float driftPhaseY = a_drift.y + u_time * a_drift.z * 0.73;
      position.x += sin(driftPhaseX) * a_drift.w;
      position.y += cos(driftPhaseY) * a_drift.w * 0.68;
      float scatterBlend = smoothstep(0.2, 0.38, u_progress);
      float directionPhase = a_drift.x * 1.7 + a_drift.y * 0.45;
      vec2 impulseDirection = normalize(vec2(cos(directionPhase), sin(a_drift.y * 1.3 + a_drift.x)));
      float individualImpulse = 0.35 + fract(sin(a_drift.x * 12.9898 + a_drift.y * 78.233) * 43758.5453) * 0.65;
      position += impulseDirection * u_scroll_impulse * scatterBlend * individualImpulse;
      gl_Position = vec4(position.x * 2.0 - 1.0, 1.0 - position.y * 2.0, 0.0, 1.0);
      gl_PointSize = a_size * u_pixel_ratio * mix(1.0, cPerspective, cTilt);
      float copperSettle = 1.0 - smoothstep(0.82, 0.94, u_progress);
      v_blue = a_blue * clamp((u_progress - 0.6) * 1.2, 0.0, 0.48) * mix(1.0, 0.18, copperSettle);
    }`);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float; in float v_blue; out vec4 color;
    void main() {
      float distanceFromCenter = length(gl_PointCoord - vec2(0.5)) * 2.0;
      float halo = 1.0 - smoothstep(0.18, 1.0, distanceFromCenter);
      float core = 1.0 - smoothstep(0.0, 0.42, distanceFromCenter);
      float alpha = halo * 0.38 + core * 0.66;
      vec3 copper = vec3(0.902, 0.643, 0.478);
      vec3 blue = vec3(0.608, 0.737, 0.984);
      vec3 particle = mix(copper, blue, v_blue);
      color = vec4(particle * alpha, alpha);
    }`);
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertexShader); gl.attachShader(program, fragmentShader); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Unable to link particle shader');
  gl.deleteShader(vertexShader); gl.deleteShader(fragmentShader);
  canvas.dataset.shaderCompileMs = (performance.now() - compilationStarted).toFixed(3);

  const count = config.count;
  const frames = keyframesFor(count);
  const sizes = new Float32Array(count);
  const blue = new Float32Array(count);
  const drift = makeDrift(count);
  for (let index = 0; index < count; index += 1) {
    sizes[index] = index % 11 === 0 ? config.pointSize * 1.22 : config.pointSize;
    blue[index] = index % 17 === 6 ? 1 : 0;
  }

  const buffers: WebGLBuffer[] = [];
  const attribute = (name: string, size: number, data: Float32Array, usage: number) => {
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error(`Unable to create ${name} particle buffer`);
    buffers.push(buffer); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    return buffer;
  };
  const fromBuffer = attribute('a_from', 2, frames[0].points, gl.DYNAMIC_DRAW);
  const toBuffer = attribute('a_to', 2, frames[1].points, gl.DYNAMIC_DRAW);
  attribute('a_size', 1, sizes, gl.STATIC_DRAW); attribute('a_blue', 1, blue, gl.STATIC_DRAW); attribute('a_drift', 4, drift, gl.STATIC_DRAW);

  const amountLocation = gl.getUniformLocation(program, 'u_amount');
  const progressLocation = gl.getUniformLocation(program, 'u_progress');
  const timeLocation = gl.getUniformLocation(program, 'u_time');
  const scrollImpulseLocation = gl.getUniformLocation(program, 'u_scroll_impulse');
  const ratioLocation = gl.getUniformLocation(program, 'u_pixel_ratio');
  const aspectLocation = gl.getUniformLocation(program, 'u_aspect');
  const pointerLocation = gl.getUniformLocation(program, 'u_pointer');
  let currentSegment = -1;
  let width = 1;
  let height = 1;
  let ratio = 1;
  gl.useProgram(program); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return {
    name: 'webgl2',
    resize(nextWidth, nextHeight, nextRatio) {
      width = nextWidth; height = nextHeight; ratio = nextRatio; gl.viewport(0, 0, canvas.width, canvas.height);
    },
    draw(value, pointerX, pointerY, time, scrollImpulse) {
      const segment = segmentFor(value);
      const previous = frames[segment - 1];
      const next = frames[segment];
      if (segment !== currentSegment) {
        gl.bindBuffer(gl.ARRAY_BUFFER, fromBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, previous.points);
        gl.bindBuffer(gl.ARRAY_BUFFER, toBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, next.points);
        currentSegment = segment;
      }
      gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(program);
      gl.uniform1f(amountLocation, (value - previous.at) / Math.max(.001, next.at - previous.at));
      gl.uniform1f(progressLocation, value); gl.uniform1f(timeLocation, time); gl.uniform1f(scrollImpulseLocation, scrollImpulse); gl.uniform1f(ratioLocation, ratio); gl.uniform1f(aspectLocation, width / Math.max(1, height));
      gl.uniform2f(pointerLocation, pointerX, pointerY); gl.drawArrays(gl.POINTS, 0, count);
    },
    dispose() {
      for (const buffer of buffers) gl.deleteBuffer(buffer);
      gl.deleteProgram(program); gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}

function createCanvasRenderer(canvas: HTMLCanvasElement, quality: ParticleQuality): Renderer | null {
  const context = canvas.getContext('2d');
  if (!context) return null;
  const config = QUALITY[quality];
  const frames = keyframesFor(config.count);
  const drift = makeDrift(config.count);
  let width = 1;
  let height = 1;
  return {
    name: 'canvas2d',
    resize(nextWidth, nextHeight, ratio) { width = nextWidth; height = nextHeight; context.setTransform(ratio, 0, 0, ratio, 0, 0); },
    draw(value, pointerX, pointerY, time, scrollImpulse) {
      const segment = segmentFor(value);
      const previous = frames[segment - 1];
      const next = frames[segment];
      const raw = (value - previous.at) / Math.max(.001, next.at - previous.at);
      const amount = raw * raw * (3 - 2 * raw);
      const earlyTide = smoothstep(.04, .16, value) * (1 - smoothstep(.18, .3, value));
      context.clearRect(0, 0, width, height); context.fillStyle = `rgba(230, 164, 122, ${config.canvasAlpha})`; context.beginPath();
      for (let index = 0; index < config.count; index += 1) {
        const offset = index * 2;
        let x = previous.points[offset] + (next.points[offset] - previous.points[offset]) * amount;
        let y = previous.points[offset + 1] + (next.points[offset + 1] - previous.points[offset + 1]) * amount;
        const cTilt = smoothstep(.02, .09, value) * (1 - smoothstep(.17, .24, value));
        const cAngle = .58 * cTilt;
        const cLocalX = x - .72;
        const cLocalY = y - .49;
        const cDepth = cLocalY * Math.sin(cAngle);
        const cPerspective = 1 / Math.max(.78, 1 + cDepth * 1.4);
        x = (.72 + (cLocalX + cDepth * .18) * cPerspective) * width;
        y = (.49 + cLocalY * Math.cos(cAngle) * cPerspective + .018 * cTilt) * height;
        const dx = x - pointerX * width; const dy = y - pointerY * height; const squared = dx * dx + dy * dy;
        const reach = Math.min(width, height) * .095;
        if (squared > 0 && squared < reach * reach) {
          const distance = Math.sqrt(squared); const force = (1 - distance / reach) * 18;
          x += dx / distance * force; y += dy / distance * force;
        }
        const radius = (index % 11 === 0 ? config.largeRadius : config.radius) * (1 + (cPerspective - 1) * cTilt);
        const driftOffset = index * 4;
        const earlyTideX = Math.sin(drift[driftOffset] * 1.8 + drift[driftOffset + 1] * .65 + time * .0009 + value * 10) * earlyTide * .012 * width;
        const earlyTideY = Math.cos(drift[driftOffset + 1] * 1.2 + drift[driftOffset] * .45 + time * .0007 + value * 7) * earlyTide * .0035 * height;
        const driftPhaseX = drift[driftOffset] + time * drift[driftOffset + 2];
        const driftPhaseY = drift[driftOffset + 1] + time * drift[driftOffset + 2] * .73;
        const driftX = Math.sin(driftPhaseX) * drift[driftOffset + 3] * width;
        const driftY = Math.cos(driftPhaseY) * drift[driftOffset + 3] * .68 * height;
        const scatterBlend = Math.max(0, Math.min(1, (value - .2) / .18));
        const directionPhase = drift[driftOffset] * 1.7 + drift[driftOffset + 1] * .45;
        const impulseSeed = Math.sin(drift[driftOffset] * 12.9898 + drift[driftOffset + 1] * 78.233) * 43758.5453;
        const individualImpulse = .35 + (impulseSeed - Math.floor(impulseSeed)) * .65;
        const impulseX = Math.cos(directionPhase) * scrollImpulse * scatterBlend * individualImpulse * width;
        const impulseY = Math.sin(drift[driftOffset + 1] * 1.3 + drift[driftOffset]) * scrollImpulse * scatterBlend * individualImpulse * height;
        context.moveTo(x + radius + earlyTideX + driftX + impulseX, y + earlyTideY + driftY + impulseY); context.arc(x + earlyTideX + driftX + impulseX, y + earlyTideY + driftY + impulseY, radius, 0, Math.PI * 2);
      }
      context.fill();
    },
    dispose() {},
  };
}

export function ParticleWorld({ progress, quality = 'high' }: { progress: MotionValue<number>; quality?: ParticleQuality }) {
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
    let pointerX = -2;
    let pointerY = -2;
    let elapsed = 0;
    let previousTimestamp = performance.now();
    let targetProgress = progress.get();
    let particleProgress = targetProgress;
    let previousParticleProgress = particleProgress;
    let scrollImpulse = 0;

    const draw = (timestamp: number) => {
      frame = 0;
      if (!visible || document.hidden) return;
      const deltaTime = Math.min(50, Math.max(0, timestamp - previousTimestamp));
      elapsed += Math.min(32, deltaTime);
      previousTimestamp = timestamp;
      targetProgress = Math.max(0, Math.min(1, progress.get()));
      const response = 1 - Math.exp(-deltaTime / PARTICLE_RESPONSE_MS);
      particleProgress += (targetProgress - particleProgress) * response;
      if (Math.abs(targetProgress - particleProgress) < .0001) particleProgress = targetProgress;
      const delta = Math.max(-.08, Math.min(.08, particleProgress - previousParticleProgress));
      scrollImpulse = scrollImpulse * Math.pow(SCROLL_IMPULSE_DAMPING, deltaTime / 16.667) + delta * SCROLL_IMPULSE_GAIN;
      previousParticleProgress = particleProgress;
      renderer.draw(particleProgress, pointerX, pointerY, elapsed, scrollImpulse);
      schedule();
    };
    const schedule = () => { if (!frame && visible && !document.hidden) frame = window.requestAnimationFrame(draw); };
    const resize = () => {
      const bounds = activeCanvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, config.dpr);
      width = Math.max(1, bounds.width); height = Math.max(1, bounds.height);
      activeCanvas.width = Math.max(1, Math.round(width * ratio)); activeCanvas.height = Math.max(1, Math.round(height * ratio));
      renderer.resize(width, height, ratio); schedule();
    };
    const onPointerMove = (event: PointerEvent) => { pointerX = event.clientX / width; pointerY = event.clientY / height; schedule(); };
    const onVisibility = () => schedule();
    const resizeObserver = new ResizeObserver(resize);
    const intersectionObserver = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? false; if (visible) schedule(); });
    const unsubscribe = progress.on('change', () => {
      schedule();
    });
    resizeObserver.observe(activeCanvas); intersectionObserver.observe(activeCanvas);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    document.addEventListener('visibilitychange', onVisibility, { passive: true });
    resize();

    return () => {
      unsubscribe();
      if (frame) window.cancelAnimationFrame(frame);
      resizeObserver.disconnect(); intersectionObserver.disconnect();
      window.removeEventListener('pointermove', onPointerMove); document.removeEventListener('visibilitychange', onVisibility);
      renderer.dispose();
      for (const element of [canvas, fallbackCanvas]) {
        element.removeAttribute('data-renderer');
        element.removeAttribute('data-particle-count');
        element.removeAttribute('data-quality');
        element.removeAttribute('data-shader-compile-ms');
        element.style.display = '';
      }
    };
  }, [progress, quality]);

  return <>
    <canvas ref={canvasRef} className="cm-particle-world" data-particle-count={QUALITY[quality].count} data-quality={quality} aria-hidden="true"/>
    <canvas ref={fallbackCanvasRef} className="cm-particle-world cm-particle-world--fallback" aria-hidden="true"/>
  </>;
}
