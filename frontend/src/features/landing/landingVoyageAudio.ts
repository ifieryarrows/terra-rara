import { getObservatoryAudioCore } from '../../utils/cosmicVision';

const LANDING_NEAR_WAV_URL = '/audio/landing-voyage-near.wav';
const LANDING_DEEP_WAV_URL = '/audio/landing-voyage-deep.wav';
const LANDING_GAIN = 0.15;

let landingActive = false;
let landingSessionId = 0;
let landingFadeTimer: number | null = null;
let landingMasterGain: GainNode | null = null;
let landingFilter: BiquadFilterNode | null = null;
let landingNearGain: GainNode | null = null;
let landingDeepGain: GainNode | null = null;
let landingNearSrc: AudioBufferSourceNode | null = null;
let landingDeepSrc: AudioBufferSourceNode | null = null;
let landingFallbackNear: OscillatorNode | null = null;
let landingFallbackDeep: OscillatorNode | null = null;
let landingProgress = 0;
let landingVelocity = 0;
let sfxUnsubscribe: (() => void) | null = null;

function computeLandingMix(progress: number, velocity: number): {
  near: number;
  deep: number;
  cutoff: number;
  rate: number;
} {
  const p = Math.max(0, Math.min(1, Number.isFinite(progress) ? progress : 0));
  const v = Math.max(0, Math.min(1, Number.isFinite(velocity) ? velocity : 0));

  // Smooth crossfade from Hero Solar Orbit (near) -> Flyby crest -> Milky Way Deep Choir (deep)
  const flybyCrest = Math.exp(-Math.pow((p - 0.78) / 0.11, 2));
  const near = Math.max(0.14, 0.86 * (1 - p * 0.82) + 0.24 * flybyCrest);
  const deep = Math.min(1.0, 0.24 + 0.74 * Math.pow(p, 0.82));
  const baseCutoff =
    p < 0.70
      ? 2400 - p * 1215
      : p < 0.84
        ? 1550 + ((p - 0.70) / 0.14) * 1650
        : 3200 - ((p - 0.84) / 0.16) * 550;
  const cutoff = Math.min(4400, baseCutoff + v * 650);
  const rate = 1 + p * 0.004;
  return { near, deep, cutoff, rate };
}

function applyLandingMix(nowOffset = 0.08): void {
  const ctx = getObservatoryAudioCore().getContext();
  if (!ctx || !landingActive) return;
  const now = ctx.currentTime;
  const { near, deep, cutoff, rate } = computeLandingMix(landingProgress, landingVelocity);

  try {
    landingNearGain?.gain.setTargetAtTime(near, now, nowOffset);
    landingDeepGain?.gain.setTargetAtTime(deep, now, nowOffset);
    landingFilter?.frequency.setTargetAtTime(cutoff, now, nowOffset);
    landingNearSrc?.playbackRate.setTargetAtTime(rate, now, 0.14);
    landingDeepSrc?.playbackRate.setTargetAtTime(rate, now, 0.14);
  } catch {
    // Ignore parameter automation errors in minimal mocks
  }
}

function teardownLandingGraph(): void {
  const nodes: Array<{ stop?: () => void; disconnect: () => void } | null> = [
    landingNearSrc,
    landingDeepSrc,
    landingFallbackNear,
    landingFallbackDeep,
  ];
  for (const node of nodes) {
    if (!node) continue;
    try {
      node.stop?.();
    } catch {
      // Ignore
    }
    try {
      node.disconnect();
    } catch {
      // Ignore
    }
  }
  const gains = [landingNearGain, landingDeepGain, landingFilter, landingMasterGain];
  for (const g of gains) {
    if (!g) continue;
    try {
      g.disconnect();
    } catch {
      // Ignore
    }
  }
  landingNearSrc = null;
  landingDeepSrc = null;
  landingFallbackNear = null;
  landingFallbackDeep = null;
  landingNearGain = null;
  landingDeepGain = null;
  landingFilter = null;
  landingMasterGain = null;
}

function startProceduralLanding(
  ctx: AudioContext,
  nearGain: GainNode,
  deepGain: GainNode,
): void {
  const now = ctx.currentTime;
  const oscNear = ctx.createOscillator();
  oscNear.type = 'sine';
  oscNear.frequency.setValueAtTime(261.63, now);
  oscNear.connect(nearGain);
  oscNear.start(now);
  landingFallbackNear = oscNear;

  const oscDeep = ctx.createOscillator();
  oscDeep.type = 'sine';
  oscDeep.frequency.setValueAtTime(87.31, now);
  oscDeep.connect(deepGain);
  oscDeep.start(now);
  landingFallbackDeep = oscDeep;
}

/** Start Landing's scroll-reactive cosmic voyage stems (never fetches milky-way-observatory.wav). */
export async function startLandingVoyageAudio(): Promise<boolean> {
  const core = getObservatoryAudioCore();
  if (typeof window === 'undefined' || core.isSfxMuted()) return false;
  if (landingFadeTimer !== null) {
    window.clearTimeout(landingFadeTimer);
    landingFadeTimer = null;
  }

  if (!sfxUnsubscribe) {
    sfxUnsubscribe = core.subscribeSfx(() => {
      if (core.isSfxMuted() && landingActive) {
        stopLandingVoyageAudio();
      }
    });
  }

  const session = ++landingSessionId;
  landingActive = true;
  core.setExternalStreamActive(true);
  core.markUnlocked();

  try {
    const ctx = core.ensureContext();
    if (!ctx) return false;
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    if (!landingActive || core.isSfxMuted() || landingSessionId !== session) return false;

    teardownLandingGraph();

    const now = ctx.currentTime;
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(0.0001, now);
    masterGain.gain.exponentialRampToValueAtTime(LANDING_GAIN, now + 1.05);
    masterGain.connect(ctx.destination);
    landingMasterGain = masterGain;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.Q.setValueAtTime(0.65, now);
    filter.connect(masterGain);
    landingFilter = filter;

    const nearGain = ctx.createGain();
    nearGain.connect(filter);
    landingNearGain = nearGain;

    const deepGain = ctx.createGain();
    deepGain.connect(filter);
    landingDeepGain = deepGain;

    applyLandingMix(0.02);

    const [nearBuf, deepBuf] = await Promise.all([
      core.loadUrl(LANDING_NEAR_WAV_URL, ctx),
      core.loadUrl(LANDING_DEEP_WAV_URL, ctx),
    ]);
    if (
      !landingActive ||
      core.isSfxMuted() ||
      landingSessionId !== session ||
      landingMasterGain !== masterGain
    ) {
      return false;
    }

    if (nearBuf && deepBuf) {
      const startAt = ctx.currentTime;
      const nearSrc = ctx.createBufferSource();
      nearSrc.buffer = nearBuf;
      nearSrc.loop = true;
      nearSrc.connect(nearGain);
      nearSrc.start(startAt);
      landingNearSrc = nearSrc;

      const deepSrc = ctx.createBufferSource();
      deepSrc.buffer = deepBuf;
      deepSrc.loop = true;
      deepSrc.connect(deepGain);
      deepSrc.start(startAt);
      landingDeepSrc = deepSrc;

      applyLandingMix(0.05);
    } else {
      startProceduralLanding(ctx, nearGain, deepGain);
    }
    return true;
  } catch {
    return false;
  }
}

/** Modulate Landing's dual cosmic stems and lowpass cutoff with scroll progress and velocity. */
export function updateLandingVoyageScroll(progress: number, velocity = 0): void {
  landingProgress = Math.max(0, Math.min(1, Number.isFinite(progress) ? progress : 0));
  landingVelocity = Math.max(0, Math.min(1, Number.isFinite(velocity) ? velocity : 0));
  if (landingActive && !getObservatoryAudioCore().isSfxMuted()) {
    applyLandingMix(0.09);
  }
}

/** Stop and clean up Landing's scroll-synced soundscape when muted or leaving Landing. */
export function stopLandingVoyageAudio(): void {
  if (typeof window === 'undefined') return;
  const core = getObservatoryAudioCore();
  landingSessionId += 1;
  landingActive = false;
  core.setExternalStreamActive(false);

  if (landingFadeTimer !== null) {
    window.clearTimeout(landingFadeTimer);
    landingFadeTimer = null;
  }

  const ctx = core.getContext();
  const masterGain = landingMasterGain;
  if (ctx && masterGain && ctx.state === 'running') {
    try {
      const now = ctx.currentTime;
      masterGain.gain.cancelScheduledValues(now);
      masterGain.gain.setValueAtTime(Math.max(0.0001, masterGain.gain.value), now);
      masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.35);
      landingFadeTimer = window.setTimeout(() => {
        teardownLandingGraph();
        landingFadeTimer = null;
      }, 390);
      return;
    } catch {
      // Fall through to synchronous teardown
    }
  }
  teardownLandingGraph();
}
