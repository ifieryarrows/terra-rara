import { useCallback, useEffect, useSyncExternalStore } from 'react';

const DRONE_WAV_URL = '/audio/milky-way-observatory.wav';
const CHIME_WAV_URL = '/audio/observatory-chime.wav';
const NOTIFICATION_WAV_URL = '/audio/observatory-notification.wav';
const HOVER_WAV_URL = '/audio/observatory-hover.wav';
const TRANSITION_WAV_URL = '/audio/observatory-transition.wav';
const DISPERSE_WAV_URL = '/audio/observatory-disperse.wav';
const CUPRUM_CHIRP_WAV_URL = '/audio/cuprum-chirp.wav';
const CUPRUM_TAP_WAV_URL = '/audio/cuprum-tap.wav';
const CUPRUM_SPEAK_WAV_URL = '/audio/cuprum-speak.wav';

const AUDIO_URLS = {
  drone: DRONE_WAV_URL,
  chime: CHIME_WAV_URL,
  notification: NOTIFICATION_WAV_URL,
  hover: HOVER_WAV_URL,
  transition: TRANSITION_WAV_URL,
  disperse: DISPERSE_WAV_URL,
  cuprumChirp: CUPRUM_CHIRP_WAV_URL,
  cuprumTap: CUPRUM_TAP_WAV_URL,
  cuprumSpeak: CUPRUM_SPEAK_WAV_URL,
} as const;

type AudioCacheKey = keyof typeof AUDIO_URLS;
type OneShotCooldownKey = Exclude<AudioCacheKey, 'drone'>;

const DRONE_GAIN = 0.16;
const HOVER_GAIN = 0.15;
const CHIME_GAIN = 0.25;
const NOTIFICATION_GAIN = 0.18;
const TRANSITION_GAIN = 0.28;
const DISPERSE_GAIN = 0.28;
const CUPRUM_CHIRP_GAIN = 0.24;
const CUPRUM_TAP_GAIN = 0.22;
const CUPRUM_SPEAK_GAIN = 0.20;
const HOVER_COOLDOWN_MS = 75;
const CHIME_COOLDOWN_MS = 110;
const NOTIFICATION_COOLDOWN_MS = 180;
const TRANSITION_COOLDOWN_MS = 350;
const DISPERSE_COOLDOWN_MS = 350;
const CUPRUM_CHIRP_COOLDOWN_MS = 120;
const CUPRUM_TAP_COOLDOWN_MS = 85;
const CUPRUM_SPEAK_COOLDOWN_MS = 42;
const INTERACTIVE_SELECTOR = 'button, a[href], summary, [role="button"], [role="tab"]';
const SPEAK_PITCH_CURVE = [0.94, 1.04, 1.12, 0.99, 1.09, 0.96, 1.16] as const;

type OneShotConfig = {
  key: Exclude<AudioCacheKey, 'drone'>;
  url: string;
  cooldownKey: OneShotCooldownKey;
  cooldownMs: number;
  defaultVolume: number;
  maxVolume: number;
  playbackRate?: number;
  sculpted?: boolean;
  choke?: boolean;
  requireActivation?: boolean;
  fallback: (ctx: AudioContext, gain: GainNode) => boolean;
};

type Listener = () => void;

function isMockedFn(fn: unknown): boolean {
  return typeof fn === 'function' && 'mock' in fn;
}

function canUseHtmlAudio(): boolean {
  if (typeof window === 'undefined' || typeof Audio === 'undefined') return false;
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  if (!/\bjsdom\b/i.test(ua)) return true;
  return (
    isMockedFn(Audio) ||
    (typeof HTMLMediaElement !== 'undefined' && isMockedFn(HTMLMediaElement.prototype.play))
  );
}

/**
 * Shared Milky Way Observatory audio engine.
 * - Starts ambient drone muted/inactive; never autoplays or blocks the initial bundle.
 * - Lazy-fetches `/audio/milky-way-observatory.wav` only on the first explicit unmute toggle.
 * - Provides non-fatiguing stardust hover whispers (`observatory-hover.wav`) and
 *   custom cosmic-fantasy CopperMind selection chimes (`observatory-chime.wav`) once
 *   user activation unlocks audio (unless muted by toggling sound OFF).
 */
class ObservatoryAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private bufferSource: AudioBufferSourceNode | null = null;
  private fallbackOsc1: OscillatorNode | null = null;
  private fallbackOsc2: OscillatorNode | null = null;
  private fallbackNoise: AudioBufferSourceNode | null = null;
  private htmlAudio: HTMLAudioElement | null = null;

  private urlBuffers: Record<string, AudioBuffer> = {};
  private urlPromises: Record<string, Promise<AudioBuffer | null>> = {};

  private fadeTimer: number | null = null;
  private externalStreamActive = false;
  private activeOneShots: Partial<Record<OneShotCooldownKey, { src: AudioBufferSourceNode; gain: GainNode }>> = {};

  private desiredActive = false;
  private sfxMuted = false;
  private unlocked = false;
  private sessionId = 0;
  private lastPlayedAt: Record<OneShotCooldownKey, number> = {
    chime: 0,
    notification: 0,
    hover: 0,
    transition: 0,
    disperse: 0,
    cuprumChirp: 0,
    cuprumTap: 0,
    cuprumSpeak: 0,
  };
  private listeners = new Set<Listener>();
  private sfxListeners = new Set<Listener>();
  private mountCount = 0;

  public subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  public subscribeSfx = (listener: Listener): (() => void) => {
    this.sfxListeners.add(listener);
    return () => {
      this.sfxListeners.delete(listener);
    };
  };

  public getSnapshot = (): boolean => this.desiredActive;

  public getSfxSnapshot = (): boolean => !this.sfxMuted;

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  private emitSfx(): void {
    for (const listener of this.sfxListeners) {
      listener();
    }
  }

  public registerMount(): () => void {
    this.mountCount += 1;
    return () => {
      this.mountCount = Math.max(0, this.mountCount - 1);
      if (this.mountCount === 0 && this.desiredActive) {
        this.stop();
      }
    };
  }

  public isRunning(): boolean {
    return this.desiredActive;
  }

  public isSfxMuted(): boolean {
    return this.sfxMuted;
  }

  public setSfxEnabled(enabled: boolean): void {
    const nextMuted = !enabled;
    if (nextMuted === this.sfxMuted && (enabled || !this.desiredActive)) return;
    this.sfxMuted = nextMuted;
    this.emitSfx();
    if (!enabled && this.desiredActive) {
      this.stop();
    }
  }

  public toggleSfx(force?: boolean): void {
    const next = typeof force === 'boolean' ? force : this.sfxMuted;
    this.setSfxEnabled(next);
  }

  public setEnabled(enabled: boolean): void {
    if (enabled === this.desiredActive) return;
    if (enabled) {
      if (this.sfxMuted) {
        this.sfxMuted = false;
        this.emitSfx();
      }
      void this.start();
    } else {
      if (!this.sfxMuted) {
        this.sfxMuted = true;
        this.emitSfx();
      }
      this.stop();
    }
  }

  public toggle(force?: boolean): void {
    const next = typeof force === 'boolean' ? force : !this.desiredActive;
    this.setEnabled(next);
  }

  public getContext(): AudioContext | null {
    return this.ctx;
  }

  public setExternalStreamActive(active: boolean): void {
    this.externalStreamActive = active;
  }

  public markUnlocked(): void {
    this.unlocked = true;
  }

  public ensureAudioContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return null;
    if (!this.ctx || this.ctx.state === 'closed') {
      this.ctx = new AudioCtx();
    }
    return this.ctx;
  }

  public loadDecodedUrl(url: string, ctx: AudioContext): Promise<AudioBuffer | null> {
    const cached = this.urlBuffers[url];
    if (cached) return Promise.resolve(cached);

    const pending = this.urlPromises[url];
    if (pending) return pending;

    const promise = (async () => {
      try {
        if (typeof fetch !== 'function') return null;
        const response = await fetch(url);
        if (!response.ok) return null;
        const arrayBuffer = await response.arrayBuffer();
        if (!arrayBuffer || arrayBuffer.byteLength === 0) return null;
        const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
        this.urlBuffers[url] = decoded;
        return decoded;
      } catch {
        return null;
      } finally {
        delete this.urlPromises[url];
      }
    })();

    this.urlPromises[url] = promise;
    return promise;
  }

  private loadDecodedBuffer(cacheKey: AudioCacheKey, ctx: AudioContext): Promise<AudioBuffer | null> {
    return this.loadDecodedUrl(AUDIO_URLS[cacheKey], ctx);
  }

  public unlockOnGesture(): void {
    if (typeof window === 'undefined' || this.sfxMuted) return;
    this.unlocked = true;
    try {
      const ctx = this.ensureAudioContext();
      if (!ctx) return;
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }
      void this.loadDecodedBuffer('hover', ctx);
      void this.loadDecodedBuffer('chime', ctx);
    } catch {
      // Ignore in restricted environments
    }
  }

  public async start(): Promise<void> {
    if (typeof window === 'undefined') return;

    if (this.fadeTimer !== null) {
      window.clearTimeout(this.fadeTimer);
      this.fadeTimer = null;
    }

    const currentSession = ++this.sessionId;
    this.sfxMuted = false;
    this.unlocked = true;
    if (!this.desiredActive) {
      this.desiredActive = true;
      this.emit();
    }

    try {
      const ctx = this.ensureAudioContext();
      if (!ctx) {
        this.startHtmlAudioFallback();
        return;
      }

      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      if (!this.desiredActive || this.sessionId !== currentSession) return;

      // Clear any previous active graph before creating fresh nodes
      this.teardownGraph(false);

      const now = ctx.currentTime;
      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(0.0001, now);
      masterGain.gain.exponentialRampToValueAtTime(DRONE_GAIN, now + 1.25);
      masterGain.connect(ctx.destination);
      this.masterGain = masterGain;

      const decoded = await this.loadDecodedBuffer('drone', ctx);
      if (!this.desiredActive || this.sessionId !== currentSession || this.masterGain !== masterGain) {
        return;
      }

      if (decoded) {
        const source = ctx.createBufferSource();
        source.buffer = decoded;
        source.loop = true;
        source.connect(masterGain);
        source.start(ctx.currentTime);
        this.bufferSource = source;
        void this.loadDecodedBuffer('chime', ctx);
        void this.loadDecodedBuffer('hover', ctx);
      } else {
        this.startProceduralDrone(ctx, masterGain);
      }
    } catch {
      // Keep UI toggle resilient in headless / restricted environments
    }
  }

  private startHtmlAudioFallback(): void {
    if (!canUseHtmlAudio()) return;
    try {
      if (!this.htmlAudio) {
        const audio = new Audio(DRONE_WAV_URL);
        audio.loop = true;
        audio.volume = 0.22;
        this.htmlAudio = audio;
      }
      void this.htmlAudio.play()?.catch(() => {});
    } catch {
      // Ignore in jsdom / restricted environments
    }
  }

  private startProceduralDrone(ctx: AudioContext, masterGain: GainNode): void {
    const now = ctx.currentTime;

    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(73.5, now);
    const osc1Gain = ctx.createGain();
    osc1Gain.gain.setValueAtTime(0.42, now);
    osc1.connect(osc1Gain);
    osc1Gain.connect(masterGain);
    osc1.start(now);
    this.fallbackOsc1 = osc1;

    const osc2 = ctx.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(110.25, now);
    const osc2Gain = ctx.createGain();
    osc2Gain.gain.setValueAtTime(0.28, now);
    osc2.connect(osc2Gain);
    osc2Gain.connect(masterGain);
    osc2.start(now);
    this.fallbackOsc2 = osc2;

    const sampleRate = ctx.sampleRate || 44100;
    const bufferLength = Math.floor(sampleRate * 2.5);
    const noiseBuffer = ctx.createBuffer(1, bufferLength, sampleRate);
    const channelData = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferLength; i += 1) {
      channelData[i] = (Math.random() * 2 - 1) * 0.28;
    }

    const noiseSource = ctx.createBufferSource();
    noiseSource.buffer = noiseBuffer;
    noiseSource.loop = true;

    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'lowpass';
    noiseFilter.frequency.setValueAtTime(160, now);
    noiseFilter.Q.setValueAtTime(0.9, now);

    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.16, now);

    noiseSource.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(masterGain);
    noiseSource.start(now);
    this.fallbackNoise = noiseSource;
  }

  public stop(): void {
    if (typeof window === 'undefined') return;
    this.sessionId += 1;

    if (this.desiredActive) {
      this.desiredActive = false;
      this.emit();
    }

    if (this.fadeTimer !== null) {
      window.clearTimeout(this.fadeTimer);
      this.fadeTimer = null;
    }

    if (this.htmlAudio) {
      try {
        this.htmlAudio.pause();
        this.htmlAudio.currentTime = 0;
      } catch {
        // Ignore
      }
    }

    const ctx = this.ctx;
    const masterGain = this.masterGain;

    if (ctx && masterGain && ctx.state === 'running') {
      try {
        const now = ctx.currentTime;
        masterGain.gain.cancelScheduledValues(now);
        masterGain.gain.setValueAtTime(Math.max(0.0001, masterGain.gain.value), now);
        masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.55);

        this.fadeTimer = window.setTimeout(() => {
          this.teardownGraph(true);
          this.fadeTimer = null;
        }, 600);
      } catch {
        this.teardownGraph(true);
      }
    } else {
      this.teardownGraph(true);
    }
  }

  private teardownGraph(suspendContext: boolean): void {
    const nodes = [this.bufferSource, this.fallbackOsc1, this.fallbackOsc2, this.fallbackNoise];
    for (const node of nodes) {
      if (!node) continue;
      try {
        node.stop();
      } catch {
        // Ignore if already stopped
      }
      try {
        node.disconnect();
      } catch {
        // Ignore
      }
    }
    try {
      this.masterGain?.disconnect();
    } catch {
      // Ignore
    }

    this.bufferSource = null;
    this.fallbackOsc1 = null;
    this.fallbackOsc2 = null;
    this.fallbackNoise = null;
    this.masterGain = null;

    if (suspendContext && !this.externalStreamActive && this.ctx && this.ctx.state === 'running') {
      try {
        void this.ctx.suspend().catch(() => {});
      } catch {
        // Ignore
      }
    }
  }

  private playHtmlOneShot(url: string, volume: number): boolean {
    if (!canUseHtmlAudio()) return false;
    try {
      const sound = new Audio(url);
      sound.volume = volume;
      void sound.play()?.catch(() => {});
      return true;
    } catch {
      return false;
    }
  }

  private playProceduralHover(ctx: AudioContext, gain: GainNode): boolean {
    const start = ctx.currentTime;
    const sampleRate = ctx.sampleRate || 24000;
    const length = Math.max(1, Math.floor(sampleRate * 0.42));
    const buffer = ctx.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i += 1) {
      data[i] = Math.random() * 2 - 1;
    }

    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(950, start);
    filter.Q.setValueAtTime(0.7, start);

    const shelf = ctx.createBiquadFilter();
    shelf.type = 'highshelf';
    shelf.frequency.setValueAtTime(2800, start);
    shelf.gain.setValueAtTime(-12, start);

    const airEnv = ctx.createGain();
    airEnv.gain.setValueAtTime(0.0001, start);
    airEnv.gain.exponentialRampToValueAtTime(0.14, start + 0.09);
    airEnv.gain.exponentialRampToValueAtTime(0.0001, start + 0.40);

    // Slow, spacey D-Lydian cosmic pad voices matching milky-way-observatory.wav
    const rootOsc = ctx.createOscillator();
    rootOsc.type = 'sine';
    rootOsc.frequency.setValueAtTime(220.5, start);

    const octaveOsc = ctx.createOscillator();
    octaveOsc.type = 'sine';
    octaveOsc.frequency.setValueAtTime(294, start);

    const padEnv = ctx.createGain();
    padEnv.gain.setValueAtTime(0.0001, start);
    padEnv.gain.exponentialRampToValueAtTime(0.32, start + 0.095);
    padEnv.gain.exponentialRampToValueAtTime(0.0001, start + 0.41);

    noise.connect(filter);
    filter.connect(shelf);
    shelf.connect(airEnv);
    airEnv.connect(gain);

    rootOsc.connect(padEnv);
    octaveOsc.connect(padEnv);
    padEnv.connect(gain);

    noise.start(start);
    noise.stop(start + 0.41);
    rootOsc.start(start);
    rootOsc.stop(start + 0.42);
    octaveOsc.start(start);
    octaveOsc.stop(start + 0.42);
    octaveOsc.onended = () => {
      try {
        noise.disconnect();
        filter.disconnect();
        shelf.disconnect();
        airEnv.disconnect();
        rootOsc.disconnect();
        octaveOsc.disconnect();
        padEnv.disconnect();
        gain.disconnect();
      } catch {
        // Ignore
      }
    };
    return true;
  }

  public playHover(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'hover',
        url: HOVER_WAV_URL,
        cooldownKey: 'hover',
        cooldownMs: HOVER_COOLDOWN_MS,
        defaultVolume: HOVER_GAIN,
        maxVolume: 0.35,
        choke: true,
        requireActivation: true,
        fallback: (ctx, gain) => this.playProceduralHover(ctx, gain),
      },
      options,
    );
  }

  private playProceduralVoices(
    ctx: AudioContext,
    gain: GainNode,
    voices: Array<[OscillatorType, number, number, number, number, number, number, number, number?]>,
  ): boolean {
    const t0 = ctx.currentTime;
    const cleanup: Array<{ disconnect: () => void }> = [gain];
    let lastOsc: OscillatorNode | null = null;
    for (const [type, f0, f1, rampAt, peak, atk, decay, stopAt, delay = 0] of voices) {
      const s = t0 + delay;
      const osc = ctx.createOscillator();
      const env = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(f0, s);
      osc.frequency.exponentialRampToValueAtTime(f1, t0 + rampAt);
      env.gain.setValueAtTime(0.0001, s);
      env.gain.exponentialRampToValueAtTime(peak, s + atk);
      env.gain.exponentialRampToValueAtTime(0.0001, t0 + decay);
      osc.connect(env);
      env.connect(gain);
      osc.start(s);
      osc.stop(t0 + stopAt);
      cleanup.push(osc, env);
      lastOsc = osc;
    }
    if (lastOsc) {
      lastOsc.onended = () => {
        for (const n of cleanup) {
          try {
            n.disconnect();
          } catch {
            // Ignore
          }
        }
      };
    }
    return true;
  }

  private async playOneShot(
    config: OneShotConfig,
    options?: { force?: boolean; volume?: number },
  ): Promise<boolean> {
    if (typeof window === 'undefined') return false;
    const allowWhenMuted = Boolean(options?.force);
    if (this.sfxMuted && !allowWhenMuted) return false;

    if (config.requireActivation && !allowWhenMuted) {
      const nav = typeof navigator !== 'undefined'
        ? (navigator as Navigator & { userActivation?: { hasBeenActive: boolean } })
        : null;
      const hasActivation =
        this.unlocked ||
        this.ctx?.state === 'running' ||
        Boolean(nav?.userActivation?.hasBeenActive);
      if (!hasActivation) return false;
    } else {
      this.unlocked = true;
    }

    const nowMs = Date.now();
    if (nowMs - this.lastPlayedAt[config.cooldownKey] < config.cooldownMs) return false;
    this.lastPlayedAt[config.cooldownKey] = nowMs;

    const targetVolume = Math.max(0.01, Math.min(config.maxVolume, options?.volume ?? config.defaultVolume));

    try {
      const ctx = this.ensureAudioContext();
      if (!ctx) return this.playHtmlOneShot(config.url, targetVolume);

      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      const decoded = await this.loadDecodedBuffer(config.key, ctx);
      if (this.sfxMuted && !allowWhenMuted) return false;

      const now = ctx.currentTime;
      const activePrev = config.choke ? this.activeOneShots[config.cooldownKey] : undefined;
      if (activePrev) {
        try {
          activePrev.gain.gain.cancelScheduledValues?.(now);
          activePrev.gain.gain.setTargetAtTime(0.0001, now, 0.003);
          activePrev.src.stop(now + 0.014);
        } catch {
          // Ignore already stopped source
        }
        delete this.activeOneShots[config.cooldownKey];
      }

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, now);
      if (typeof gain.gain.linearRampToValueAtTime === 'function') {
        gain.gain.linearRampToValueAtTime(targetVolume, now + 0.004);
      } else {
        gain.gain.setValueAtTime(targetVolume, now);
      }
      gain.connect(ctx.destination);

      let entryNode: AudioNode = gain;
      const chainNodes: AudioNode[] = [gain];
      if (config.sculpted && typeof ctx.createBiquadFilter === 'function') {
        // Cut sub-140 Hz mud and soften transients above 7500 Hz for gentle repeated clicks.
        const hp = ctx.createBiquadFilter();
        hp.type = 'highpass';
        hp.frequency.setValueAtTime(140, now);
        hp.Q.setValueAtTime(0.7, now);
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.setValueAtTime(7500, now);
        lp.Q.setValueAtTime(0.7, now);
        hp.connect(lp);
        lp.connect(gain);
        entryNode = hp;
        chainNodes.push(hp, lp);
      }

      if (decoded) {
        const source = ctx.createBufferSource();
        source.buffer = decoded;
        if (config.playbackRate && source.playbackRate) {
          source.playbackRate.value = config.playbackRate;
        }
        source.connect(entryNode);
        if (config.choke) {
          this.activeOneShots[config.cooldownKey] = { src: source, gain };
        }
        source.onended = () => {
          if (this.activeOneShots[config.cooldownKey]?.src === source) {
            delete this.activeOneShots[config.cooldownKey];
          }
          try {
            source.disconnect();
            for (const node of chainNodes) node.disconnect();
          } catch {
            // Ignore
          }
        };
        source.start(now);
        return true;
      }

      return config.fallback(ctx, entryNode as GainNode);
    } catch {
      return false;
    }
  }

  public playChime(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    const rate = 0.98 + Math.random() * 0.04;
    return this.playOneShot(
      {
        key: 'chime',
        url: CHIME_WAV_URL,
        cooldownKey: 'chime',
        cooldownMs: CHIME_COOLDOWN_MS,
        defaultVolume: CHIME_GAIN,
        maxVolume: 0.3,
        playbackRate: rate,
        sculpted: true,
        choke: true,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [
            ['sine', 960 * rate, 1176 * rate, 0.026, 0.16, 0.011, 0.24, 0.25],
            ['sine', 1480 * rate, 1480 * rate, 0.02, 0.06, 0.014, 0.3, 0.31, 0.008],
            ['sine', 1764 * rate, 1764 * rate, 0.02, 0.035, 0.018, 0.31, 0.32, 0.012],
          ]),
      },
      options,
    );
  }

  public playNotification(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'notification',
        url: NOTIFICATION_WAV_URL,
        cooldownKey: 'notification',
        cooldownMs: NOTIFICATION_COOLDOWN_MS,
        defaultVolume: NOTIFICATION_GAIN,
        maxVolume: 0.45,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [
            ['sine', 520, 588, 0.014, 0.12, 0.007, 0.11, 0.12],
            ['sine', 784, 882, 0.074, 0.28, 0.009, 0.255, 0.265, 0.058],
            ['sine', 1568, 1764, 0.074, 0.07, 0.009, 0.22, 0.23, 0.058],
          ]),
      },
      options,
    );
  }

  public playTransition(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'transition',
        url: TRANSITION_WAV_URL,
        cooldownKey: 'transition',
        cooldownMs: TRANSITION_COOLDOWN_MS,
        defaultVolume: TRANSITION_GAIN,
        maxVolume: 0.65,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [
            ['sine', 147, 294, 1.35, 0.2, 1.35, 2.35, 2.4],
            ['triangle', 220.5, 441, 1.35, 0.2, 1.35, 2.35, 2.4],
          ]),
      },
      options,
    );
  }

  public playDisperse(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'disperse',
        url: DISPERSE_WAV_URL,
        cooldownKey: 'disperse',
        cooldownMs: DISPERSE_COOLDOWN_MS,
        defaultVolume: DISPERSE_GAIN,
        maxVolume: 0.6,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [
            ['sine', 588, 294, 1.35, 0.16, 0.05, 1.48, 1.5],
            ['triangle', 441, 220.5, 1.35, 0.16, 0.05, 1.48, 1.5],
          ]),
      },
      options,
    );
  }

  public playCuprumSpeakBlip(options?: { pitchStep?: number; force?: boolean; volume?: number }): Promise<boolean> {
    const idx = Math.abs(options?.pitchStep ?? 0) % SPEAK_PITCH_CURVE.length;
    const rate = SPEAK_PITCH_CURVE[idx];
    return this.playOneShot(
      {
        key: 'cuprumSpeak',
        url: CUPRUM_SPEAK_WAV_URL,
        cooldownKey: 'cuprumSpeak',
        cooldownMs: CUPRUM_SPEAK_COOLDOWN_MS,
        defaultVolume: CUPRUM_SPEAK_GAIN,
        maxVolume: 0.35,
        playbackRate: rate,
        requireActivation: true,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [['triangle', 485 * rate, 542 * rate, 0.032, 0.28, 0.006, 0.058, 0.062]]),
      },
      options,
    );
  }

  public playCuprumChirp(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'cuprumChirp',
        url: CUPRUM_CHIRP_WAV_URL,
        cooldownKey: 'cuprumChirp',
        cooldownMs: CUPRUM_CHIRP_COOLDOWN_MS,
        defaultVolume: CUPRUM_CHIRP_GAIN,
        maxVolume: 0.55,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [
            ['triangle', 392, 507, 0.06, 0.34, 0.015, 0.095, 0.105],
            ['sine', 588, 670, 0.16, 0.42, 0.016, 0.23, 0.24, 0.074],
          ]),
      },
      options,
    );
  }

  public playCuprumTap(options?: { force?: boolean; volume?: number }): Promise<boolean> {
    return this.playOneShot(
      {
        key: 'cuprumTap',
        url: CUPRUM_TAP_WAV_URL,
        cooldownKey: 'cuprumTap',
        cooldownMs: CUPRUM_TAP_COOLDOWN_MS,
        defaultVolume: CUPRUM_TAP_GAIN,
        maxVolume: 0.5,
        fallback: (ctx, gain) =>
          this.playProceduralVoices(ctx, gain, [['triangle', 740, 588, 0.025, 0.42, 0.005, 0.105, 0.115]]),
      },
      options,
    );
  }
}

const sharedAudioEngine = new ObservatoryAudioEngine();

function findInteractiveControl(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) return null;
  const control = target.closest(INTERACTIVE_SELECTOR);
  if (!(control instanceof HTMLElement)) return null;
  if ((control as HTMLButtonElement).disabled || control.getAttribute('aria-disabled') === 'true') {
    return null;
  }
  if (control.dataset.cmNoSound === 'true') return null;
  return control;
}

/**
 * Bind delegated UI hover whisper & cosmic-fantasy click feedback across all buttons and interactive tabs/CTAs.
 */
export function attachObservatoryInteractionAudio(doc: Document = document): () => void {
  let lastHoverControl: HTMLElement | null = null;

  const onFirstActivation = () => {
    sharedAudioEngine.unlockOnGesture();
  };

  const onPointerOver = (event: PointerEvent) => {
    if (event.pointerType === 'touch' || event.buttons !== 0) return;
    const control = findInteractiveControl(event.target);
    if (!control || control === lastHoverControl) return;
    lastHoverControl = control;
    void sharedAudioEngine.playHover();
  };

  const onPointerOut = (event: PointerEvent) => {
    if (!lastHoverControl) return;
    if (findInteractiveControl(event.relatedTarget) !== lastHoverControl) {
      lastHoverControl = null;
    }
  };

  const onClick = (event: MouseEvent) => {
    if (doc.documentElement.dataset.cmSuppressClickSound === '1') {
      delete doc.documentElement.dataset.cmSuppressClickSound;
      return;
    }
    const control = findInteractiveControl(event.target);
    if (!control || control.dataset.cmClickSound === 'off') return;
    const customSound =
      control.dataset.cmSound ??
      control.closest<HTMLElement>('[data-cm-sound-scope]')?.dataset.cmSoundScope;
    if (customSound === 'cuprum-pet') {
      void sharedAudioEngine.playCuprumChirp();
      return;
    }
    if (customSound === 'cuprum-tap') {
      void sharedAudioEngine.playCuprumTap();
      return;
    }
    void sharedAudioEngine.playChime();
  };

  doc.addEventListener('pointerdown', onFirstActivation, { passive: true, capture: true });
  doc.addEventListener('keydown', onFirstActivation, { passive: true, capture: true });
  doc.addEventListener('pointerover', onPointerOver, { passive: true });
  doc.addEventListener('pointerout', onPointerOut, { passive: true });
  doc.addEventListener('click', onClick, { passive: true });

  return () => {
    doc.removeEventListener('pointerdown', onFirstActivation, true);
    doc.removeEventListener('keydown', onFirstActivation, true);
    doc.removeEventListener('pointerover', onPointerOver);
    doc.removeEventListener('pointerout', onPointerOut);
    doc.removeEventListener('click', onClick);
  };
}

/** React hook mounting global button hover/click observatory sound effects once at app root. */
export function useObservatoryInteractionAudio(): void {
  useEffect(() => attachObservatoryInteractionAudio(document), []);
}

/** Returns whether Observatory ambient sound is currently active. */
export function isObservatorySoundEnabled(): boolean {
  return sharedAudioEngine.isRunning();
}

/** Explicitly enable or disable Observatory ambient music + UI SFX. */
export function setObservatorySound(enabled: boolean): void {
  sharedAudioEngine.setEnabled(enabled);
}

/** Explicitly enable or disable Observatory UI SFX without starting ambient drone music. */
export function setObservatorySfxEnabled(enabled: boolean): void {
  sharedAudioEngine.setSfxEnabled(enabled);
}

/** Stop any playing ambient observatory song while leaving UI SFX untouched. */
export function stopObservatoryAmbient(): void {
  sharedAudioEngine.stop();
}

/**
 * Play the slow, spacey D-Lydian pad breath (`observatory-hover.wav`) on button hover.
 */
export function playObservatoryHover(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playHover(options);
}

/**
 * Play the slender, serene Milky Way cosmic selection thread (`observatory-chime.wav`).
 */
export function playObservatoryChime(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playChime(options);
}

/**
 * Play the reserved ascending "din-DIN" notification chime (`observatory-notification.wav`).
 */
export function playObservatoryNotification(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playNotification(options);
}

/**
 * Play the particle gather transition sweep (`observatory-transition.wav`).
 */
export function playObservatoryTransition(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playTransition(options);
}

/**
 * Play the outward particle dispersion shimmer (`observatory-disperse.wav`).
 */
export function playObservatoryDisperse(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playDisperse(options);
}

/**
 * Play Cuprum's expressive "mi" speech blip while typing dialogue (`cuprum-speak.wav`).
 */
export function playCuprumSpeakBlip(options?: { pitchStep?: number; force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playCuprumSpeakBlip(options);
}

/** Expose shared Web Audio core primitives for lazy route-specific audio subsystems. */
export function getObservatoryAudioCore() {
  return {
    getContext: () => sharedAudioEngine.getContext(),
    ensureContext: () => sharedAudioEngine.ensureAudioContext(),
    loadUrl: (url: string, ctx: AudioContext) => sharedAudioEngine.loadDecodedUrl(url, ctx),
    isSfxMuted: () => sharedAudioEngine.isSfxMuted(),
    subscribeSfx: (listener: () => void) => sharedAudioEngine.subscribeSfx(listener),
    markUnlocked: () => sharedAudioEngine.markUnlocked(),
    setExternalStreamActive: (active: boolean) => sharedAudioEngine.setExternalStreamActive(active),
  };
}

/**
 * Play Cuprum's warm pixel-copper chirp (`cuprum-chirp.wav`) when clicking the companion pet.
 */
export function playCuprumChirp(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playCuprumChirp(options);
}

/**
 * Play Cuprum's crisp patina button tap (`cuprum-tap.wav`) when clicking pet UI controls.
 */
export function playCuprumTap(options?: { force?: boolean; volume?: number }): Promise<boolean> {
  return sharedAudioEngine.playCuprumTap(options);
}

/**
 * UI SFX-only sound toggle hook (starts enabled ON; never fetches or plays ambient background drone).
 * Used on the Landing experience where background music is disabled while hover/click/transition SFX remain active.
 */
export function useObservatorySfx(): [boolean, (force?: boolean) => void] {
  const sfxEnabled = useSyncExternalStore(
    sharedAudioEngine.subscribeSfx,
    sharedAudioEngine.getSfxSnapshot,
    () => true,
  );

  const toggleSfx = useCallback((force?: boolean) => {
    sharedAudioEngine.toggleSfx(force);
  }, []);

  return [sfxEnabled, toggleSfx];
}

/**
 * Shared Observatory Sound Engine hook (ambient drone + UI SFX).
 * Starts false (OFF). When toggled ON, lazy-fetches and loops `/audio/milky-way-observatory.wav`.
 * Synchronizes across toggles, and tears down cleanly when the last consumer unmounts.
 */
export function useObservatorySound(): [boolean, (force?: boolean) => void] {
  const soundActive = useSyncExternalStore(
    sharedAudioEngine.subscribe,
    sharedAudioEngine.getSnapshot,
    () => false,
  );

  useEffect(() => {
    return sharedAudioEngine.registerMount();
  }, []);

  const toggleSound = useCallback((force?: boolean) => {
    sharedAudioEngine.toggle(force);
  }, []);

  return [soundActive, toggleSound];
}
