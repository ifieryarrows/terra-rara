// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type AudioSpyContext = {
  urlsFetched: string[];
  playedSources: Array<{ bufferLength: number; duration: number; playbackRate: number }>;
  oscillatorsCreated: number;
  oscFrequencies: number[];
  filtersCreated: Array<{ type: BiquadFilterType; freq: number; q?: number }>;
  hoverGainValues: number[];
  gainLinearRamps: Array<{ value: number; endTime: number }>;
  stoppedSources: number[];
};

function installMockWebAudio(options?: { failFetch?: boolean }): AudioSpyContext {
  const spy: AudioSpyContext = {
    urlsFetched: [],
    playedSources: [],
    oscillatorsCreated: 0,
    oscFrequencies: [],
    filtersCreated: [],
    hoverGainValues: [],
    gainLinearRamps: [],
    stoppedSources: [],
  };

  class MockAudioContext {
    public state: AudioContextState = 'running';
    public sampleRate = 24_000;
    public currentTime = 0;
    public destination = {};

    public resume = vi.fn(async () => {
      this.state = 'running';
    });

    public suspend = vi.fn(async () => {
      this.state = 'suspended';
    });

    public createGain = vi.fn(() => ({
      gain: {
        value: 1,
        setValueAtTime: vi.fn((val: number) => {
          spy.hoverGainValues.push(val);
        }),
        linearRampToValueAtTime: vi.fn((val: number, endTime: number) => {
          spy.gainLinearRamps.push({ value: val, endTime });
        }),
        exponentialRampToValueAtTime: vi.fn(),
        setTargetAtTime: vi.fn(),
        cancelScheduledValues: vi.fn(),
      },
      connect: vi.fn(),
      disconnect: vi.fn(),
    }));

    public createBuffer = vi.fn((channels: number, length: number, sampleRate: number) => ({
      numberOfChannels: channels,
      length,
      sampleRate,
      duration: length / sampleRate,
      getChannelData: () => new Float32Array(length),
    }));

    public createBufferSource = vi.fn(() => {
      const node = {
        buffer: null as unknown as { length: number; duration: number } | null,
        loop: false,
        playbackRate: {
          value: 1,
          setTargetAtTime: vi.fn(),
        },
        onended: null as null | (() => void),
        connect: vi.fn(),
        disconnect: vi.fn(),
        start: vi.fn(() => {
          if (node.buffer) {
            spy.playedSources.push({
              bufferLength: node.buffer.length,
              duration: node.buffer.duration,
              playbackRate: node.playbackRate.value,
            });
          }
        }),
        stop: vi.fn((when?: number) => {
          spy.stoppedSources.push(when ?? 0);
        }),
      };
      return node;
    });

    public createBiquadFilter = vi.fn(() => {
      const entry: { type: BiquadFilterType; freq: number; q?: number } = { type: 'bandpass', freq: 0 };
      spy.filtersCreated.push(entry);
      const filterObj = {
        get type(): BiquadFilterType {
          return entry.type;
        },
        set type(next: BiquadFilterType) {
          entry.type = next;
        },
        frequency: {
          setValueAtTime: vi.fn((f: number) => {
            entry.freq = f;
          }),
        },
        Q: {
          setValueAtTime: vi.fn((q: number) => {
            entry.q = q;
          }),
        },
        gain: { setValueAtTime: vi.fn() },
        connect: vi.fn(),
        disconnect: vi.fn(),
      };
      return filterObj;
    });

    public createOscillator = vi.fn(() => {
      spy.oscillatorsCreated += 1;
      return {
        type: 'sine' as OscillatorType,
        frequency: {
          setValueAtTime: vi.fn((f: number) => {
            spy.oscFrequencies.push(f);
          }),
          exponentialRampToValueAtTime: vi.fn((f: number) => {
            spy.oscFrequencies.push(f);
          }),
        },
        connect: vi.fn(),
        disconnect: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
        onended: null,
      };
    });

    public decodeAudioData = vi.fn(async (buf: ArrayBuffer) => ({
      numberOfChannels: 1,
      length: Math.max(1, Math.floor(buf.byteLength / 2)),
      sampleRate: 24_000,
      duration: Math.max(0.1, buf.byteLength / 48_000),
      getChannelData: () => new Float32Array(128),
    }));
  }

  vi.stubGlobal('AudioContext', MockAudioContext);
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      spy.urlsFetched.push(url);
      if (options?.failFetch) {
        return new Response(null, { status: 404 });
      }
      return new Response(new Uint8Array(4096).buffer, { status: 200 });
    }),
  );

  return spy;
}

describe('cosmicVision Observatory Audio Engine', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    delete document.documentElement.dataset.cmSuppressClickSound;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('keeps hover silent prior to user activation, then unlocks stardust hover & fantasy click with cooldown', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const { playObservatoryHover, playObservatoryChime, playObservatoryNotification } = await import('./cosmicVision');

    // Before any gesture, hover does not trigger blocked Web Audio calls or fetch WAVs.
    expect(await playObservatoryHover()).toBe(false);
    expect(spy.urlsFetched).toHaveLength(0);

    // Clicking activates audio and plays the custom click chime WAV.
    expect(await playObservatoryChime()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/observatory-chime.wav');
    expect(spy.playedSources).toHaveLength(1);

    // Notification sound stays separately callable and fetches observatory-notification.wav.
    expect(await playObservatoryNotification()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/observatory-notification.wav');

    // Chime inside 110ms cooldown is throttled cleanly.
    expect(await playObservatoryChime()).toBe(false);

    // Now that the user clicked, hovering plays the delicate cosmic hover WAV and respects 75ms cooldown.
    expect(await playObservatoryHover()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/observatory-hover.wav');
    expect(await playObservatoryHover()).toBe(false);

    vi.advanceTimersByTime(90);
    expect(await playObservatoryHover()).toBe(true);
  });

  it('synthesizes non-fatiguing stardust hover, slender cosmic click, and reserved din-DIN notification when WAV fetch fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio({ failFetch: true });
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.5);

    const { playObservatoryHover, playObservatoryChime, playObservatoryNotification } = await import('./cosmicVision');

    expect(await playObservatoryChime()).toBe(true);
    randomSpy.mockRestore();
    // Slender serene cosmic button click: gentle rising crystal glide (960 -> 1176 Hz) + F#6/A6 void halo
    expect(spy.oscFrequencies).toContain(960);
    expect(spy.oscFrequencies).toContain(1176);
    expect(spy.oscFrequencies).toContain(1480);
    expect(spy.oscFrequencies).toContain(1764);

    expect(await playObservatoryNotification()).toBe(true);
    // Preserved two-step ascending din-DIN notification motif: D5 (520 -> 588 Hz) -> A5 (784 -> 882 Hz + 1764 Hz)
    expect(spy.oscFrequencies).toContain(520);
    expect(spy.oscFrequencies).toContain(588);
    expect(spy.oscFrequencies).toContain(784);
    expect(spy.oscFrequencies).toContain(882);

    vi.advanceTimersByTime(100);
    expect(await playObservatoryHover()).toBe(true);
    // Warm D-Lydian lowpass breath + highshelf attenuation matching milky-way-observatory.wav
    expect(spy.filtersCreated.some((f) => f.type === 'lowpass' && f.freq === 950)).toBe(true);
    expect(spy.filtersCreated.some((f) => f.type === 'highshelf' && f.freq === 2800)).toBe(true);
    expect(spy.oscFrequencies).toContain(220.5);
    expect(spy.oscFrequencies).toContain(294);
  });

  it('plays distinct particle transition, Cuprum chirp, and Cuprum button tap sounds cleanly', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const {
      playObservatoryTransition,
      playCuprumChirp,
      playCuprumTap,
      setObservatorySfxEnabled,
      stopObservatoryAmbient,
      attachObservatoryInteractionAudio,
    } = await import('./cosmicVision');

    expect(await playObservatoryTransition()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/observatory-transition.wav');

    vi.advanceTimersByTime(150);
    expect(await playCuprumChirp()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/cuprum-chirp.wav');

    vi.advanceTimersByTime(150);
    expect(await playCuprumTap()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/cuprum-tap.wav');
    expect(spy.urlsFetched).not.toContain('/audio/milky-way-observatory.wav');

    // data-cm-sound attributes route delegated clicks to Cuprum voices cleanly
    document.body.innerHTML = `
      <button id="pet" data-cm-sound="cuprum-pet">Pet</button>
      <div data-cm-sound-scope="cuprum-tap"><button id="choice">Choice</button></div>
    `;
    const detach = attachObservatoryInteractionAudio(document);
    const countBeforeDelegate = spy.playedSources.length;

    vi.advanceTimersByTime(150);
    document.getElementById('pet')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(countBeforeDelegate + 1);

    vi.advanceTimersByTime(150);
    document.getElementById('choice')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(countBeforeDelegate + 2);

    // Muting UI SFX via setObservatorySfxEnabled(false) silences one-shots; stopObservatoryAmbient never fetches ambient
    setObservatorySfxEnabled(false);
    stopObservatoryAmbient();
    vi.advanceTimersByTime(150);
    expect(await playCuprumTap()).toBe(false);
    expect(spy.urlsFetched).not.toContain('/audio/milky-way-observatory.wav');

    detach();
  });

  it('delegates hover whispers and click chimes across interactive buttons while ignoring disabled and suppressed controls', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const { attachObservatoryInteractionAudio, setObservatorySound } = await import('./cosmicVision');

    document.body.innerHTML = `
      <button id="active-btn"><span>Analyze</span></button>
      <button id="disabled-btn" disabled>Disabled</button>
      <button id="mute-btn" data-cm-click-sound="off">Mute</button>
    `;
    const activeBtn = document.getElementById('active-btn') as HTMLButtonElement;
    const innerSpan = activeBtn.querySelector('span') as HTMLSpanElement;
    const disabledBtn = document.getElementById('disabled-btn') as HTMLButtonElement;
    const muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;

    const detach = attachObservatoryInteractionAudio(document);

    // First click unlocks audio & chimes
    activeBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(spy.urlsFetched).toContain('/audio/observatory-chime.wav');

    const countAfterClick = spy.playedSources.length;

    // Hovering active button triggers hover sound once (moving onto child span inside same button does not re-trigger)
    vi.advanceTimersByTime(120);
    const overEvent = new MouseEvent('pointerover', { bubbles: true });
    Object.defineProperty(overEvent, 'pointerType', { value: 'mouse' });
    Object.defineProperty(overEvent, 'buttons', { value: 0 });
    activeBtn.dispatchEvent(overEvent);
    await vi.runAllTimersAsync();
    expect(spy.urlsFetched).toContain('/audio/observatory-hover.wav');
    expect(spy.playedSources.length).toBe(countAfterClick + 1);

    vi.advanceTimersByTime(120);
    const innerOver = new MouseEvent('pointerover', { bubbles: true });
    Object.defineProperty(innerOver, 'pointerType', { value: 'mouse' });
    Object.defineProperty(innerOver, 'buttons', { value: 0 });
    innerSpan.dispatchEvent(innerOver);
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(countAfterClick + 1);

    // Hovering disabled button is ignored
    const disabledOver = new MouseEvent('pointerover', { bubbles: true });
    Object.defineProperty(disabledOver, 'pointerType', { value: 'mouse' });
    Object.defineProperty(disabledOver, 'buttons', { value: 0 });
    disabledBtn.dispatchEvent(disabledOver);
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(countAfterClick + 1);

    // Mute control opt-out suppresses automatic click chime
    vi.advanceTimersByTime(200);
    muteBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(countAfterClick + 1);

    // Explicitly toggling Observatory Sound ON -> OFF mutes all UI SFX until re-enabled
    setObservatorySound(true);
    await vi.runAllTimersAsync();
    setObservatorySound(false);
    await vi.runAllTimersAsync();

    const afterMuteCount = spy.playedSources.length;
    vi.advanceTimersByTime(200);
    activeBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(spy.playedSources.length).toBe(afterMuteCount);

    detach();
  });

  it('plays outward particle dispersion and pitched Cuprum speech blips cleanly', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const { playObservatoryDisperse, playCuprumSpeakBlip } = await import('./cosmicVision');

    expect(await playObservatoryDisperse({ volume: 0.36 })).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/observatory-disperse.wav');

    expect(await playCuprumSpeakBlip({ pitchStep: 0 })).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/cuprum-speak.wav');

    vi.advanceTimersByTime(60);
    expect(await playCuprumSpeakBlip({ pitchStep: 3 })).toBe(true);
    expect(spy.playedSources.length).toBeGreaterThanOrEqual(3);
  });

  it('runs Landing dual-stem voyage audio synchronized to scroll without loading dashboard track', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const {
      startLandingVoyageAudio,
      updateLandingVoyageScroll,
      stopLandingVoyageAudio,
    } = await import('../features/landing/landingVoyageAudio');

    expect(await startLandingVoyageAudio()).toBe(true);
    expect(spy.urlsFetched).toContain('/audio/landing-voyage-near.wav');
    expect(spy.urlsFetched).toContain('/audio/landing-voyage-deep.wav');
    expect(spy.urlsFetched).not.toContain('/audio/milky-way-observatory.wav');

    updateLandingVoyageScroll(0.55, 0.35);
    stopLandingVoyageAudio();
  });

  it('sculpts click chime with 140Hz HP / 7500Hz LP filters, 4ms ramp to 0.25 gain, random 0.98-1.02 playbackRate, and overlap choke', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const randomSpy = vi.spyOn(Math, 'random');
    randomSpy.mockReturnValueOnce(0).mockReturnValueOnce(1);

    const { playObservatoryChime } = await import('./cosmicVision');

    expect(await playObservatoryChime()).toBe(true);
    expect(spy.filtersCreated).toEqual(
      expect.arrayContaining([
        { type: 'highpass', freq: 140, q: 0.7 },
        { type: 'lowpass', freq: 7500, q: 0.7 },
      ]),
    );
    expect(spy.gainLinearRamps).toContainEqual({ value: 0.25, endTime: 0.004 });
    expect(spy.playedSources[0].playbackRate).toBeCloseTo(0.98, 4);

    vi.advanceTimersByTime(150);
    expect(await playObservatoryChime()).toBe(true);
    expect(spy.stoppedSources.length).toBe(1);
    expect(spy.playedSources[1].playbackRate).toBeCloseTo(1.02, 4);

    randomSpy.mockRestore();
  });

  it('suppresses hover and click observatory audio inside data-cm-no-sound="true" containers such as the heatmap stage', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-13T10:00:00Z'));
    const spy = installMockWebAudio();

    const { attachObservatoryInteractionAudio, playObservatoryChime } = await import('./cosmicVision');
    const detach = attachObservatoryInteractionAudio(document);

    // Unlock interaction audio first
    expect(await playObservatoryChime()).toBe(true);
    const initialCount = spy.playedSources.length;

    const heatmapStage = document.createElement('div');
    heatmapStage.dataset.cmNoSound = 'true';
    const tile = document.createElement('div');
    tile.setAttribute('role', 'button');
    const tickerSpan = document.createElement('span');
    tickerSpan.textContent = 'NVDA';
    tile.appendChild(tickerSpan);
    heatmapStage.appendChild(tile);
    document.body.appendChild(heatmapStage);

    const tileOver = new MouseEvent('pointerover', { bubbles: true });
    Object.defineProperty(tileOver, 'pointerType', { value: 'mouse' });
    Object.defineProperty(tileOver, 'buttons', { value: 0 });
    tickerSpan.dispatchEvent(tileOver);
    await vi.runAllTimersAsync();

    vi.advanceTimersByTime(200);
    tile.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.runAllTimersAsync();

    expect(spy.playedSources).toHaveLength(initialCount);

    detach();
  });
});
