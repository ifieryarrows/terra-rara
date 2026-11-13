"""Generate Milky Way Observatory tactile UI & companion SFX.

Tuning matches milky-way-observatory.wav (D Lydian / D major 9 rooted at 73.5 Hz / 110.2 Hz / 147 Hz / 220.5 Hz / 294 Hz).

Outputs:
  - frontend/public/audio/observatory-hover.wav
      Slow, warm, airy D-Lydian space pad breath for interactive hover (~420 ms, 24 kHz).
  - frontend/public/audio/observatory-chime.wav
      Slender, serene D-Lydian cosmic thread for button selection (~320 ms, 24 kHz).
  - frontend/public/audio/observatory-notification.wav
      Reserved ascending two-note D-Lydian "din-DIN" chime for future notifications (D5 -> A5, ~265 ms, 24 kHz).
  - frontend/public/audio/observatory-transition.wav
      Cosmic particle gather and convergence bloom for Landing <-> Dashboard warp (~1920 ms, 24 kHz).
  - frontend/public/audio/observatory-disperse.wav
      Luminous particle release and outward scatter shimmer (~1520 ms, 24 kHz).
  - frontend/public/audio/cuprum-chirp.wav
      Warm two-syllable pixel-copper companion voice chirp (~240 ms, 24 kHz).
  - frontend/public/audio/cuprum-tap.wav
      Compact pixel-patina tactile button tap for Cuprum controls (~115 ms, 24 kHz).
  - frontend/public/audio/cuprum-speak.wav
      Warm "mi" speech blip for Cuprum's typewriter chatter (~62 ms, 24 kHz).
  - frontend/public/audio/landing-voyage-near.wav
      Seamless 12s A-Lydian near-orbit solar shimmer stem for Landing scroll audio.
  - frontend/public/audio/landing-voyage-deep.wav
      Seamless 12s deep-space Milky Way choir stem for Landing scroll audio.
"""

from __future__ import annotations

from pathlib import Path
import wave
import numpy as np


SAMPLE_RATE = 24_000
ROOT_DIR = Path(__file__).resolve().parent.parent / "public" / "audio"


def _write_wav_16bit_mono(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -0.98, 0.98)
    # Apply tiny linear safety tapers on the boundary samples to guarantee zero DC/click edge
    taper = min(48, len(clipped) // 16)
    if taper > 0:
        clipped[:taper] *= np.linspace(0.0, 1.0, taper)
        clipped[-taper:] *= np.linspace(1.0, 0.0, taper)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _spectral_shape(noise: np.ndarray, sample_rate: int, gain_fn) -> np.ndarray:
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), d=1.0 / sample_rate)
    weights = gain_fn(freqs)
    return np.fft.irfft(spectrum * weights, n=len(noise))


def synthesize_hover(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Slow, warm, spacey D-Lydian observatory hover breath matching milky-way-observatory.wav."""
    duration = 0.420
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261113)

    # Velvety low-mid nebula air wash (no harsh 2.4-4.4 kHz fizz or high-frequency hiss)
    raw_noise = rng.standard_normal(n)

    def warm_air_curve(f: np.ndarray) -> np.ndarray:
        warm_band = np.exp(-0.5 * ((f - 620.0) / 260.0) ** 2)
        upper_sheen = 0.34 * np.exp(-0.5 * ((f - 1280.0) / 380.0) ** 2)
        sub_cut = 1.0 / (1.0 + (220.0 / np.maximum(f, 1.0)) ** 4)
        top_cut = 1.0 / (1.0 + (f / 2150.0) ** 5)
        return (warm_band + upper_sheen) * sub_cut * top_cut

    nebula_air = _spectral_shape(raw_noise, sample_rate, warm_air_curve)
    nebula_air /= max(np.max(np.abs(nebula_air)), 1e-6)

    # Slow weightless swell attack (92 ms) + lingering spacey release (328 ms)
    atk_n = int(0.092 * sample_rate)
    env_pad = np.zeros_like(t)
    env_pad[:atk_n] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, atk_n)))
    tail_u = np.linspace(0.0, 1.0, n - atk_n)
    env_pad[atk_n:] = (1.0 - tail_u) ** 1.75 * np.exp(-2.1 * tail_u)

    # Chorused cosmic pad voices tuned directly to milky-way-observatory.wav partials (D3, A3, D4, A4, D5, F#5)
    lfo = 0.90 + 0.10 * np.sin(2.0 * np.pi * 4.2 * t)
    pad = (
        0.34 * np.sin(2.0 * np.pi * 147.0 * t)
        + 0.28 * np.sin(2.0 * np.pi * 220.5 * t + 0.4)
        + 0.24 * (0.5 * np.sin(2.0 * np.pi * 293.4 * t + 0.2) + 0.5 * np.sin(2.0 * np.pi * 294.7 * t + 1.1))
        + 0.15 * np.sin(2.0 * np.pi * 441.0 * t + 0.8)
        + 0.10 * np.sin(2.0 * np.pi * 588.0 * t + 1.6)
        + 0.05 * np.sin(2.0 * np.pi * 740.0 * t + 2.2)
    )

    mix = (0.68 * pad * lfo + 0.14 * nebula_air) * env_pad
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.30


def synthesize_click(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Slender, serene D-Lydian cosmic selection click carrying Milky Way stillness and open space (~320 ms)."""
    duration = 0.320
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(771199)

    # Silky 10 ms bloom onset rising cleanly without percussive click or notification bell strike
    onset = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.010, 0.0, 1.0)))

    # Slender high-register crystal thread lifting smoothly into D6 (960 -> 1176 Hz), free of heavy low-mid partials
    freq_root = 1176.0 - 216.0 * np.exp(-t / 0.018)
    phi_root = 2.0 * np.pi * np.cumsum(freq_root) / sample_rate
    filament = (
        0.78 * np.sin(phi_root) * np.exp(-t / 0.072)
        + 0.24 * np.sin(2.0 * np.pi * 1480.0 * t + 0.4) * np.exp(-t / 0.095)
        + 0.12 * np.sin(2.0 * np.pi * 1764.0 * t + 0.9) * np.exp(-t / 0.118)
    ) * onset

    # Weightless cosmic void mist tail: quiet open-space atmosphere dissolving into silence
    raw_void = rng.standard_normal(n)

    def void_curve(f: np.ndarray) -> np.ndarray:
        calm_glow = np.exp(-0.5 * ((f - 1060.0) / 340.0) ** 2)
        distant_dust = 0.28 * np.exp(-0.5 * ((f - 4850.0) / 950.0) ** 2)
        sub_cut = 1.0 / (1.0 + (480.0 / np.maximum(f, 1.0)) ** 4)
        harsh_dip = 1.0 - 0.90 * np.exp(-0.5 * ((f - 3300.0) / 780.0) ** 2)
        top_cut = 1.0 / (1.0 + (f / 7000.0) ** 5)
        return (calm_glow + distant_dust) * sub_cut * harsh_dip * top_cut

    void_air = _spectral_shape(raw_void, sample_rate, void_curve)
    void_air /= max(np.max(np.abs(void_air)), 1e-6)
    void_atk = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.022, 0.0, 1.0)))
    void_env = void_atk * np.exp(-t / 0.115)

    fade_out = np.clip((duration - t) / 0.070, 0.0, 1.0)
    mix = (0.86 * filament + 0.11 * void_air * void_env) * fade_out

    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.29


def synthesize_notification(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Reserved ascending two-note D-Lydian 'din-DIN' notification chime (D5 -> A5, ~265 ms)."""
    duration = 0.265
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(771199)

    # 1. Beat 1 ("din"): quieter lower pitch (520 -> 588 Hz D5) with soft 7 ms cosine onset
    onset1 = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.007, 0.0, 1.0)))
    freq1 = 588.0 - 68.0 * np.exp(-t / 0.007)
    phi1 = 2.0 * np.pi * np.cumsum(freq1) / sample_rate
    env1 = onset1 * np.exp(-t / 0.036)
    note1 = (
        0.74 * np.sin(phi1)
        + 0.24 * np.sin(0.5 * phi1 + 0.25)
        + 0.14 * np.sin(2.0 * phi1 + 0.4)
    ) * env1

    # 2. Beat 2 ("DIN"): stronger ascending fifth (784 -> 882 Hz A5) at +58 ms, then smooth decay
    t2_offset = 0.058
    t2 = np.maximum(0.0, t - t2_offset)
    mask2 = (t >= t2_offset).astype(np.float64)
    onset2 = 0.5 * (1.0 - np.cos(np.pi * np.clip(t2 / 0.009, 0.0, 1.0)))
    freq2 = 882.0 - 98.0 * np.exp(-t2 / 0.008)
    phi2 = 2.0 * np.pi * np.cumsum(freq2) / sample_rate
    env2 = mask2 * onset2 * np.exp(-t2 / 0.078)
    note2 = (
        0.72 * np.sin(phi2)
        + 0.22 * np.sin(0.5 * phi2 + 0.3)
        + 0.20 * np.sin(2.0 * phi2 + 0.6)
        + 0.08 * np.sin(1.5 * phi2 + 0.9)
    ) * env2

    # 3. Delicate stardust sheen beneath Beat 2 glow (attenuated 2.4-4.4 kHz band)
    raw_air = rng.standard_normal(n)

    def air_curve(f: np.ndarray) -> np.ndarray:
        glow = 0.85 * np.exp(-0.5 * ((f - 1320.0) / 360.0) ** 2)
        dip = 1.0 - 0.88 * np.exp(-0.5 * ((f - 3350.0) / 760.0) ** 2)
        return glow * dip / (1.0 + (380.0 / np.maximum(f, 1.0)) ** 3) * (1.0 / (1.0 + (f / 2300.0) ** 4))

    air_texture = _spectral_shape(raw_air, sample_rate, air_curve)
    air_texture /= max(np.max(np.abs(air_texture)), 1e-6)
    air_env = mask2 * onset2 * np.exp(-t2 / 0.022)

    fade_out = np.clip((duration - t) / 0.055, 0.0, 1.0)
    mix = (0.42 * note1 + 1.00 * note2 + 0.05 * air_texture * air_env) * fade_out

    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.38


def synthesize_transition(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Particle gather -> convergence lock for Landing <-> Dashboard warp (~1920 ms)."""
    duration = 1.92
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261114)
    climax = 1.62

    def low_dust_curve(f: np.ndarray) -> np.ndarray:
        band = np.exp(-0.5 * ((f - 480.0) / 230.0) ** 2)
        sub_cut = 1.0 / (1.0 + (140.0 / np.maximum(f, 1.0)) ** 4)
        top_cut = 1.0 / (1.0 + (f / 1600.0) ** 4)
        return band * sub_cut * top_cut

    def high_sparkle_curve(f: np.ndarray) -> np.ndarray:
        band = np.exp(-0.5 * ((f - 1850.0) / 640.0) ** 2)
        harsh_dip = 1.0 - 0.80 * np.exp(-0.5 * ((f - 3200.0) / 700.0) ** 2)
        top_cut = 1.0 / (1.0 + (f / 5400.0) ** 5)
        return band * harsh_dip * top_cut

    low_dust = _spectral_shape(rng.standard_normal(n), sample_rate, low_dust_curve)
    low_dust /= max(np.max(np.abs(low_dust)), 1e-6)
    high_dust = _spectral_shape(rng.standard_normal(n), sample_rate, high_sparkle_curve)
    high_dust /= max(np.max(np.abs(high_dust)), 1e-6)

    gather_u = np.clip(t / climax, 0.0, 1.0)
    gather_env = (gather_u ** 1.6) * np.exp(-np.clip(t - climax, 0.0, 1.0) / 0.22)
    dust_mix = (0.24 * low_dust + 0.14 * high_dust) * (0.38 + 0.85 * gather_env)

    # Gravitational harmonic pull: detuned intervals locking cleanly into D-Lydian at t = climax
    pull = np.exp(-((t / 1.15) ** 2.2))
    glide_factor_a = 1.0 - 0.16 * pull * (t < climax)
    glide_factor_b = 1.0 + 0.19 * pull * (t < climax)
    phase_d3 = 2.0 * np.pi * np.cumsum(147.0 * glide_factor_a) / sample_rate
    phase_a3 = 2.0 * np.pi * np.cumsum(220.5 * glide_factor_b) / sample_rate
    phase_d4 = 2.0 * np.pi * np.cumsum(294.0 * (1.0 - 0.09 * pull)) / sample_rate
    phase_f4 = 2.0 * np.pi * np.cumsum(370.0 * (1.0 + 0.11 * pull)) / sample_rate

    gather_env_pad = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 1.15
    swarm_harmonics = (
        0.34 * np.sin(phase_d3)
        + 0.28 * np.sin(phase_a3 + 0.5)
        + 0.22 * np.sin(phase_d4 + 0.9)
        + 0.16 * np.sin(phase_f4 + 1.3)
    ) * gather_env_pad

    # Convergence lock chord holding briefly into the release moment
    tc = np.maximum(0.0, t - (climax - 0.08))
    bloom_mask = (t >= (climax - 0.08)).astype(np.float64)
    env_bloom = bloom_mask * (1.0 - np.exp(-tc / 0.055)) * np.exp(-tc / 0.34)
    bloom_chord = (
        0.32 * np.sin(2.0 * np.pi * 294.0 * tc)
        + 0.28 * np.sin(2.0 * np.pi * 441.0 * tc + 0.4)
        + 0.22 * np.sin(2.0 * np.pi * 588.0 * tc + 0.9)
    ) * env_bloom

    sub_swell = 0.22 * np.sin(2.0 * np.pi * 73.5 * t) * np.exp(-((t - climax) / 0.55) ** 2)

    mix = dust_mix + 0.46 * swarm_harmonics + 0.38 * bloom_chord + sub_swell
    mix *= np.clip(t / 0.045, 0.0, 1.0) * np.clip((duration - t) / 0.14, 0.0, 1.0)
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.56


def synthesize_disperse(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Outward particle dispersion shimmer when logo releases across dashboard (~1520 ms)."""
    duration = 1.52
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261115)

    def disperse_air_curve(f: np.ndarray) -> np.ndarray:
        warm = np.exp(-0.5 * ((f - 780.0) / 360.0) ** 2)
        starlight = 0.52 * np.exp(-0.5 * ((f - 5400.0) / 1200.0) ** 2)
        harsh_dip = 1.0 - 0.84 * np.exp(-0.5 * ((f - 3250.0) / 720.0) ** 2)
        sub_cut = 1.0 / (1.0 + (160.0 / np.maximum(f, 1.0)) ** 4)
        top_cut = 1.0 / (1.0 + (f / 8200.0) ** 5)
        return (warm + starlight) * harsh_dip * sub_cut * top_cut

    stardust = _spectral_shape(rng.standard_normal(n), sample_rate, disperse_air_curve)
    stardust /= max(np.max(np.abs(stardust)), 1e-6)

    # Smooth bloom attack -> wide airy outward drift
    atk = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.042, 0.0, 1.0)))
    drift_env = atk * ((1.0 - t / duration) ** 1.35) * np.exp(-t / 0.82)

    # Outward Doppler harmonic release: descending warm D-Lydian chord opening upward/downward
    spread = 1.0 - np.exp(-t / 0.24)
    p_d5 = 2.0 * np.pi * np.cumsum(588.0 * (1.0 - 0.06 * spread)) / sample_rate
    p_a4 = 2.0 * np.pi * np.cumsum(441.0 * (1.0 + 0.04 * spread)) / sample_rate
    p_f5 = 2.0 * np.pi * np.cumsum(740.0 * (1.0 - 0.08 * spread)) / sample_rate
    p_d4 = 2.0 * np.pi * np.cumsum(294.0 * (1.0 - 0.03 * spread)) / sample_rate
    p_sub = 2.0 * np.pi * 147.0 * t

    chord = (
        0.30 * np.sin(p_d4) * np.exp(-t / 0.90)
        + 0.28 * np.sin(p_a4 + 0.4) * np.exp(-t / 0.72)
        + 0.24 * np.sin(p_d5 + 0.9) * np.exp(-t / 0.52)
        + 0.16 * np.sin(p_f5 + 1.4) * np.exp(-t / 0.42)
        + 0.18 * np.sin(p_sub) * np.exp(-t / 0.65)
    ) * atk

    # Delicate twinkling particle glints staggered as particles fly to cards
    twinkle = np.zeros_like(t)
    for idx, (delay, freq) in enumerate([(0.05, 1176.0), (0.14, 1480.0), (0.25, 1764.0), (0.38, 1480.0), (0.54, 1176.0)]):
        tu = np.maximum(0.0, t - delay)
        mask = (t >= delay).astype(np.float64)
        twinkle_env = mask * (1.0 - np.exp(-tu / 0.018)) * np.exp(-tu / 0.24)
        twinkle += (0.09 / (1.0 + 0.18 * idx)) * np.sin(2.0 * np.pi * freq * tu) * twinkle_env

    mix = (0.30 * stardust * drift_env + 0.52 * chord + twinkle) * np.clip((duration - t) / 0.18, 0.0, 1.0)
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.52


def synthesize_cuprum_chirp(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Warm two-syllable pixel-copper companion pet chirp when clicking Cuprum."""
    duration = 0.240
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Syllable 1: cheerful upward F#4 -> A4 pixel glide (0..92 ms)
    t1 = np.clip(t, 0.0, 0.11)
    env1 = (t < 0.098).astype(np.float64) * np.sin(np.pi * np.clip(t / 0.095, 0.0, 1.0)) ** 1.15 * np.exp(-t1 / 0.065)
    f1 = 392.0 + 115.0 * (1.0 - np.exp(-t1 / 0.022))
    p1 = 2.0 * np.pi * np.cumsum(f1) / sample_rate
    # Soft retro square-triangle warmth (odd harmonics heavily rolled off so it's cozy, not harsh)
    syl1 = (0.76 * np.sin(p1) + 0.20 * np.sin(2.0 * p1) + 0.07 * np.sin(3.0 * p1)) * env1

    # Syllable 2: bright copper D5 -> E5 curious trill (74..240 ms)
    t2 = np.maximum(0.0, t - 0.074)
    m2 = (t >= 0.074).astype(np.float64)
    env2 = m2 * (1.0 - np.exp(-t2 / 0.009)) * np.exp(-t2 / 0.058) * np.clip((duration - t) / 0.022, 0.0, 1.0)
    f2 = 588.0 + 82.0 * (1.0 - np.exp(-t2 / 0.028)) + 8.0 * np.sin(2.0 * np.pi * 24.0 * t2)
    p2 = 2.0 * np.pi * np.cumsum(f2) / sample_rate
    syl2 = (0.74 * np.sin(p2) + 0.22 * np.sin(2.0 * p2) + 0.06 * np.sin(3.0 * p2)) * env2

    mix = 0.72 * syl1 + 0.94 * syl2
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.56


def synthesize_cuprum_tap(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Crisp compact pixel-patina button tap for Cuprum companion UI controls."""
    duration = 0.115
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Tactile copper-wood micro-blip: 740 Hz -> 588 Hz fast settling envelope
    f_tap = 588.0 + 190.0 * np.exp(-t / 0.011)
    p_tap = 2.0 * np.pi * np.cumsum(f_tap) / sample_rate
    env_tap = (1.0 - np.exp(-t / 0.0025)) * np.exp(-t / 0.026) * np.clip((duration - t) / 0.014, 0.0, 1.0)
    blip = (0.72 * np.sin(p_tap) + 0.22 * np.sin(2.0 * p_tap) + 0.08 * np.sin(2.0 * np.pi * 1176.0 * t)) * env_tap

    # Subtle 12 ms patina click skin
    env_snap = np.exp(-t / 0.004)
    snap = np.sin(2.0 * np.pi * 1480.0 * t) * env_snap * 0.20

    mix = blip + snap
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.50


def synthesize_cuprum_speak(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Warm pixel-companion 'mi' speech blip (soft nasal onset + vowel formant inflect)."""
    duration = 0.062
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Envelope: smooth 5 ms cosine attack, rounded sustain, 18 ms decay
    atk = 0.5 * (1.0 - np.cos(np.pi * np.clip(t / 0.0055, 0.0, 1.0)))
    rel = np.clip((duration - t) / 0.020, 0.0, 1.0) ** 1.3
    env = atk * rel

    # Slight rising/settling "mi" inflection (485 -> 542 -> 515 Hz)
    u = t / duration
    pitch = 485.0 + 57.0 * np.sin(np.pi * np.clip(u / 0.72, 0.0, 1.0)) - 20.0 * (u ** 2)
    phase = 2.0 * np.pi * np.cumsum(pitch) / sample_rate

    # Rounded square/vowel timbre: warm fundamental + gentle 2nd/3rd harmonics (~1.0-1.5 kHz 'i' formant)
    consonant = 0.24 * np.sin(2.0 * np.pi * 245.0 * t) * np.exp(-t / 0.011)
    vowel = (
        0.64 * np.sin(phase)
        + 0.24 * np.sin(2.0 * phase + 0.25)
        + 0.12 * np.sin(3.0 * phase + 0.5)
    )
    mix = (consonant + vowel) * env
    peak = max(np.max(np.abs(mix)), 1e-6)
    return (mix / peak) * 0.44


def synthesize_landing_voyage(sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Generate two phase-coherent 12s stems in A Lydian / F#m9 for scroll-synced Landing audio."""
    duration = 12.0
    n = int(round(duration * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261116)

    def _loop_freq(f: float) -> float:
        # Snap frequency to exact integer cycles over 12s for seamless loop boundaries
        return round(f * duration) / duration

    # Near orbit stem: solar-lit crystalline glass arpeggio pulses + luminous mid pad
    lfo_pulse = 0.82 + 0.18 * np.sin(2.0 * np.pi * _loop_freq(0.50) * t)
    lfo_drift = 0.90 + 0.10 * np.sin(2.0 * np.pi * _loop_freq(0.25) * t + 1.1)

    near_pad = (
        0.30 * np.sin(2.0 * np.pi * _loop_freq(220.0) * t)
        + 0.24 * np.sin(2.0 * np.pi * _loop_freq(330.0) * t + 0.4)
        + 0.20 * np.sin(2.0 * np.pi * _loop_freq(415.25) * t + 0.9)
        + 0.14 * np.sin(2.0 * np.pi * _loop_freq(494.0) * t + 1.5)
    ) * lfo_drift

    # Gentle glass shimmer notes cycling rhythmically at integer divisions of 12s
    pulse_a = (0.5 + 0.5 * np.sin(2.0 * np.pi * _loop_freq(0.3333) * t - 0.5)) ** 2.2
    pulse_b = (0.5 + 0.5 * np.sin(2.0 * np.pi * _loop_freq(0.50) * t + 1.4)) ** 2.6
    near_glass = (
        0.18 * np.sin(2.0 * np.pi * _loop_freq(554.5) * t) * pulse_a
        + 0.15 * np.sin(2.0 * np.pi * _loop_freq(740.0) * t + 0.7) * pulse_b
        + 0.09 * np.sin(2.0 * np.pi * _loop_freq(880.0) * t + 1.8) * pulse_a
    ) * lfo_pulse

    def solar_air_curve(f: np.ndarray) -> np.ndarray:
        band = np.exp(-0.5 * ((f - 760.0) / 300.0) ** 2)
        sub_cut = 1.0 / (1.0 + (220.0 / np.maximum(f, 1.0)) ** 4)
        top_cut = 1.0 / (1.0 + (f / 2100.0) ** 5)
        return band * sub_cut * top_cut

    solar_air = _spectral_shape(rng.standard_normal(n), sample_rate, solar_air_curve)
    solar_air /= max(np.max(np.abs(solar_air)), 1e-6)

    near_mix = 0.64 * near_pad + 0.42 * near_glass + 0.09 * solar_air * lfo_drift
    near_mix /= max(np.max(np.abs(near_mix)), 1e-6)

    # Deep space stem: expansive galactic sub-fifth foundation + suspended F#m9 starlight choir
    lfo_deep = 0.85 + 0.15 * np.sin(2.0 * np.pi * _loop_freq(0.1667) * t + 0.6)
    lfo_choir = 0.80 + 0.20 * np.sin(2.0 * np.pi * _loop_freq(0.25) * t + 2.2)

    deep_choir = (
        0.34 * np.sin(2.0 * np.pi * _loop_freq(55.0) * t)
        + 0.30 * np.sin(2.0 * np.pi * _loop_freq(110.0) * t + 0.3)
        + 0.26 * np.sin(2.0 * np.pi * _loop_freq(165.0) * t + 0.8)
        + 0.22 * np.sin(2.0 * np.pi * _loop_freq(185.0) * t + 1.3)
        + 0.18 * np.sin(2.0 * np.pi * _loop_freq(277.25) * t + 1.9)
        + 0.12 * np.sin(2.0 * np.pi * _loop_freq(370.0) * t + 2.4)
        + 0.08 * np.sin(2.0 * np.pi * _loop_freq(659.25) * t + 0.5)
    ) * lfo_choir

    def void_air_curve(f: np.ndarray) -> np.ndarray:
        warm = np.exp(-0.5 * ((f - 320.0) / 160.0) ** 2)
        mist = 0.38 * np.exp(-0.5 * ((f - 1350.0) / 380.0) ** 2)
        top_cut = 1.0 / (1.0 + (f / 2200.0) ** 5)
        return (warm + mist) * top_cut

    void_air = _spectral_shape(rng.standard_normal(n), sample_rate, void_air_curve)
    void_air /= max(np.max(np.abs(void_air)), 1e-6)

    deep_mix = 0.78 * deep_choir * lfo_deep + 0.14 * void_air * lfo_choir
    deep_mix /= max(np.max(np.abs(deep_mix)), 1e-6)

    return near_mix * 0.42, deep_mix * 0.44


def main() -> None:
    landing_near, landing_deep = synthesize_landing_voyage()
    artifacts = [
        ("hover", ROOT_DIR / "observatory-hover.wav", synthesize_hover(), False),
        ("click", ROOT_DIR / "observatory-chime.wav", synthesize_click(), False),
        ("notification", ROOT_DIR / "observatory-notification.wav", synthesize_notification(), False),
        ("transition", ROOT_DIR / "observatory-transition.wav", synthesize_transition(), False),
        ("disperse", ROOT_DIR / "observatory-disperse.wav", synthesize_disperse(), False),
        ("cuprum_chirp", ROOT_DIR / "cuprum-chirp.wav", synthesize_cuprum_chirp(), False),
        ("cuprum_tap", ROOT_DIR / "cuprum-tap.wav", synthesize_cuprum_tap(), False),
        ("cuprum_speak", ROOT_DIR / "cuprum-speak.wav", synthesize_cuprum_speak(), False),
        ("landing_near", ROOT_DIR / "landing-voyage-near.wav", landing_near, True),
        ("landing_deep", ROOT_DIR / "landing-voyage-deep.wav", landing_deep, True),
    ]

    for label, path, arr, is_loop in artifacts:
        if is_loop:
            path.parent.mkdir(parents=True, exist_ok=True)
            clipped = np.clip(arr, -0.98, 0.98)
            pcm = np.round(clipped * 32767.0).astype("<i2")
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
        else:
            _write_wav_16bit_mono(path, arr)
        spec = np.abs(np.fft.rfft(arr)) ** 2
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / SAMPLE_RATE)
        total_e = max(float(np.sum(spec)), 1e-12)
        low_pct = 100.0 * float(np.sum(spec[freqs < 2200])) / total_e
        harsh_pct = 100.0 * float(np.sum(spec[(freqs >= 2400) & (freqs <= 4400)])) / total_e
        mist_pct = 100.0 * float(np.sum(spec[freqs > 4400])) / total_e
        rms_db = 20.0 * float(np.log10(max(float(np.sqrt(np.mean(arr ** 2))), 1e-9)))
        peak_db = 20.0 * float(np.log10(max(float(np.max(np.abs(arr))), 1e-9)))
        print(
            f"{label}: {path} | {len(arr)/SAMPLE_RATE*1000:.0f}ms | {path.stat().st_size}B | "
            f"RMS={rms_db:.1f}dBFS Peak={peak_db:.1f}dBFS | "
            f"<2.2k={low_pct:.1f}% Harsh(2.4-4.4k)={harsh_pct:.1f}% Air(>4.4k)={mist_pct:.1f}%"
        )


if __name__ == "__main__":
    main()
