import { useId } from 'react';
import './BrandMark.css';

export type BrandMarkVariant = 'primary' | 'on-dark' | 'on-light' | 'monochrome' | 'small';

type BrandMarkProps = {
  size?: number;
  variant?: BrandMarkVariant;
  className?: string;
  ariaLabel?: string;
  x?: number;
  y?: number;
};

// Architectural Venus emblem in 48x48 viewBox:
// Copper planetary ring & subtle inner track + cool Galactic Silver-Blue sweeping orbit (ZERO star, ZERO crescent moon)
const RING_TOP = 'M 11.06 31.19 A 14.8 14.8 0 1 1 38.77 23.12';
const RING_BOT = 'M 37.68 29.65 A 14.8 14.8 0 0 1 16.25 36.61';
const INNER_TOP = 'M 14.53 31.54 A 12.1 12.1 0 1 1 36.05 25.05';
const INNER_BOT = 'M 32.93 32.17 A 12.1 12.1 0 0 1 22.23 35.97';
const SWOOSH_MAIN =
  'M 2.20 30.80 C 17.03 33.75 30.56 31.83 45.80 19.80 C 32.40 32.44 16.26 38.90 2.20 30.80 Z';
const SWOOSH_ECHO =
  'M 7.40 33.80 C 18.96 37.63 30.40 35.05 42.60 24.40 C 30.91 35.10 18.72 39.11 7.40 33.80 Z';

type OrbitalPalette = {
  sphere: [string, string, string] | null;
  copperRing: [string, string, string];
  copperTrack: string;
  orbitSwoosh: [string, string, string, string];
  orbitEcho: string;
};

const PALETTES: Record<'primary' | 'on-dark' | 'on-light', OrbitalPalette> = {
  primary: {
    sphere: ['#3a2519', '#131926', '#060a11'],
    copperRing: ['#f7dcb8', '#dda174', '#b26b3e'],
    copperTrack: '#dda174',
    orbitSwoosh: ['#7c9dcf', '#bdd2f1', '#edf4fe', '#96b4e2'],
    orbitEcho: '#8eadde',
  },
  'on-dark': {
    sphere: ['#3d291d', '#141b29', '#060a11'],
    copperRing: ['#fae4c4', '#e8b68c', '#c47b4d'],
    copperTrack: '#e8b68c',
    orbitSwoosh: ['#93b2e0', '#d0e0f7', '#f5f9ff', '#adc6ec'],
    orbitEcho: '#a7c1e9',
  },
  'on-light': {
    sphere: null,
    copperRing: ['#b87956', '#8f593f', '#603929'],
    copperTrack: '#8f593f',
    orbitSwoosh: ['#4a6fa3', '#36588a', '#243f6b', '#416699'],
    orbitEcho: '#416699',
  },
};

export function BrandMark({
  size = 36,
  variant = 'primary',
  className = '',
  ariaLabel,
  x,
  y,
}: BrandMarkProps) {
  const id = useId().replace(/:/g, '');
  const simplified = variant === 'small' || size <= 20;
  const monochrome = variant === 'monochrome';
  const classNames = [
    'cm-brand-mark-svg',
    `cm-brand-mark-svg--${variant}`,
    simplified ? 'cm-brand-mark-svg--simplified' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  const palette =
    variant === 'on-light'
      ? PALETTES['on-light']
      : variant === 'on-dark'
        ? PALETTES['on-dark']
        : PALETTES.primary;

  const flatRingColor =
    variant === 'on-light'
      ? '#8f593f'
      : variant === 'on-dark'
        ? '#fff1df'
        : variant === 'primary'
          ? '#dda174'
          : 'currentColor';
  const flatSwooshColor =
    monochrome
      ? 'currentColor'
      : variant === 'on-light'
        ? '#36588a'
        : '#c8daf4';

  return (
    <svg
      className={classNames}
      width={size}
      height={size}
      viewBox="0 0 48 48"
      x={x}
      y={y}
      fill="none"
      role={ariaLabel ? 'img' : undefined}
      aria-label={ariaLabel}
      aria-hidden={ariaLabel ? undefined : true}
      focusable="false"
    >
      {!simplified && !monochrome && (
        <defs>
          {palette.sphere && (
            <radialGradient id={`${id}-sp`} cx=".34" cy=".28" r=".74">
              <stop offset="0" stopColor={palette.sphere[0]} stopOpacity=".62" />
              <stop offset=".55" stopColor={palette.sphere[1]} stopOpacity=".76" />
              <stop offset="1" stopColor={palette.sphere[2]} stopOpacity=".9" />
            </radialGradient>
          )}
          <linearGradient id={`${id}-cu`} x1=".08" y1=".08" x2=".92" y2=".92">
            <stop offset="0" stopColor={palette.copperRing[0]} />
            <stop offset=".48" stopColor={palette.copperRing[1]} />
            <stop offset="1" stopColor={palette.copperRing[2]} />
          </linearGradient>
          <linearGradient id={`${id}-sw`} x1=".05" y1=".75" x2=".95" y2=".2">
            <stop offset="0" stopColor={palette.orbitSwoosh[0]} />
            <stop offset=".44" stopColor={palette.orbitSwoosh[1]} />
            <stop offset=".8" stopColor={palette.orbitSwoosh[2]} />
            <stop offset="1" stopColor={palette.orbitSwoosh[3]} />
          </linearGradient>
        </defs>
      )}

      {simplified || monochrome ? (
        <>
          <path
            d={RING_TOP}
            stroke={flatRingColor}
            strokeWidth={simplified ? '2.6' : '2.3'}
            strokeLinecap="round"
          />
          <path
            d={RING_BOT}
            stroke={flatRingColor}
            strokeWidth={simplified ? '2.6' : '2.3'}
            strokeLinecap="round"
          />
          <path d={SWOOSH_MAIN} fill={flatSwooshColor} />
        </>
      ) : (
        <>
          {palette.sphere && <circle cx="24" cy="24" r="14.2" fill={`url(#${id}-sp)`} />}
          <path
            d={RING_TOP}
            stroke={`url(#${id}-cu)`}
            strokeWidth="2.1"
            strokeLinecap="round"
          />
          <path
            d={RING_BOT}
            stroke={`url(#${id}-cu)`}
            strokeWidth="2.1"
            strokeLinecap="round"
          />
          <path
            d={INNER_TOP}
            stroke={palette.copperTrack}
            strokeOpacity=".34"
            strokeWidth=".75"
            strokeLinecap="round"
          />
          <path
            d={INNER_BOT}
            stroke={palette.copperTrack}
            strokeOpacity=".24"
            strokeWidth=".75"
            strokeLinecap="round"
          />
          <path d={SWOOSH_MAIN} fill={`url(#${id}-sw)`} />
          <path d={SWOOSH_ECHO} fill={palette.orbitEcho} fillOpacity=".62" />
        </>
      )}
    </svg>
  );
}
