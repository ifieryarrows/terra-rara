import { useId } from 'react';
import {
  TERRA_RARA_MARK_COMPACT_PATH,
  TERRA_RARA_MARK_STAR_PATH,
  TERRA_RARA_MARK_VIEWBOX,
} from './brand-mark-geometry';
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
  const classNames = ['cm-brand-mark-svg', `cm-brand-mark-svg--${variant}`, simplified ? 'cm-brand-mark-svg--simplified' : '', className]
    .filter(Boolean)
    .join(' ');

  return (
    <svg
      className={classNames}
      width={size}
      height={size}
      viewBox={TERRA_RARA_MARK_VIEWBOX}
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
          <linearGradient id={`${id}-ring`} x1="0" y1="0" x2="1" y2="1">
            {variant === 'on-light' ? (
              <>
                <stop offset="0" stopColor="#a87351" stopOpacity=".92"/>
                <stop offset=".58" stopColor="#89543b" stopOpacity=".84"/>
                <stop offset="1" stopColor="#57382e" stopOpacity=".94"/>
              </>
            ) : (
              <>
                <stop offset="0" stopColor="#f2c39f" stopOpacity=".9"/>
                <stop offset=".58" stopColor="#c9825b" stopOpacity=".64"/>
                <stop offset="1" stopColor="#764832" stopOpacity=".82"/>
              </>
            )}
          </linearGradient>
          {variant === 'primary' && (
            <linearGradient id={`${id}-copper`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#f2c39f"/>
              <stop offset=".34" stopColor="#d9956c"/>
              <stop offset=".68" stopColor="#a96545"/>
              <stop offset="1" stopColor="#704431"/>
            </linearGradient>
          )}
          {variant === 'on-light' && (
            <linearGradient id={`${id}-light`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#b87956"/>
              <stop offset=".55" stopColor="#89543b"/>
              <stop offset="1" stopColor="#57382e"/>
            </linearGradient>
          )}
        </defs>
      )}

      {simplified ? (
        <path
          d={TERRA_RARA_MARK_COMPACT_PATH}
          fill={variant === 'on-light' ? '#784832' : variant === 'on-dark' ? '#fff1df' : 'currentColor'}
        />
      ) : (
        <>
          <circle
            cx="24"
            cy="24"
            r="15.75"
            stroke={monochrome ? 'currentColor' : `url(#${id}-ring)`}
            strokeWidth="1.35"
          />
          {!monochrome && variant !== 'on-dark' && (
            <path d={TERRA_RARA_MARK_STAR_PATH} transform="translate(.35 .65)" fill="#4c2b25" opacity=".48"/>
          )}
          <path
            d={TERRA_RARA_MARK_STAR_PATH}
            fill={monochrome ? 'currentColor' : variant === 'on-dark' ? '#fff1df' : variant === 'on-light' ? `url(#${id}-light)` : `url(#${id}-copper)`}
          />
          {!monochrome && variant === 'primary' && size > 32 && (
            <path d="M24 11.25 26.55 21.45 36.75 24" fill="none" stroke="#fff4e7" strokeWidth=".6" strokeLinecap="round" opacity=".58"/>
          )}
          {!monochrome && <circle cx="24" cy="24" r="1.15" fill={variant === 'on-light' ? '#f3dfcf' : '#fff8ef'} opacity=".88"/>}
        </>
      )}
    </svg>
  );
}
