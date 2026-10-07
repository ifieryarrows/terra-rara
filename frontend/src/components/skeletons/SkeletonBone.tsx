import type { CSSProperties, HTMLAttributes } from 'react';
import clsx from 'clsx';

export interface SkeletonBoneProps extends HTMLAttributes<HTMLElement> {
  as?: 'div' | 'span';
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'full';
  className?: string;
  style?: CSSProperties;
}

export function SkeletonBone({
  as: Component = 'span',
  width,
  height,
  rounded = 'md',
  className,
  style,
  ...rest
}: SkeletonBoneProps) {
  const roundedClass =
    rounded === 'full'
      ? 'rounded-full'
      : rounded === 'xl'
      ? 'rounded-xl'
      : rounded === 'lg'
      ? 'rounded-lg'
      : rounded === 'sm'
      ? 'rounded-sm'
      : rounded === 'none'
      ? 'rounded-none'
      : 'rounded-md';

  return (
    <Component
      aria-hidden="true"
      className={clsx('cm-skeleton-bone', roundedClass, className)}
      style={{
        ...(width !== undefined ? { width } : {}),
        ...(height !== undefined ? { height } : {}),
        ...style,
      }}
      {...rest}
    />
  );
}
