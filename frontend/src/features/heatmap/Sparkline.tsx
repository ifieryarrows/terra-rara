import { memo, useMemo } from 'react';

interface Props {
  values?: number[] | null;
  positive?: boolean;
  width?: number;
  height?: number;
}

export const Sparkline = memo(function Sparkline({ values, positive, width = 72, height = 24 }: Props) {
  const points = useMemo(() => {
    if (!values || values.length < 2) return '';
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(0.0001, max - min);
    return values.map((value, index) => {
      const x = (index / (values.length - 1)) * (width - 2) + 1;
      const y = height - 2 - ((value - min) / span) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
  }, [height, values, width]);
  if (!points) return <span className="block" style={{ width, height }} aria-hidden="true" />;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <polyline
        points={points}
        fill="none"
        stroke={positive ? '#34d399' : '#fb7185'}
        strokeWidth="1.4"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
});
