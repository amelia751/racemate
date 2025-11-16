'use client';

/**
 * Hero Metrics Panel
 * Large, prominent display of key racing metrics
 */

import { memo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { useCogniraceStore } from '@/lib/store';

// Memoized individual metric component to prevent unnecessary re-renders
const MetricDisplay = memo(function MetricDisplay({
  label,
  value,
  unit,
  color,
  size
}: {
  label: string;
  value: string | number;
  unit: string;
  color: string;
  size: string;
}) {
  const getSize = (s: string) => {
    switch (s) {
      case 'large': return 'text-7xl';
      case 'medium': return 'text-5xl';
      case 'small': return 'text-4xl';
      default: return 'text-5xl';
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="text-[10px] text-muted-foreground tracking-wider mb-1">
        {label}
      </div>
      <div
        className={`${getSize(size)} font-black font-mono leading-none transition-all duration-200`}
        style={{ color }}
      >
        {value}
      </div>
      <div className="text-xs text-muted-foreground mt-1">
        {unit}
      </div>
    </div>
  );
});

export default memo(function HeroMetrics() {
  const { telemetryData } = useCogniraceStore();

  const metrics = [
    {
      label: 'SPEED',
      value: Math.round(telemetryData?.speed || 0),
      unit: 'km/h',
      color: '#00ffff',
      size: 'large'
    },
    {
      label: 'RPM',
      value: Math.round(telemetryData?.rpm || telemetryData?.nmot || 0),
      unit: '',
      color: '#facc15',
      size: 'medium'
    },
    {
      label: 'GEAR',
      value: telemetryData?.gear || 0,
      unit: '',
      color: '#22c55e',
      size: 'small'
    },
    {
      label: 'THROTTLE',
      value: Math.round(telemetryData?.throttle || telemetryData?.aps || 0),
      unit: '%',
      color: '#22c55e',
      size: 'small'
    },
    {
      label: 'FUEL',
      value: (telemetryData?.fuel_level || 0).toFixed(1),
      unit: 'L',
      color: '#fb923c',
      size: 'small'
    },
  ];

  return (
    <Card className="bg-black/40 border-cyan-500/30 h-full">
      <CardContent className="pt-4 h-full">
        <div className="text-cyan-400 text-xs font-bold mb-3 tracking-wider">CURRENT STATUS</div>
        <div className="grid grid-cols-5 gap-3 h-[calc(100%-30px)]">
          {metrics.map((metric) => (
            <MetricDisplay
              key={metric.label}
              label={metric.label}
              value={metric.value}
              unit={metric.unit}
              color={metric.color}
              size={metric.size}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
})

