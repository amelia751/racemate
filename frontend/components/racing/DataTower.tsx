'use client';

/**
 * Data Tower - Real-time Racing Metrics
 */

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useRaceMateStore } from '@/lib/store';

interface Metric {
  label: string;
  value: string | number;
  unit?: string;
  color: string;
  trend?: 'up' | 'down' | 'stable';
}

export default function DataTower() {
  const { telemetryData } = useRaceMateStore();
  
  const metrics: Metric[] = [
    { label: 'SPEED', value: telemetryData?.speed || 0, unit: 'km/h', color: 'cyan', trend: 'up' },
    { label: 'RPM', value: telemetryData?.rpm || telemetryData?.nmot || 0, unit: '', color: 'yellow', trend: 'stable' },
    { label: 'GEAR', value: telemetryData?.gear || 0, unit: '', color: 'green', trend: 'stable' },
    { label: 'THROTTLE', value: telemetryData?.throttle || telemetryData?.aps || 0, unit: '%', color: 'green', trend: 'up' },
    { label: 'FUEL', value: telemetryData?.fuel_level || 0, unit: 'L', color: 'orange', trend: 'down' },
  ];

  const getColorClass = (color: string) => {
    const colors: Record<string, string> = {
      cyan: 'text-cyan-400 border-cyan-500/30',
      yellow: 'text-yellow-400 border-yellow-500/30',
      green: 'text-green-400 border-green-500/30',
      orange: 'text-orange-400 border-orange-500/30',
      red: 'text-red-400 border-red-500/30',
      purple: 'text-purple-400 border-purple-500/30',
    };
    return colors[color] || 'text-white';
  };

  const getTrendIcon = (trend?: string) => {
    if (trend === 'up') return '↗';
    if (trend === 'down') return '↘';
    return '→';
  };

  return (
    <div className="grid grid-cols-1 gap-2">
      <AnimatePresence>
        {metrics.map((metric, idx) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.03 }}
          >
            <Card className={`bg-black/60 border ${getColorClass(metric.color).split(' ')[1]} backdrop-blur-sm`}>
              <div className="p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] text-muted-foreground tracking-wider">
                    {metric.label}
                  </span>
                  <span className="text-xs">
                    {getTrendIcon(metric.trend)}
                  </span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className={`text-xl font-black font-mono ${getColorClass(metric.color).split(' ')[0]}`}>
                    {typeof metric.value === 'number' ? Math.round(metric.value) : metric.value}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {metric.unit}
                  </span>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

