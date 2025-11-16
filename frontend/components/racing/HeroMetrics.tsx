'use client';

/**
 * Hero Metrics Panel
 * Large, prominent display of key racing metrics
 */

import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { useCogniraceStore } from '@/lib/store';

export default function HeroMetrics() {
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

  const getSize = (size: string) => {
    switch (size) {
      case 'large': return 'text-7xl';
      case 'medium': return 'text-5xl';
      case 'small': return 'text-4xl';
      default: return 'text-5xl';
    }
  };

  return (
    <Card className="bg-black/40 border-cyan-500/30 h-full">
      <CardContent className="pt-4 h-full">
        <div className="text-cyan-400 text-xs font-bold mb-3 tracking-wider">CURRENT STATUS</div>
        <div className="grid grid-cols-5 gap-3 h-[calc(100%-30px)]">
          {metrics.map((metric, idx) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="flex flex-col items-center justify-center"
            >
              <div className="text-[10px] text-muted-foreground tracking-wider mb-1">
                {metric.label}
              </div>
              <div 
                className={`${getSize(metric.size)} font-black font-mono leading-none`}
                style={{ color: metric.color }}
              >
                {metric.value}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {metric.unit}
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

