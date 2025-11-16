'use client';

/**
 * Real-Time Telemetry Charts
 * F1-style racing data visualization
 */

import { useEffect, useState } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { useCogniraceStore } from '@/lib/store';

interface TelemetryPoint {
  time: number;
  speed: number;
  rpm: number;
  throttle: number;
  brake: number;
  gForce: number;
}

export default function TelemetryCharts() {
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    // Use real telemetry data from store
    if (telemetryData) {
      const now = Date.now();
      const newPoint: TelemetryPoint = {
        time: now,
        speed: telemetryData.speed || 150,
        rpm: telemetryData.rpm || telemetryData.nmot || 7000,
        throttle: telemetryData.throttle || telemetryData.aps || 60,
        brake: Math.random() * 100, // Mock brake data
        gForce: -1 + Math.random() * 3 // Mock g-force
      };

      setData(prev => {
        const updated = [...prev, newPoint];
        // Keep last 30 points
        return updated.slice(-30);
      });
    }
  }, [telemetryData]);

  return (
    <div className="grid grid-cols-2 gap-4 h-full">
      {/* Speed Chart */}
      <Card className="bg-black/40 border-cyan-500/30">
        <CardContent className="pt-4 h-full">
          <div className="text-cyan-400 text-xs font-bold mb-2 tracking-wider">SPEED (KM/H)</div>
          <ResponsiveContainer width="100%" height="90%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00ffff" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#00ffff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis hide />
              <YAxis stroke="#00ffff" domain={[100, 250]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000', border: '1px solid #00ffff' }}
                labelStyle={{ color: '#00ffff' }}
              />
              <Area 
                type="monotone" 
                dataKey="speed" 
                stroke="#00ffff" 
                strokeWidth={2}
                fill="url(#speedGradient)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* RPM Chart */}
      <Card className="bg-black/40 border-yellow-500/30">
        <CardContent className="pt-4 h-full">
          <div className="text-yellow-400 text-xs font-bold mb-2 tracking-wider">ENGINE RPM</div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis hide />
              <YAxis stroke="#facc15" domain={[6000, 10000]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000', border: '1px solid #facc15' }}
                labelStyle={{ color: '#facc15' }}
              />
              <Line 
                type="monotone" 
                dataKey="rpm" 
                stroke="#facc15" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}

