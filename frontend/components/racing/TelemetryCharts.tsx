'use client';

/**
 * Real-Time Telemetry Charts
 * F1-style racing data visualization
 */

import { useEffect, useState, memo } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { useRaceMateStore } from '@/lib/store';

interface TelemetryPoint {
  id: number; // Stable key for chart data
  time: number;
  speed: number;
  rpm: number;
  throttle: number;
  brake: number;
  gForce: number;
}

const SpeedChart = memo(function SpeedChart({ data }: { data: TelemetryPoint[] }) {
  return (
    <Card className="bg-black/40">
      <CardContent className="pt-4 h-full">
        <div className="text-cyan-400 text-xs font-bold mb-2 tracking-wider">SPEED (KM/H)</div>
        <ResponsiveContainer width="100%" height="90%">
          <AreaChart data={data} key="speed-chart">
            <defs>
              <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00ffff" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#00ffff" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis hide />
            <YAxis 
              stroke="#00ffff" 
              domain={[100, 250]} 
              tickFormatter={(value) => Math.round(value).toString()}
            />
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
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
});

const RPMChart = memo(function RPMChart({ data }: { data: TelemetryPoint[] }) {
  return (
    <Card className="bg-black/40">
      <CardContent className="pt-4 h-full">
        <div className="text-yellow-400 text-xs font-bold mb-2 tracking-wider">ENGINE RPM</div>
        <ResponsiveContainer width="100%" height="90%">
          <AreaChart data={data} key="rpm-chart">
            <defs>
              <linearGradient id="rpmGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#facc15" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#facc15" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis hide />
            <YAxis 
              stroke="#facc15" 
              domain={[6000, 10000]} 
              tickFormatter={(value) => Math.round(value).toString()}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#000', border: '1px solid #facc15' }}
              labelStyle={{ color: '#facc15' }}
            />
            <Area
              type="monotone"
              dataKey="rpm"
              stroke="#facc15"
              strokeWidth={2}
              fill="url(#rpmGradient)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
});

export default memo(function TelemetryCharts() {
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const [pointId, setPointId] = useState(0);
  const { telemetryData } = useRaceMateStore();

  useEffect(() => {
    // Use real telemetry data from store
    if (telemetryData) {
      const now = Date.now();
      const newPoint: TelemetryPoint = {
        id: pointId,
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

      setPointId(id => id + 1);
    }
  }, [telemetryData]);

  return (
    <div className="grid grid-cols-2 gap-6 h-full">
      <SpeedChart data={data} />
      <RPMChart data={data} />
    </div>
  );
})

