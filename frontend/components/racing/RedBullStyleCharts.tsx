'use client';

/**
 * Red Bull Style Charts - More Time-Series Focus
 */

import { useEffect, useState, memo } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { useCogniraceStore } from '@/lib/store';

interface DataPoint {
  time: string;
  value: number;
}

export const ThrottleBrakeTimeSeries = memo(function ThrottleBrakeTimeSeries() {
  const [data, setData] = useState<any[]>([]);
  const [pointId, setPointId] = useState(0);
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    if (telemetryData) {
      const now = new Date().toLocaleTimeString();
      const newPoint = {
        id: pointId,
        time: now,
        throttle: telemetryData.throttle || telemetryData.aps || 0,
        brake: Math.random() * 100
      };

      setData(prev => [...prev, newPoint].slice(-20));
      setPointId(id => id + 1);
    }
  }, [telemetryData]);

  return (
    <Card className="bg-black/40  h-full">
      <CardContent className="pt-3 h-full">
        <div className="text-green-400 text-xs font-bold mb-2 tracking-wider">THROTTLE / BRAKE</div>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={data} key="throttle-brake-chart">
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis stroke="#22c55e" tick={{ fontSize: 10 }} />
            <YAxis stroke="#22c55e" domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#000', border: '1px solid #22c55e' }}
              labelStyle={{ color: '#22c55e' }}
            />
            <Line type="monotone" dataKey="throttle" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="brake" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
})

export const GForceTimeSeries = memo(function GForceTimeSeries() {
  const [data, setData] = useState<any[]>([]);
  const [pointId, setPointId] = useState(0);
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    // Only update when streaming (when telemetryData is being updated)
    if (telemetryData) {
      const now = new Date().toLocaleTimeString();
      const newPoint = {
        id: pointId,
        time: now,
        lateral: -1 + Math.random() * 2,
        longitudinal: -1 + Math.random() * 2
      };

      setData(prev => [...prev, newPoint].slice(-20));
      setPointId(id => id + 1);
    }
  }, [telemetryData]);

  return (
    <Card className="bg-black/40  h-full">
      <CardContent className="pt-3 h-full">
        <div className="text-purple-400 text-xs font-bold mb-2 tracking-wider">G-FORCE</div>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={data} key="gforce-chart">
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis stroke="#a855f7" tick={{ fontSize: 10 }} />
            <YAxis stroke="#a855f7" domain={[-2, 2]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#000', border: '1px solid #a855f7' }}
              labelStyle={{ color: '#a855f7' }}
            />
            <Line type="monotone" dataKey="lateral" stroke="#a855f7" strokeWidth={2} dot={false} name="Lateral" isAnimationActive={false} />
            <Line type="monotone" dataKey="longitudinal" stroke="#22c55e" strokeWidth={2} dot={false} name="Long" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
})

export function FuelBar() {
  const { telemetryData } = useCogniraceStore();
  const fuelLevel = telemetryData?.fuel_level || 35.2;
  const fuelPercentage = (fuelLevel / 50) * 100;
  
  return (
    <Card className="bg-black/40  h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="text-orange-400 text-xs font-bold mb-2 tracking-wider">FUEL</div>
        <div className="flex-1 flex items-center gap-3">
          <div className="flex-1 h-6 bg-gray-800 rounded-full overflow-hidden relative">
            <div 
              className="h-full transition-all duration-300"
              style={{ 
                width: `${fuelPercentage}%`,
                background: fuelPercentage > 40 ? '#22c55e' : fuelPercentage > 20 ? '#fb923c' : '#ef4444'
              }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-bold">
              {fuelLevel.toFixed(1)}L
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            {Math.floor(fuelLevel / 2.5)} laps
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function TirePressureBars() {
  const [pressures, setPressures] = useState({ FL: 32, FR: 32, RL: 30, RR: 30 });

  useEffect(() => {
    const interval = setInterval(() => {
      setPressures({
        FL: 30 + Math.random() * 4,
        FR: 30 + Math.random() * 4,
        RL: 28 + Math.random() * 4,
        RR: 28 + Math.random() * 4
      });
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const getTireColor = (psi: number) => {
    if (psi < 29 || psi > 34) return '#ef4444';
    if (psi < 30 || psi > 33) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40  h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="text-cyan-400 text-xs font-bold mb-2 tracking-wider">TIRE PRESSURE</div>
        <div className="flex-1 flex items-center justify-between gap-2">
          {Object.entries(pressures).map(([tire, psi]) => (
            <div key={tire} className="flex-1 flex flex-col items-center gap-1">
              <div className="text-[10px] text-muted-foreground">{tire}</div>
              <div className="w-full h-12 bg-gray-800 rounded overflow-hidden relative">
                <div 
                  className="w-full absolute bottom-0 transition-all duration-300"
                  style={{ 
                    height: `${(psi / 40) * 100}%`,
                    background: getTireColor(psi)
                  }}
                />
              </div>
              <div className="text-xs font-bold" style={{ color: getTireColor(psi) }}>
                {psi.toFixed(1)}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function BrakeTemperatureBars() {
  const [temps, setTemps] = useState({ FL: 420, FR: 425, RL: 410, RR: 415 });

  useEffect(() => {
    const interval = setInterval(() => {
      setTemps({
        FL: 400 + Math.random() * 100,
        FR: 400 + Math.random() * 100,
        RL: 380 + Math.random() * 100,
        RR: 380 + Math.random() * 100
      });
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const getTempColor = (temp: number) => {
    if (temp > 500) return '#ef4444';
    if (temp > 450) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40  h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="text-red-400 text-xs font-bold mb-2 tracking-wider">BRAKE TEMP (°C)</div>
        <div className="flex-1 flex items-center justify-between gap-2">
          {Object.entries(temps).map(([corner, temp]) => (
            <div key={corner} className="flex-1 flex flex-col items-center gap-1">
              <div className="text-[10px] text-muted-foreground">{corner}</div>
              <div className="w-full h-12 bg-gray-800 rounded overflow-hidden relative">
                <div 
                  className="w-full absolute bottom-0 transition-all duration-300"
                  style={{ 
                    height: `${((temp - 300) / 300) * 100}%`,
                    background: getTempColor(temp)
                  }}
                />
              </div>
              <div className="text-xs font-bold" style={{ color: getTempColor(temp) }}>
                {Math.round(temp)}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

