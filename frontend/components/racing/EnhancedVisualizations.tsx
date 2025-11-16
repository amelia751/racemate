'use client';

/**
 * Enhanced Visualizations - True F1 Style
 * Based on actual Red Bull Racing telemetry
 */

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { useCogniraceStore } from '@/lib/store';

// Fuel Consumption Over Time (Line Chart)
export function FuelConsumptionChart() {
  const [data, setData] = useState<any[]>([]);
  const { telemetryData } = useCogniraceStore();
  
  useEffect(() => {
    if (telemetryData) {
      const lap = telemetryData.lap || 13;
      const fuel = telemetryData.fuel_level || 35;
      
      setData(prev => {
        const newData = [...prev, { lap, fuel }];
        return newData.slice(-10); // Keep last 10 laps
      });
    }
  }, [telemetryData?.lap]);

  const currentFuel = telemetryData?.fuel_level || 0;
  const lapsRemaining = Math.floor(currentFuel / 2.5);

  return (
    <Card className="bg-black/40 h-full">
      <CardContent className="pt-3 pb-2 h-full flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="text-orange-400 text-xs font-bold tracking-wider">FUEL CONSUMPTION</div>
          <div className="text-right">
            <div className="text-orange-400 text-lg font-black font-mono">{currentFuel.toFixed(1)}L</div>
            <div className="text-[10px] text-muted-foreground">~{lapsRemaining} laps</div>
          </div>
        </div>
        <div className="flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="fuelGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#fb923c" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#fb923c" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="lap" stroke="#fb923c" tick={{ fontSize: 10 }} />
              <YAxis stroke="#fb923c" domain={[0, 50]} tick={{ fontSize: 10 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000', border: '1px solid #fb923c' }}
                labelStyle={{ color: '#fb923c' }}
              />
              <Area 
                type="monotone" 
                dataKey="fuel" 
                stroke="#fb923c" 
                strokeWidth={2}
                fill="url(#fuelGradient)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Tire Temperature - Circular Vehicle View (Red Bull Style)
export function TireTemperatureDisplay() {
  const [temps, setTemps] = useState({
    FL: 88,
    FR: 89,
    RL: 85,
    RR: 86
  });
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    // Only update when streaming (when telemetryData is being updated)
    if (telemetryData) {
      setTemps({
        FL: 80 + Math.random() * 20,
        FR: 80 + Math.random() * 20,
        RL: 78 + Math.random() * 20,
        RR: 78 + Math.random() * 20
      });
    }
  }, [telemetryData]);

  const getTempColor = (temp: number) => {
    if (temp > 100) return '#dc2626'; // Deep red - overheating
    if (temp > 95) return '#ef4444';  // Red
    if (temp > 85) return '#eab308';  // Yellow
    if (temp > 75) return '#22c55e';  // Green
    return '#0ea5e9';                 // Blue - cold
  };

  // SVG circle positions: FL (top-left), FR (top-right), RL (bottom-left), RR (bottom-right)
  const tirePositions = [
    { key: 'FL', label: 'FL', x: 30, y: 30 },
    { key: 'FR', label: 'FR', x: 170, y: 30 },
    { key: 'RL', label: 'RL', x: 30, y: 170 },
    { key: 'RR', label: 'RR', x: 170, y: 170 }
  ];

  return (
    <Card className="bg-black/40 h-full">
      <CardContent className="pt-3 pb-2 h-full flex flex-col">
        <div className="text-cyan-400 text-xs font-bold mb-2 tracking-wider">TIRE TEMPERATURE</div>

        {/* 2x2 Grid layout for tires */}
        <div className="grid grid-cols-2 gap-4 flex-1 p-2">
          {tirePositions.map((pos) => {
            const temp = temps[pos.key as keyof typeof temps];
            const color = getTempColor(temp);

            // Determine temperature status
            let status = 'Cold';
            if (temp > 95) status = 'Hot';
            else if (temp > 85) status = 'Warm';
            else if (temp > 75) status = 'Good';

            return (
              <div key={pos.key} className="flex items-center mb-2">
                <div
                  className="w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 flex-shrink-0"
                  style={{ backgroundColor: color, opacity: 0.85, padding: '12px' }}
                >
                  <div className="text-2xl font-bold text-white text-center">{Math.round(temp)}</div>
                </div>
                <div className="text-left flex-1 ml-6">
                  <div className="text-[10px] text-muted-foreground font-bold">{pos.label}</div>
                  <div className="text-[8px] text-muted-foreground">{status}</div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// Brake System Status (Comprehensive)
export function BrakeSystemStatus() {
  const [brakes, setBrakes] = useState({
    FL: { temp: 420, pressure: 85, wear: 15 },
    FR: { temp: 425, pressure: 86, wear: 16 },
    RL: { temp: 410, pressure: 82, wear: 12 },
    RR: { temp: 415, pressure: 83, wear: 13 }
  });

  const [tempHistory, setTempHistory] = useState<any[]>([]);
  const [historyId, setHistoryId] = useState(0);
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    // Only update when streaming (when telemetryData is being updated)
    if (telemetryData) {
      const newBrakes = {
        FL: {
          temp: 400 + Math.random() * 100,
          pressure: 80 + Math.random() * 10,
          wear: 10 + Math.random() * 10
        },
        FR: {
          temp: 400 + Math.random() * 100,
          pressure: 80 + Math.random() * 10,
          wear: 10 + Math.random() * 10
        },
        RL: {
          temp: 380 + Math.random() * 100,
          pressure: 78 + Math.random() * 10,
          wear: 8 + Math.random() * 10
        },
        RR: {
          temp: 380 + Math.random() * 100,
          pressure: 78 + Math.random() * 10,
          wear: 8 + Math.random() * 10
        }
      };
      setBrakes(newBrakes);

      // Add to history with stable ID
      const avgTemp = (newBrakes.FL.temp + newBrakes.FR.temp + newBrakes.RL.temp + newBrakes.RR.temp) / 4;
      setTempHistory(prev => [...prev, { id: historyId, time: Date.now(), temp: avgTemp }].slice(-20));
      setHistoryId(id => id + 1);
    }
  }, [telemetryData]);

  const getTempColor = (temp: number) => {
    if (temp > 500) return '#ef4444';
    if (temp > 450) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40 h-full">
      <CardContent className="pt-3 pb-2 h-full flex flex-col">
        <div className="text-red-400 text-xs font-bold mb-2 tracking-wider">BRAKE SYSTEM</div>
        <div className="flex-1 flex gap-3">
          {/* Temperature Trend */}
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={tempHistory} key="brake-temp-chart">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis hide />
                <YAxis stroke="#ef4444" domain={[300, 600]} tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#000', border: '1px solid #ef4444' }}
                  labelStyle={{ color: '#ef4444' }}
                />
                <Line
                  type="monotone"
                  dataKey="temp"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Corner Status */}
          <div className="flex flex-col gap-1 justify-center">
            {Object.entries(brakes).map(([corner, data]) => (
              <div key={corner} className="flex items-center gap-2 text-xs">
                <span className="w-6 text-muted-foreground font-bold">{corner}</span>
                <div
                  className="w-10 h-2.5 rounded-full"
                  style={{ backgroundColor: getTempColor(data.temp) }}
                />
                <span className="font-mono font-bold text-[10px]" style={{ color: getTempColor(data.temp) }}>
                  {Math.round(data.temp)}°C
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

