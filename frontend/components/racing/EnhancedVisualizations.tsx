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
    <Card className="bg-black/40 border-orange-500/30 h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="text-orange-400 text-xs font-bold tracking-wider">FUEL CONSUMPTION</div>
          <div className="text-right">
            <div className="text-orange-400 text-2xl font-black font-mono">{currentFuel.toFixed(1)}L</div>
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

// Tire Temperature Heat Map (4 Corners)
export function TireTemperatureDisplay() {
  const [temps, setTemps] = useState({
    FL: { inner: 85, middle: 88, outer: 90 },
    FR: { inner: 86, middle: 89, outer: 91 },
    RL: { inner: 82, middle: 85, outer: 87 },
    RR: { inner: 83, middle: 86, outer: 88 }
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setTemps({
        FL: { 
          inner: 80 + Math.random() * 15, 
          middle: 85 + Math.random() * 15, 
          outer: 88 + Math.random() * 15 
        },
        FR: { 
          inner: 80 + Math.random() * 15, 
          middle: 85 + Math.random() * 15, 
          outer: 88 + Math.random() * 15 
        },
        RL: { 
          inner: 78 + Math.random() * 15, 
          middle: 83 + Math.random() * 15, 
          outer: 85 + Math.random() * 15 
        },
        RR: { 
          inner: 78 + Math.random() * 15, 
          middle: 83 + Math.random() * 15, 
          outer: 85 + Math.random() * 15 
        }
      });
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const getTempColor = (temp: number) => {
    if (temp > 95) return '#ef4444';
    if (temp > 85) return '#22c55e';
    if (temp > 75) return '#fb923c';
    return '#3b82f6';
  };

  const renderTire = (position: string, temps: { inner: number, middle: number, outer: number }) => (
    <div className="flex flex-col items-center gap-1">
      <div className="text-[10px] text-muted-foreground font-bold">{position}</div>
      <div className="flex gap-0.5">
        {[temps.inner, temps.middle, temps.outer].map((temp, idx) => (
          <div 
            key={idx}
            className="w-6 h-14 rounded-sm transition-colors duration-300"
            style={{ backgroundColor: getTempColor(temp) }}
          />
        ))}
      </div>
      <div className="text-xs font-mono font-bold" style={{ color: getTempColor(temps.middle) }}>
        {Math.round(temps.middle)}°C
      </div>
    </div>
  );

  return (
    <Card className="bg-black/40 border-cyan-500/30 h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="text-cyan-400 text-xs font-bold mb-3 tracking-wider">TIRE TEMPERATURE</div>
        <div className="flex-1 grid grid-cols-2 gap-4 items-center justify-items-center">
          {renderTire('FL', temps.FL)}
          {renderTire('FR', temps.FR)}
          {renderTire('RL', temps.RL)}
          {renderTire('RR', temps.RR)}
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

  useEffect(() => {
    const interval = setInterval(() => {
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

      // Add to history
      const avgTemp = (newBrakes.FL.temp + newBrakes.FR.temp + newBrakes.RL.temp + newBrakes.RR.temp) / 4;
      setTempHistory(prev => [...prev, { time: Date.now(), temp: avgTemp }].slice(-20));
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const getTempColor = (temp: number) => {
    if (temp > 500) return '#ef4444';
    if (temp > 450) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40 border-red-500/30 h-full">
      <CardContent className="pt-3 h-full flex flex-col">
        <div className="text-red-400 text-xs font-bold mb-2 tracking-wider">BRAKE SYSTEM</div>
        <div className="flex-1 flex gap-3">
          {/* Temperature Trend */}
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={tempHistory}>
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
                  className="w-12 h-3 rounded-full"
                  style={{ backgroundColor: getTempColor(data.temp) }}
                />
                <span className="font-mono font-bold" style={{ color: getTempColor(data.temp) }}>
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

