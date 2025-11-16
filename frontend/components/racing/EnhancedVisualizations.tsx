'use client';

/**
 * Enhanced Visualizations - True F1 Style
 * Based on actual Red Bull Racing telemetry
 */

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { useCogniraceStore } from '@/lib/store';

// Fuel Consumption - Red Bull Style Vertical Bars
export function FuelConsumptionChart() {
  const [fuelHistory, setFuelHistory] = useState<number[]>([35, 35, 35]);
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    if (telemetryData) {
      const fuel = telemetryData.fuel_level || 35;
      setFuelHistory(prev => [...prev, fuel].slice(-3)); // Keep last 3 readings
    }
  }, [telemetryData?.fuel_level]);

  const currentFuel = telemetryData?.fuel_level || 0;
  const maxFuel = 50;
  const fuelPercentage = (currentFuel / maxFuel) * 100;
  const lapsRemaining = Math.floor(currentFuel / 2.5);

  // Determine fuel color based on level
  const getFuelColor = (percentage: number) => {
    if (percentage > 80) return '#22c55e'; // Green - plenty
    if (percentage > 50) return '#eab308'; // Yellow - good
    if (percentage > 30) return '#fb923c'; // Orange - warning
    return '#ef4444'; // Red - critical
  };

  const barColor = getFuelColor(fuelPercentage);

  return (
    <Card className="bg-black/40 h-full py-2">
      <CardContent className="py-0 px-3 h-full flex flex-col">
        <div className="text-orange-400 text-xs font-bold mb-1 tracking-wider">FUEL</div>

        {/* Main fuel display area */}
        <div className="flex-1 flex items-end justify-center gap-3">
          {/* Fuel history bars - Vertical */}
          <div className="flex items-end gap-2 h-full flex-1">
            {fuelHistory.map((fuel, idx) => {
              const percent = (fuel / maxFuel) * 100;
              const color = getFuelColor(percent);
              return (
                <div key={idx} className="flex-1 flex flex-col items-center justify-end h-full">
                  <div className="w-full bg-gray-800 rounded-t-sm overflow-hidden flex-1 relative mb-0.5">
                    <div
                      className="w-full transition-all duration-300 absolute bottom-0"
                      style={{
                        height: `${percent}%`,
                        backgroundColor: color,
                        opacity: 0.85
                      }}
                    />
                  </div>
                  <div className="text-[7px] text-muted-foreground">{Math.round(fuel)}L</div>
                </div>
              );
            })}
          </div>

          {/* Current fuel display */}
          <div className="text-center">
            <div className="text-orange-400 text-2xl font-black font-mono">{currentFuel.toFixed(0)}</div>
            <div className="text-[7px] text-muted-foreground">L</div>
            <div className="text-[7px] text-muted-foreground mt-0.5">{lapsRemaining}L</div>
          </div>
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

      // Add to history with stable ID - track all 4 corners
      const dataPoint = {
        id: historyId,
        time: Date.now(),
        FL: newBrakes.FL.temp,
        FR: newBrakes.FR.temp,
        RL: newBrakes.RL.temp,
        RR: newBrakes.RR.temp
      };
      setTempHistory(prev => [...prev, dataPoint].slice(-20));
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
                  dataKey="FL"
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="FR"
                  stroke="#fb923c"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="RL"
                  stroke="#22c55e"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="RR"
                  stroke="#06b6d4"
                  strokeWidth={1.5}
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

// Brake System - Option 2: Circular Gauge View
export function BrakeCircularGauges() {
  const [brakes, setBrakes] = useState({
    FL: { temp: 420, pressure: 85, wear: 15 },
    FR: { temp: 425, pressure: 86, wear: 16 },
    RL: { temp: 410, pressure: 82, wear: 12 },
    RR: { temp: 415, pressure: 83, wear: 13 }
  });
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    if (telemetryData) {
      setBrakes({
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
      });
    }
  }, [telemetryData]);

  const getTempColor = (temp: number) => {
    if (temp > 500) return '#ef4444';
    if (temp > 450) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40 h-full !py-0 !gap-0">
      <CardContent className="!py-0 !px-2 h-full flex flex-col">
        <div className="text-red-400 text-[10px] font-bold mb-0.5 tracking-wider">GAUGES</div>

        {/* 2x2 Grid of Circular Gauges */}
        <div className="grid grid-cols-2 gap-1 flex-1">
          {Object.entries(brakes).map(([corner, data]) => {
            const color = getTempColor(data.temp);

            return (
              <div key={corner} className="flex items-center gap-2 min-h-0">
                {/* Compact circle */}
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300"
                  style={{ backgroundColor: color, opacity: 0.85, padding: '6px' }}
                >
                  <div className="text-sm font-bold text-white text-center">{Math.round(data.temp)}</div>
                </div>

                {/* Corner Label and Pressure */}
                <div className="min-w-0">
                  <div className="text-[7px] font-bold text-white">{corner}</div>
                  <div className="text-[6px] text-muted-foreground">{Math.round(data.pressure)}bar</div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// Brake System - Option 4: Compact Matrix
export function BrakeCompactMatrix() {
  const [brakes, setBrakes] = useState({
    FL: { temp: 420, pressure: 85, wear: 15 },
    FR: { temp: 425, pressure: 86, wear: 16 },
    RL: { temp: 410, pressure: 82, wear: 12 },
    RR: { temp: 415, pressure: 83, wear: 13 }
  });
  const { telemetryData } = useCogniraceStore();

  useEffect(() => {
    if (telemetryData) {
      setBrakes({
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
      });
    }
  }, [telemetryData]);

  const getTempColor = (temp: number) => {
    if (temp > 500) return 'bg-red-600/30 border-red-500';
    if (temp > 450) return 'bg-orange-600/30 border-orange-500';
    return 'bg-green-600/30 border-green-500';
  };

  const getTextColor = (temp: number) => {
    if (temp > 500) return '#ef4444';
    if (temp > 450) return '#fb923c';
    return '#22c55e';
  };

  return (
    <Card className="bg-black/40 h-full !py-0 !gap-0">
      <CardContent className="!py-0 !px-2 h-full flex flex-col">
        <div className="text-red-400 text-[10px] font-bold mb-0.5 tracking-wider">STATUS</div>

        {/* 2x2 Grid of Brake Status */}
        <div className="grid grid-cols-2 gap-1 flex-1">
          {Object.entries(brakes).map(([corner, data]) => (
            <div
              key={corner}
              className={`flex flex-col p-1.5 rounded border ${getTempColor(data.temp)}`}
            >
              {/* Corner label */}
              <div className="text-[7px] font-bold text-white mb-0.5">{corner}</div>

              {/* Temperature */}
              <div className="text-center mb-0.5">
                <div style={{ color: getTextColor(data.temp) }} className="text-xs font-black font-mono">
                  {Math.round(data.temp)}°
                </div>
              </div>

              {/* Pressure bar */}
              <div className="mb-0.5">
                <div className="text-[5px] text-muted-foreground mb-0.5">P</div>
                <div className="h-0.5 bg-gray-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-cyan-500"
                    style={{ width: `${(data.pressure / 100) * 100}%` }}
                  />
                </div>
              </div>

              {/* Wear bar */}
              <div>
                <div className="text-[5px] text-muted-foreground mb-0.5">W</div>
                <div className="h-0.5 bg-gray-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-yellow-500"
                    style={{ width: `${(data.wear / 20) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

