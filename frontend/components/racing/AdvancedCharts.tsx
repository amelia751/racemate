'use client';

/**
 * Advanced Racing Charts - Red Bull Style
 * More diverse and sophisticated visualizations
 */

import { useEffect, useState } from 'react';
import { BarChart, Bar, RadialBarChart, RadialBar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { useCogniraceStore } from '@/lib/store';

export function GForceGauge() {
  const [gForce, setGForce] = useState({ lateral: 0, longitudinal: 0 });

  useEffect(() => {
    const interval = setInterval(() => {
      setGForce({
        lateral: -1 + Math.random() * 2,
        longitudinal: -1 + Math.random() * 2
      });
    }, 100);
    return () => clearInterval(interval);
  }, []);

  const data = [
    { name: 'Lateral', value: Math.abs(gForce.lateral) * 50, fill: '#a855f7' },
    { name: 'Long', value: Math.abs(gForce.longitudinal) * 50, fill: '#22c55e' }
  ];

  return (
    <Card className="bg-black/40 border-purple-500/30 h-full">
      <CardContent className="pt-4 h-full flex flex-col">
        <div className="text-purple-400 text-xs font-bold mb-2 tracking-wider">G-FORCE</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="relative w-full h-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart 
                innerRadius="30%" 
                outerRadius="90%" 
                data={data}
                startAngle={180}
                endAngle={0}
              >
                <PolarGrid gridType="circle" stroke="#333" />
                <RadialBar dataKey="value" cornerRadius={10} />
                <text 
                  x="50%" 
                  y="50%" 
                  textAnchor="middle" 
                  dominantBaseline="middle"
                  className="text-2xl font-bold fill-purple-400"
                >
                  {gForce.lateral.toFixed(2)}G
                </text>
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute bottom-0 left-0 right-0 text-center">
              <div className="text-xs text-muted-foreground">
                LAT: {gForce.lateral.toFixed(2)} | LONG: {gForce.longitudinal.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function BrakeTemperature() {
  const [temps, setTemps] = useState({
    FL: 420, FR: 425, RL: 410, RR: 415
  });

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

  const data = [
    { name: 'FL', temp: temps.FL, fill: temps.FL > 500 ? '#ef4444' : temps.FL > 450 ? '#fb923c' : '#22c55e' },
    { name: 'FR', temp: temps.FR, fill: temps.FR > 500 ? '#ef4444' : temps.FR > 450 ? '#fb923c' : '#22c55e' },
    { name: 'RL', temp: temps.RL, fill: temps.RL > 500 ? '#ef4444' : temps.RL > 450 ? '#fb923c' : '#22c55e' },
    { name: 'RR', temp: temps.RR, fill: temps.RR > 500 ? '#ef4444' : temps.RR > 450 ? '#fb923c' : '#22c55e' }
  ];

  return (
    <Card className="bg-black/40 border-red-500/30 h-full">
      <CardContent className="pt-4 h-full">
        <div className="text-red-400 text-xs font-bold mb-2 tracking-wider">BRAKE TEMP (°C)</div>
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="name" stroke="#ef4444" />
            <YAxis stroke="#ef4444" domain={[300, 600]} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#000', border: '1px solid #ef4444' }}
              labelStyle={{ color: '#ef4444' }}
            />
            <Bar dataKey="temp" radius={[8, 8, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

export function TirePressureGrid() {
  const [pressures, setPressures] = useState({
    FL: 32, FR: 32, RL: 30, RR: 30
  });

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
    if (psi < 29 || psi > 34) return 'text-red-400 border-red-500';
    if (psi < 30 || psi > 33) return 'text-orange-400 border-orange-500';
    return 'text-green-400 border-green-500';
  };

  return (
    <Card className="bg-black/40 border-cyan-500/30 h-full">
      <CardContent className="pt-4 h-full">
        <div className="text-cyan-400 text-xs font-bold mb-3 tracking-wider">TIRE PRESSURE (PSI)</div>
        <div className="grid grid-cols-2 gap-4 h-[calc(100%-30px)]">
          {Object.entries(pressures).map(([tire, psi]) => (
            <motion.div
              key={tire}
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
              className={`border-2 rounded-lg p-3 flex flex-col items-center justify-center ${getTireColor(psi)}`}
            >
              <div className="text-xs text-muted-foreground mb-1">{tire}</div>
              <div className="text-3xl font-black font-mono">{psi.toFixed(1)}</div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function FuelGauge() {
  const { telemetryData } = useCogniraceStore();
  const fuelLevel = telemetryData?.fuel_level || 35.2;
  const fuelPercentage = (fuelLevel / 50) * 100;

  const data = [
    { name: 'Fuel', value: fuelPercentage, fill: fuelPercentage > 40 ? '#22c55e' : fuelPercentage > 20 ? '#fb923c' : '#ef4444' }
  ];

  return (
    <Card className="bg-black/40 border-orange-500/30 h-full">
      <CardContent className="pt-4 h-full flex flex-col">
        <div className="text-orange-400 text-xs font-bold mb-2 tracking-wider">FUEL REMAINING</div>
        <div className="flex-1 flex items-center justify-center">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              innerRadius="60%"
              outerRadius="100%"
              data={data}
              startAngle={90}
              endAngle={-270}
            >
              <PolarGrid gridType="circle" stroke="#333" />
              <RadialBar
                dataKey="value"
                cornerRadius={10}
                fill={data[0].fill}
              />
              <text 
                x="50%" 
                y="50%" 
                textAnchor="middle" 
                dominantBaseline="middle"
                className="text-3xl font-black fill-orange-400"
              >
                {fuelLevel.toFixed(1)}L
              </text>
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
        <div className="text-center text-xs text-muted-foreground">
          {fuelPercentage.toFixed(0)}% • ~{Math.floor(fuelLevel / 2.5)} laps
        </div>
      </CardContent>
    </Card>
  );
}

