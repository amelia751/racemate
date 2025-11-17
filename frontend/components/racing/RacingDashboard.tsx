'use client';

/**
 * Racing Dashboard - F1/Red Bull Inspired
 * Professional racing telemetry interface
 */

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import TelemetryCharts from './TelemetryCharts';
import LapTimeDisplay from './LapTimeDisplay';
import StreamingControls from './StreamingControls';
import { ThrottleBrakeTimeSeries, GForceTimeSeries } from './RedBullStyleCharts';
import { FuelConsumptionChart, TireTemperatureDisplay, BrakeSystemStatus } from './EnhancedVisualizations';
import HeroMetrics from './HeroMetrics';
import VoiceStrategist from '@/components/VoiceStrategist';
// import DebugLayer from '@/components/DebugLayer'; // Hidden for production demo

export default function RacingDashboard() {
  const [mounted, setMounted] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-cyan-400 text-2xl font-bold animate-pulse">
          INITIALIZING RACEMATE SYSTEMS...
        </div>
      </div>
    );
  }

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white overflow-hidden flex flex-col">
      {/* Animated Background Grid */}
      <div className="fixed inset-0 pointer-events-none opacity-10 z-0">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(to right, cyan 1px, transparent 1px),
            linear-gradient(to bottom, cyan 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }} />
      </div>

      {/* Main Content Grid - Full Screen */}
      <div className="flex-1 relative z-10 overflow-hidden px-6 py-6">
        <div className="grid grid-cols-12 gap-8 h-full">
          
          {/* LEFT COLUMN - Telemetry Charts with Header */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="col-span-9 flex flex-col h-full overflow-hidden"
          >
            {/* Dashboard Header */}
            <div className="bg-black/60 backdrop-blur-md px-6 py-4 flex-shrink-0">
              <div className="flex items-center gap-4">
                <div className="text-2xl font-black tracking-tight">
                  <span className="bg-gradient-to-r from-cyan-400 to-cyan-300 bg-clip-text text-transparent">RACE</span>
                  <span className="bg-gradient-to-r from-yellow-400 to-amber-400 bg-clip-text text-transparent">MATE</span>
                </div>
                {isStreaming && (
                  <Badge variant="default" className="text-xs bg-green-500 text-white animate-pulse">
                    ● LIVE
                  </Badge>
                )}
              </div>
            </div>

            {/* Scrollable Dashboard Content */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden pr-2 space-y-4 pt-4 pb-6" style={{ minHeight: 0 }}>
            {/* Lap Time & Current Status - Side by Side */}
            <div className="grid grid-cols-2 gap-4 h-auto">
              {/* Lap Time Info - Left */}
              <Card className="bg-black/40 backdrop-blur-sm">
                <CardContent className="pt-4 h-full">
                  <LapTimeDisplay />
                </CardContent>
              </Card>

              {/* Hero Metrics - Right */}
              <div className="h-full">
                <HeroMetrics />
              </div>
            </div>

            {/* Real-time Charts - Speed & RPM */}
            <div className="h-[200px]">
              <TelemetryCharts />
            </div>

            {/* Time-Series Charts */}
            <div className="grid grid-cols-2 gap-4 h-[150px]">
              <ThrottleBrakeTimeSeries />
              <GForceTimeSeries />
            </div>

            {/* Brake System Status - Full Width */}
            <div className="h-[120px]">
              <BrakeSystemStatus />
            </div>

            {/* Advanced System Visualizations */}
            <div className="grid grid-cols-2 gap-4 h-[220px] flex-shrink-0">
              <div className="h-full overflow-hidden">
                <FuelConsumptionChart />
              </div>
              <div className="h-full overflow-hidden">
                <TireTemperatureDisplay />
              </div>
            </div>

            {/* Streaming Controls */}
            <StreamingControls onStreamingChange={setIsStreaming} />
            </div>
          </motion.div>

          {/* RIGHT COLUMN - AI Assistant - Full Height */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="col-span-3 flex flex-col h-full"
            style={{ minHeight: 0 }}
          >
            <VoiceStrategist />
          </motion.div>

        </div>
      </div>

      {/* Debug Layer - Hidden for clean demo */}
      {/* <DebugLayer /> */}
    </div>
  );
}

