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
import VapiVoiceChat from '@/components/VapiVoiceChat';
import DebugLayer from '@/components/DebugLayer';

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
          INITIALIZING COGNIRACE SYSTEMS...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white overflow-hidden">
      {/* Animated Background Grid */}
      <div className="fixed inset-0 pointer-events-none opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(to right, cyan 1px, transparent 1px),
            linear-gradient(to bottom, cyan 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }} />
      </div>

      {/* Header */}
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="relative z-10 bg-black/60 backdrop-blur-md"
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-3xl font-black tracking-tighter">
                <span className="text-cyan-400">COGNI</span>
                <span className="text-yellow-400">RACE</span>
              </div>
              {isStreaming && (
                <Badge variant="default" className="text-xs bg-green-500 text-white animate-pulse">
                  ● LIVE
                </Badge>
              )}
            </div>
            
            <div className="flex items-center gap-6">
              <div className="text-right">
                <div className="text-xs text-muted-foreground">TOYOTA GR CUP</div>
                <div className="text-sm font-bold text-cyan-400">SERIES 2025</div>
              </div>
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-cyan-500 to-yellow-500 flex items-center justify-center font-black">
                AI
              </div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-6 relative z-10">
        <div className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">
          
          {/* LEFT COLUMN - Telemetry Charts */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="col-span-9 space-y-4 overflow-y-auto pr-2"
            style={{ maxHeight: 'calc(100vh - 140px)' }}
          >
            {/* Lap Time & Current Status - Side by Side */}
            <div className="grid grid-cols-2 gap-4 h-[180px]">
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

            {/* Advanced System Visualizations */}
            <div className="grid grid-cols-2 gap-4 h-[120px]">
              <FuelConsumptionChart />
              <TireTemperatureDisplay />
            </div>

            <div className="h-[120px]">
              <BrakeSystemStatus />
            </div>

            {/* Streaming Controls */}
            <StreamingControls onStreamingChange={setIsStreaming} />
          </motion.div>

          {/* RIGHT COLUMN - AI Assistant */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="col-span-3"
          >
            <Card className="bg-black/40 backdrop-blur-sm h-full flex flex-col">
              <div className="px-4 py-3 bg-yellow-500/5">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                  <span className="text-xs font-bold tracking-wider text-yellow-400">
                    AI RACE STRATEGIST
                  </span>
                </div>
              </div>
              <div className="flex-1 overflow-hidden">
                <VapiVoiceChat />
              </div>
            </Card>
          </motion.div>

        </div>
      </div>

      {/* Animated Corner Accents */}
      <div className="fixed top-0 left-0 w-32 h-32 border-t-2 border-l-2 border-cyan-500/50 pointer-events-none" />
      <div className="fixed top-0 right-0 w-32 h-32 border-t-2 border-r-2 border-yellow-500/50 pointer-events-none" />
      <div className="fixed bottom-0 left-0 w-32 h-32 border-b-2 border-l-2 border-purple-500/50 pointer-events-none" />
      <div className="fixed bottom-0 right-0 w-32 h-32 border-b-2 border-r-2 border-red-500/50 pointer-events-none" />

      {/* Debug Layer */}
      <DebugLayer />
    </div>
  );
}

