'use client';

/**
 * Streaming Controls - F1 Style
 * Controls for telemetry simulation
 */

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Square, Zap, Settings } from 'lucide-react';
import { motion } from 'framer-motion';
import { useCogniraceStore } from '@/lib/store';

interface StreamingControlsProps {
  onStreamingChange?: (isStreaming: boolean) => void;
}

export default function StreamingControls({ onStreamingChange }: StreamingControlsProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [dataRate, setDataRate] = useState(10); // Hz
  const [currentFuel, setCurrentFuel] = useState(50); // Start with full tank
  const { addDebugLog } = useCogniraceStore();

  const { setTelemetry } = useCogniraceStore();

  const startStreaming = () => {
    setIsStreaming(true);
    onStreamingChange?.(true);
    setCurrentFuel(50); // Reset fuel to full tank when streaming starts
    addDebugLog('info', 'Telemetry streaming started', { rate: dataRate });

    // Simulate telemetry updates
    const interval = setInterval(() => {
      // Fuel decreases by ~0.3-0.7L per update (noticeable decrease)
      setCurrentFuel(prev => Math.max(0, prev - (0.3 + Math.random() * 0.4)));

      setCurrentFuel(fuel => {
        const telemetry = {
          speed: 150 + Math.random() * 50,
          rpm: 7000 + Math.random() * 2000,
          nmot: 7000 + Math.random() * 2000,
          gear: Math.floor(Math.random() * 6) + 1,
          throttle: 60 + Math.random() * 40,
          aps: 60 + Math.random() * 40,
          lap: 13,
          fuel_level: fuel,
          cum_brake_energy: 25000 + Math.random() * 5000,
          cum_lateral_load: 45000 + Math.random() * 10000,
          air_temp: 26 + Math.random() * 2,
        };

        // Update store with real data
        setTelemetry(telemetry);
        addDebugLog('success', 'Telemetry updated', {
          speed: Math.round(telemetry.speed),
          rpm: Math.round(telemetry.rpm || 0),
          gear: telemetry.gear,
          fuel: fuel.toFixed(1)
        });

        return fuel;
      });
    }, 1000 / dataRate);

    // Store interval for cleanup
    (window as any).telemetryInterval = interval;
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    onStreamingChange?.(false);
    if ((window as any).telemetryInterval) {
      clearInterval((window as any).telemetryInterval);
    }
    addDebugLog('warn', 'Telemetry streaming stopped');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
    >
      <Card className="bg-gradient-to-r from-cyan-900/30 to-purple-900/30 backdrop-blur-sm">
        <CardContent className="pt-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              <span className="text-sm font-bold tracking-wider text-cyan-400">
                TELEMETRY STREAM
              </span>
            </div>
            {isStreaming && (
              <Badge variant="default" className="animate-pulse bg-green-500">
                <div className="w-2 h-2 rounded-full bg-white mr-1" />
                LIVE
              </Badge>
            )}
          </div>

          <div className="space-y-3">
            {/* Start/Stop Controls */}
            <div className="flex gap-2">
              {!isStreaming ? (
                <Button
                  onClick={startStreaming}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white font-bold"
                  size="lg"
                >
                  <Play className="w-4 h-4 mr-2" />
                  START STREAMING
                </Button>
              ) : (
                <Button
                  onClick={stopStreaming}
                  variant="destructive"
                  className="flex-1 font-bold"
                  size="lg"
                >
                  <Square className="w-4 h-4 mr-2" />
                  STOP STREAMING
                </Button>
              )}
            </div>

            {/* Data Rate Control */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Update Rate</span>
                <span className="text-cyan-400 font-bold">{dataRate} Hz</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={dataRate}
                onChange={(e) => setDataRate(Number(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-3 gap-2 pt-2">
              <Button
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={() => addDebugLog('info', 'Lap 13 started')}
              >
                📍 Lap 13
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={() => addDebugLog('warn', 'Pit window open')}
              >
                🏁 Pit
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={() => addDebugLog('error', 'FCY deployed')}
              >
                🚨 FCY
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

