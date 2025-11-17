'use client';

/**
 * Streaming Controls - F1 Style
 * Controls for telemetry simulation with ROBUST scenarios
 */

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Square, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { useCogniraceStore } from '@/lib/store';

interface StreamingControlsProps {
  onStreamingChange?: (isStreaming: boolean) => void;
}

// Robust scenario generator
const generateScenario = (frameCount: number) => {
  const scenarios = [
    // Scenario 1: FUEL CRISIS (immediate)
    {
      name: 'FUEL_CRISIS',
      data: {
        speed: 180 + Math.random() * 20,
        rpm: 8000 + Math.random() * 1000,
        gear: 5,
        throttle: 75 + Math.random() * 15,
        fuel_level: 4 + Math.random() * 2, // CRITICAL LOW FUEL
        lap: 18,
        cum_brake_energy: 30000 + Math.random() * 5000,
        cum_lateral_load: 50000 + Math.random() * 10000,
        air_temp: 28
      }
    },
    // Scenario 2: HIGH SPEED + LOW FUEL
    {
      name: 'HIGH_SPEED_LOW_FUEL',
      data: {
        speed: 195 + Math.random() * 10, // HIGH SPEED
        rpm: 9000 + Math.random() * 500,
        gear: 6,
        throttle: 90 + Math.random() * 10,
        fuel_level: 8 + Math.random() * 2, // LOW FUEL
        lap: 15,
        cum_brake_energy: 28000 + Math.random() * 3000,
        cum_lateral_load: 48000 + Math.random() * 8000,
        air_temp: 27
      }
    },
    // Scenario 3: ANOMALY (high RPM, low speed - potential issue)
    {
      name: 'ANOMALY_DETECTED',
      data: {
        speed: 95 + Math.random() * 20, // LOW SPEED
        rpm: 10500 + Math.random() * 500, // VERY HIGH RPM - ANOMALY
        gear: 3,
        throttle: 85 + Math.random() * 10,
        fuel_level: 22 + Math.random() * 5,
        lap: 10,
        cum_brake_energy: 35000 + Math.random() * 5000,
        cum_lateral_load: 55000 + Math.random() * 10000,
        air_temp: 29
      }
    },
    // Scenario 4: OPTIMAL PERFORMANCE (for contrast)
    {
      name: 'OPTIMAL',
      data: {
        speed: 170 + Math.random() * 15,
        rpm: 7500 + Math.random() * 1000,
        gear: 5,
        throttle: 70 + Math.random() * 15,
        fuel_level: 30 + Math.random() * 5,
        lap: 8,
        cum_brake_energy: 20000 + Math.random() * 3000,
        cum_lateral_load: 40000 + Math.random() * 8000,
        air_temp: 26
      }
    },
    // Scenario 5: EXTREME CONDITIONS
    {
      name: 'EXTREME',
      data: {
        speed: 200 + Math.random() * 5, // VERY HIGH SPEED
        rpm: 10000 + Math.random() * 1000,
        gear: 6,
        throttle: 100, // FULL THROTTLE
        fuel_level: 6 + Math.random() * 2, // LOW FUEL
        lap: 20,
        cum_brake_energy: 40000 + Math.random() * 5000,
        cum_lateral_load: 60000 + Math.random() * 10000,
        air_temp: 32 // HIGH TEMP
      }
    },
    // Scenario 6: TIRE STRESS
    {
      name: 'TIRE_STRESS',
      data: {
        speed: 185 + Math.random() * 15,
        rpm: 8500 + Math.random() * 1000,
        gear: 5,
        throttle: 80 + Math.random() * 15,
        fuel_level: 15 + Math.random() * 5,
        lap: 22, // HIGH LAP COUNT - TIRE WEAR
        cum_brake_energy: 45000 + Math.random() * 5000, // HIGH BRAKE ENERGY
        cum_lateral_load: 65000 + Math.random() * 10000, // HIGH LATERAL LOAD
        air_temp: 30
      }
    }
  ];
  
  // Cycle through scenarios every 5 frames (5 seconds at 1Hz)
  const scenarioIndex = Math.floor(frameCount / 5) % scenarios.length;
  const scenario = scenarios[scenarioIndex];
  
  // Add common fields
  const telemetry = {
    ...scenario.data,
    nmot: scenario.data.rpm,
    aps: scenario.data.throttle
  };
  
  return { scenario: scenario.name, telemetry };
};

export default function StreamingControls({ onStreamingChange }: StreamingControlsProps) {
  const [dataRate, setDataRate] = useState(1); // 1 Hz for better observation
  const [frameCount, setFrameCount] = useState(0);
  const [currentScenario, setCurrentScenario] = useState('');
  const { addDebugLog, setTelemetry, isStreaming, setIsStreaming: setStreamingStore } = useCogniraceStore();
  
  // Simulate telemetry updates when streaming - SEND TO BACKEND VIA API
  useEffect(() => {
    if (!isStreaming) return;
    
    let frameCounter = 0;
    
    const interval = setInterval(() => {
      frameCounter++;
      
      // Generate scenario-based telemetry
      const { scenario, telemetry } = generateScenario(frameCounter);
      
      // Update UI state in next tick to avoid render conflicts
      setTimeout(() => {
        setFrameCount(frameCounter);
        setCurrentScenario(scenario);
        setTelemetry(telemetry);
      }, 0);
      
      // Log frame to debug panel (only every 10 frames to reduce noise)
      if (frameCounter % 10 === 0) {
        setTimeout(() => {
          addDebugLog('info', `📊 Frame ${frameCounter}: ${scenario}`, {
            fuel: telemetry.fuel_level,
            speed: telemetry.speed
          });
        }, 0);
      }
      
      // Send to backend via API route
      fetch('/api/telemetry/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry)
      })
      .then(res => res.json())
      .then(data => {
        if (data.success && data.recommendations) {
          // Store recommendation in window for VoiceStrategist to display
          if (typeof window !== 'undefined') {
            (window as any).__latestRecommendation = {
              ...data.recommendations,
              scenario: scenario,
              timestamp: Date.now()
            };
          }
          
          // Only log significant events
          const critical = data.recommendations.severity_summary?.critical || 0;
          const high = data.recommendations.severity_summary?.high || 0;
          if (critical > 0 || high >= 2) {
            setTimeout(() => {
              addDebugLog('warn', `⚠️ ${scenario}: C:${critical} H:${high}`, {
                events: data.events?.length || 0
              });
            }, 0);
          }
        }
      })
      .catch(err => {
        setTimeout(() => {
          addDebugLog('error', `❌ Backend API Error`, {
            message: err.message,
            scenario
          });
        }, 0);
      });
    }, 1000 / dataRate);

    return () => clearInterval(interval);
  }, [isStreaming, dataRate, setTelemetry, addDebugLog]);

  const startStreaming = () => {
    console.log('[StreamingControls] START STREAMING clicked');
    
    setStreamingStore(true); // Update global store
    onStreamingChange?.(true);
    setFrameCount(0);
    
    // Clear previous logs to focus on this streaming session (deferred)
    setTimeout(() => {
      (window as any).__clearDebugLogs?.();
      addDebugLog('success', '🏁 START STREAMING - Robust scenario testing begins', { 
        scenarios: 6,
        rate: `${dataRate} Hz`,
        note: 'Cycling through: FUEL_CRISIS, HIGH_SPEED, ANOMALY, OPTIMAL, EXTREME, TIRE_STRESS'
      });
    }, 100);
  };

  const stopStreaming = () => {
    console.log('[StreamingControls] STOP STREAMING clicked');
    
    setStreamingStore(false); // Update global store
    onStreamingChange?.(false);
    
    setTimeout(() => {
      addDebugLog('warn', '⏸️ STOP STREAMING - Session ended', { frames_sent: frameCount });
      setFrameCount(0);
    }, 100);
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
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-bold"
                  size="lg"
                >
                  <Square className="w-4 h-4 mr-2" />
                  STOP STREAMING
                </Button>
              )}
            </div>

            {/* Status Info */}
            {isStreaming && (
              <div className="text-xs space-y-1 p-3 bg-black/40 rounded border border-cyan-500/20">
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Scenario:</span>
                  <span className="text-cyan-400 font-mono">{currentScenario}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frames Sent:</span>
                  <span className="text-cyan-400 font-mono">{frameCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Update Rate:</span>
                  <span className="text-cyan-400 font-mono">{dataRate} Hz</span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
