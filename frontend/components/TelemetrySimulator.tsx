'use client';

/**
 * Telemetry Simulator Component
 * Simulates streaming telemetry data to test real-time agent interaction
 */

import { useState, useRef, useEffect } from 'react';
import { useCogniraceStore } from '@/lib/store';
import { apiClient } from '@/lib/api-client';
import { debugLogger } from './DebugPanel';
import { useLiveKitContext } from '@/lib/livekit/LiveKitContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

export default function TelemetrySimulator() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentLap, setCurrentLap] = useState(1);
  const [speed, setSpeed] = useState(150);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { setTelemetry } = useCogniraceStore();
  const { sendTextMessage, isAgentReady } = useLiveKitContext();

  // Mock telemetry data generator
  const generateTelemetryData = (lap: number) => {
    const baseSpeed = 180 + Math.random() * 100;
    const baseRPM = 8000 + Math.random() * 4000;
    
    return {
      speed: Math.round(baseSpeed),
      nmot: Math.round(baseRPM),
      gear: Math.floor(baseSpeed / 40) + 1,
      aps: Math.round(70 + Math.random() * 30),
      lap,
      fuel_level: Math.max(10, 50 - (lap * 2.5)),
      cum_brake_energy: lap * 1250 + Math.random() * 200,
      cum_lateral_load: lap * 3400 + Math.random() * 500,
      air_temp: 26 + Math.random() * 4
    };
  };

  const startStreaming = () => {
    debugLogger.info('SIMULATOR', 'Starting telemetry stream');
    setIsStreaming(true);
    
    let lap = currentLap;
    let tickCount = 0;

    intervalRef.current = setInterval(async () => {
      tickCount++;

      const telemetry = generateTelemetryData(lap);
      setTelemetry(telemetry);
      setSpeed(telemetry.speed);

      debugLogger.info('TELEMETRY', `Lap ${lap} | Speed: ${telemetry.speed} km/h | RPM: ${telemetry.nmot} | Fuel: ${telemetry.fuel_level.toFixed(1)}L`, telemetry);

      if (tickCount % 10 === 0) {
        lap++;
        setCurrentLap(lap);
        debugLogger.success('SIMULATOR', `Lap ${lap - 1} completed! Starting Lap ${lap}`);

        if (isAgentReady) {
          try {
            await sendTextMessage(`I just completed lap ${lap - 1}`);
            debugLogger.success('AGENT', `Sent lap completion message to agent`);
          } catch (error) {
            debugLogger.error('AGENT', 'Failed to send lap message', error);
          }
        }
      }

      if (tickCount % 5 === 0) {
        try {
          debugLogger.info('API', 'Requesting fuel prediction...');
          const fuelPred = await apiClient.predictFuel(telemetry);
          debugLogger.success('API', 'Fuel prediction received', fuelPred);

          debugLogger.info('API', 'Requesting tire prediction...');
          const tirePred = await apiClient.predictTire({
            cum_brake_energy: telemetry.cum_brake_energy,
            cum_lateral_load: telemetry.cum_lateral_load,
            air_temp: telemetry.air_temp,
            telemetry_sequence: [[telemetry.speed, telemetry.nmot, telemetry.gear, telemetry.aps]]
          });
          debugLogger.success('API', 'Tire prediction received', tirePred);

        } catch (error) {
          debugLogger.error('API', 'Prediction API call failed', error);
        }
      }

      if (lap > 50) {
        stopStreaming();
        debugLogger.warning('SIMULATOR', 'Reached lap 50, stopping stream');
      }
    }, 1000);
  };

  const stopStreaming = () => {
    debugLogger.info('SIMULATOR', 'Stopping telemetry stream');
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsStreaming(false);
  };

  const resetSimulator = () => {
    debugLogger.info('SIMULATOR', 'Resetting simulator');
    stopStreaming();
    setCurrentLap(1);
    setSpeed(150);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">🏁</span>
          Telemetry Simulator
        </CardTitle>
        <CardDescription>
          Simulate real-time telemetry streaming
        </CardDescription>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-4">
          {/* Status Card */}
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Status</p>
                  <Badge variant={isStreaming ? "default" : "secondary"} className="text-sm">
                    {isStreaming ? '🟢 STREAMING' : '⚫ STOPPED'}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Current Lap</p>
                  <p className="text-2xl font-bold">{currentLap}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Current Speed</p>
                  <p className="text-xl font-bold text-primary">{speed} km/h</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Data Rate</p>
                  <Badge variant="outline">1 Hz</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Controls */}
          <div className="space-y-3">
            {!isStreaming ? (
              <Button 
                onClick={startStreaming} 
                className="w-full h-14 text-lg"
                size="lg"
              >
                <span className="text-2xl mr-2">🚀</span>
                START STREAMING
              </Button>
            ) : (
              <Button 
                onClick={stopStreaming} 
                variant="destructive"
                className="w-full h-14 text-lg"
                size="lg"
              >
                <span className="text-2xl mr-2">⏹️</span>
                STOP STREAMING
              </Button>
            )}

            <Button
              onClick={resetSimulator}
              disabled={isStreaming}
              variant="outline"
              className="w-full"
            >
              🔄 Reset to Lap 1
            </Button>
          </div>

          <Separator />

          {/* Quick Actions */}
          <div>
            <p className="text-sm font-semibold mb-3">Quick Test Actions</p>
            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={() => {
                  const telemetry = generateTelemetryData(currentLap);
                  apiClient.predictFuel(telemetry).then(result => {
                    debugLogger.success('API TEST', 'Manual fuel prediction', result);
                  }).catch(error => {
                    debugLogger.error('API TEST', 'Manual fuel prediction failed', error);
                  });
                }}
                disabled={isStreaming}
                variant="secondary"
                size="sm"
              >
                Test Fuel API
              </Button>
              <Button
                onClick={() => {
                  apiClient.checkHealth().then(result => {
                    debugLogger.success('API TEST', 'Health check passed', result);
                  }).catch(error => {
                    debugLogger.error('API TEST', 'Health check failed', error);
                  });
                }}
                variant="secondary"
                size="sm"
              >
                Test Health
              </Button>
            </div>
          </div>

          <Separator />

          {/* Info */}
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="pt-6">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <span>ℹ️</span>
                How it works
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs text-muted-foreground">
                <li>Streams telemetry data at 1 Hz (once per second)</li>
                <li>Updates lap every 10 seconds</li>
                <li>Makes API predictions every 5 seconds</li>
                <li>Sends lap completion messages to agent</li>
                <li>All events logged to debug panel below</li>
                <li>Simulates 50 laps total (stops automatically)</li>
              </ul>
            </CardContent>
          </Card>
        </CardContent>
      </ScrollArea>
    </Card>
  );
}
