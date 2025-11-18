'use client';

/**
 * Telemetry Display Component
 * Shows real-time telemetry data in a fixed-height scrollable area
 */

import { useRaceMateStore } from '@/lib/store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';

export default function TelemetryDisplay() {
  const telemetry = useRaceMateStore((state) => state.telemetryData);

  if (!telemetry) {
    return (
      <Card className="h-[400px] flex items-center justify-center">
        <CardContent>
          <div className="text-center space-y-4">
            <div className="text-6xl">📡</div>
            <p className="text-muted-foreground">
              No telemetry data yet
            </p>
            <p className="text-sm text-muted-foreground">
              Start the simulator to begin streaming
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getSpeedColor = (speed: number) => {
    if (speed > 250) return 'text-red-500';
    if (speed > 200) return 'text-orange-500';
    if (speed > 150) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getFuelColor = (fuel: number) => {
    if (fuel < 15) return 'text-red-500';
    if (fuel < 30) return 'text-yellow-500';
    return 'text-green-500';
  };

  const fuelPercentage = (telemetry.fuel_level / 50) * 100;

  return (
    <Card className="h-[400px] flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Real-Time Telemetry
          </span>
          <Badge variant="default" className="animate-pulse">
            🔴 LIVE
          </Badge>
        </CardTitle>
        <CardDescription>
          Current vehicle data stream
        </CardDescription>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-4">
          {/* Primary Metrics */}
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="pt-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground uppercase mb-2">Speed</p>
                  <p className={`text-4xl font-bold ${getSpeedColor(telemetry.speed)}`}>
                    {telemetry.speed}
                  </p>
                  <p className="text-xs text-muted-foreground">km/h</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase mb-2">RPM</p>
                  <p className="text-4xl font-bold text-primary">
                    {telemetry.nmot?.toLocaleString()}
                  </p>
                  <p className="text-xs text-muted-foreground">rev/min</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Secondary Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground mb-1">Gear</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold">{telemetry.gear}</span>
                  <Badge variant="outline" className="text-xs">GEAR</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground mb-1">Throttle</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-primary">{telemetry.aps}</span>
                  <span className="text-sm text-muted-foreground">%</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground mb-1">Lap</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-primary">{telemetry.lap}</span>
                  <Badge variant="secondary" className="text-xs">CURRENT</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground mb-1">Air Temp</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold">{telemetry.air_temp?.toFixed(1)}</span>
                  <span className="text-sm text-muted-foreground">°C</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <Separator />

          {/* Fuel Status */}
          <Card className="border-orange-500/30 bg-orange-500/5">
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold mb-1">⛽ Fuel Level</p>
                  <p className={`text-3xl font-bold ${getFuelColor(telemetry.fuel_level)}`}>
                    {telemetry.fuel_level.toFixed(1)} L
                  </p>
                </div>
                <Badge 
                  variant={telemetry.fuel_level < 15 ? "destructive" : "secondary"}
                  className="h-6"
                >
                  {Math.round(fuelPercentage)}%
                </Badge>
              </div>
              <Progress value={fuelPercentage} className="h-3" />
              {telemetry.fuel_level < 15 && (
                <p className="text-xs text-destructive font-semibold animate-pulse">
                  ⚠️ LOW FUEL WARNING
                </p>
              )}
            </CardContent>
          </Card>

          <Separator />

          {/* Tire Stress Metrics */}
          <Card className="border-blue-500/30 bg-blue-500/5">
            <CardContent className="pt-6">
              <p className="text-sm font-semibold mb-3">🏎️ Tire Stress Indicators</p>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted-foreground">Brake Energy</span>
                    <span className="font-mono">
                      {telemetry.cum_brake_energy?.toFixed(0)} kJ
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(((telemetry.cum_brake_energy || 0) / 50000) * 100, 100)} 
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted-foreground">Lateral Load</span>
                    <span className="font-mono">
                      {telemetry.cum_lateral_load?.toFixed(0)} N
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(((telemetry.cum_lateral_load || 0) / 100000) * 100, 100)} 
                    className="h-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Timestamp */}
          <Card className="bg-muted/50">
            <CardContent className="pt-4">
              <p className="text-xs text-muted-foreground text-center font-mono">
                Last Update: {new Date().toLocaleTimeString()}
              </p>
            </CardContent>
          </Card>
        </CardContent>
      </ScrollArea>
    </Card>
  );
}
