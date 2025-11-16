'use client';

/**
 * Race Context Panel Component  
 * Displays and allows editing of current race context
 */

import { useCogniraceStore } from '@/lib/store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

export default function RaceContextPanel() {
  const { raceContext, setRaceContext } = useCogniraceStore();

  const updateField = (field: string, value: string) => {
    setRaceContext({
      ...raceContext,
      [field]: value,
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">🏁</span>
          Race Context
        </CardTitle>
        <CardDescription>
          Configure current race conditions and strategy
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Track Info */}
        <Card className="border-primary/20">
          <CardContent className="pt-6">
            <div className="space-y-3">
              <div>
                <label className="text-xs text-muted-foreground uppercase font-semibold mb-2 block">
                  🏎️ Track
                </label>
                <input
                  type="text"
                  value={raceContext.track}
                  onChange={(e) => updateField('track', e.target.value)}
                  className="w-full px-3 py-2 rounded-md bg-background border border-input focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="Circuit name"
                />
              </div>

              <div>
                <label className="text-xs text-muted-foreground uppercase font-semibold mb-2 block">
                  🌤️ Weather
                </label>
                <input
                  type="text"
                  value={raceContext.weather}
                  onChange={(e) => updateField('weather', e.target.value)}
                  className="w-full px-3 py-2 rounded-md bg-background border border-input focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="Weather conditions"
                />
              </div>

              <div>
                <label className="text-xs text-muted-foreground uppercase font-semibold mb-2 block">
                  🎯 Strategy
                </label>
                <input
                  type="text"
                  value={raceContext.strategy}
                  onChange={(e) => updateField('strategy', e.target.value)}
                  className="w-full px-3 py-2 rounded-md bg-background border border-input focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="Race strategy"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Separator />

        {/* Current Values Display */}
        <Card className="bg-muted/50">
          <CardContent className="pt-6">
            <p className="text-sm font-semibold mb-3">📋 Current Configuration</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Track:</span>
                <Badge variant="secondary">{raceContext.track || 'Not set'}</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Weather:</span>
                <Badge variant="secondary">{raceContext.weather || 'Not set'}</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Strategy:</span>
                <Badge variant="secondary">{raceContext.strategy || 'Not set'}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Info */}
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="pt-6">
            <p className="font-semibold mb-2 flex items-center gap-2 text-sm">
              <span>ℹ️</span>
              About Race Context
            </p>
            <p className="text-xs text-muted-foreground">
              This context is shared with the AI agent to provide more relevant 
              and accurate strategic recommendations based on track conditions and your race plan.
            </p>
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  );
}
