'use client';

/**
 * Main Racing Dashboard
 * Layout: Telemetry (left) | Voice Agent Chat (right)
 */

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import TelemetryDisplay from '@/components/TelemetryDisplay';
import RaceContextPanel from '@/components/RaceContextPanel';
import VapiVoiceChat from '@/components/VapiVoiceChat';
import TelemetrySimulator from '@/components/TelemetrySimulator';
import DebugLayer from '@/components/DebugLayer';
import { useCogniraceStore } from '@/lib/store';

export default function RaceDashboard() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className="h-screen w-screen bg-background flex flex-col overflow-hidden">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-3xl">🏁</div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">Cognirace</h1>
                <p className="text-sm text-muted-foreground">
                  Real-Time AI Race Strategy Platform
                </p>
              </div>
            </div>
            <Badge variant="default" className="animate-pulse">
              <span className="mr-1">●</span> LIVE
            </Badge>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="flex-1 overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 h-full">
          {/* LEFT SIDE: Telemetry & Race Context */}
          <div className="border-r border-border bg-background overflow-y-auto">
            <div className="p-4 space-y-4">
              {/* Race Context Panel */}
              <RaceContextPanel />

              {/* Telemetry Display */}
              <TelemetryDisplay />

              {/* Simulator Controls */}
              <TelemetrySimulator />
            </div>
          </div>

          {/* RIGHT SIDE: Voice Agent Chat Interface */}
          <div className="bg-card/30 flex flex-col">
            <VapiVoiceChat />
          </div>
        </div>
      </div>
      
      {/* Debug Layer */}
      <DebugLayer />
    </div>
  );
}

