'use client';

/**
 * Real-Time AI Race Strategist - WebSocket Version
 * Connects to backend event-driven prediction engine
 * No direct Gemini calls - backend handles all AI
 */

import { useEffect, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bot, Activity, AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCogniraceStore } from '@/lib/store';

interface Recommendation {
  id: string;
  type: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: Date;
  events?: any[];
}

export default function VoiceStrategist() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('Connecting to AI...');
  const { telemetryData } = useCogniraceStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Connect to WebSocket backend
  useEffect(() => {
    connectToBackend();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  const connectToBackend = () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8005';
      const wsUrl = backendUrl.replace('http', 'ws') + '/realtime/ws/telemetry';
      
      console.log('[WebSocket] Connecting to:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('[WebSocket] Connected to event-driven AI backend!');
        setIsConnected(true);
        setCurrentStatus('AI Strategist Online');
        // Don't add connection message to chat - badge shows connection status
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'recommendation') {
            // Event detected with AI recommendation
            const strategy = data.recommendations?.strategy || 'No strategy available';
            const eventCount = data.event_count || 0;
            const events = data.events || [];
            
            // Determine severity
            let type: 'info' | 'warning' | 'critical' = 'info';
            if (eventCount > 2 || events.some((e: any) => e.severity === 'critical')) {
              type = 'critical';
            } else if (eventCount > 0) {
              type = 'warning';
            }
            
            addRecommendation(type, strategy, events);
          } else if (data.type === 'status') {
            // No events, all nominal
            setCurrentStatus('Monitoring');
          }
        } catch (e) {
          console.error('[WebSocket] Parse error:', e);
        }
      };
      
      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setCurrentStatus('Connection error');
      };
      
      ws.onclose = () => {
        console.log('[WebSocket] Disconnected from backend');
        setIsConnected(false);
        setCurrentStatus('Reconnecting...');
        
        // Auto-reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectToBackend();
        }, 5000);
      };
      
      wsRef.current = ws;
      
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      setCurrentStatus('Failed to connect to backend');
      setIsConnected(false);
      // Error shown in badge and status, no need for chat message
    }
  };

  // Send telemetry to backend via WebSocket
  useEffect(() => {
    if (!isConnected || !telemetryData || !wsRef.current) return;
    
    // Only send if speed > 0 (car is moving)
    if (telemetryData.speed > 0) {
      try {
        const message = {
          telemetry: {
            lap: telemetryData.lap,
            speed: telemetryData.speed,
            rpm: telemetryData.rpm || telemetryData.nmot,
            nmot: telemetryData.nmot || telemetryData.rpm,
            gear: telemetryData.gear,
            throttle: telemetryData.throttle || telemetryData.aps,
            aps: telemetryData.aps || telemetryData.throttle,
            fuel_level: telemetryData.fuel_level,
            air_temp: telemetryData.air_temp,
            cum_brake_energy: telemetryData.cum_brake_energy,
            cum_lateral_load: telemetryData.cum_lateral_load,
            timestamp: new Date().toISOString()
          },
          timestamp: new Date().toISOString()
        };
        
        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('[WebSocket] Send error:', error);
      }
    }
  }, [telemetryData, isConnected]);

  const addRecommendation = (
    type: 'info' | 'warning' | 'critical', 
    message: string,
    events?: any[]
  ) => {
    const rec: Recommendation = {
      id: Date.now().toString() + Math.random(),
      type,
      message,
      timestamp: new Date(),
      events
    };
    setRecommendations(prev => [...prev, rec].slice(-15)); // Keep last 15
  };

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [recommendations]);

  const getIcon = (type: string) => {
    switch (type) {
      case 'critical': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'warning': return <Activity className="w-4 h-4 text-orange-400" />;
      default: return <Bot className="w-4 h-4 text-cyan-400" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'critical': return 'border-red-500/30 bg-red-500/5';
      case 'warning': return 'border-orange-500/30 bg-orange-500/5';
      default: return 'border-cyan-500/30 bg-cyan-500/5';
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-cyan-900/50 to-purple-900/50 border-b border-cyan-500/30 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-4 h-4 text-cyan-400" />
            <p className="text-sm font-medium">{recommendations.length} alert{recommendations.length !== 1 ? 's' : ''}</p>
          </div>
          {isConnected ? (
            <Badge variant="default" className="bg-green-500 animate-pulse text-xs gap-1">
              <Wifi className="w-3 h-3" />
              Connected
            </Badge>
          ) : (
            <Badge variant="destructive" className="text-xs gap-1">
              <WifiOff className="w-3 h-3" />
              Offline
            </Badge>
          )}
        </div>
      </div>

      {/* Recommendations Feed - Fixed Height with Scroll */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full px-4">
          <div className="py-4 space-y-3">
          {recommendations.length === 0 ? (
            <Card className="border-dashed bg-black/20">
              <CardContent className="pt-6 text-center">
                <Bot className="w-12 h-12 mx-auto mb-3 text-cyan-400 opacity-50" />
                <p className="text-sm text-muted-foreground mb-2">
                  {isConnected
                    ? "Monitoring active - AI will alert on critical events"
                    : "Connecting to AI backend..."}
                </p>
                {isConnected && (
                  <p className="text-xs text-muted-foreground/70">
                    Start streaming to see event-driven recommendations
                  </p>
                )}
              </CardContent>
            </Card>
          ) : (
            <AnimatePresence>
              {recommendations.map((rec, index) => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className={`${getTypeColor(rec.type)} border transition-all hover:border-opacity-60`}>
                    <CardContent className="p-4">
                      <div className="flex gap-3">
                        <div className="flex-shrink-0 mt-1">
                          {getIcon(rec.type)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">
                            {rec.message}
                          </p>
                          {rec.events && rec.events.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-white/10">
                              <p className="text-xs text-muted-foreground mb-1">
                                Detected Events:
                              </p>
                              {rec.events.slice(0, 3).map((event: any, i: number) => (
                                <div key={i} className="text-xs text-cyan-400/70">
                                  • {event.event_type}
                                </div>
                              ))}
                            </div>
                          )}
                          <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                            <span>{rec.timestamp.toLocaleTimeString()}</span>
                            {rec.type === 'critical' && (
                              <Badge variant="destructive" className="text-xs px-1.5 py-0">
                                CRITICAL
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>

    </div>
  );
}
