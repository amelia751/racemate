'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bot, Activity, AlertTriangle } from 'lucide-react';
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
  const [currentStatus, setCurrentStatus] = useState('Waiting for data...');
  const { isStreaming, addDebugLog } = useCogniraceStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const checkIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const addRecommendation = useCallback((
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
    
    console.log(`[VoiceStrategist] 💬 Adding ${type.toUpperCase()} to chat (${events?.length || 0} events)`);
    
    setRecommendations(prev => {
      const updated = [...prev, rec].slice(-20); // Keep last 20
      
      // Log to debug panel only for critical
      if (type === 'critical') {
        setTimeout(() => {
          addDebugLog('error', `🚨 CRITICAL: ${events?.length || 0} events`, {
            total: updated.length
          });
        }, 0);
      }
      
      return updated;
    });
  }, [addDebugLog]);

  // Poll for recommendations from StreamingControls via window object
  useEffect(() => {
    if (!isStreaming) {
      setCurrentStatus('Waiting for data...');
      setTimeout(() => {
        addDebugLog('info', '⏸️ VoiceStrategist: Not streaming', {});
      }, 0);
      return;
    }
    
    setCurrentStatus('Monitoring telemetry...');
    setTimeout(() => {
      addDebugLog('info', '▶️ VoiceStrategist: Started polling', {
        interval: '500ms'
      });
    }, 0);
    
    let pollCount = 0;
    
    const interval = setInterval(() => {
      pollCount++;
      
      // Check if StreamingControls has a new recommendation
      const hasRecommendation = typeof window !== 'undefined' && (window as any).__latestRecommendation;
      
      if (hasRecommendation) {
        const data = (window as any).__latestRecommendation;
        
        // FILTER: Only show significant recommendations
        const criticalCount = data.severity_summary?.critical || 0;
        const highCount = data.severity_summary?.high || 0;
        
        // Skip if:
        // 1. OPTIMAL scenario with only 1 high event (just fuel consumption spike)
        // 2. Less than 2 high-severity events
        const isOptimalWithMinorEvent = data.scenario === 'OPTIMAL' && highCount === 1;
        const hasSignificantEvents = criticalCount > 0 || highCount >= 2;
        
        if (!hasSignificantEvents || isOptimalWithMinorEvent) {
          console.log(`[VoiceStrategist] Skipping non-critical recommendation (${data.scenario}, C:${criticalCount}, H:${highCount})`);
          delete (window as any).__latestRecommendation;
          return;
        }
        
        console.log(`[VoiceStrategist] ✅ SIGNIFICANT EVENT: ${data.scenario} (C:${criticalCount}, H:${highCount})`);
        
        setTimeout(() => {
          addDebugLog('success', `🎯 SIGNIFICANT: ${data.scenario}`, {
            critical: criticalCount,
            high: highCount,
            events: data.events?.length || 0
          });
        }, 0);
        
        // Determine type based on severity
        let type: 'info' | 'warning' | 'critical' = 'info';
        if (criticalCount > 0) {
          type = 'critical';
        } else if (highCount > 0) {
          type = 'warning';
        }
        
        addRecommendation(type, data.strategy, data.events);
        setCurrentStatus(`${data.scenario} - Action required!`);
        
        // Clear the recommendation so we don't show it again
        delete (window as any).__latestRecommendation;
        
        setTimeout(() => {
          addDebugLog('info', '✓ Recommendation added to chat', {});
        }, 0);
      } else {
        // Log every 20 polls to show we're still alive (every 10 seconds)
        if (pollCount % 20 === 0) {
          setTimeout(() => {
            addDebugLog('info', `🔄 Polling... (${pollCount} checks)`, {
              status: 'Waiting for events'
            });
          }, 0);
        }
      }
    }, 500); // Check every 500ms
    
    checkIntervalRef.current = interval;
    
    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
        setTimeout(() => {
          addDebugLog('info', '⏹️ VoiceStrategist: Stopped polling', {});
        }, 0);
      }
    };
  }, [isStreaming, addRecommendation, addDebugLog]);

  // Scroll to bottom on new recommendations
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [recommendations]);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'critical': return 'border-red-500/30 bg-red-500/5';
      case 'warning': return 'border-orange-500/30 bg-orange-500/5';
      default: return 'border-cyan-500/30 bg-cyan-500/5';
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-gray-900 via-black to-black border border-cyan-500/20 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2 bg-gradient-to-r from-cyan-900/30 to-purple-900/30 border-b border-cyan-500/20 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <div>
              <h2 className="text-sm font-bold text-cyan-400">AI RACE STRATEGIST</h2>
              <p className="text-xs text-muted-foreground">{currentStatus}</p>
            </div>
          </div>
          {isStreaming ? (
            <Badge variant="default" className="bg-green-500 animate-pulse text-xs">
              <Activity className="w-3 h-3 mr-1" />
              Monitoring
            </Badge>
          ) : (
            <Badge variant="outline" className="text-xs">
              Standby
            </Badge>
          )}
        </div>
      </div>

      {/* Recommendations Feed - Scrollable */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden px-3 py-3 space-y-2" style={{ minHeight: 0 }}>
          {recommendations.length === 0 ? (
            <Card className="border-dashed bg-black/20">
              <CardContent className="pt-4 text-center">
                <Bot className="w-8 h-8 mx-auto mb-2 text-cyan-400 opacity-50" />
                <p className="text-xs text-muted-foreground">
                  {isStreaming
                    ? "🎯 Monitoring - AI will alert on critical events"
                    : "⚠️ Click START STREAMING to begin"}
                </p>
              </CardContent>
            </Card>
          ) : (
            <AnimatePresence>
              {recommendations.map((rec, index) => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Card className={`${getTypeColor(rec.type)} border`}>
                    <CardContent className="p-2">
                      <div className="flex items-start gap-2">
                        <div className="flex-shrink-0 mt-0.5">
                          {rec.type === 'critical' ? (
                            <AlertTriangle className="w-3 h-3 text-red-400" />
                          ) : (
                            <Bot className="w-3 h-3 text-cyan-400" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <Badge 
                              variant={rec.type === 'critical' ? 'destructive' : 'default'}
                              className="text-[10px] px-1 py-0"
                            >
                              {rec.type.toUpperCase()}
                            </Badge>
                            <span className="text-[10px] text-muted-foreground">
                              {rec.timestamp.toLocaleTimeString()}
                            </span>
                          </div>
                          <p className="text-xs text-white leading-snug whitespace-pre-wrap">
                            {rec.message}
                          </p>
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
    </div>
  );
}
