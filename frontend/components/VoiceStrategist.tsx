'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bot, Activity, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRaceMateStore } from '@/lib/store';

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
  const { isStreaming, addDebugLog } = useRaceMateStore();
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
    <div className="flex flex-col h-full bg-gradient-to-b from-gray-900 via-black to-black rounded-t-lg overflow-hidden">
      {/* Header */}
      <div className="px-3 py-1 bg-black/40 border-b border-cyan-500/10 flex-shrink-0">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: isStreaming ? '#22c55e' : '#6b7280' }} />
            <h2 className="text-[11px] font-semibold tracking-wide text-cyan-300">RACE STRATEGIST</h2>
          </div>
          {isStreaming && (
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Badge variant="default" className="bg-green-500/20 text-green-300 border border-green-500/30 text-[9px] px-2 py-0.5">
                🟢 LIVE
              </Badge>
            </motion.div>
          )}
        </div>
      </div>

      {/* Recommendations Feed - Scrollable */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-2 space-y-2" style={{ minHeight: 0 }}>
          {recommendations.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center mx-auto mb-3">
                  <Bot className="w-6 h-6 text-cyan-400/70" />
                </div>
                <p className="text-[11px] text-cyan-300/60 tracking-wide">
                  {isStreaming
                    ? "🎯 MONITORING FOR CRITICAL EVENTS"
                    : "START STREAMING TO BEGIN"}
                </p>
              </div>
            </div>
          ) : (
            <AnimatePresence>
              {recommendations.map((rec, index) => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2, delay: index * 0.02 }}
                >
                  <Card className={`${getTypeColor(rec.type)} border`}>
                    <CardContent className="p-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <Badge
                            className={`text-[9px] px-3 py-1 h-6 whitespace-nowrap ${
                              rec.type === 'critical' 
                                ? 'bg-red-500/20 text-red-300 border border-red-500/40' 
                                : rec.type === 'warning'
                                ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40'
                                : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                            }`}
                          >
                            {rec.type === 'critical' ? '🔴 CRITICAL' : rec.type === 'warning' ? '🟡 WARNING' : '🔵 INFO'}
                          </Badge>
                          <span className="text-[9px] text-muted-foreground/70 flex-shrink-0">
                            {rec.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-200 leading-tight whitespace-pre-wrap break-words">
                          {rec.message}
                        </p>
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
