'use client';

/**
 * Enhanced Debug Layer Component
 * Comprehensive debugging tools with Real-Time monitoring
 */

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Copy, Trash2, X, Maximize2, Minimize2, Activity, Wifi, Server, Zap, CheckCircle, Info } from 'lucide-react';
import { motion } from 'framer-motion';
import { useRaceMateStore } from '@/lib/store';

interface LogEntry {
  timestamp: Date;
  level: 'info' | 'warn' | 'error' | 'success';
  category: string;
  message: string;
  data?: any;
}

let globalLogs: LogEntry[] = [];
let logListeners: Array<(logs: LogEntry[]) => void> = [];

export const debugLog = {
  info: (category: string, message: string, data?: any) => {
    addLog('info', category, message, data);
    console.log(`[${category}]`, message, data || '');
  },
  warn: (category: string, message: string, data?: any) => {
    addLog('warn', category, message, data);
    console.warn(`[${category}]`, message, data || '');
  },
  error: (category: string, message: string, data?: any) => {
    addLog('error', category, message, data);
    console.error(`[${category}]`, message, data || '');
  },
  success: (category: string, message: string, data?: any) => {
    addLog('success', category, message, data);
    console.log(`✅ [${category}]`, message, data || '');
  },
  clearAll: () => {
    globalLogs = [];
    logListeners.forEach(listener => listener([]));
  }
};

// Make clearAll available globally for StreamingControls
if (typeof window !== 'undefined') {
  (window as any).__clearDebugLogs = debugLog.clearAll;
}

function addLog(level: LogEntry['level'], category: string, message: string, data?: any) {
  const entry: LogEntry = {
    timestamp: new Date(),
    level,
    category,
    message,
    data
  };
  
  globalLogs.push(entry);
  
  // Keep last 200 logs
  if (globalLogs.length > 200) {
    globalLogs = globalLogs.slice(-200);
  }
  
  // Notify listeners
  logListeners.forEach(listener => listener([...globalLogs]));
}

export default function DebugLayer() {
  const [logs, setLogs] = useState<LogEntry[]>(globalLogs);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState<'all' | LogEntry['level']>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [systemInfo, setSystemInfo] = useState<any>({});
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [wsConnected, setWsConnected] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { telemetryData, debugLogs: storeDebugLogs } = useRaceMateStore();

  // Subscribe to log updates
  useEffect(() => {
    const listener = (newLogs: LogEntry[]) => setLogs(newLogs);
    logListeners.push(listener);
    
    return () => {
      logListeners = logListeners.filter(l => l !== listener);
    };
  }, []);

  // Sync store debugLogs to local state
  useEffect(() => {
    if (storeDebugLogs && storeDebugLogs.length > 0) {
      // Parse store logs and add to global logs
      const parsedLogs: LogEntry[] = storeDebugLogs.map((logStr, idx) => {
        // Parse format: [timestamp] [LEVEL] message
        const match = logStr.match(/\[(.*?)\] \[(.*?)\] ([\s\S]+)/);
        if (match) {
          return {
            timestamp: new Date(match[1]),
            level: match[2].toLowerCase() as LogEntry['level'],
            category: 'STREAM',
            message: match[3].split('\n')[0],
            data: match[3].includes('\n') ? match[3].split('\n').slice(1).join('\n') : undefined
          };
        }
        return {
          timestamp: new Date(),
          level: 'info' as const,
          category: 'STREAM',
          message: logStr,
          data: undefined
        };
      });
      
      // Merge with global logs
      setLogs([...globalLogs, ...parsedLogs]);
    }
  }, [storeDebugLogs]);

  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8005'}/health`);
        if (response.ok) {
          const wasOffline = backendStatus === 'offline';
          setBackendStatus('online');
          if (wasOffline) {
            debugLog.success('BACKEND', 'Backend API reconnected');
          }
        } else {
          setBackendStatus('offline');
        }
      } catch (error) {
        const wasOnline = backendStatus === 'online';
        setBackendStatus('offline');
        if (wasOnline) {
          debugLog.error('BACKEND', 'Backend API connection lost');
        }
      }
    };
    
    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10s
    
    return () => clearInterval(interval);
  }, [backendStatus]);

  // Monitor WebSocket connection
  useEffect(() => {
    const checkWs = () => {
      const wsState = (window as any).__ws_connected;
      setWsConnected(!!wsState);
    };
    
    const interval = setInterval(checkWs, 2000);
    return () => clearInterval(interval);
  }, []);

  // Get system info
  useEffect(() => {
    const info = {
      userAgent: navigator.userAgent,
      language: navigator.language,
      online: navigator.onLine,
      cookieEnabled: navigator.cookieEnabled,
      platform: navigator.platform,
      screenSize: `${window.screen.width}x${window.screen.height}`,
      windowSize: `${window.innerWidth}x${window.innerHeight}`,
      googleApiKey: process.env.NEXT_PUBLIC_GOOGLE_API_KEY?.substring(0, 8) + '...',
      geminiModel: process.env.NEXT_PUBLIC_GEMINI_MODEL || 'gemini-2.5-pro',
      backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL
    };
    setSystemInfo(info);
    
    console.log('🐛 Debug layer initialized', info);
  }, []);

  const filteredLogs = logs.filter(log => {
    const levelMatch = selectedLevel === 'all' || log.level === selectedLevel;
    const searchMatch = searchTerm === '' || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.category.toLowerCase().includes(searchTerm.toLowerCase());
    return levelMatch && searchMatch;
  });

  const copyLogs = () => {
    // Copy ALL logs, not just filtered ones
    const logsText = logs.map(log => 
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.category}] ${log.message}${log.data ? '\n' + JSON.stringify(log.data, null, 2) : ''}`
    ).join('\n\n');
    navigator.clipboard.writeText(logsText);
    debugLog.success('CLIPBOARD', `Copied entire log history (${logs.length} entries)`);
  };

  const clearLogs = () => {
    globalLogs = [];
    setLogs([]);
    console.log('🧹 Debug logs cleared manually');
  };

  const testBackend = async () => {
    debugLog.info('TEST', 'Testing backend connection...');
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8005'}/health`);
      const data = await response.json();
      debugLog.success('TEST', 'Backend connection successful', data);
    } catch (error: any) {
      debugLog.error('TEST', 'Backend connection failed', {
        message: error.message
      });
    }
  };

  const getLevelBadge = (level: LogEntry['level']) => {
    const configs = {
      info: { variant: 'secondary' as const, icon: 'ℹ️' },
      success: { variant: 'default' as const, icon: '✅' },
      warn: { variant: 'outline' as const, icon: '⚠️' },
      error: { variant: 'destructive' as const, icon: '❌' }
    };
    const config = configs[level];
    return <Badge variant={config.variant}>{config.icon}</Badge>;
  };

  const logCounts = {
    all: logs.length,
    info: logs.filter(l => l.level === 'info').length,
    success: logs.filter(l => l.level === 'success').length,
    warn: logs.filter(l => l.level === 'warn').length,
    error: logs.filter(l => l.level === 'error').length
  };

  if (isMinimized) {
    return (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        className="fixed bottom-6 right-6 z-[9999]"
      >
        <Button
          onClick={() => setIsMinimized(false)}
          className="rounded-full shadow-2xl w-16 h-16 bg-gradient-to-br from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500"
          size="lg"
        >
          <div className="text-center">
            <div className="text-2xl">🐛</div>
            <div className="text-xs">{logs.length}</div>
          </div>
        </Button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`fixed ${isExpanded ? 'inset-8' : 'bottom-6 right-6 w-[700px] h-[500px]'} z-[9999] transition-all`}
    >
      <Card className="h-full flex flex-col shadow-2xl border-2 border-cyan-500 bg-black/95 backdrop-blur-xl">
        <CardHeader className="pb-3 bg-gradient-to-r from-cyan-900/50 to-purple-900/50 border-b border-cyan-500/30">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center text-2xl">
                🐛
              </div>
              <div>
                <div className="text-lg font-black tracking-tight text-cyan-400">DEBUG CONSOLE</div>
                <div className="text-xs text-muted-foreground">
                  {filteredLogs.length} / {logs.length} events
                </div>
              </div>
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                onClick={() => setIsExpanded(!isExpanded)}
                variant="ghost"
                size="sm"
                className="hover:bg-cyan-500/20"
              >
                {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </Button>
              <Button
                onClick={() => setIsMinimized(true)}
                variant="ghost"
                size="sm"
                className="hover:bg-red-500/20"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col space-y-3 p-4 overflow-hidden">
          <Tabs defaultValue="logs" className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-4 bg-black/40">
              <TabsTrigger value="logs" className="text-xs">
                <Info className="w-3 h-3 mr-1" />
                Logs ({logCounts.all})
              </TabsTrigger>
              <TabsTrigger value="realtime" className="text-xs">
                <Activity className="w-3 h-3 mr-1" />
                Real-Time
              </TabsTrigger>
              <TabsTrigger value="system" className="text-xs">
                <Server className="w-3 h-3 mr-1" />
                System
              </TabsTrigger>
              <TabsTrigger value="tests" className="text-xs">
                <Zap className="w-3 h-3 mr-1" />
                Tests
              </TabsTrigger>
            </TabsList>

            {/* Real-Time Tab */}
            <TabsContent value="realtime" className="flex-1 overflow-hidden">
              <ScrollArea className="h-full">
                <div className="space-y-3">
                  <Card className="bg-black/40 border-cyan-500/30">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Activity className="w-4 h-4 text-cyan-400" />
                        System Status
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {/* Connection Status */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="text-xs text-gray-400">Backend API</div>
                          <div className="flex items-center gap-2">
                            {backendStatus === 'online' ? (
                              <>
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                                <span className="text-xs text-green-400 font-mono">ONLINE</span>
                              </>
                            ) : backendStatus === 'offline' ? (
                              <>
                                <div className="w-2 h-2 rounded-full bg-red-500" />
                                <span className="text-xs text-red-400 font-mono">OFFLINE</span>
                              </>
                            ) : (
                              <>
                                <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                                <span className="text-xs text-yellow-400 font-mono">CHECKING</span>
                              </>
                            )}
                          </div>
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs text-gray-400">WebSocket</div>
                          <div className="flex items-center gap-2">
                            {wsConnected ? (
                              <>
                                <Wifi className="w-3 h-3 text-green-400" />
                                <span className="text-xs text-green-400 font-mono">CONNECTED</span>
                              </>
                            ) : (
                              <>
                                <Wifi className="w-3 h-3 text-red-400" />
                                <span className="text-xs text-red-400 font-mono">DISCONNECTED</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Current Telemetry */}
                      <div className="pt-3 border-t border-gray-700">
                        <div className="text-xs text-gray-400 mb-2">Current Telemetry</div>
                        <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                          <div>
                            <span className="text-gray-500">Speed:</span>
                            <span className="text-cyan-400 ml-2">{Math.round(telemetryData?.speed || 0)} km/h</span>
                          </div>
                          <div>
                            <span className="text-gray-500">RPM:</span>
                            <span className="text-cyan-400 ml-2">{Math.round(telemetryData?.rpm || telemetryData?.nmot || 0)}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Gear:</span>
                            <span className="text-cyan-400 ml-2">{telemetryData?.gear || 0}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Fuel:</span>
                            <span className="text-cyan-400 ml-2">{(telemetryData?.fuel_level || 0).toFixed(1)}L</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Throttle:</span>
                            <span className="text-cyan-400 ml-2">{Math.round(telemetryData?.throttle || telemetryData?.aps || 0)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Lap:</span>
                            <span className="text-cyan-400 ml-2">{telemetryData?.lap || 1}</span>
                          </div>
                        </div>
                      </div>

                      {/* ML Models Status */}
                      <div className="pt-3 border-t border-gray-700">
                        <div className="text-xs text-gray-400 mb-2">ML Models (8 total)</div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {[
                            'Fuel Consumption',
                            'Tire Degradation',
                            'Anomaly Detector',
                            'FCY Hazard',
                            'Lap-Time Transformer',
                            'Pit Loss',
                            'Driver Embedding',
                            'Traffic GNN'
                          ].map((model, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <CheckCircle className="w-3 h-3 text-green-400" />
                              <span className="text-gray-300 font-mono">{model}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Rate Limiting */}
                      <div className="pt-3 border-t border-gray-700">
                        <div className="text-xs text-gray-400 mb-2">AI Rate Limiting</div>
                        <div className="space-y-1 text-xs font-mono">
                          <div className="flex justify-between">
                            <span className="text-gray-500">Gemini Model:</span>
                            <span className="text-cyan-400">{process.env.NEXT_PUBLIC_GEMINI_MODEL || 'gemini-2.5-pro'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Call Interval:</span>
                            <span className="text-cyan-400">10 seconds</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Max Calls/min:</span>
                            <span className="text-cyan-400">6</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </TabsContent>

            {/* Logs Tab */}
            <TabsContent value="logs" className="flex-1 flex flex-col space-y-2 overflow-hidden">
              {/* Controls */}
              <div className="flex gap-2 flex-wrap items-center">
                <select
                  value={selectedLevel}
                  onChange={(e) => setSelectedLevel(e.target.value as any)}
                  className="px-2 py-1 text-xs bg-background border border-input rounded"
                >
                  <option value="all">All ({logCounts.all})</option>
                  <option value="info">Info ({logCounts.info})</option>
                  <option value="success">Success ({logCounts.success})</option>
                  <option value="warn">Warn ({logCounts.warn})</option>
                  <option value="error">Error ({logCounts.error})</option>
                </select>
                <input
                  type="text"
                  placeholder="Search logs..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="flex-1 px-2 py-1 text-xs bg-background border border-input rounded min-w-[120px]"
                />
                <Button onClick={copyLogs} size="sm" variant="default" className="text-xs bg-cyan-600 hover:bg-cyan-500">
                  <Copy className="h-3 w-3 mr-1" />
                  Copy All ({logs.length})
                </Button>
                <Button onClick={clearLogs} size="sm" variant="outline" className="text-xs">
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>
              
              {/* Helpful Note */}
              <div className="text-xs text-muted-foreground bg-cyan-500/5 border border-cyan-500/20 rounded px-3 py-2">
                💡 <strong>Tip:</strong> Logs auto-clear when you click "START STREAMING" to focus on current session
              </div>

              {/* Logs List */}
              <ScrollArea className="flex-1" ref={scrollRef}>
                <div className="space-y-2 pr-4">
                  {filteredLogs.length === 0 ? (
                    <div className="text-center text-sm text-muted-foreground py-8">
                      No logs match your filters
                    </div>
                  ) : (
                    filteredLogs.map((log, i) => (
                      <Card key={i} className="bg-black/40 border-gray-700">
                        <CardContent className="p-3">
                          <div className="flex items-start gap-2">
                            {getLevelBadge(log.level)}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge variant="outline" className="text-xs">
                                  {log.category}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {log.timestamp.toLocaleTimeString()}
                                </span>
                              </div>
                              <p className="text-sm mt-1 break-words">{log.message}</p>
                              {log.data && (
                                <pre className="text-xs mt-2 p-2 bg-black/60 rounded overflow-x-auto">
                                  {JSON.stringify(log.data, null, 2)}
                                </pre>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            {/* System Tab */}
            <TabsContent value="system" className="flex-1 overflow-hidden">
              <ScrollArea className="h-full">
                <Card className="bg-black/40 border-cyan-500/30">
                  <CardHeader>
                    <CardTitle className="text-sm">System Information</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <dl className="space-y-2 text-xs">
                      {Object.entries(systemInfo).map(([key, value]) => (
                        <div key={key} className="flex justify-between py-1 border-b border-gray-800">
                          <dt className="text-muted-foreground font-mono">{key}:</dt>
                          <dd className="text-foreground font-mono break-all max-w-[60%] text-right">
                            {String(value)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </CardContent>
                </Card>
              </ScrollArea>
            </TabsContent>

            {/* Tests Tab */}
            <TabsContent value="tests" className="flex-1 overflow-hidden">
              <ScrollArea className="h-full">
                <div className="space-y-2">
                  <Card className="bg-black/40 border-cyan-500/30">
                    <CardHeader>
                      <CardTitle className="text-sm">Diagnostic Tests</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <Button onClick={testBackend} className="w-full" size="sm">
                        Test Backend Connection
                      </Button>
                      <p className="text-xs text-muted-foreground">
                        Check logs tab for test results
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  );
}
