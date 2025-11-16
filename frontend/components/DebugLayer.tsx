'use client';

/**
 * Debug Layer Component
 * Comprehensive debugging tools for Vapi integration
 */

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Copy, Trash2, X, Maximize2, Minimize2 } from 'lucide-react';
import { motion } from 'framer-motion';

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
  }
};

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
  const scrollRef = useRef<HTMLDivElement>(null);

  // Subscribe to log updates
  useEffect(() => {
    const listener = (newLogs: LogEntry[]) => setLogs(newLogs);
    logListeners.push(listener);
    
    return () => {
      logListeners = logListeners.filter(l => l !== listener);
    };
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
      geminiModel: process.env.NEXT_PUBLIC_GEMINI_MODEL || 'gemini-2.0-flash-exp',
      backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL
    };
    setSystemInfo(info);
    debugLog.info('DEBUG_LAYER', 'Debug layer initialized', info);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current && !isMinimized) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isMinimized]);

  // Filter logs
  const filteredLogs = logs.filter(log => {
    const matchesLevel = selectedLevel === 'all' || log.level === selectedLevel;
    const matchesSearch = !searchTerm || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.category.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  const clearLogs = () => {
    globalLogs = [];
    setLogs([]);
    debugLog.info('DEBUG_LAYER', 'Logs cleared');
  };

  const copyLogs = () => {
    const logsText = logs.map(log => 
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.category}] ${log.message}${log.data ? '\n' + JSON.stringify(log.data, null, 2) : ''}`
    ).join('\n\n');
    
    navigator.clipboard.writeText(logsText);
    debugLog.success('DEBUG_LAYER', 'Logs copied to clipboard');
  };

  const testGeminiConnection = async () => {
    debugLog.info('GEMINI_TEST', 'Testing Gemini connection...');
    
    try {
      // Check if Google AI SDK loaded
      debugLog.info('GEMINI_TEST', 'Checking Google AI SDK...');
      const { GoogleGenerativeAI } = await import('@google/generative-ai');
      debugLog.success('GEMINI_TEST', 'Google AI SDK loaded successfully');
      
      // Check API key
      const apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
      if (!apiKey) {
        debugLog.error('GEMINI_TEST', 'GOOGLE API KEY IS MISSING!');
        return;
      }
      debugLog.success('GEMINI_TEST', `API key found: ${apiKey.substring(0, 8)}...`);
      
      // Try to create instance
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
      debugLog.success('GEMINI_TEST', 'Gemini model initialized successfully');
      
      debugLog.success('GEMINI_TEST', 'All Gemini checks passed!');
      
    } catch (error: any) {
      debugLog.error('GEMINI_TEST', 'Gemini connection test failed', {
        message: error.message,
        stack: error.stack?.substring(0, 200)
      });
    }
  };

  const testMicrophoneAccess = async () => {
    debugLog.info('MIC_TEST', 'Testing microphone access...');
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      debugLog.success('MIC_TEST', 'Microphone access granted!');
      
      // Stop the stream
      stream.getTracks().forEach(track => track.stop());
      
    } catch (error: any) {
      debugLog.error('MIC_TEST', 'Microphone access denied or failed', {
        message: error.message,
        name: error.name
      });
    }
  };

  const testBackendConnection = async () => {
    debugLog.info('BACKEND_TEST', 'Testing backend connection...');
    
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
    
    if (!backendUrl) {
      debugLog.error('BACKEND_TEST', 'Backend URL not configured!');
      return;
    }
    
    try {
      const response = await fetch(`${backendUrl}/health`);
      const data = await response.json();
      debugLog.success('BACKEND_TEST', 'Backend is healthy', data);
    } catch (error: any) {
      debugLog.error('BACKEND_TEST', 'Backend connection failed', {
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
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="logs">
                Logs ({logCounts.all})
              </TabsTrigger>
              <TabsTrigger value="system">System Info</TabsTrigger>
              <TabsTrigger value="tests">Tests</TabsTrigger>
            </TabsList>

            <TabsContent value="logs" className="flex-1 flex flex-col space-y-2 overflow-hidden">
              {/* Controls */}
              <div className="flex gap-2 flex-wrap">
                <input
                  type="text"
                  placeholder="🔍 Search logs..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="flex-1 min-w-[200px] px-3 py-1 text-sm rounded-md bg-background border border-input"
                />
                <Button onClick={copyLogs} variant="secondary" size="sm">
                  <Copy className="h-3 w-3 mr-1" />
                  Copy
                </Button>
                <Button onClick={clearLogs} variant="destructive" size="sm">
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>

              {/* Level Filter */}
              <div className="flex gap-1 flex-wrap">
                {(['all', 'info', 'success', 'warn', 'error'] as const).map(level => (
                  <Button
                    key={level}
                    onClick={() => setSelectedLevel(level)}
                    variant={selectedLevel === level ? "default" : "outline"}
                    size="sm"
                    className="text-xs"
                  >
                    {level === 'all' ? 'All' : level} ({logCounts[level]})
                  </Button>
                ))}
              </div>

              {/* Logs */}
              <div ref={scrollRef} className="flex-1 overflow-y-auto rounded-md border bg-background/50">
                <div className="p-2 space-y-1">
                  {filteredLogs.length === 0 ? (
                    <div className="text-center text-muted-foreground text-sm py-4">
                      No logs yet
                    </div>
                  ) : (
                    filteredLogs.map((log, idx) => (
                      <div key={idx} className="text-xs font-mono p-2 rounded border bg-card hover:bg-accent">
                        <div className="flex items-start gap-2">
                          <span className="text-muted-foreground text-[10px] whitespace-nowrap">
                            {log.timestamp.toLocaleTimeString()}.{log.timestamp.getMilliseconds()}
                          </span>
                          {getLevelBadge(log.level)}
                          <Badge variant="outline" className="text-[10px]">
                            {log.category}
                          </Badge>
                        </div>
                        <div className="mt-1">{log.message}</div>
                        {log.data && (
                          <pre className="mt-1 text-[10px] overflow-x-auto bg-muted p-1 rounded">
                            {JSON.stringify(log.data, null, 2)}
                          </pre>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="system" className="flex-1 overflow-auto">
              <div className="space-y-3">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Environment</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs font-mono">
                    {Object.entries(systemInfo).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-muted-foreground">{key}:</span>
                        <span className="font-semibold">{String(value)}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Browser APIs</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>getUserMedia:</span>
                      <Badge variant={navigator.mediaDevices ? "default" : "destructive"}>
                        {navigator.mediaDevices ? 'Available' : 'Not Available'}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Clipboard:</span>
                      <Badge variant={navigator.clipboard ? "default" : "destructive"}>
                        {navigator.clipboard ? 'Available' : 'Not Available'}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Online:</span>
                      <Badge variant={navigator.onLine ? "default" : "destructive"}>
                        {navigator.onLine ? 'Yes' : 'No'}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="tests" className="flex-1 overflow-auto">
              <div className="space-y-2">
                <Card>
                  <CardContent className="pt-4 space-y-2">
                    <Button onClick={testGeminiConnection} className="w-full" size="sm">
                      🔌 Test Gemini Connection
                    </Button>
                    <Button onClick={testMicrophoneAccess} className="w-full" size="sm">
                      🎤 Test Microphone Access
                    </Button>
                    <Button onClick={testBackendConnection} className="w-full" size="sm">
                      🖥️ Test Backend Connection
                    </Button>
                  </CardContent>
                </Card>

                <Card className="bg-primary/5">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground">
                      Click the test buttons above to diagnose issues. Results will appear in the Logs tab.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  );
}

