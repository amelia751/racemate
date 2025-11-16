'use client';

/**
 * Debug Panel Component
 * Comprehensive system logging and debugging interface
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

type LogLevel = 'info' | 'success' | 'warning' | 'error';

interface LogEntry {
  timestamp: Date;
  level: LogLevel;
  category: string;
  message: string;
  data?: any;
}

let logStore: LogEntry[] = [];
let listeners: Array<(logs: LogEntry[]) => void> = [];

export const debugLogger = {
  info: (category: string, message: string, data?: any) => {
    addLog('info', category, message, data);
  },
  success: (category: string, message: string, data?: any) => {
    addLog('success', category, message, data);
  },
  warning: (category: string, message: string, data?: any) => {
    addLog('warning', category, message, data);
  },
  error: (category: string, message: string, data?: any) => {
    addLog('error', category, message, data);
  },
};

function addLog(level: LogLevel, category: string, message: string, data?: any) {
  const entry: LogEntry = {
    timestamp: new Date(),
    level,
    category,
    message,
    data,
  };
  
  logStore.push(entry);
  
  // Keep only last 500 logs
  if (logStore.length > 500) {
    logStore = logStore.slice(-500);
  }
  
  // Notify all listeners
  listeners.forEach(listener => listener([...logStore]));
  
  // Also log to console
  const prefix = `[${category}]`;
  switch (level) {
    case 'info':
      console.log(prefix, message, data || '');
      break;
    case 'success':
      console.log(`✅ ${prefix}`, message, data || '');
      break;
    case 'warning':
      console.warn(`⚠️ ${prefix}`, message, data || '');
      break;
    case 'error':
      console.error(`❌ ${prefix}`, message, data || '');
      break;
  }
}

export default function DebugPanel() {
  const [logs, setLogs] = useState<LogEntry[]>(logStore);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLevel, setSelectedLevel] = useState<LogLevel | 'all'>('all');

  useEffect(() => {
    // Subscribe to log updates
    const listener = (newLogs: LogEntry[]) => setLogs(newLogs);
    listeners.push(listener);
    
    return () => {
      listeners = listeners.filter(l => l !== listener);
    };
  }, []);

  const clearLogs = () => {
    logStore = [];
    setLogs([]);
    debugLogger.info('SYSTEM', 'Debug logs cleared');
  };

  const exportLogs = () => {
    const logsText = logs.map(log => 
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.category}] ${log.message}${log.data ? '\n' + JSON.stringify(log.data, null, 2) : ''}`
    ).join('\n\n');
    
    const blob = new Blob([logsText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cognirace-logs-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    debugLogger.success('SYSTEM', 'Logs exported successfully');
  };

  // Filter logs
  const filteredLogs = logs.filter(log => {
    const matchesLevel = selectedLevel === 'all' || log.level === selectedLevel;
    const matchesSearch = !searchTerm || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.category.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  const getLevelBadge = (level: LogLevel) => {
    const variants = {
      info: { variant: 'secondary' as const, icon: 'ℹ️', label: 'INFO' },
      success: { variant: 'default' as const, icon: '✅', label: 'SUCCESS' },
      warning: { variant: 'outline' as const, icon: '⚠️', label: 'WARNING' },
      error: { variant: 'destructive' as const, icon: '❌', label: 'ERROR' },
    };
    
    const config = variants[level];
    return (
      <Badge variant={config.variant} className="text-xs">
        {config.icon} {config.label}
      </Badge>
    );
  };

  const logCounts = {
    all: logs.length,
    info: logs.filter(l => l.level === 'info').length,
    success: logs.filter(l => l.level === 'success').length,
    warning: logs.filter(l => l.level === 'warning').length,
    error: logs.filter(l => l.level === 'error').length,
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <span className="text-2xl">🐛</span>
            Debug Console
          </span>
          <Badge variant="outline">{filteredLogs.length} / {logs.length}</Badge>
        </CardTitle>
        <CardDescription>
          System logs and real-time debugging information
        </CardDescription>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col space-y-3 pb-4">
        {/* Controls */}
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="🔍 Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 px-3 py-2 text-sm rounded-md bg-background border border-input focus:outline-none focus:ring-2 focus:ring-ring"
          />
          <Button onClick={exportLogs} variant="secondary" size="sm">
            💾 Export
          </Button>
          <Button onClick={clearLogs} variant="destructive" size="sm">
            🗑️ Clear
          </Button>
        </div>

        {/* Filter Tabs */}
        <Tabs value={selectedLevel} onValueChange={(v) => setSelectedLevel(v as any)} className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="all" className="text-xs">
              All ({logCounts.all})
            </TabsTrigger>
            <TabsTrigger value="info" className="text-xs">
              Info ({logCounts.info})
            </TabsTrigger>
            <TabsTrigger value="success" className="text-xs">
              ✅ ({logCounts.success})
            </TabsTrigger>
            <TabsTrigger value="warning" className="text-xs">
              ⚠️ ({logCounts.warning})
            </TabsTrigger>
            <TabsTrigger value="error" className="text-xs">
              ❌ ({logCounts.error})
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Log Display */}
        <ScrollArea className="flex-1 rounded-md border">
          <div className="p-3 space-y-2 font-mono text-xs">
            {filteredLogs.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="pt-6">
                  <div className="text-center space-y-2">
                    <div className="text-3xl">📋</div>
                    <p className="text-sm text-muted-foreground">
                      {searchTerm ? 'No logs match your search' : 'No logs yet'}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              filteredLogs.map((log, idx) => (
                <Card key={idx} className={`
                  ${log.level === 'error' ? 'border-destructive bg-destructive/5' : ''}
                  ${log.level === 'warning' ? 'border-yellow-500/30 bg-yellow-500/5' : ''}
                  ${log.level === 'success' ? 'border-green-500/30 bg-green-500/5' : ''}
                  ${log.level === 'info' ? 'border-blue-500/30 bg-blue-500/5' : ''}
                `}>
                  <CardContent className="pt-3 pb-3">
                    <div className="space-y-2">
                      <div className="flex items-start gap-2 flex-wrap">
                        <span className="text-muted-foreground text-[10px]">
                          {log.timestamp.toLocaleTimeString()}.{log.timestamp.getMilliseconds()}
                        </span>
                        {getLevelBadge(log.level)}
                        <Badge variant="outline" className="text-[10px]">
                          {log.category}
                        </Badge>
                      </div>
                      <p className="text-xs">{log.message}</p>
                      {log.data && (
                        <>
                          <Separator />
                          <pre className="text-[10px] overflow-x-auto bg-muted p-2 rounded">
                            {JSON.stringify(log.data, null, 2)}
                          </pre>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer Stats */}
        <Card className="bg-muted/50">
          <CardContent className="pt-3 pb-3">
            <div className="flex justify-between items-center text-[10px] text-muted-foreground font-mono">
              <span>Total: {logs.length} | Filtered: {filteredLogs.length}</span>
              <span>Last: {logs.length > 0 ? logs[logs.length - 1].timestamp.toLocaleTimeString() : 'N/A'}</span>
            </div>
          </CardContent>
        </Card>
      </CardContent>
    </Card>
  );
}
