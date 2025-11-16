'use client';

/**
 * Vapi Voice Chat Component
 * Simple voice interface using Vapi AI
 */

import { useEffect, useState, useRef } from 'react';
import Vapi from '@vapi-ai/web';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Mic, MicOff, Loader2, Bot, Phone, PhoneOff } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useCogniraceStore } from '@/lib/store';
import { debugLog } from './DebugLayer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function VapiVoiceChat() {
  const [vapi, setVapi] = useState<Vapi | null>(null);
  const [isCallActive, setIsCallActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [connectionState, setConnectionState] = useState<string>('disconnected');
  const [volumeLevel, setVolumeLevel] = useState(0);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { raceContext } = useCogniraceStore();
  
  // Initialize Vapi
  useEffect(() => {
    debugLog.info('VAPI_INIT', 'Initializing Vapi...');
    
    const publicKey = process.env.NEXT_PUBLIC_VAPI_PUBLIC_KEY;
    
    if (!publicKey) {
      debugLog.error('VAPI_INIT', 'Vapi public key not found in environment!');
      setConnectionState('error');
      return;
    }
    
    debugLog.success('VAPI_INIT', `Public key found: ${publicKey.substring(0, 8)}...`);
    
    try {
      const vapiInstance = new Vapi(publicKey);
      setVapi(vapiInstance);
      debugLog.success('VAPI_INIT', 'Vapi instance created successfully');
      
      // Event listeners
      vapiInstance.on('call-start', () => {
        debugLog.success('VAPI_EVENT', 'Call started');
        setIsCallActive(true);
        setConnectionState('connected');
        // Add welcome message
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: "Hey! I'm Camry, your AI race strategist. I'm monitoring your race. How can I help you?",
          timestamp: new Date()
        }]);
      });
      
      vapiInstance.on('call-end', () => {
        debugLog.info('VAPI_EVENT', 'Call ended');
        setIsCallActive(false);
        setConnectionState('disconnected');
        setIsSpeaking(false);
      });
      
      vapiInstance.on('speech-start', () => {
        debugLog.info('VAPI_EVENT', 'Assistant started speaking');
        setIsSpeaking(true);
      });
      
      vapiInstance.on('speech-end', () => {
        debugLog.info('VAPI_EVENT', 'Assistant stopped speaking');
        setIsSpeaking(false);
      });
      
      vapiInstance.on('message', (message: any) => {
        debugLog.info('VAPI_MESSAGE', 'Message received', { type: message.type, role: message.role });
        
        // Handle transcripts
        if (message.type === 'transcript' && message.transcriptType === 'final') {
          if (message.role === 'user') {
            debugLog.success('TRANSCRIPT', `User: ${message.transcript}`);
            setMessages(prev => [...prev, {
              role: 'user',
              content: message.transcript,
              timestamp: new Date()
            }]);
          } else if (message.role === 'assistant') {
            debugLog.success('TRANSCRIPT', `Assistant: ${message.transcript}`);
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: message.transcript,
              timestamp: new Date()
            }]);
          }
        }
      });
      
      vapiInstance.on('volume-level', (level: number) => {
        setVolumeLevel(level);
      });
      
      vapiInstance.on('error', (error: any) => {
        debugLog.error('VAPI_ERROR', 'Vapi error occurred', {
          message: error.message,
          type: error.type,
          details: error
        });
        setConnectionState('error');
      });
      
      debugLog.success('VAPI_INIT', 'All event listeners registered');
      
    } catch (error: any) {
      debugLog.error('VAPI_INIT', 'Failed to initialize Vapi', {
        message: error.message,
        stack: error.stack
      });
      setConnectionState('error');
    }
    
    return () => {
      if (vapi) {
        debugLog.info('VAPI_CLEANUP', 'Cleaning up Vapi instance');
        vapi.stop();
      }
    };
  }, []);
  
  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const startCall = async () => {
    debugLog.info('CALL_START', 'Start call button clicked');
    
    if (!vapi) {
      debugLog.error('CALL_START', 'Vapi instance not initialized!');
      return;
    }
    
    try {
      setConnectionState('connecting');
      debugLog.info('CALL_START', 'Connection state set to connecting');
      
      const assistantId = process.env.NEXT_PUBLIC_VAPI_ASSISTANT_ID || 'camry';
      debugLog.info('CALL_START', `Using assistant ID: ${assistantId}`);
      
      debugLog.info('CALL_START', 'Calling vapi.start()...');
      await vapi.start(assistantId);
      debugLog.success('CALL_START', 'vapi.start() completed successfully');
      
    } catch (error: any) {
      debugLog.error('CALL_START', 'Failed to start call', {
        message: error.message,
        name: error.name,
        stack: error.stack,
        details: error
      });
      setConnectionState('error');
    }
  };
  
  const endCall = () => {
    debugLog.info('CALL_END', 'End call button clicked');
    if (vapi) {
      vapi.stop();
      debugLog.success('CALL_END', 'vapi.stop() called');
    } else {
      debugLog.warn('CALL_END', 'Vapi instance not available');
    }
  };
  
  const getStatusBadge = () => {
    if (connectionState === 'connected') {
      return (
        <Badge variant="default" className="animate-pulse">
          <div className="w-2 h-2 rounded-full bg-green-500 mr-1" />
          Connected
        </Badge>
      );
    } else if (connectionState === 'connecting') {
      return (
        <Badge variant="secondary">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          Connecting...
        </Badge>
      );
    } else if (connectionState === 'error') {
      return <Badge variant="destructive">Error</Badge>;
    }
    return <Badge variant="outline">Not Connected</Badge>;
  };
  
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border bg-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
              <Bot className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold">Camry - AI Race Strategist</h2>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {getStatusBadge()}
                {isSpeaking && (
                  <Badge variant="secondary" className="text-xs">
                    <Mic className="w-3 h-3 mr-1" />
                    Speaking
                  </Badge>
                )}
              </div>
            </div>
          </div>
          <Badge variant="outline" className="gap-1">
            <span className="text-xl">🏁</span>
            {raceContext.track}
          </Badge>
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea className="flex-1 px-4">
        <div className="py-4 space-y-4 max-w-3xl mx-auto">
          {messages.length === 0 && !isCallActive && (
            <Card className="border-dashed">
              <CardContent className="pt-6 text-center space-y-3">
                <div className="text-5xl">🎙️</div>
                <p className="text-muted-foreground text-sm">
                  Click the phone icon below to start talking with Camry
                </p>
                <p className="text-xs text-muted-foreground">
                  Powered by Vapi AI
                </p>
              </CardContent>
            </Card>
          )}

          {messages.map((message, idx) => (
            <div
              key={idx}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <Card
                className={cn(
                  'max-w-[80%]',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                <CardContent className="pt-4 pb-3 px-4">
                  <div className="flex items-start gap-2">
                    {message.role === 'assistant' && (
                      <Bot className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="text-sm leading-relaxed">
                        {message.content}
                      </p>
                      <p className={cn(
                        "text-xs mt-1",
                        message.role === 'user' 
                          ? 'text-primary-foreground/70' 
                          : 'text-muted-foreground'
                      )}>
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
          
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Call Controls */}
      <div className="border-t border-border bg-card p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center justify-center space-x-4">
            {!isCallActive ? (
              <Button
                onClick={startCall}
                disabled={connectionState === 'connecting'}
                className="rounded-full w-20 h-20 shadow-lg"
                size="lg"
              >
                {connectionState === 'connecting' ? (
                  <Loader2 className="h-8 w-8 animate-spin" />
                ) : (
                  <Phone className="h-8 w-8" />
                )}
              </Button>
            ) : (
              <Button
                onClick={endCall}
                variant="destructive"
                className="rounded-full w-20 h-20 shadow-lg"
                size="lg"
              >
                <PhoneOff className="h-8 w-8" />
              </Button>
            )}
          </div>

          {/* Volume Level Indicator (when speaking) */}
          {isCallActive && volumeLevel > 0 && (
            <div className="mt-4">
              <div className="flex items-center justify-center gap-1">
                {[...Array(10)].map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      "w-1 h-8 rounded-full transition-all",
                      i < Math.floor(volumeLevel * 10)
                        ? "bg-primary"
                        : "bg-muted"
                    )}
                    style={{
                      height: `${(i + 1) * 8}px`
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Status Text */}
          <p className="text-center text-xs text-muted-foreground mt-3">
            {!isCallActive ? (
              "Tap phone icon to start voice call"
            ) : isSpeaking ? (
              <span className="text-primary font-medium">Camry is speaking...</span>
            ) : (
              "Listening... speak now"
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

