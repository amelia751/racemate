'use client';

/**
 * Voice Chat Interface Component
 * Auto-connects, agent greets first, voice-first interaction
 */

import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useCogniraceStore } from '@/lib/store';
import { useLiveKitContext } from '@/lib/livekit/LiveKitContext';
import { Mic, MicOff, Keyboard, ChevronDown, Loader2, Bot } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Message {
  sender: 'driver' | 'agent';
  content: string;
  timestamp: Date;
}

export default function VoiceChatInterface() {
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [textInput, setTextInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [hasGreeted, setHasGreeted] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  const { messages, addMessage, raceContext } = useCogniraceStore();
  const { isConnected, isAgentReady, isConnecting, connect, sendTextMessage, error: connectionError } = useLiveKitContext();

  // Auto-connect on mount
  useEffect(() => {
    if (!isConnected && !isConnecting) {
      const roomName = `cognirace-${Date.now()}`;
      const driverName = 'Driver-1';
      connect(roomName, driverName);
    }
  }, [isConnected, isConnecting, connect]);

  // Agent greets first when ready
  useEffect(() => {
    if (isAgentReady && !hasGreeted && messages.length === 0) {
      setTimeout(() => {
        addMessage({
          sender: 'agent',
          content: `Hello! I'm your AI race strategist. I'm monitoring your telemetry from ${raceContext.track}. Current conditions: ${raceContext.weather}. How can I help you today?`
        });
        setHasGreeted(true);
      }, 1000);
    }
  }, [isAgentReady, hasGreeted, messages.length, addMessage, raceContext]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle scroll detection
  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.target as HTMLDivElement;
    const isNearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 100;
    setShowScrollButton(!isNearBottom);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleMicClick = () => {
    if (!isAgentReady) return;
    setIsListening(!isListening);
    // Toggle microphone via LiveKit
    // In production, this would enable/disable audio track
  };

  const handleSendMessage = async () => {
    if (!textInput.trim() || !isAgentReady) return;

    const messageText = textInput;
    setTextInput('');

    try {
      // sendTextMessage will add the message to the store automatically
      await sendTextMessage(messageText);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Re-add to input on failure
      setTextInput(messageText);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
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
              <h2 className="font-semibold">AI Race Strategist</h2>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {!isAgentReady ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    <span>Connecting...</span>
                  </>
                ) : (
                  <>
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span>Online</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <Badge variant="outline" className="gap-1">
            <Mic className="w-3 h-3" />
            Voice Enabled
          </Badge>
        </div>
      </div>

      {/* Connection Loading Overlay */}
      {!isAgentReady && (
        <div className="absolute inset-0 bg-background/95 backdrop-blur-sm flex items-center justify-center z-50">
          <Card className="max-w-sm">
            <CardContent className="pt-6 text-center space-y-4">
              <div className="relative w-20 h-20 mx-auto">
                <div className="absolute inset-0 border-4 border-primary/20 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-transparent border-t-primary border-r-primary rounded-full animate-spin"></div>
                <div className="absolute inset-3 bg-primary/10 rounded-full flex items-center justify-center">
                  <Bot className="h-8 w-8 text-primary" />
                </div>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">Connecting to AI Strategist</h3>
                <p className="text-sm text-muted-foreground">
                  Initializing voice connection and analyzing race data...
                </p>
              </div>
              <div className="flex justify-center space-x-1">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Messages Area */}
      <ScrollArea 
        className="flex-1 px-4" 
        onScroll={handleScroll}
        ref={scrollAreaRef}
      >
        <div className="py-4 space-y-4 max-w-3xl mx-auto">
          {messages.length === 0 && isAgentReady && (
            <Card className="border-dashed">
              <CardContent className="pt-6 text-center space-y-3">
                <div className="text-5xl">🎙️</div>
                <p className="text-muted-foreground text-sm">
                  Waiting for agent greeting...
                </p>
              </CardContent>
            </Card>
          )}

          {messages.map((message, idx) => (
            <div
              key={idx}
              className={cn(
                'flex',
                message.sender === 'driver' ? 'justify-end' : 'justify-start'
              )}
            >
              <Card
                className={cn(
                  'max-w-[80%]',
                  message.sender === 'driver'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                )}
              >
                <CardContent className="pt-4 pb-3 px-4">
                  <div className="flex items-start gap-2">
                    {message.sender === 'agent' && (
                      <Bot className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {message.content}
                      </p>
                      <p className={cn(
                        "text-xs mt-1",
                        message.sender === 'driver' 
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

      {/* Scroll to Bottom Button */}
      {showScrollButton && (
        <div className="absolute bottom-32 right-4 z-10">
          <Button
            onClick={scrollToBottom}
            size="sm"
            className="rounded-full w-10 h-10 p-0 shadow-lg"
          >
            <ChevronDown className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-border bg-card p-4">
        <div className="max-w-3xl mx-auto">
          {inputMode === 'voice' ? (
            // Voice Mode
            <div className="flex items-center justify-center space-x-4">
              <Button
                onClick={() => setInputMode('text')}
                variant="outline"
                size="icon"
                className="rounded-full w-12 h-12"
                disabled={!isAgentReady}
              >
                <Keyboard className="h-5 w-5" />
              </Button>

              {/* Main Mic Button */}
              <Button
                onClick={handleMicClick}
                disabled={!isAgentReady}
                className={cn(
                  'rounded-full w-20 h-20 transition-all duration-300',
                  isListening && 'scale-110 shadow-lg shadow-primary/50'
                )}
              >
                {!isAgentReady ? (
                  <Loader2 className="h-8 w-8 animate-spin" />
                ) : isListening ? (
                  <div className="flex items-center gap-0.5">
                    <div className="w-1 h-6 bg-white rounded-sm animate-equalizer"></div>
                    <div className="w-1 h-6 bg-white rounded-sm animate-equalizer" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-1 h-6 bg-white rounded-sm animate-equalizer" style={{ animationDelay: '0.4s' }}></div>
                    <div className="w-1 h-6 bg-white rounded-sm animate-equalizer" style={{ animationDelay: '0.6s' }}></div>
                  </div>
                ) : (
                  <Mic className="h-8 w-8" />
                )}
              </Button>

              <div className="w-12" /> {/* Spacer for symmetry */}
            </div>
          ) : (
            // Text Mode
            <div className="flex items-center space-x-2">
              <Button
                onClick={() => setInputMode('voice')}
                variant="default"
                size="icon"
                className="rounded-full w-12 h-12 flex-shrink-0"
                disabled={!isAgentReady}
              >
                <Mic className="h-5 w-5" />
              </Button>

              <input
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  !isAgentReady
                    ? 'Connecting...'
                    : 'Ask about fuel, tires, strategy...'
                }
                className="flex-1 px-4 py-3 rounded-full border border-input bg-background focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                disabled={!isAgentReady}
              />

              <Button
                onClick={handleSendMessage}
                disabled={!textInput.trim() || !isAgentReady}
                size="icon"
                className="rounded-full w-12 h-12 flex-shrink-0"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="m6 12-3 9 18-9L3 3zm0 0h6" />
                </svg>
              </Button>
            </div>
          )}

          {/* Status Text */}
          <p className="text-center text-xs text-muted-foreground mt-3">
            {isListening ? (
              <span className="text-primary font-medium">🎤 Listening...</span>
            ) : isAgentReady ? (
              'Tap the mic to start talking'
            ) : (
              'Connecting to AI strategist...'
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

