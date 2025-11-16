'use client';

/**
 * Voice Agent Panel Component
 * Handles voice agent connection and text input
 */

import { useState } from 'react';
import { useLiveKitContext } from '@/lib/livekit/LiveKitContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

export default function VoiceAgentPanel() {
  const [textInput, setTextInput] = useState('');
  const {
    isConnected,
    isAgentReady,
    connectionState,
    connectToRoom,
    disconnectFromRoom,
    sendTextMessage,
    error,
  } = useLiveKitContext();

  const handleSendMessage = async () => {
    if (!textInput.trim()) return;
    
    try {
      await sendTextMessage(textInput);
      setTextInput('');
    } catch (err) {
      console.error('Failed to send message:', err);
    }
  };

  const getStatusBadge = () => {
    if (isAgentReady) {
      return <Badge variant="default" className="animate-pulse">🟢 Agent Ready</Badge>;
    }
    if (isConnected) {
      return <Badge variant="secondary">🟡 Connecting...</Badge>;
    }
    return <Badge variant="outline">⚫ Disconnected</Badge>;
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <span className="text-2xl">🎙️</span>
            AI Race Strategist
          </span>
          {getStatusBadge()}
        </CardTitle>
        <CardDescription>
          Real-time voice and text interaction with AI race engineer
        </CardDescription>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-4">
          {/* Connection Status */}
          <Card className="border-primary/20">
            <CardContent className="pt-6">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Connection</span>
                  <Badge variant={isConnected ? "default" : "outline"}>
                    {connectionState}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Agent Status</span>
                  <Badge variant={isAgentReady ? "default" : "secondary"}>
                    {isAgentReady ? 'Ready' : 'Not Ready'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="border-destructive bg-destructive/10">
              <CardContent className="pt-6">
                <p className="text-sm text-destructive font-semibold mb-1">⚠️ Connection Error</p>
                <p className="text-xs text-destructive/80">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Connection Controls */}
          <div className="space-y-2">
            {!isConnected ? (
              <Button
                onClick={connectToRoom}
                className="w-full h-12 text-base"
                size="lg"
              >
                <span className="text-xl mr-2">🔌</span>
                Connect to Agent
              </Button>
            ) : (
              <Button
                onClick={disconnectFromRoom}
                variant="destructive"
                className="w-full h-12 text-base"
                size="lg"
              >
                <span className="text-xl mr-2">🔌</span>
                Disconnect
              </Button>
            )}
          </div>

          <Separator />

          {/* Text Input */}
          <div className="space-y-3">
            <label className="text-sm font-semibold">💬 Send Text Message</label>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Type a message to the AI strategist... (Press Enter to send)"
              disabled={!isAgentReady}
              className="w-full min-h-[100px] p-3 rounded-md bg-background border border-input resize-none focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!isAgentReady || !textInput.trim()}
              className="w-full"
              size="lg"
            >
              Send Message →
            </Button>
          </div>

          <Separator />

          {/* Quick Commands */}
          <Card className="bg-muted/50">
            <CardContent className="pt-6">
              <p className="text-sm font-semibold mb-3">⚡ Quick Commands</p>
              <div className="grid grid-cols-1 gap-2">
                <Button
                  onClick={() => sendTextMessage("What's my fuel status?")}
                  disabled={!isAgentReady}
                  variant="secondary"
                  size="sm"
                >
                  Check Fuel
                </Button>
                <Button
                  onClick={() => sendTextMessage("How are my tires doing?")}
                  disabled={!isAgentReady}
                  variant="secondary"
                  size="sm"
                >
                  Check Tires
                </Button>
                <Button
                  onClick={() => sendTextMessage("Give me a race status update")}
                  disabled={!isAgentReady}
                  variant="secondary"
                  size="sm"
                >
                  Race Status
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Info */}
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="pt-6">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <span>ℹ️</span>
                How to use
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs text-muted-foreground">
                <li>Click "Connect to Agent" to start</li>
                <li>Type questions or use quick commands</li>
                <li>Agent analyzes telemetry in real-time</li>
                <li>Responses appear in message panel</li>
                <li>Voice coming soon!</li>
              </ul>
            </CardContent>
          </Card>
        </CardContent>
      </ScrollArea>
    </Card>
  );
}
