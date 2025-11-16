'use client';

/**
 * Message Display Component
 * Shows conversation history between driver and agent
 */

import { useEffect, useRef } from 'react';
import { useCogniraceStore } from '@/lib/store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

export default function MessageDisplay() {
  const messages = useCogniraceStore((state) => state.messages);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card className="h-[500px] flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <span className="text-2xl">💬</span>
            Conversation
          </span>
          <Badge variant="outline">{messages.length} messages</Badge>
        </CardTitle>
        <CardDescription>
          Strategic dialogue with AI race engineer
        </CardDescription>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-3">
          {messages.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="pt-6">
                <div className="text-center space-y-3">
                  <div className="text-5xl">💭</div>
                  <p className="text-muted-foreground text-sm">
                    No messages yet
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Connect to the agent and start a conversation
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx}>
                {msg.sender === 'driver' ? (
                  // Driver message (right-aligned)
                  <div className="flex justify-end">
                    <Card className="max-w-[80%] border-primary/30 bg-primary/10">
                      <CardContent className="pt-4 pb-3">
                        <div className="flex items-start gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="default" className="text-xs">
                                👤 YOU
                              </Badge>
                              <span className="text-xs text-muted-foreground font-mono">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            <p className="text-sm">{msg.content}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  // Agent message (left-aligned)
                  <div className="flex justify-start">
                    <Card className="max-w-[80%] border-green-500/30 bg-green-500/10">
                      <CardContent className="pt-4 pb-3">
                        <div className="flex items-start gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="secondary" className="text-xs">
                                🤖 AI STRATEGIST
                              </Badge>
                              <span className="text-xs text-muted-foreground font-mono">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            ))
          )}

          {/* Auto-scroll anchor */}
          <div ref={scrollRef} />
        </CardContent>
      </ScrollArea>

      <Separator />

      {/* Footer Stats */}
      <CardContent className="pt-3 pb-4">
        <div className="flex justify-between items-center text-xs text-muted-foreground">
          <span>Total Messages: {messages.length}</span>
          <span>
            Driver: {messages.filter(m => m.sender === 'driver').length} | 
            Agent: {messages.filter(m => m.sender === 'agent').length}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
