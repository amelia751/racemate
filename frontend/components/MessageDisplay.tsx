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
          <span>Chat</span>
          <Badge variant="outline" className="text-xs">{messages.length}</Badge>
        </CardTitle>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-3">
          {messages.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="pt-6">
                <div className="text-center space-y-2">
                  <p className="text-muted-foreground text-sm">
                    No messages yet
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Connect to start a conversation
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
                      <CardContent className="pt-3 pb-3 px-4">
                        <p className="text-sm">{msg.content}</p>
                        <span className="text-xs text-muted-foreground mt-1 block">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  // Agent message (left-aligned)
                  <div className="flex justify-start">
                    <Card className="max-w-[80%] border-green-500/30 bg-green-500/10">
                      <CardContent className="pt-3 pb-3 px-4">
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                        <span className="text-xs text-muted-foreground mt-1 block">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
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

    </Card>
  );
}
