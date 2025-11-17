'use client';

/**
 * Message Display Component
 * Shows conversation history between driver and agent
 */

import { useEffect, useRef } from 'react';
import { useCogniraceStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bot } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

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
    <div className="h-[500px] flex flex-col bg-gradient-to-b from-gray-900 via-black to-black border border-cyan-500/20 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-cyan-900/30 to-purple-900/30 border-b border-cyan-500/20 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <h2 className="text-lg font-bold tracking-wide text-cyan-400">CHAT</h2>
          </div>
          <Badge variant="outline" className="text-[10px] font-bold tracking-wider bg-black/50 border-cyan-500/30 text-cyan-400">
            {messages.length}
          </Badge>
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea className="flex-1">
        <div className="px-4 py-4 space-y-3">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full min-h-[200px]">
              <div className="text-center space-y-3">
                <div className="w-16 h-16 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto">
                  <Bot className="w-8 h-8 text-cyan-400" />
                </div>
                <p className="text-sm text-gray-300 font-medium">
                  No messages yet
                </p>
                <p className="text-[11px] text-gray-500 tracking-wide">
                  CONNECT TO START
                </p>
              </div>
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`flex ${msg.sender === 'driver' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.sender === 'driver' ? (
                    // Driver message (right-aligned)
                    <div className="max-w-[80%] group">
                      <div className="bg-gradient-to-br from-cyan-600 to-cyan-700 rounded-xl rounded-tr-sm px-3 py-2 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30 transition-all">
                        <p className="text-xs text-white leading-relaxed font-medium">{msg.content}</p>
                        <span className="text-[9px] text-cyan-200/60 mt-1.5 block font-mono">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ) : (
                    // Agent message (left-aligned)
                    <div className="max-w-[80%] group">
                      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700/50 rounded-xl rounded-tl-sm px-3 py-2 shadow-lg shadow-gray-800/50 hover:border-cyan-500/30 hover:shadow-cyan-500/20 transition-all">
                        <p className="text-xs text-gray-100 leading-relaxed whitespace-pre-wrap font-medium">{msg.content}</p>
                        <span className="text-[9px] text-gray-500 mt-1.5 block font-mono">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          )}

          {/* Auto-scroll anchor */}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
