'use client';

/**
 * Lap Time Display - F1 Style
 */

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default function LapTimeDisplay() {
  const [lapTime, setLapTime] = useState('1:05.031');
  const [currentLap, setCurrentLap] = useState(13);
  const [position, setPosition] = useState(3);
  const [sector, setSector] = useState(2);

  return (
    <div className="relative">
      {/* Main Lap Time */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="text-center"
      >
        <div className="text-yellow-400 text-6xl md:text-8xl font-black tracking-tighter font-mono">
          {lapTime}
        </div>
        <div className="flex items-center justify-center gap-4 mt-4">
          <Badge variant="outline" className="text-2xl px-4 py-2 bg-blue-500/20 border-blue-500">
            LAP {currentLap}
          </Badge>
          <Badge variant="outline" className="text-2xl px-4 py-2 bg-purple-500/20 border-purple-500">
            P{position}
          </Badge>
          <Badge variant="outline" className="text-2xl px-4 py-2 bg-cyan-500/20 border-cyan-500">
            S{sector}
          </Badge>
        </div>
      </motion.div>

      {/* Track Info */}
      <div className="mt-6 text-center">
        <div className="text-cyan-400 text-lg tracking-wider">CIRCUIT OF THE AMERICAS</div>
        <div className="text-muted-foreground text-sm">TEXAS • 5.513km • 20 laps</div>
      </div>
    </div>
  );
}

