'use client';

/**
 * Lap Time Display - F1 Style
 * Left column display only
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';

export default function LapTimeDisplay() {
  const [lapTime, setLapTime] = useState('1:05.031');
  const [currentLap, setCurrentLap] = useState(13);
  const [position, setPosition] = useState(3);
  const [sector, setSector] = useState(2);

  return (
    <div className="h-full flex flex-col justify-between">
      {/* Main Lap Time */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="text-center"
      >
        <div className="text-yellow-400 text-5xl font-black tracking-tighter font-mono mb-3">
          {lapTime}
        </div>
      </motion.div>

      {/* Badges */}
      <div className="flex flex-col gap-2 items-center">
        <Badge variant="outline" className="text-lg px-3 py-2 bg-blue-500/20 border-blue-500 w-full justify-center">
          LAP {currentLap}
        </Badge>
        <Badge variant="outline" className="text-lg px-3 py-2 bg-purple-500/20 border-purple-500 w-full justify-center">
          P{position}
        </Badge>
        <Badge variant="outline" className="text-lg px-3 py-2 bg-cyan-500/20 border-cyan-500 w-full justify-center">
          S{sector}
        </Badge>
      </div>

      {/* Track Info */}
      <div className="text-center">
        <div className="text-cyan-400 text-sm tracking-wider font-bold">CIRCUIT OF THE AMERICAS</div>
        <div className="text-muted-foreground text-xs">TEXAS • 5.513km • 20 laps</div>
      </div>
    </div>
  );
}

