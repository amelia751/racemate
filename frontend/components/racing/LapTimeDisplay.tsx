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
    <div className="flex flex-col gap-3">
      {/* Main Lap Time */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="text-center"
      >
        <div className="text-yellow-400 text-5xl font-black tracking-tighter font-mono">
          {lapTime}
        </div>
      </motion.div>

      {/* Badges - Single Line */}
      <div className="flex gap-2 items-center justify-center">
        <Badge variant="outline" className="text-sm px-2 py-1 bg-blue-500/20 border-blue-500">
          LAP {currentLap}
        </Badge>
        <span className="text-muted-foreground">·</span>
        <Badge variant="outline" className="text-sm px-2 py-1 bg-purple-500/20 border-purple-500">
          P{position}
        </Badge>
        <span className="text-muted-foreground">·</span>
        <Badge variant="outline" className="text-sm px-2 py-1 bg-cyan-500/20 border-cyan-500">
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

