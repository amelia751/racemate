'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { 
  Cpu, 
  Zap, 
  Gauge, 
  TrendingUp, 
  AlertTriangle, 
  Network, 
  User, 
  Fuel,
  ChevronRight,
  Home
} from 'lucide-react';
import Link from 'next/link';
import OverviewSection from '@/components/documentation/OverviewSection';
import FuelSection from '@/components/documentation/FuelSection';
import LaptimeSection from '@/components/documentation/LaptimeSection';
import TireSection from '@/components/documentation/TireSection';

const sections = [
  { id: 'overview', title: 'System Overview', icon: Home },
  { id: 'fuel', title: 'Fuel Consumption Model', icon: Fuel },
  { id: 'laptime', title: 'Lap Time Transformer', icon: Zap },
  { id: 'tire', title: 'Tire Degradation Model', icon: Gauge },
  { id: 'fcy', title: 'FCY Hazard Predictor', icon: AlertTriangle },
  { id: 'pit', title: 'Pit Loss Model', icon: TrendingUp },
  { id: 'anomaly', title: 'Anomaly Detector', icon: AlertTriangle },
  { id: 'driver', title: 'Driver Embedding Model', icon: User },
  { id: 'traffic', title: 'Traffic GNN', icon: Network },
  { id: 'architecture', title: 'System Architecture', icon: Cpu },
];

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-black to-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-black/40 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-4">
              <div className="text-2xl font-black tracking-tighter">
                <span className="text-cyan-400">COGNI</span>
                <span className="text-yellow-400">RACE</span>
              </div>
              <span className="text-sm text-muted-foreground">/ Documentation</span>
            </Link>
            <Badge variant="outline" className="text-xs">
              ML Pipeline v1.0
            </Badge>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-12 gap-8">
          {/* Sidebar */}
          <aside className="col-span-3 space-y-2">
            <div className="sticky top-24">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">
                📚 Contents
              </h3>
              <nav className="space-y-1">
                {sections.map((section) => {
                  const Icon = section.icon;
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${
                        activeSection === section.id
                          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                          : 'text-gray-400 hover:bg-gray-800/50 hover:text-white'
                      }`}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{section.title}</span>
                      {activeSection === section.id && (
                        <ChevronRight className="w-3 h-3 ml-auto" />
                      )}
                    </button>
                  );
                })}
              </nav>
            </div>
          </aside>

          {/* Main Content */}
          <main className="col-span-9 space-y-8">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {activeSection === 'overview' && <OverviewSection />}
              {activeSection === 'fuel' && <FuelSection />}
              {activeSection === 'laptime' && <LaptimeSection />}
              {activeSection === 'tire' && <TireSection />}
              {activeSection === 'fcy' && <PlaceholderSection title="FCY Hazard Predictor" />}
              {activeSection === 'pit' && <PlaceholderSection title="Pit Loss Model" />}
              {activeSection === 'anomaly' && <PlaceholderSection title="Anomaly Detector" />}
              {activeSection === 'driver' && <PlaceholderSection title="Driver Embedding Model" />}
              {activeSection === 'traffic' && <PlaceholderSection title="Traffic GNN" />}
              {activeSection === 'architecture' && <PlaceholderSection title="System Architecture" />}
            </motion.div>
          </main>
        </div>
      </div>
    </div>
  );
}

// Placeholder for sections not yet created
function PlaceholderSection({ title }: { title: string }) {
  return (
    <div className="bg-black/40 border border-cyan-500/20 rounded-lg p-12 text-center">
      <h2 className="text-3xl font-bold text-cyan-400 mb-4">{title}</h2>
      <p className="text-muted-foreground">
        Detailed documentation component coming soon...
      </p>
      <p className="text-sm text-muted-foreground mt-2">
        Use Overview and Fuel Consumption sections as reference for comprehensive content structure
      </p>
    </div>
  );
}
