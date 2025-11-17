'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Cpu,
  Zap,
  Gauge,
  TrendingUp,
  AlertTriangle,
  Network,
  User,
  Fuel,
  Home,
  Loader2,
} from 'lucide-react';

// Loading component (must be defined before dynamic imports)
function LoadingSection() {
  return (
    <div className="flex flex-col items-center justify-center py-24 space-y-4">
      <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      <p className="text-slate-400 text-sm">Loading documentation...</p>
    </div>
  );
}

// Lazy load documentation sections using Next.js dynamic imports
const OverviewSection = dynamic(() => import('@/components/documentation/OverviewSection'), {
  loading: () => <LoadingSection />
});
const FuelSection = dynamic(() => import('@/components/documentation/FuelSection'), {
  loading: () => <LoadingSection />
});
const LaptimeSection = dynamic(() => import('@/components/documentation/LaptimeSection'), {
  loading: () => <LoadingSection />
});
const TireSection = dynamic(() => import('@/components/documentation/TireSection'), {
  loading: () => <LoadingSection />
});
const FCYSection = dynamic(() => import('@/components/documentation/FCYSection'), {
  loading: () => <LoadingSection />
});
const PitSection = dynamic(() => import('@/components/documentation/PitSection'), {
  loading: () => <LoadingSection />
});
const AnomalySection = dynamic(() => import('@/components/documentation/AnomalySection'), {
  loading: () => <LoadingSection />
});
const DriverSection = dynamic(() => import('@/components/documentation/DriverSection'), {
  loading: () => <LoadingSection />
});
const TrafficSection = dynamic(() => import('@/components/documentation/TrafficSection'), {
  loading: () => <LoadingSection />
});
const ArchitectureSection = dynamic(() => import('@/components/documentation/ArchitectureSection'), {
  loading: () => <LoadingSection />
});

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
    <div className="h-screen flex bg-black text-white overflow-hidden p-6">
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-black to-slate-900" />
        <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-cyan-500/5 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-purple-500/5 rounded-full blur-3xl animate-pulse" />
      </div>

      {/* Sidebar Navigation */}
      <aside className="w-64 rounded-lg flex flex-col bg-black/40 backdrop-blur-sm flex-shrink-0">
        {/* Logo Section */}
        <div className="px-6 py-6 border-b border-slate-800/50">
          <div className="text-xl font-black tracking-tight">
            <span className="bg-gradient-to-r from-cyan-400 to-cyan-300 bg-clip-text text-transparent">RACE</span>
            <span className="bg-gradient-to-r from-yellow-400 to-amber-400 bg-clip-text text-transparent">MATE</span>
          </div>
          <div className="text-xs text-white mt-1">Documentation</div>
        </div>

        {/* Navigation */}
        <ScrollArea className="flex-1">
          <nav className="px-4 py-4 space-y-1">
            {sections.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                    isActive
                      ? 'bg-slate-800 text-cyan-400'
                      : 'text-white hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 transition-colors ${
                    isActive ? 'text-cyan-400' : 'text-slate-400'
                  }`} />
                  <span className="truncate text-left">{section.title}</span>
                </button>
              );
            })}
          </nav>
        </ScrollArea>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto rounded-lg bg-black/40 backdrop-blur-sm ml-6">
        <div className="px-8 py-8">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="space-y-8"
          >
            {activeSection === 'overview' && <OverviewSection />}
            {activeSection === 'fuel' && <FuelSection />}
            {activeSection === 'laptime' && <LaptimeSection />}
            {activeSection === 'tire' && <TireSection />}
            {activeSection === 'fcy' && <FCYSection />}
            {activeSection === 'pit' && <PitSection />}
            {activeSection === 'anomaly' && <AnomalySection />}
            {activeSection === 'driver' && <DriverSection />}
            {activeSection === 'traffic' && <TrafficSection />}
            {activeSection === 'architecture' && <ArchitectureSection />}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
