import { LucideIcon } from 'lucide-react';
import MetricCard from './MetricCard';

interface Metric {
  label: string;
  value: string | number;
}

interface DocumentationHeaderProps {
  icon: LucideIcon;
  title: string;
  subtitle: string;
  metrics: Metric[];
  color: 'cyan' | 'amber' | 'purple' | 'red' | 'orange' | 'green' | 'rose' | 'blue';
}

export default function DocumentationHeader({
  icon: Icon,
  title,
  subtitle,
  metrics,
  color
}: DocumentationHeaderProps) {
  // Color mappings for gradients
  const colorMap = {
    cyan: {
      primary: 'from-cyan-500 to-purple-600',
      text: 'from-cyan-300 via-cyan-400 to-purple-400',
      overlay: 'from-cyan-500/5 to-purple-500/5',
      shadow: 'shadow-cyan-500/20 group-hover:shadow-cyan-500/40'
    },
    amber: {
      primary: 'from-amber-500 to-orange-600',
      text: 'from-amber-300 via-amber-400 to-orange-400',
      overlay: 'from-amber-500/5 to-orange-500/5',
      shadow: 'shadow-amber-500/20 group-hover:shadow-amber-500/40'
    },
    purple: {
      primary: 'from-purple-500 to-blue-600',
      text: 'from-purple-300 via-purple-400 to-blue-400',
      overlay: 'from-purple-500/5 to-blue-500/5',
      shadow: 'shadow-purple-500/20 group-hover:shadow-purple-500/40'
    },
    red: {
      primary: 'from-red-500 to-orange-600',
      text: 'from-red-300 via-red-400 to-orange-400',
      overlay: 'from-red-500/5 to-orange-500/5',
      shadow: 'shadow-red-500/20 group-hover:shadow-red-500/40'
    },
    orange: {
      primary: 'from-orange-500 to-yellow-600',
      text: 'from-orange-300 via-orange-400 to-yellow-400',
      overlay: 'from-orange-500/5 to-yellow-500/5',
      shadow: 'shadow-orange-500/20 group-hover:shadow-orange-500/40'
    },
    green: {
      primary: 'from-green-500 to-emerald-600',
      text: 'from-green-300 via-green-400 to-emerald-400',
      overlay: 'from-green-500/5 to-emerald-500/5',
      shadow: 'shadow-green-500/20 group-hover:shadow-green-500/40'
    },
    rose: {
      primary: 'from-rose-500 to-pink-600',
      text: 'from-rose-300 via-rose-400 to-pink-400',
      overlay: 'from-rose-500/5 to-pink-500/5',
      shadow: 'shadow-rose-500/20 group-hover:shadow-rose-500/40'
    },
    blue: {
      primary: 'from-blue-500 to-cyan-600',
      text: 'from-blue-300 via-blue-400 to-cyan-400',
      overlay: 'from-blue-500/5 to-cyan-500/5',
      shadow: 'shadow-blue-500/20 group-hover:shadow-blue-500/40'
    }
  };

  const colors = colorMap[color];

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900/50 via-black to-slate-950 p-8 group`}>
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:30px_30px]" />
      <div className={`absolute inset-0 bg-gradient-to-br ${colors.overlay} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
      <div className="relative z-10">
        <div className="flex items-center gap-4 mb-3">
          <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${colors.primary} flex items-center justify-center shadow-lg ${colors.shadow} transition-shadow flex-shrink-0`}>
            <Icon className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-4xl font-black text-white leading-none">
            {title}
          </h1>
        </div>
        <p className="text-white mt-2 ml-[72px]">
          {subtitle}
        </p>
        
        <div className={`grid ${metrics.length === 3 ? 'grid-cols-3' : 'grid-cols-4'} gap-4 mt-8`}>
          {metrics.map((metric, index) => (
            <MetricCard 
              key={index}
              label={metric.label}
              value={metric.value}
              color={color}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

