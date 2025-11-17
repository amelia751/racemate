interface MetricCardProps {
  label: string;
  value: string | number;
  color?: 'cyan' | 'amber' | 'purple' | 'red' | 'orange' | 'green' | 'rose' | 'blue';
}

export default function MetricCard({ label, value, color = 'cyan' }: MetricCardProps) {
  // Map colors to full Tailwind class strings for JIT compilation
  const colorClasses = {
    cyan: {
      bg: 'from-cyan-500/10',
      overlay: 'from-cyan-500/0 via-cyan-500/5 to-cyan-500/0',
      text: 'from-cyan-400 to-cyan-300'
    },
    amber: {
      bg: 'from-amber-500/10',
      overlay: 'from-amber-500/0 via-amber-500/5 to-amber-500/0',
      text: 'from-amber-400 to-amber-300'
    },
    purple: {
      bg: 'from-purple-500/10',
      overlay: 'from-purple-500/0 via-purple-500/5 to-purple-500/0',
      text: 'from-purple-400 to-purple-300'
    },
    red: {
      bg: 'from-red-500/10',
      overlay: 'from-red-500/0 via-red-500/5 to-red-500/0',
      text: 'from-red-400 to-red-300'
    },
    orange: {
      bg: 'from-orange-500/10',
      overlay: 'from-orange-500/0 via-orange-500/5 to-orange-500/0',
      text: 'from-orange-400 to-orange-300'
    },
    green: {
      bg: 'from-green-500/10',
      overlay: 'from-green-500/0 via-green-500/5 to-green-500/0',
      text: 'from-green-400 to-green-300'
    },
    rose: {
      bg: 'from-rose-500/10',
      overlay: 'from-rose-500/0 via-rose-500/5 to-rose-500/0',
      text: 'from-rose-400 to-rose-300'
    },
    blue: {
      bg: 'from-blue-500/10',
      overlay: 'from-blue-500/0 via-blue-500/5 to-blue-500/0',
      text: 'from-blue-400 to-blue-300'
    }
  };

  const classes = colorClasses[color];

  return (
    <div className={`group/stat relative overflow-hidden bg-gradient-to-br ${classes.bg} to-slate-950 rounded-xl p-5 transition-all duration-300`}>
      <div className={`absolute inset-0 bg-gradient-to-r ${classes.overlay} opacity-0 group-hover/stat:opacity-100 transition-opacity`} />
      <div className="relative z-10">
        <div className="text-2xl font-bold text-white">
          {value}
        </div>
        <div className="text-xs text-white mt-1 font-medium">{label}</div>
      </div>
    </div>
  );
}

