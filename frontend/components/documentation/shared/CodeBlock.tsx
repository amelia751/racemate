'use client';

import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
}

export default function CodeBlock({ code, language = 'python', title }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-950/50 rounded-lg overflow-hidden">
      {title && (
        <div className="px-4 py-2 bg-slate-900/50 flex items-center justify-between">
          <span className="text-xs font-medium text-slate-400">{title}</span>
          <span className="text-xs text-slate-500">{language}</span>
        </div>
      )}
      <div className="relative">
        <button
          onClick={handleCopy}
          className="absolute top-3 right-3 z-10 p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 transition-all duration-200 group"
          aria-label="Copy code"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-400" />
          ) : (
            <Copy className="w-4 h-4 text-slate-400 group-hover:text-slate-300" />
          )}
        </button>
        <pre className="p-4 overflow-x-auto text-sm leading-relaxed">
          <code className="text-slate-300 font-mono">{code}</code>
        </pre>
      </div>
    </div>
  );
}

