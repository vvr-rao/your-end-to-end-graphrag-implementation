import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { EvidenceItem } from "../api/types";

const kindColor: Record<string, string> = {
  chunk: "bg-sky-900 text-sky-200",
  artifact: "bg-violet-900 text-violet-200",
  class: "bg-emerald-900 text-emerald-200",
  entity: "bg-amber-900 text-amber-200",
};

export function EvidenceList({ evidence }: { evidence: EvidenceItem[] }) {
  const [open, setOpen] = useState(false);
  if (!evidence || evidence.length === 0) return null;
  return (
    <div className="mt-6 border-t border-slate-800 pt-4">
      <button
        onClick={() => setOpen((x) => !x)}
        className="flex items-center gap-2 text-sm text-slate-300 hover:text-white"
      >
        {open ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        Evidence ({evidence.length})
      </button>
      {open && (
        <ul className="mt-3 space-y-2">
          {evidence.map((ev, i) => (
            <li
              key={i}
              className="text-sm border border-slate-800 rounded p-3 bg-slate-900"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-slate-500 text-xs">#{ev.rank ?? i + 1}</span>
                <span
                  className={`text-xs px-1.5 py-0.5 rounded ${
                    kindColor[ev.kind] || "bg-slate-800 text-slate-300"
                  }`}
                >
                  {ev.kind}
                </span>
                {ev.iri && (
                  <code className="text-xs text-slate-400 truncate">{ev.iri}</code>
                )}
              </div>
              {ev.snippet && (
                <p className="text-slate-300 whitespace-pre-wrap">
                  {String(ev.snippet)}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
