import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { EvidenceItem } from "../api/types";

const kindColor: Record<string, string> = {
  chunk: "bg-sky-100 text-sky-800",
  artifact: "bg-violet-100 text-violet-800",
  class: "bg-emerald-100 text-emerald-800",
  entity: "bg-amber-100 text-amber-800",
};

export function EvidenceList({ evidence }: { evidence: EvidenceItem[] }) {
  const [open, setOpen] = useState(false);
  if (!evidence || evidence.length === 0) return null;
  return (
    <div className="mt-6 border-t border-stone-200 pt-4">
      <button
        onClick={() => setOpen((x) => !x)}
        className="flex items-center gap-2 text-sm text-stone-700 hover:text-stone-900"
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
              className="text-sm border border-stone-200 rounded p-3 bg-stone-50"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-stone-500 text-xs">#{ev.rank ?? i + 1}</span>
                <span
                  className={`text-xs px-1.5 py-0.5 rounded ${
                    kindColor[ev.kind] || "bg-stone-200 text-stone-700"
                  }`}
                >
                  {ev.kind}
                </span>
                {ev.iri && (
                  <code className="text-xs text-stone-600 truncate">{ev.iri}</code>
                )}
              </div>
              {ev.snippet && (
                <p className="text-stone-800 whitespace-pre-wrap">
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
