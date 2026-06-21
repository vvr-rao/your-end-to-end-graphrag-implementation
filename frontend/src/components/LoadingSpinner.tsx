import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

/** Shared spinner. After 5s of waiting it swaps copy to warn about the
 * Render Free-tier cold start (30-60s wakeup after 15min idle). */
export function LoadingSpinner({ label = "Loading..." }: { label?: string }) {
  const [coldStart, setColdStart] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setColdStart(true), 5000);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="flex items-center gap-3 text-slate-400">
      <Loader2 className="h-5 w-5 animate-spin" />
      <span>
        {coldStart
          ? "Waking up the backend — first request after idle can take 30-60s on Render Free tier..."
          : label}
      </span>
    </div>
  );
}
