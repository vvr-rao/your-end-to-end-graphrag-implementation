import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";

type HealthState =
  | { status: "loading" }
  | { status: "ok" }
  | { status: "error"; message: string };

function App() {
  const [health, setHealth] = useState<HealthState>({ status: "loading" });

  useEffect(() => {
    fetch("/health")
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = await r.json();
        if (body.status !== "ok") throw new Error(`unexpected body: ${JSON.stringify(body)}`);
        setHealth({ status: "ok" });
      })
      .catch((err: Error) => setHealth({ status: "error", message: err.message }));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
      <div className="max-w-xl w-full space-y-6">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">
            your-personal-knowledge-graph-creator
          </h1>
          <p className="text-slate-400 mt-1">
            GraphRAG ontology + document management — bootstrap UI
          </p>
        </header>

        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="flex items-center gap-3">
            {health.status === "loading" && (
              <>
                <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
                <span className="text-slate-400">Pinging backend /health…</span>
              </>
            )}
            {health.status === "ok" && (
              <>
                <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                <span>Backend healthy — REST + MCP reachable.</span>
              </>
            )}
            {health.status === "error" && (
              <>
                <XCircle className="h-5 w-5 text-rose-400" />
                <span>Backend unreachable: {health.message}</span>
              </>
            )}
          </div>
        </div>

        <p className="text-xs text-slate-500">
          Phase 0 placeholder. Real UI (Upload / Graph / Review / QA) lands in Phase 3.
        </p>
      </div>
    </div>
  );
}

export default App;
