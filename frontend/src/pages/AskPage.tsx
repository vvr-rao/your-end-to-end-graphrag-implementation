import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Send } from "lucide-react";
import { api } from "../api/client";
import { AnswerView } from "../components/AnswerView";
import { EvidenceList } from "../components/EvidenceList";
import { LoadingSpinner } from "../components/LoadingSpinner";
import type { Mode, TurnResponse } from "../api/types";

/** Asking a question creates a new conversation (so it shows up in
 * /conversations) and adds the first turn. The single-turn-only path
 * (POST /qa) is intentionally not used here — we want every Q&A to be
 * replayable from history. */
export function AskPage() {
  const navigate = useNavigate();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("deep_research");
  const [result, setResult] = useState<TurnResponse | null>(null);

  const ask = useMutation({
    mutationFn: async (): Promise<TurnResponse> => {
      const conv = await api.startConversation(
        question.length > 60 ? question.slice(0, 57) + "..." : question,
      );
      return api.conversationTurn(conv.iri, {
        question,
        mode,
        max_cost_usd: 0.2,
      });
    },
    onSuccess: (data) => setResult(data),
  });

  return (
    <div className="space-y-6">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!question.trim() || ask.isPending) return;
          setResult(null);
          ask.mutate();
        }}
        className="space-y-3"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask your knowledge graph anything..."
          rows={3}
          className="w-full rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500"
        />
        <div className="flex items-center gap-3">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
          >
            <option value="deep_research">deep_research (default)</option>
            <option value="simple_qa">simple_qa</option>
          </select>
          <button
            type="submit"
            disabled={ask.isPending || !question.trim()}
            className="ml-auto inline-flex items-center gap-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 text-sm font-medium"
          >
            <Send className="h-4 w-4" /> Ask
          </button>
        </div>
      </form>

      {ask.isPending && (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <LoadingSpinner label="Retrieving + synthesizing..." />
        </div>
      )}
      {ask.isError && (
        <div className="rounded-lg border border-rose-800 bg-rose-950 p-4 text-rose-200">
          {(ask.error as Error)?.message || "Request failed."}
        </div>
      )}
      {result && (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <div className="flex items-center justify-between text-xs text-slate-500 mb-3">
            <span>mode: {result.mode}</span>
            <span>
              ${result.cost_usd.toFixed(4)} · {result.wall_seconds.toFixed(1)}s
            </span>
          </div>
          <AnswerView answer={result.answer} mode={result.mode} />
          <EvidenceList evidence={result.evidence} />
          <div className="mt-5 flex items-center gap-3 text-sm">
            <button
              className="text-emerald-400 hover:text-emerald-300"
              onClick={() =>
                navigate(
                  `/conversations/${encodeURIComponent(
                    result.conversation_iri,
                  )}`,
                )
              }
            >
              Continue this thread →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
