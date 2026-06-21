import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams, Link } from "react-router-dom";
import { Send } from "lucide-react";
import { api } from "../api/client";
import { AnswerView } from "../components/AnswerView";
import { LoadingSpinner } from "../components/LoadingSpinner";
import type { ConversationTurn, Mode } from "../api/types";

export function ConversationPage() {
  const { iri = "" } = useParams<{ iri: string }>();
  const decoded = decodeURIComponent(iri);
  const qc = useQueryClient();
  const [followUp, setFollowUp] = useState("");
  const [mode, setMode] = useState<Mode>("deep_research");

  const conv = useQuery({
    queryKey: ["conversation", decoded],
    queryFn: () => api.getConversation(decoded),
  });

  const addTurn = useMutation({
    mutationFn: () =>
      api.conversationTurn(decoded, {
        question: followUp,
        mode,
        max_cost_usd: 0.2,
      }),
    onSuccess: () => {
      setFollowUp("");
      qc.invalidateQueries({ queryKey: ["conversation", decoded] });
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
  });

  if (conv.isPending) {
    return (
      <div className="rounded-lg border border-stone-200 bg-white p-4 shadow-sm">
        <LoadingSpinner label="Loading conversation..." />
      </div>
    );
  }
  if (conv.isError) {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
        {(conv.error as Error).message}
      </div>
    );
  }
  const data = conv.data!;
  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <Link
          to="/conversations"
          className="text-xs text-stone-500 hover:text-stone-700"
        >
          ← back to history
        </Link>
        <h1 className="text-xl font-semibold text-stone-900">
          {data.title || "(untitled)"}
        </h1>
        <div className="text-xs text-stone-500">
          {data.turn_count} turn{data.turn_count === 1 ? "" : "s"} · started{" "}
          {new Date(data.created_at).toLocaleString()}
        </div>
      </header>

      <div className="space-y-4">
        {data.turns.map((t: ConversationTurn) => (
          <article
            key={t.turn_index}
            className="rounded-lg border border-stone-200 bg-white p-5 shadow-sm"
          >
            <div className="flex items-start gap-3 mb-3">
              <span className="text-xs text-stone-500 mt-1">
                #{t.turn_index}
              </span>
              <div className="flex-1">
                <div className="text-sm text-stone-900 font-medium">
                  {t.user_question}
                </div>
                {t.follow_up_resolved && t.resolved_question &&
                  t.resolved_question !== t.user_question && (
                    <div className="text-xs text-stone-500 mt-1 italic">
                      resolved: {t.resolved_question}
                    </div>
                  )}
              </div>
              <span className="text-xs text-stone-500">{t.mode}</span>
            </div>
            <AnswerView answer={t.answer} mode={t.mode} />
          </article>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!followUp.trim() || addTurn.isPending) return;
          addTurn.mutate();
        }}
        className="space-y-3"
      >
        <h2 className="text-sm font-semibold text-stone-700">Continue thread</h2>
        <textarea
          value={followUp}
          onChange={(e) => setFollowUp(e.target.value)}
          placeholder="Ask a follow-up..."
          rows={3}
          className="w-full rounded-lg border border-stone-300 bg-white px-4 py-3 text-stone-900 placeholder:text-stone-400 focus:outline-none focus:border-emerald-500 shadow-sm"
        />
        <div className="flex items-center gap-3">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            className="rounded border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900"
          >
            <option value="deep_research">deep_research</option>
            <option value="simple_qa">simple_qa</option>
          </select>
          <button
            type="submit"
            disabled={addTurn.isPending || !followUp.trim()}
            className="ml-auto inline-flex items-center gap-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 text-sm font-medium text-white"
          >
            <Send className="h-4 w-4" /> Send
          </button>
        </div>
        {addTurn.isPending && (
          <LoadingSpinner label="Retrieving + synthesizing follow-up..." />
        )}
        {addTurn.isError && (
          <div className="text-rose-700 text-sm">
            {(addTurn.error as Error).message}
          </div>
        )}
      </form>
    </div>
  );
}
