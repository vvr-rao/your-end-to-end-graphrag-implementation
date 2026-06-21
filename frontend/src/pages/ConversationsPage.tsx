import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { MessageSquare } from "lucide-react";
import { api } from "../api/client";
import { LoadingSpinner } from "../components/LoadingSpinner";

export function ConversationsPage() {
  const q = useQuery({
    queryKey: ["conversations"],
    queryFn: () => api.listConversations(50, 0),
  });

  if (q.isPending) {
    return (
      <div className="rounded-lg border border-stone-200 bg-white p-4 shadow-sm">
        <LoadingSpinner label="Loading conversations..." />
      </div>
    );
  }
  if (q.isError) {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-700">
        {(q.error as Error).message}
      </div>
    );
  }
  const items = q.data || [];
  if (items.length === 0) {
    return (
      <div className="text-stone-600 text-sm">
        No conversations yet. Head over to{" "}
        <Link to="/ask" className="text-emerald-700 hover:text-emerald-600 font-medium">
          Ask
        </Link>{" "}
        to start one.
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {items.map((c) => (
        <Link
          key={c.iri}
          to={`/conversations/${encodeURIComponent(c.iri)}`}
          className="block rounded border border-stone-200 bg-white p-4 hover:border-emerald-500 hover:shadow-sm transition-all"
        >
          <div className="flex items-center gap-3">
            <MessageSquare className="h-4 w-4 text-emerald-600 shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-sm text-stone-900 truncate">
                {c.title || "(untitled)"}
              </div>
              <div className="text-xs text-stone-500 mt-0.5">
                {c.turn_count} turn{c.turn_count === 1 ? "" : "s"} ·{" "}
                {new Date(c.created_at).toLocaleString()}
              </div>
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}
