// Fetch wrapper. Reads bearer token + API base URL from localStorage,
// falling back to the Vite build-time VITE_API_BASE_URL env var. Throws
// `UnauthorizedError` on 401 so callers can clear auth + redirect to /settings.

import type {
  AskRequest,
  AskResponse,
  ConversationListItem,
  ConversationView,
  TurnResponse,
} from "./types";

export const LS_BEARER = "graphrag.bearer";
export const LS_API_BASE = "graphrag.apiBase";

const BUILD_DEFAULT =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  "http://localhost:8000";

export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
    this.name = "UnauthorizedError";
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API ${status}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
  }
}

export function getApiBase(): string {
  return localStorage.getItem(LS_API_BASE) || BUILD_DEFAULT;
}

export function getBearer(): string {
  return localStorage.getItem(LS_BEARER) || "";
}

export function clearAuth(): void {
  localStorage.removeItem(LS_BEARER);
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const base = getApiBase().replace(/\/$/, "");
  const bearer = getBearer();
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  if (bearer) headers["Authorization"] = `Bearer ${bearer}`;
  if (init.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const resp = await fetch(`${base}${path}`, { ...init, headers });
  if (resp.status === 401) {
    clearAuth();
    throw new UnauthorizedError();
  }
  if (!resp.ok) {
    throw new ApiError(resp.status, await resp.text());
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

// ---- API surface used by the UI ------------------------------------------

export const api = {
  health: () => request<{ status: string }>("/health"),

  ask: (body: AskRequest) =>
    request<AskResponse>("/qa", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  startConversation: (title?: string | null) =>
    request<{ iri: string; id: string; title: string | null }>(
      "/conversations",
      {
        method: "POST",
        body: JSON.stringify({ title: title ?? null }),
      },
    ),

  conversationTurn: (
    iri: string,
    body: { question: string; mode?: string; max_cost_usd?: number },
  ) =>
    request<TurnResponse>(
      `/conversations/${encodeURIComponent(iri)}/turns`,
      {
        method: "POST",
        body: JSON.stringify(body),
      },
    ),

  listConversations: (limit = 50, offset = 0) =>
    request<ConversationListItem[]>(
      `/conversations?limit=${limit}&offset=${offset}`,
    ),

  getConversation: (iri: string) =>
    request<ConversationView>(
      `/conversations/${encodeURIComponent(iri)}`,
    ),
};
