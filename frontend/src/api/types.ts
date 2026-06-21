// TS mirrors of the backend Pydantic models. Hand-maintained — keep in
// sync with backend/app/api/{qa.py, conversations.py, trace.py}.

export type Mode = "simple_qa" | "deep_research";

export interface EvidenceItem {
  kind: string;            // "chunk" / "artifact" / "class" / "entity"
  iri?: string | null;
  rank: number;
  score?: number | null;
  snippet?: string | null;
  // backend includes a free-form bag of extras; carry it through.
  [key: string]: unknown;
}

export interface AskRequest {
  question: string;
  mode?: Mode;
  top_k?: number | null;
  hops?: number;
  max_cost_usd?: number;
}

export interface AskResponse {
  question: string;
  resolved_query: string;
  mode: Mode;
  answer: string | null;
  evidence: EvidenceItem[];
  retrieval_run_id: string | null;
  cost_usd: number;
  wall_seconds: number;
}

export interface ConversationListItem {
  iri: string;
  title: string | null;
  created_at: string;       // ISO 8601
  turn_count: number;
  last_turn_at: string | null;
}

export interface ConversationTurn {
  turn_index: number;
  user_question: string;
  resolved_question: string | null;
  mode: Mode;
  answer: string | null;
  follow_up_resolved: boolean;
  created_at: string;
}

export interface ConversationView {
  iri: string;
  title: string | null;
  created_at: string;
  turn_count: number;
  turns: ConversationTurn[];
}

export interface TurnResponse {
  conversation_iri: string;
  conversation_turn_id: string;
  turn_index: number;
  follow_up_resolved: boolean;
  user_question: string;
  resolved_question: string;
  mode: Mode;
  answer: string | null;
  evidence: EvidenceItem[];
  retrieval_run_id: string | null;
  cost_usd: number;
  wall_seconds: number;
}
