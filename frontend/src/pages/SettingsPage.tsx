import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, XCircle } from "lucide-react";
import {
  LS_API_BASE,
  LS_BEARER,
  api,
  getApiBase,
  getBearer,
} from "../api/client";

/** Single-user auth: paste the BEARER_TOKEN from the backend env into
 * the form. Stored in localStorage. */
export function SettingsPage() {
  const navigate = useNavigate();
  const [bearer, setBearer] = useState(getBearer());
  const [apiBase, setApiBase] = useState(getApiBase());
  const [status, setStatus] = useState<
    { ok: boolean; message: string } | null
  >(null);

  const save = () => {
    if (bearer.trim()) {
      localStorage.setItem(LS_BEARER, bearer.trim());
    } else {
      localStorage.removeItem(LS_BEARER);
    }
    if (apiBase.trim()) {
      localStorage.setItem(LS_API_BASE, apiBase.trim());
    } else {
      localStorage.removeItem(LS_API_BASE);
    }
    setStatus({ ok: true, message: "Saved." });
  };

  const test = async () => {
    setStatus({ ok: true, message: "Pinging /health..." });
    try {
      const out = await api.health();
      setStatus({
        ok: out.status === "ok",
        message:
          out.status === "ok"
            ? "Backend reachable + token accepted."
            : `Backend responded with status=${out.status}`,
      });
    } catch (e) {
      setStatus({
        ok: false,
        message: (e as Error).message,
      });
    }
  };

  return (
    <div className="max-w-xl space-y-6 bg-white border border-stone-200 rounded-lg p-6 shadow-sm">
      <header>
        <h1 className="text-xl font-semibold text-stone-900">Settings</h1>
        <p className="text-sm text-stone-600 mt-1">
          The browser keeps these values in <code>localStorage</code>. Anyone
          with this device can use the token — single-user only.
        </p>
      </header>

      <div className="space-y-2">
        <label className="text-sm font-medium text-stone-800">
          API base URL
        </label>
        <input
          type="text"
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
          placeholder="http://localhost:8000"
          className="w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-stone-800">
          Bearer token
        </label>
        <input
          type="password"
          value={bearer}
          onChange={(e) => setBearer(e.target.value)}
          placeholder="paste the BEARER_TOKEN from .env / Render dashboard"
          className="w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm font-mono text-stone-900"
        />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={save}
          className="rounded bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-sm font-medium text-white"
        >
          Save
        </button>
        <button
          onClick={test}
          className="rounded border border-stone-300 hover:bg-stone-100 px-4 py-2 text-sm text-stone-800"
        >
          Test connection
        </button>
        <button
          onClick={() => navigate("/ask")}
          className="rounded border border-stone-300 hover:bg-stone-100 px-4 py-2 text-sm text-stone-800 ml-auto"
        >
          Done →
        </button>
      </div>

      {status && (
        <div
          className={`flex items-center gap-2 text-sm ${
            status.ok ? "text-emerald-700" : "text-rose-700"
          }`}
        >
          {status.ok ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <XCircle className="h-4 w-4" />
          )}
          {status.message}
        </div>
      )}
    </div>
  );
}
