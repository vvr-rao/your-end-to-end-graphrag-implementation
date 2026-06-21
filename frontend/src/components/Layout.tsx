import { Link, NavLink, Outlet } from "react-router-dom";
import { MessageSquare, History, Settings as SettingsIcon } from "lucide-react";

const navItem =
  "px-3 py-2 rounded text-sm flex items-center gap-2 hover:bg-slate-800";
const navActive = "bg-slate-800 text-white";

export function Layout() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-6">
          <Link
            to="/ask"
            className="text-lg font-semibold tracking-tight"
          >
            Your End-to-End GraphRAG
          </Link>
          <nav className="flex items-center gap-1 ml-auto">
            <NavLink
              to="/ask"
              className={({ isActive }) =>
                `${navItem} ${isActive ? navActive : "text-slate-400"}`
              }
            >
              <MessageSquare className="h-4 w-4" /> Ask
            </NavLink>
            <NavLink
              to="/conversations"
              className={({ isActive }) =>
                `${navItem} ${isActive ? navActive : "text-slate-400"}`
              }
            >
              <History className="h-4 w-4" /> History
            </NavLink>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `${navItem} ${isActive ? navActive : "text-slate-400"}`
              }
            >
              <SettingsIcon className="h-4 w-4" /> Settings
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="max-w-5xl mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
