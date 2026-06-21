import { Navigate, useLocation } from "react-router-dom";
import { getBearer } from "../api/client";

/** Redirects to /settings if no bearer token is stored.
 * Wrap protected routes in `<AuthGate>{children}</AuthGate>`. */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const bearer = getBearer();
  if (!bearer) {
    return <Navigate to="/settings" state={{ from: location.pathname }} replace />;
  }
  return <>{children}</>;
}
