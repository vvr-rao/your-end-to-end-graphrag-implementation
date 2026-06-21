import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AuthGate } from "./components/AuthGate";
import { AskPage } from "./pages/AskPage";
import { ConversationsPage } from "./pages/ConversationsPage";
import { ConversationPage } from "./pages/ConversationPage";
import { SettingsPage } from "./pages/SettingsPage";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/ask" replace />} />
        <Route
          path="/ask"
          element={
            <AuthGate>
              <AskPage />
            </AuthGate>
          }
        />
        <Route
          path="/conversations"
          element={
            <AuthGate>
              <ConversationsPage />
            </AuthGate>
          }
        />
        <Route
          path="/conversations/:iri"
          element={
            <AuthGate>
              <ConversationPage />
            </AuthGate>
          }
        />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/ask" replace />} />
      </Route>
    </Routes>
  );
}

export default App;
