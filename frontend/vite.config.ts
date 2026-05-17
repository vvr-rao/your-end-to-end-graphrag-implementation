import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_BASE_URL || "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/health": apiTarget,
        "/api": apiTarget,
        "/mcp": apiTarget,
      },
    },
  };
});
