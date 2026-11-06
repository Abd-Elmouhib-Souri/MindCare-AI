// Vite config: the dev server forwards /api/* to the FastAPI server on port 8000,
// so the page and the API look like one site (no CORS problem in development).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
