import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// During development (npm run dev) /api and /ws are proxied to the Python backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
