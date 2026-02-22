import { defineConfig } from "vite";
const env = globalThis.process?.env ?? {};
export default defineConfig({
    clearScreen: false,
    server: {
        port: 5173,
        strictPort: true,
    },
    envPrefix: ["VITE_", "TAURI_"],
    build: {
        target: env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
        minify: env.TAURI_ENV_DEBUG ? false : "esbuild",
        sourcemap: Boolean(env.TAURI_ENV_DEBUG),
    },
});
