import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

const repositoryRoot = path.resolve(__dirname, "../../..");
const e2eDataRoot = path.join(repositoryRoot, "build/desktop/e2e-data");

export default defineConfig({
  testDir: "./test/e2e",
  outputDir: path.join(repositoryRoot, "build/desktop/playwright/results"),
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { outputFolder: path.join(repositoryRoot, "build/desktop/playwright/report") }]] : "list",
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: "python -m uvicorn app.main:app --app-dir ../api --host 127.0.0.1 --port 8000",
      url: "http://127.0.0.1:8000/api/v1/health",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        ...process.env,
        LLM_STUDIO_DATA_DIR: path.join(e2eDataRoot, "data"),
        LLM_STUDIO_CACHE_DIR: path.join(e2eDataRoot, "cache"),
        LLM_STUDIO_LOG_DIR: path.join(e2eDataRoot, "logs"),
        LLM_STUDIO_SERVE_WEB: "0",
      },
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
});

