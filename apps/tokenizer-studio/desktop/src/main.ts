import { invoke } from "@tauri-apps/api/core";

interface StartupResponse {
  base_url: string;
  health_url: string;
  runtime_dir: string;
  log_file: string;
}

const statusEl = document.querySelector<HTMLParagraphElement>("#status");
const detailsEl = document.querySelector<HTMLParagraphElement>("#details");
const errorEl = document.querySelector<HTMLElement>("#error");
const retryButton = document.querySelector<HTMLButtonElement>("#retry");

function setStatus(message: string): void {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function setDetails(message: string): void {
  if (detailsEl) {
    detailsEl.textContent = message;
  }
}

function setError(message: string): void {
  if (!errorEl) {
    return;
  }
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError(): void {
  if (!errorEl) {
    return;
  }
  errorEl.textContent = "";
  errorEl.hidden = true;
}

async function boot(): Promise<void> {
  clearError();
  if (retryButton) {
    retryButton.hidden = true;
    retryButton.disabled = true;
  }

  setStatus("Starting local API sidecar...");
  setDetails("Preparing runtime and health checks.");

  try {
    const result = await invoke<StartupResponse>("start_backend");
    setStatus("Opening Tokenizer Studio...");
    setDetails(`Runtime: ${result.runtime_dir}`);
    window.location.replace(result.base_url);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : typeof error === "string" ? error : JSON.stringify(error);
    setStatus("Startup failed");
    setDetails("Review the log path below and retry startup.");
    setError(message);
    if (retryButton) {
      retryButton.hidden = false;
      retryButton.disabled = false;
    }
  }
}

if (retryButton) {
  retryButton.addEventListener("click", () => {
    void boot();
  });
}

void boot();
