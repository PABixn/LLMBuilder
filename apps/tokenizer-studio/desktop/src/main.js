import { invoke } from "@tauri-apps/api/core";
const statusEl = document.querySelector("#status");
const detailsEl = document.querySelector("#details");
const errorEl = document.querySelector("#error");
const retryButton = document.querySelector("#retry");
function setStatus(message) {
    if (statusEl) {
        statusEl.textContent = message;
    }
}
function setDetails(message) {
    if (detailsEl) {
        detailsEl.textContent = message;
    }
}
function setError(message) {
    if (!errorEl) {
        return;
    }
    errorEl.textContent = message;
    errorEl.hidden = false;
}
function clearError() {
    if (!errorEl) {
        return;
    }
    errorEl.textContent = "";
    errorEl.hidden = true;
}
async function boot() {
    clearError();
    if (retryButton) {
        retryButton.hidden = true;
        retryButton.disabled = true;
    }
    setStatus("Starting local API sidecar...");
    setDetails("Preparing runtime and health checks.");
    try {
        const result = await invoke("start_backend");
        setStatus("Opening Tokenizer Studio...");
        setDetails(`Runtime: ${result.runtime_dir}`);
        window.location.replace(result.base_url);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : typeof error === "string" ? error : JSON.stringify(error);
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
