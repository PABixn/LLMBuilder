export type DesktopRuntimeBootstrap = {
  environment: "desktop";
  api_base_url: string;
  runtime_token: string;
  capabilities: {
    native_save: boolean;
    open_logs: boolean;
    open_data: boolean;
    reveal_artifact: boolean;
    diagnostics_export: boolean;
  };
  versions: Record<string, string>;
};

type TauriInvoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
type TauriUnlisten = () => void;
type TauriListen = <T>(
  event: string,
  handler: (event: { payload: T }) => void
) => Promise<TauriUnlisten>;

export type DesktopCloseRequest = {
  active_jobs: {
    active: boolean;
    tokenizer_jobs: unknown[];
    training_jobs: unknown[];
    has_active_local_training: boolean;
    has_active_runpod_training: boolean;
  };
  message: string;
};

export type DesktopStartupProgress = {
  stage: string;
  message: string;
};

export type DesktopRuntimeStatus = {
  lifecycle: "starting" | "ready" | "failed" | "stopping" | "stopped";
  last_error: string | null;
  start_attempts: number;
};

declare global {
  interface Window {
    __TAURI__?: {
      core?: { invoke?: TauriInvoke };
      event?: { listen?: TauriListen };
    };
    __TAURI_INTERNALS__?: { invoke?: TauriInvoke };
  }
}

function tauriInvoke(): TauriInvoke | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.__TAURI__?.core?.invoke ?? window.__TAURI_INTERNALS__?.invoke ?? null;
}

export function isDesktopShell(): boolean {
  return tauriInvoke() !== null;
}

async function invokeDesktop<T>(
  command: string,
  args?: Record<string, unknown>
): Promise<T> {
  const invoke = tauriInvoke();
  if (!invoke) {
    throw new Error("Desktop bridge is unavailable.");
  }
  return invoke<T>(command, args);
}

export function bootstrapDesktopRuntime(): Promise<DesktopRuntimeBootstrap> {
  return invokeDesktop<DesktopRuntimeBootstrap>("runtime_bootstrap");
}

export function retryDesktopRuntime(): Promise<DesktopRuntimeBootstrap> {
  return invokeDesktop<DesktopRuntimeBootstrap>("retry_runtime");
}

export function cancelDesktopRuntimeStart(): Promise<void> {
  return invokeDesktop<void>("cancel_runtime_start");
}

export function getDesktopRuntimeStatus(): Promise<DesktopRuntimeStatus> {
  return invokeDesktop<DesktopRuntimeStatus>("runtime_status");
}

export function saveDesktopFile(
  suggestedName: string,
  bytes: Uint8Array
): Promise<string | null> {
  return invokeDesktop<string | null>("save_file", {
    suggested_name: suggestedName,
    bytes: Array.from(bytes),
  });
}

export function saveDesktopApiArtifact(
  apiPath: string,
  suggestedName: string
): Promise<string | null> {
  return invokeDesktop<string | null>("save_api_artifact", {
    api_path: apiPath,
    suggested_name: suggestedName,
  });
}

export function revealDesktopApiArtifact(apiPath: string): Promise<void> {
  return invokeDesktop<void>("reveal_api_artifact", {
    api_path: apiPath,
  });
}

export function openDesktopLogs(): Promise<void> {
  return invokeDesktop<void>("open_logs_folder");
}

export function openDesktopData(): Promise<void> {
  return invokeDesktop<void>("open_data_folder");
}

export function exportDesktopDiagnostics(): Promise<string | null> {
  return invokeDesktop<string | null>("export_diagnostics");
}

export function quitDesktopApp(): Promise<void> {
  return invokeDesktop<void>("quit_app");
}

export function stopAndExitDesktopApp(): Promise<void> {
  return invokeDesktop<void>("stop_and_exit");
}

export function listenForDesktopCloseRequest(
  handler: (request: DesktopCloseRequest) => void
): Promise<TauriUnlisten> {
  const listen = typeof window === "undefined" ? null : window.__TAURI__?.event?.listen;
  if (!listen) {
    return Promise.resolve(() => undefined);
  }
  return listen<DesktopCloseRequest>("desktop-close-requested", (event) =>
    handler(event.payload)
  );
}

export function listenForDesktopStartupProgress(
  handler: (progress: DesktopStartupProgress) => void
): Promise<TauriUnlisten> {
  const listen = typeof window === "undefined" ? null : window.__TAURI__?.event?.listen;
  if (!listen) {
    return Promise.resolve(() => undefined);
  }
  return listen<DesktopStartupProgress>("desktop-startup-progress", (event) =>
    handler(event.payload)
  );
}

export function listenForDesktopRetryRequest(
  handler: () => void
): Promise<TauriUnlisten> {
  const listen = typeof window === "undefined" ? null : window.__TAURI__?.event?.listen;
  if (!listen) {
    return Promise.resolve(() => undefined);
  }
  return listen<void>("desktop-retry-runtime-requested", () => handler());
}
