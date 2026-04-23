import type { TauriInvokeFn } from "../types";

export function getTauriInvoke(): TauriInvokeFn | null {
  if (typeof window === "undefined") {
    return null;
  }

  const tauriWindow = window as Window & {
    __TAURI__?: {
      core?: {
        invoke?: TauriInvokeFn;
      };
    };
    __TAURI_INTERNALS__?: {
      invoke?: TauriInvokeFn;
    };
  };

  return tauriWindow.__TAURI__?.core?.invoke ?? tauriWindow.__TAURI_INTERNALS__?.invoke ?? null;
}

export function triggerBlobDownload(blob: Blob, fileName: string): void {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  link.rel = "noreferrer";
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => {
    URL.revokeObjectURL(objectUrl);
  }, 1_000);
}

export function readStoredValue(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function readStoredJson<T = unknown>(key: string): T | null {
  const raw = readStoredValue(key);
  if (raw === null || raw.trim() === "") {
    return null;
  }
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function readStoredStringArray(key: string): string[] {
  const raw = readStoredJson<unknown[]>(key);
  if (!Array.isArray(raw)) {
    return [];
  }
  const values = raw
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry !== "");
  return Array.from(new Set(values));
}

export function writeStoredValue(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}

export function writeStoredJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}

export function removeStoredValue(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}
