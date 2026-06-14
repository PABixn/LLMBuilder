import {
  revealDesktopApiArtifact,
  saveDesktopApiArtifact,
  saveDesktopFile,
} from "./desktopBridge";
import { getRuntimeConfig, runtimeRequest } from "./runtimeConfig";

export async function downloadApiArtifact(
  path: string,
  suggestedName: string
): Promise<"native" | "browser" | "cancelled"> {
  if (getRuntimeConfig().environment === "desktop") {
    const saved = await saveDesktopApiArtifact(path, suggestedName);
    return saved ? "native" : "cancelled";
  }
  const response = await runtimeRequest(path, { method: "GET" });
  if (!response.ok) {
    throw new Error(await readDownloadError(response));
  }
  return saveBlob(await response.blob(), suggestedName);
}

export async function revealApiArtifact(path: string): Promise<void> {
  const config = getRuntimeConfig();
  if (config.environment !== "desktop" || !config.capabilities.reveal_artifact) {
    throw new Error("Artifact reveal is available only in the desktop app.");
  }
  await revealDesktopApiArtifact(path);
}

export async function downloadTextFile(
  suggestedName: string,
  text: string,
  type = "application/json;charset=utf-8"
): Promise<"native" | "browser" | "cancelled"> {
  return saveBlob(new Blob([text], { type }), suggestedName);
}

export async function saveBlob(
  blob: Blob,
  suggestedName: string
): Promise<"native" | "browser" | "cancelled"> {
  if (getRuntimeConfig().environment === "desktop") {
    const saved = await saveDesktopFile(
      suggestedName,
      new Uint8Array(await blob.arrayBuffer())
    );
    return saved ? "native" : "cancelled";
  }

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = suggestedName;
  anchor.rel = "noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  return "browser";
}

async function readDownloadError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
  } catch {
    // Keep the status-based fallback.
  }
  return `Download failed (${response.status}).`;
}
