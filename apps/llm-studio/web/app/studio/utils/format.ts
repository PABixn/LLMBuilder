export function integerInputValue(value: number): number | "" {
  return Number.isFinite(value) ? Math.trunc(value) : "";
}

export function numberInputValue(value: number): number | "" {
  return Number.isFinite(value) ? value : "";
}

export function parseIntegerInput(value: string, fallback: number): number {
  if (value.trim() === "") {
    return 0;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function parseNumberInput(value: string, fallback: number): number {
  if (value.trim() === "") {
    return 0;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function formatTimeAgo(timestamp: number | null): string {
  if (!timestamp) {
    return "Not yet";
  }
  const deltaMs = Date.now() - timestamp;
  if (deltaMs < 3_000) {
    return "just now";
  }
  const seconds = Math.round(deltaMs / 1_000);
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

export function formatCompactCount(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  const digits = amount >= 100 || unitIndex === 0 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(digits)} ${units[unitIndex]}`;
}

export function componentDomIdPrefix(blockIndex: number, componentIndex: number): string {
  return `block-${blockIndex}-component-${componentIndex}`;
}

export function mlpStepDomIdPrefix(
  blockIndex: number,
  componentIndex: number,
  stepIndex: number
): string {
  return `block-${blockIndex}-component-${componentIndex}-mlp-step-${stepIndex}`;
}

export function downloadTextFile(fileName: string, text: string): void {
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.rel = "noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}
