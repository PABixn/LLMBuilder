export function stripGeneratedUploadPrefix(value: string): string {
  const trimmed = value.trim();
  const separatorIndex = trimmed.indexOf("-");
  if (separatorIndex <= 0) {
    return trimmed;
  }
  const prefix = trimmed.slice(0, separatorIndex);
  if (!/^[0-9a-f]{12}$/i.test(prefix)) {
    return trimmed;
  }
  const stripped = trimmed.slice(separatorIndex + 1).trim();
  return stripped === "" ? trimmed : stripped;
}

export function fileNameFromPath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? "";
}

export function formatCharCount(value: number | null): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return new Intl.NumberFormat().format(Math.trunc(value));
}
