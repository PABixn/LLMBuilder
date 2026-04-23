export function readStoredJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeStoredJson(key: string, value: unknown): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore local storage failures
  }
}

export function cloneRecord<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

export function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
}

export function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function updateAtPath(
  source: Record<string, unknown>,
  path: string[],
  value: unknown
): Record<string, unknown> {
  const next = cloneRecord(source);
  let cursor: Record<string, unknown> = next;
  for (const segment of path.slice(0, -1)) {
    const existing = cursor[segment];
    const child = isRecord(existing) ? cloneRecord(existing) : {};
    cursor[segment] = child;
    cursor = child;
  }
  cursor[path[path.length - 1]] = value;
  return next;
}

export function deleteAtPath(
  source: Record<string, unknown>,
  path: string[]
): Record<string, unknown> {
  const next = cloneRecord(source);
  let cursor: Record<string, unknown> = next;
  for (const segment of path.slice(0, -1)) {
    const existing = cursor[segment];
    if (!isRecord(existing)) {
      return next;
    }
    const child = cloneRecord(existing);
    cursor[segment] = child;
    cursor = child;
  }
  delete cursor[path[path.length - 1]];
  return next;
}
