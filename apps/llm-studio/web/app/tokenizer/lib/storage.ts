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

export function migrateStoredValues(
  migrations: ReadonlyArray<{
    currentKey: string;
    legacyKey: string;
  }>
): void {
  for (const { currentKey, legacyKey } of migrations) {
    try {
      const current = window.localStorage.getItem(currentKey);
      if (current !== null) {
        window.localStorage.removeItem(legacyKey);
        continue;
      }
      const legacy = window.localStorage.getItem(legacyKey);
      if (legacy === null) {
        continue;
      }
      window.localStorage.setItem(currentKey, legacy);
      window.localStorage.removeItem(legacyKey);
    } catch {
      // Preserve legacy data when migration cannot complete safely.
    }
  }
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
