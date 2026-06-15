export type ThemeMode = "white" | "dark";

export const THEME_STORAGE_KEY = "llm-studio-theme";
export const LEGACY_THEME_STORAGE_KEYS = ["tokenizer-studio-theme"] as const;

type ThemeStorage = Pick<Storage, "getItem" | "removeItem" | "setItem">;

export function isThemeMode(value: unknown): value is ThemeMode {
  return value === "dark" || value === "white";
}

function removeLegacyThemeValues(storage: ThemeStorage, legacyStorageKeys: readonly string[]): void {
  for (const key of legacyStorageKeys) {
    try {
      storage.removeItem(key);
    } catch {
      // The current value is already durable; stale legacy values are harmless.
    }
  }
}

export function migrateStoredTheme(
  storage: ThemeStorage,
  legacyStorageKeys: readonly string[] = LEGACY_THEME_STORAGE_KEYS
): ThemeMode | null {
  let current: string | null;
  try {
    current = storage.getItem(THEME_STORAGE_KEY);
  } catch {
    return null;
  }

  if (isThemeMode(current)) {
    removeLegacyThemeValues(storage, legacyStorageKeys);
    return current;
  }

  for (const legacyKey of legacyStorageKeys) {
    let legacy: string | null;
    try {
      legacy = storage.getItem(legacyKey);
    } catch {
      continue;
    }
    if (!isThemeMode(legacy)) {
      continue;
    }

    try {
      storage.setItem(THEME_STORAGE_KEY, legacy);
    } catch {
      return legacy;
    }
    removeLegacyThemeValues(storage, legacyStorageKeys);
    return legacy;
  }

  return null;
}

export function writeStoredTheme(
  storage: ThemeStorage,
  theme: ThemeMode,
  legacyStorageKeys: readonly string[] = LEGACY_THEME_STORAGE_KEYS
): boolean {
  try {
    storage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    return false;
  }
  removeLegacyThemeValues(storage, legacyStorageKeys);
  return true;
}

const THEME_STORAGE_KEY_LITERAL = JSON.stringify(THEME_STORAGE_KEY);
const LEGACY_THEME_STORAGE_KEYS_LITERAL = JSON.stringify(LEGACY_THEME_STORAGE_KEYS);

export const THEME_BOOTSTRAP_SCRIPT =
  `(function(){try{var s=localStorage,k=${THEME_STORAGE_KEY_LITERAL},l=${LEGACY_THEME_STORAGE_KEYS_LITERAL},` +
  `t=s.getItem(k),p=t==="dark"||t==="white",i,v;if(!p){for(i=0;i<l.length;i++){v=s.getItem(l[i]);` +
  `if(v==="dark"||v==="white"){t=v;try{s.setItem(k,t);p=true;}catch(e){}break;}}}` +
  `if(t==="dark"||t==="white"){if(p){for(i=0;i<l.length;i++){try{s.removeItem(l[i]);}catch(e){}}}` +
  `document.documentElement.dataset.theme=t;}}catch(e){}})();`;
