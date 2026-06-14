"use client";

import {
  useEffect,
  useLayoutEffect,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import {
  LEGACY_THEME_STORAGE_KEYS,
  THEME_STORAGE_KEY,
  isThemeMode,
  migrateStoredTheme,
  writeStoredTheme,
  type ThemeMode,
} from "./themeStorage";

export { THEME_STORAGE_KEY, type ThemeMode } from "./themeStorage";
const THEME_CHANGED_EVENT = "llm-studio:theme-change";

function readThemeMode(legacyStorageKeys: readonly string[]): ThemeMode {
  if (typeof window !== "undefined") {
    const storedTheme = migrateStoredTheme(window.localStorage, legacyStorageKeys);
    if (storedTheme !== null) {
      return storedTheme;
    }
  }

  if (typeof document !== "undefined") {
    const documentTheme = document.documentElement.dataset.theme;
    if (isThemeMode(documentTheme)) {
      return documentTheme;
    }
  }

  return "white";
}

function applyThemeMode(theme: ThemeMode, legacyStorageKeys: readonly string[]): void {
  if (typeof document !== "undefined") {
    document.documentElement.dataset.theme = theme;
  }

  if (typeof window !== "undefined") {
    writeStoredTheme(window.localStorage, theme, legacyStorageKeys);

    window.dispatchEvent(
      new CustomEvent<ThemeMode>(THEME_CHANGED_EVENT, {
        detail: theme,
      })
    );
  }
}

export function useThemeMode(options: {
  legacyStorageKeys?: string[];
} = {}): [ThemeMode, Dispatch<SetStateAction<ThemeMode>>] {
  const legacyStorageKeys = Array.from(
    new Set([...LEGACY_THEME_STORAGE_KEYS, ...(options.legacyStorageKeys ?? [])])
  );
  const legacyStorageKeySignature = legacyStorageKeys.join("|");
  const [theme, setThemeState] = useState<ThemeMode>("white");

  useLayoutEffect(() => {
    const nextTheme = readThemeMode(legacyStorageKeys);
    setThemeState((current) => (current === nextTheme ? current : nextTheme));
    applyThemeMode(nextTheme, legacyStorageKeys);
  }, [legacyStorageKeySignature]);

  useEffect(() => {
    function handleStorageChange(event: StorageEvent): void {
      if (
        event.key !== THEME_STORAGE_KEY &&
        !(event.key && legacyStorageKeys.includes(event.key))
      ) {
        return;
      }

      const nextTheme = readThemeMode(legacyStorageKeys);
      setThemeState((current) => (current === nextTheme ? current : nextTheme));
      applyThemeMode(nextTheme, legacyStorageKeys);
    }

    function handleThemeChange(event: Event): void {
      const nextTheme = (event as CustomEvent<ThemeMode>).detail;
      if (!isThemeMode(nextTheme)) {
        return;
      }
      setThemeState((current) => (current === nextTheme ? current : nextTheme));
    }

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener(THEME_CHANGED_EVENT, handleThemeChange);
    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener(THEME_CHANGED_EVENT, handleThemeChange);
    };
  }, [legacyStorageKeySignature]);

  const setTheme: Dispatch<SetStateAction<ThemeMode>> = (value) => {
    setThemeState((current) => {
      const nextTheme = typeof value === "function" ? value(current) : value;
      applyThemeMode(nextTheme, legacyStorageKeys);
      return nextTheme;
    });
  };

  return [theme, setTheme];
}
