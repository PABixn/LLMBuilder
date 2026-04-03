"use client";

import {
  useEffect,
  useLayoutEffect,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

export type ThemeMode = "white" | "dark";

export const THEME_STORAGE_KEY = "llm-studio-theme";
const THEME_CHANGED_EVENT = "llm-studio:theme-change";

function isThemeMode(value: unknown): value is ThemeMode {
  return value === "dark" || value === "white";
}

function readThemeMode(legacyStorageKeys: string[]): ThemeMode {
  if (typeof document !== "undefined") {
    const documentTheme = document.documentElement.dataset.theme;
    if (isThemeMode(documentTheme)) {
      return documentTheme;
    }
  }

  if (typeof window !== "undefined") {
    for (const key of [THEME_STORAGE_KEY, ...legacyStorageKeys]) {
      try {
        const raw = window.localStorage.getItem(key);
        if (isThemeMode(raw)) {
          return raw;
        }
      } catch {
        // Ignore local storage failures in local workspace mode.
      }
    }
  }

  return "white";
}

function applyThemeMode(theme: ThemeMode): void {
  if (typeof document !== "undefined") {
    document.documentElement.dataset.theme = theme;
  }

  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore local storage failures in local workspace mode.
    }

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
  const legacyStorageKeys = options.legacyStorageKeys ?? [];
  const legacyStorageKeySignature = legacyStorageKeys.join("|");
  const [theme, setThemeState] = useState<ThemeMode>("white");

  useLayoutEffect(() => {
    const nextTheme = readThemeMode(legacyStorageKeys);
    setThemeState((current) => (current === nextTheme ? current : nextTheme));
    applyThemeMode(nextTheme);
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
      applyThemeMode(nextTheme);
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
      applyThemeMode(nextTheme);
      return nextTheme;
    });
  };

  return [theme, setTheme];
}
