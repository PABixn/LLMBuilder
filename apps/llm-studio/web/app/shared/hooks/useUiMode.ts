"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

export type UiMode = "simple" | "expert";

export const UI_MODE_STORAGE_KEY = "llm-studio-ui-mode-v1";
export const UI_MODE_CHANGED_EVENT = "llm-studio:ui-mode-change";

export function isUiMode(value: unknown): value is UiMode {
  return value === "simple" || value === "expert";
}

export function normalizeUiMode(value: unknown, fallback: UiMode = "expert"): UiMode {
  return isUiMode(value) ? value : fallback;
}

export function readStoredUiMode(fallback: UiMode = "expert"): UiMode {
  if (typeof window === "undefined") {
    return fallback;
  }

  try {
    return normalizeUiMode(window.localStorage.getItem(UI_MODE_STORAGE_KEY), fallback);
  } catch {
    return fallback;
  }
}

function applyUiMode(mode: UiMode): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(UI_MODE_STORAGE_KEY, mode);
  } catch {
    // Ignore storage failures in local workspace mode.
  }

  window.dispatchEvent(
    new CustomEvent<UiMode>(UI_MODE_CHANGED_EVENT, {
      detail: mode,
    })
  );
}

export function useUiMode(): [UiMode, Dispatch<SetStateAction<UiMode>>] {
  const [mode, setModeState] = useState<UiMode>("expert");
  const modeRef = useRef<UiMode>("expert");

  useLayoutEffect(() => {
    const nextMode = readStoredUiMode();
    modeRef.current = nextMode;
    setModeState((current) => (current === nextMode ? current : nextMode));
  }, []);

  useEffect(() => {
    function handleStorageChange(event: StorageEvent): void {
      if (event.key !== UI_MODE_STORAGE_KEY) {
        return;
      }
      const nextMode = readStoredUiMode();
      modeRef.current = nextMode;
      setModeState((current) => (current === nextMode ? current : nextMode));
    }

    function handleModeChange(event: Event): void {
      const nextMode = normalizeUiMode((event as CustomEvent<UiMode>).detail);
      modeRef.current = nextMode;
      setModeState((current) => (current === nextMode ? current : nextMode));
    }

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener(UI_MODE_CHANGED_EVENT, handleModeChange);
    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener(UI_MODE_CHANGED_EVENT, handleModeChange);
    };
  }, []);

  const setMode: Dispatch<SetStateAction<UiMode>> = useCallback((value) => {
    const currentMode = modeRef.current;
    const nextMode = normalizeUiMode(
      typeof value === "function" ? value(currentMode) : value,
      currentMode
    );
    modeRef.current = nextMode;
    setModeState((current) => (current === nextMode ? current : nextMode));
    applyUiMode(nextMode);
  }, []);

  return [mode, setMode];
}
