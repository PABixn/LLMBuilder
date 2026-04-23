"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { ToastLevel, ToastState } from "../types";

export function useTokenizerToasts() {
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const toastTimeoutsRef = useRef<Record<string, number>>({});

  const removeToast = useCallback((toastId: string) => {
    setToasts((previous) => previous.filter((toast) => toast.id !== toastId));
    const timeoutId = toastTimeoutsRef.current[toastId];
    if (typeof timeoutId === "number") {
      window.clearTimeout(timeoutId);
      delete toastTimeoutsRef.current[toastId];
    }
  }, []);

  const notify = useCallback(
    (level: ToastLevel, message: string, durationMs = 4500) => {
      const id = `toast-${Math.random().toString(36).slice(2, 10)}`;
      const toast: ToastState = { id, level, message, durationMs };
      setToasts((previous) => [...previous, toast]);

      toastTimeoutsRef.current[id] = window.setTimeout(() => {
        removeToast(id);
      }, durationMs);
    },
    [removeToast]
  );

  useEffect(() => {
    return () => {
      Object.values(toastTimeoutsRef.current).forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      toastTimeoutsRef.current = {};
    };
  }, []);

  return {
    toasts,
    notify,
    removeToast,
  };
}
