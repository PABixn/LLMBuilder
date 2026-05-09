"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { ToastLevel, ToastState } from "../types";

const TOAST_DURATION_MS = 3600;

export function useTrainingToasts() {
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const timeoutIdsRef = useRef<number[]>([]);

  const notify = useCallback((level: ToastLevel, title: string, body: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts((current) => [...current, { id, level, title, body }]);

    const timeoutId = window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
      timeoutIdsRef.current = timeoutIdsRef.current.filter((currentId) => currentId !== timeoutId);
    }, TOAST_DURATION_MS);
    timeoutIdsRef.current.push(timeoutId);
  }, []);

  useEffect(() => {
    return () => {
      timeoutIdsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
      timeoutIdsRef.current = [];
    };
  }, []);

  return {
    notify,
    toasts,
  };
}
