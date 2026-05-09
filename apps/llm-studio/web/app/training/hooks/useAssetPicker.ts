"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  fetchProjects,
  type ProjectSummary,
} from "../../../lib/api";
import {
  fetchTrainingJobs as fetchTokenizerJobs,
  type TrainingJob as TokenizerTrainingJob,
} from "../../../lib/tokenizerLegacyApi";
import {
  filterPickerProjects,
  filterPickerTokenizerJobs,
} from "../lib/assetPicker";
import type { AssetPickerKind } from "../types";

export function useAssetPicker() {
  const [pickerKind, setPickerKind] = useState<AssetPickerKind | null>(null);
  const [pickerQuery, setPickerQuery] = useState("");
  const [pickerLoading, setPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const [pickerProjects, setPickerProjects] = useState<ProjectSummary[]>([]);
  const [pickerTokenizerJobs, setPickerTokenizerJobs] = useState<TokenizerTrainingJob[]>([]);
  const pickerRequestIdRef = useRef(0);

  const visiblePickerProjects = useMemo(
    () => filterPickerProjects(pickerProjects, pickerQuery),
    [pickerProjects, pickerQuery]
  );

  const visiblePickerTokenizerJobs = useMemo(
    () => filterPickerTokenizerJobs(pickerTokenizerJobs, pickerQuery),
    [pickerQuery, pickerTokenizerJobs]
  );

  const closePicker = useCallback(() => {
    pickerRequestIdRef.current += 1;
    setPickerKind(null);
    setPickerQuery("");
    setPickerError(null);
    setPickerLoading(false);
  }, []);

  const openPicker = useCallback(async (kind: AssetPickerKind) => {
    const requestId = pickerRequestIdRef.current + 1;
    pickerRequestIdRef.current = requestId;

    setPickerKind(kind);
    setPickerQuery("");
    setPickerError(null);
    setPickerLoading(true);

    try {
      if (kind === "project") {
        const projects = await fetchProjects();
        if (pickerRequestIdRef.current !== requestId) {
          return;
        }
        startTransition(() => {
          setPickerProjects(projects);
        });
        return;
      }

      const jobs = await fetchTokenizerJobs();
      if (pickerRequestIdRef.current !== requestId) {
        return;
      }
      startTransition(() => {
        setPickerTokenizerJobs(jobs);
      });
    } catch (error) {
      if (pickerRequestIdRef.current !== requestId) {
        return;
      }
      setPickerError(error instanceof Error ? error.message : "Failed to load workspace assets.");
    } finally {
      if (pickerRequestIdRef.current === requestId) {
        setPickerLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!pickerKind) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closePicker();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [closePicker, pickerKind]);

  return {
    closePicker,
    openPicker,
    pickerError,
    pickerKind,
    pickerLoading,
    pickerQuery,
    setPickerQuery,
    visiblePickerProjects,
    visiblePickerTokenizerJobs,
  };
}
