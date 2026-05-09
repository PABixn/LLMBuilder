"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import { TrainingApiError } from "../../../lib/training/errors";
import {
  fetchTrainingCheckpoints,
  fetchTrainingDataPreview,
  fetchTrainingJob,
  fetchTrainingJobs,
  fetchTrainingLogs,
  fetchTrainingMetrics,
  fetchTrainingSamples,
} from "../../../lib/training/jobs";
import type {
  TrainingCheckpointEntry,
  TrainingDataPreview,
  TrainingJob,
  TrainingMetricPoint,
  TrainingSampleEntry,
} from "../../../lib/training/types";
import {
  POLL_INTERVAL_MS,
  RECENT_RUNS_POLL_INTERVAL_MS,
} from "../constants";
import { shouldPollTrainingRun } from "../lib/display";
import { replaceRunInOrder } from "../lib/run";
import type { ToastLevel } from "../types";

type NotifyTrainingPolling = (level: ToastLevel, title: string, body: string) => void;

interface UseTrainingPollingOptions {
  activeRunId: string | null;
  notify: NotifyTrainingPolling;
  setActiveRunOpen: Dispatch<SetStateAction<boolean>>;
  setActiveRunId: Dispatch<SetStateAction<string | null>>;
}

export function useTrainingPolling({
  activeRunId,
  notify,
  setActiveRunOpen,
  setActiveRunId,
}: UseTrainingPollingOptions) {
  const [activeRun, setActiveRun] = useState<TrainingJob | null>(null);
  const [recentRuns, setRecentRuns] = useState<TrainingJob[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [samples, setSamples] = useState<TrainingSampleEntry[]>([]);
  const [logs, setLogs] = useState<{ stdout: string[]; stderr: string[] }>({
    stdout: [],
    stderr: [],
  });
  const [dataPreview, setDataPreview] = useState<TrainingDataPreview | null>(null);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const recentRunsRequestPendingRef = useRef(false);

  const refreshRecentRuns = useCallback(async () => {
    if (recentRunsRequestPendingRef.current) {
      return;
    }
    recentRunsRequestPendingRef.current = true;
    try {
      const jobs = await fetchTrainingJobs();
      startTransition(() => {
        setRecentRuns(jobs);
        setActiveRunId((current) => current ?? jobs[0]?.id ?? null);
      });
    } catch (error) {
      notify("error", "Recent runs unavailable", error instanceof Error ? error.message : "Failed to load training jobs.");
    } finally {
      recentRunsRequestPendingRef.current = false;
    }
  }, [notify, setActiveRunId]);

  useEffect(() => {
    if (!activeRunId) {
      setActiveRun(null);
      setMetrics([]);
      setSamples([]);
      setLogs({ stdout: [], stderr: [] });
      setDataPreview(null);
      setCheckpoints([]);
      return;
    }

    let cancelled = false;
    let pollInFlight = false;
    let timeoutId: number | null = null;

    const poll = async () => {
      if (cancelled || pollInFlight) {
        return;
      }
      pollInFlight = true;
      let shouldPollAgain = true;
      try {
        const [job, fetchedMetrics, fetchedSamples, fetchedLogs, fetchedDataPreview, fetchedCheckpoints] = await Promise.all([
          fetchTrainingJob(activeRunId),
          fetchTrainingMetrics(activeRunId),
          fetchTrainingSamples(activeRunId),
          fetchTrainingLogs(activeRunId),
          fetchTrainingDataPreview(activeRunId).catch((error) => {
            if (error instanceof TrainingApiError && error.status === 404) {
              return null;
            }
            throw error;
          }),
          fetchTrainingCheckpoints(activeRunId),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setActiveRun(job);
          setMetrics(fetchedMetrics);
          setSamples(fetchedSamples);
          setLogs({
            stdout: fetchedLogs.stdout_lines,
            stderr: fetchedLogs.stderr_lines,
          });
          setDataPreview(fetchedDataPreview);
          setCheckpoints(fetchedCheckpoints);
          setRecentRuns((current) => replaceRunInOrder(current, job).slice(0, 12));
        });
        shouldPollAgain = shouldPollTrainingRun(job);
      } catch (error) {
        if (!cancelled) {
          if (error instanceof TrainingApiError && error.status === 404) {
            startTransition(() => {
              setActiveRunId(null);
              setActiveRunOpen(false);
              setActiveRun(null);
              setMetrics([]);
              setSamples([]);
              setLogs({ stdout: [], stderr: [] });
              setDataPreview(null);
              setCheckpoints([]);
              setRecentRuns((current) => current.filter((item) => item.id !== activeRunId));
            });
            shouldPollAgain = false;
            return;
          }
          notify("error", "Run polling interrupted", error instanceof Error ? error.message : "Failed to refresh the active run.");
        }
      } finally {
        pollInFlight = false;
        if (!cancelled && shouldPollAgain) {
          timeoutId = window.setTimeout(() => {
            void poll();
          }, POLL_INTERVAL_MS);
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [activeRunId, notify, setActiveRunId, setActiveRunOpen]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshRecentRuns();
    }, RECENT_RUNS_POLL_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, [refreshRecentRuns]);

  return {
    activeRun,
    checkpoints,
    dataPreview,
    logs,
    metrics,
    recentRuns,
    refreshRecentRuns,
    samples,
    setActiveRun,
    setRecentRuns,
  };
}
