"use client";

import { startTransition, useEffect, useMemo, useRef, useState } from "react";

import {
  createTrainingJob,
  fetchTrainingCheckpoints,
  fetchTrainingConfigTemplates,
  fetchTrainingJob,
  fetchTrainingJobs,
  fetchTrainingLogs,
  fetchTrainingMetrics,
  fetchTrainingSamples,
  validateTrainingPreflight,
} from "../../../lib/training/jobs";
import type {
  TrainingCheckpointEntry,
  TrainingConfigTemplates,
  TrainingFixSuggestion,
  TrainingJob,
  TrainingLogsResponse,
  TrainingMetricPoint,
  TrainingPreflightResponse,
  TrainingSampleEntry,
} from "../../../lib/training/types";
import { invalidateWorkspaceAssetInventory } from "../../../lib/workspaceAssets";
import {
  SIMPLE_POLL_INTERVAL_MS,
  SIMPLE_RECENT_RUNS_POLL_INTERVAL_MS,
} from "../constants";
import {
  applySafeTrainingFixes,
  buildSimpleTrainingConfig,
  buildSimpleTrainingDataloaderConfig,
} from "../lib/trainingProfiles";
import type {
  SimpleFlowState,
  SimpleTrainingStepState,
} from "../types";

interface UseSimpleTrainingStepOptions {
  flow: SimpleFlowState;
  projectReady: boolean;
  tokenizerReady: boolean;
  updateFlow: (updater: (current: SimpleFlowState) => SimpleFlowState) => void;
}

function shouldPollTrainingRun(job: TrainingJob | null): boolean {
  return job?.status === "pending" || job?.status === "running";
}

function fixSignature(fixes: TrainingFixSuggestion[]): string {
  return JSON.stringify(
    fixes.map((fix) => ({
      code: fix.code,
      path: fix.path,
      value: fix.value ?? null,
    }))
  );
}

export function useSimpleTrainingStep({
  flow,
  projectReady,
  tokenizerReady,
  updateFlow,
}: UseSimpleTrainingStepOptions): SimpleTrainingStepState {
  const [templates, setTemplates] = useState<TrainingConfigTemplates | null>(null);
  const [templatesError, setTemplatesError] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<TrainingPreflightResponse | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [manualPreflightRefreshId, setManualPreflightRefreshId] = useState(0);
  const [launching, setLaunching] = useState(false);
  const [cloudConfirmed, setCloudConfirmed] = useState(false);
  const [appliedFixes, setAppliedFixes] = useState<TrainingFixSuggestion[]>([]);
  const [trainingRun, setTrainingRun] = useState<TrainingJob | null>(null);
  const [recentRuns, setRecentRuns] = useState<TrainingJob[]>([]);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [samples, setSamples] = useState<TrainingSampleEntry[]>([]);
  const [logs, setLogs] = useState<TrainingLogsResponse | null>(null);
  const appliedFixSignatureRef = useRef("");
  const handledManualPreflightIdRef = useRef(0);
  const lastValidatedPreflightKeyRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchTrainingConfigTemplates()
      .then((nextTemplates) => {
        if (!cancelled) {
          startTransition(() => {
            setTemplates(nextTemplates);
            setTemplatesError(null);
          });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setTemplatesError(
            error instanceof Error ? error.message : "Training templates could not be loaded."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const profiledTraining = useMemo(() => {
    if (!templates) {
      return null;
    }
    return buildSimpleTrainingConfig(
      templates.training_config_template,
      flow.trainingProfile,
      flow.targetContextLength,
      preflight?.batch_and_lr_recommendation
    );
  }, [
    flow.targetContextLength,
    flow.trainingProfile,
    preflight?.batch_and_lr_recommendation,
    templates,
  ]);

  const baseDataloaderConfig = useMemo(() => {
    if (!templates) {
      return null;
    }
    return buildSimpleTrainingDataloaderConfig(
      templates.dataloader_config_template,
      flow.datasetSource,
      flow.localTrainFiles
    );
  }, [flow.datasetSource, flow.localTrainFiles, templates]);

  const fixedConfigs = useMemo(() => {
    if (!profiledTraining || !baseDataloaderConfig) {
      return null;
    }
    return applySafeTrainingFixes(
      profiledTraining.config,
      baseDataloaderConfig,
      appliedFixes
    );
  }, [appliedFixes, baseDataloaderConfig, profiledTraining]);

  const trainingConfig = fixedConfigs?.trainingConfig ?? null;
  const dataloaderConfig = fixedConfigs?.dataloaderConfig ?? null;
  const preflightInputKey = useMemo(
    () =>
      JSON.stringify({
        projectId: flow.projectId,
        tokenizerJobId: flow.tokenizerJobId,
        trainingConfig,
        dataloaderConfig,
      }),
    [dataloaderConfig, flow.projectId, flow.tokenizerJobId, trainingConfig]
  );

  useEffect(() => {
    const recommendedFixes = preflight?.recommended_fixes ?? [];
    if (recommendedFixes.length === 0) {
      return;
    }
    const signature = fixSignature(recommendedFixes);
    if (signature === appliedFixSignatureRef.current) {
      return;
    }
    appliedFixSignatureRef.current = signature;
    setAppliedFixes(recommendedFixes);
  }, [preflight?.recommended_fixes]);

  useEffect(() => {
    if (!projectReady || !tokenizerReady || !flow.projectId || !flow.tokenizerJobId || !trainingConfig || !dataloaderConfig) {
      setPreflight(null);
      setPreflightLoading(false);
      setPreflightError(templatesError);
      return;
    }

    if (
      manualPreflightRefreshId === 0 ||
      handledManualPreflightIdRef.current === manualPreflightRefreshId
    ) {
      return;
    }
    handledManualPreflightIdRef.current = manualPreflightRefreshId;

    const controller = new AbortController();
    const requestKey = preflightInputKey;
    setPreflightLoading(true);
    void validateTrainingPreflight(
      {
        project_id: flow.projectId,
        tokenizer_job_id: flow.tokenizerJobId,
        training_config: trainingConfig,
        dataloader_config: dataloaderConfig,
      },
      controller.signal
    )
      .then((result) => {
        startTransition(() => {
          lastValidatedPreflightKeyRef.current = requestKey;
          setPreflight(result);
          setPreflightError(null);
        });
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setPreflight(null);
          setPreflightError(
            error instanceof Error ? error.message : "Preflight validation failed."
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPreflightLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [
    dataloaderConfig,
    flow.projectId,
    flow.tokenizerJobId,
    manualPreflightRefreshId,
    preflightInputKey,
    projectReady,
    templatesError,
    tokenizerReady,
    trainingConfig,
  ]);

  useEffect(() => {
    if (!preflight || lastValidatedPreflightKeyRef.current === preflightInputKey) {
      return;
    }
    setPreflight(null);
    setPreflightError(null);
  }, [preflight, preflightInputKey]);

  useEffect(() => {
    let cancelled = false;
    let intervalId: number | null = null;

    const refresh = async () => {
      try {
        const jobs = await fetchTrainingJobs();
        if (!cancelled) {
          startTransition(() => {
            setRecentRuns(jobs);
          });
        }
      } catch {
        // The active run poll surfaces actionable errors. Recent-run refresh is best effort.
      }
    };

    void refresh();
    intervalId = window.setInterval(() => {
      void refresh();
    }, SIMPLE_RECENT_RUNS_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [flow.projectId, flow.tokenizerJobId, flow.trainingJobId, updateFlow]);

  useEffect(() => {
    if (!flow.trainingJobId) {
      setTrainingRun(null);
      setCheckpoints((current) => (current.length === 0 ? current : []));
      setMetrics((current) => (current.length === 0 ? current : []));
      setSamples((current) => (current.length === 0 ? current : []));
      setLogs(null);
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const poll = async () => {
      try {
        const [job, nextMetrics, nextSamples, nextLogs, nextCheckpoints] = await Promise.all([
          fetchTrainingJob(flow.trainingJobId as string),
          fetchTrainingMetrics(flow.trainingJobId as string),
          fetchTrainingSamples(flow.trainingJobId as string),
          fetchTrainingLogs(flow.trainingJobId as string),
          fetchTrainingCheckpoints(flow.trainingJobId as string),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setTrainingRun(job);
          setMetrics(nextMetrics);
          setSamples(nextSamples);
          setLogs(nextLogs);
          setCheckpoints(nextCheckpoints);
          setRecentRuns((current) => [job, ...current.filter((item) => item.id !== job.id)].slice(0, 12));
        });
        if (job.status === "completed") {
          updateFlow((current) => ({
            ...current,
            trainingJobId: job.id,
            lastCompletedStep: "training",
          }));
          invalidateWorkspaceAssetInventory();
        }
        if (shouldPollTrainingRun(job)) {
          timeoutId = window.setTimeout(() => {
            void poll();
          }, SIMPLE_POLL_INTERVAL_MS);
        }
      } catch (error) {
        if (!cancelled) {
          setPreflightError(
            error instanceof Error ? error.message : "Training run could not be refreshed."
          );
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
  }, [flow.trainingJobId, updateFlow]);

  const runPreflight = () => {
    setManualPreflightRefreshId((current) => current + 1);
  };

  const startTraining = async () => {
    if (!flow.projectId || !flow.tokenizerJobId || !trainingConfig || !dataloaderConfig || !preflight?.valid) {
      setPreflightError("Fix preflight blockers before starting training.");
      return;
    }
    if (flow.executionKind === "runpod_pod" && !cloudConfirmed) {
      setPreflightError("Confirm cloud execution before starting a RunPod job.");
      return;
    }

    setLaunching(true);
    setPreflightError(null);
    try {
      const job = await createTrainingJob({
        name: `${flow.modelName.trim() || "Simple model"} ${flow.trainingProfile}`,
        project_id: flow.projectId,
        tokenizer_job_id: flow.tokenizerJobId,
        training_config: trainingConfig,
        dataloader_config: dataloaderConfig,
        execution_target: {
          kind: flow.executionKind,
        },
      });
      startTransition(() => {
        setTrainingRun(job);
        setRecentRuns((current) => [job, ...current.filter((item) => item.id !== job.id)]);
        updateFlow((current) => ({
          ...current,
          trainingJobId: job.id,
        }));
      });
      invalidateWorkspaceAssetInventory();
    } catch (error) {
      setPreflightError(error instanceof Error ? error.message : "Could not start training.");
    } finally {
      setLaunching(false);
    }
  };

  return {
    trainingRun,
    recentRuns,
    checkpoints,
    metrics,
    samples,
    logs,
    trainingConfig,
    dataloaderConfig,
    preflight,
    preflightError,
    preflightLoading,
    launching,
    appliedFixes: fixedConfigs?.labels ?? [],
    profileNote: profiledTraining?.note ?? "Loading backend training defaults.",
    cloudConfirmed,
    setCloudConfirmed,
    runPreflight,
    startTraining,
  };
}
