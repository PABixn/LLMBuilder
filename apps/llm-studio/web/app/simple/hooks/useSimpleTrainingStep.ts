"use client";

import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  TrainingBatchLrRecommendation,
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
  applySimpleTrainingProfileGuardrails,
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

function recommendationSignature(
  recommendation: TrainingBatchLrRecommendation | null | undefined
): string {
  if (!recommendation) {
    return "";
  }
  return JSON.stringify({
    recommended_option_key: recommendation.recommended_option_key,
    signals: {
      total_parameters: recommendation.signals.total_parameters,
      parameter_scaled_run_token_target: recommendation.signals.parameter_scaled_run_token_target,
      recommended_run_token_budget: recommendation.signals.recommended_run_token_budget,
      recommended_batch_target: recommendation.signals.recommended_batch_target,
      max_memory_micro_batch_size: recommendation.signals.max_memory_micro_batch_size,
    },
    options: recommendation.options.map((option) => ({
      key: option.key,
      total_batch_size: option.total_batch_size,
      learning_rate: option.learning_rate,
      recommended_max_steps: option.recommended_max_steps,
      clear_manual_micro_batch: option.clear_manual_micro_batch,
    })),
  });
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
  const [batchRecommendation, setBatchRecommendation] =
    useState<TrainingBatchLrRecommendation | null>(null);
  const [appliedFixes, setAppliedFixes] = useState<TrainingFixSuggestion[]>([]);
  const [trainingRun, setTrainingRun] = useState<TrainingJob | null>(null);
  const [recentRuns, setRecentRuns] = useState<TrainingJob[]>([]);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [samples, setSamples] = useState<TrainingSampleEntry[]>([]);
  const [logs, setLogs] = useState<TrainingLogsResponse | null>(null);
  const appliedFixSignatureRef = useRef("");
  const appliedRecommendationSignatureRef = useRef("");
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
      batchRecommendation
    );
  }, [
    batchRecommendation,
    flow.targetContextLength,
    flow.trainingProfile,
    templates,
  ]);

  const baseDataloaderConfig = useMemo(() => {
    if (!templates) {
      return null;
    }
    return buildSimpleTrainingDataloaderConfig(
      templates.dataloader_config_template,
      flow.datasetSource,
      flow.localTrainFiles,
      flow.streamingPrimaryDatasetId,
      flow.streamingAdditionalDatasetIds
    );
  }, [
    flow.datasetSource,
    flow.localTrainFiles,
    flow.streamingAdditionalDatasetIds,
    flow.streamingPrimaryDatasetId,
    templates,
  ]);

  const fixedConfigs = useMemo(() => {
    if (!profiledTraining || !baseDataloaderConfig) {
      return null;
    }
    const fixed = applySafeTrainingFixes(
      profiledTraining.config,
      baseDataloaderConfig,
      appliedFixes
    );
    return {
      ...fixed,
      trainingConfig: applySimpleTrainingProfileGuardrails(
        fixed.trainingConfig,
        flow.trainingProfile,
        profiledTraining.appliedRecommendation,
        profiledTraining.targetRunTokens,
        profiledTraining.targetTotalBatchTokens
      ),
    };
  }, [appliedFixes, baseDataloaderConfig, flow.trainingProfile, profiledTraining]);

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

  const preflightBlocker = useMemo(() => {
    if (templatesError) {
      return templatesError;
    }
    if (!projectReady || !flow.projectId) {
      return "Create an architecture first.";
    }
    if (!tokenizerReady || !flow.tokenizerJobId) {
      return "Train a tokenizer first.";
    }
    if (!trainingConfig || !dataloaderConfig) {
      return templates
        ? "Training settings are still being prepared."
        : "Loading backend training defaults.";
    }
    return null;
  }, [
    dataloaderConfig,
    flow.projectId,
    flow.tokenizerJobId,
    projectReady,
    templates,
    templatesError,
    tokenizerReady,
    trainingConfig,
  ]);

  useEffect(() => {
    const recommendedFixes = preflight?.recommended_fixes ?? [];
    const signature = fixSignature(recommendedFixes);
    if (signature === appliedFixSignatureRef.current) {
      return;
    }
    appliedFixSignatureRef.current = signature;
    setAppliedFixes(recommendedFixes);
  }, [preflight?.recommended_fixes]);

  useEffect(() => {
    if (!preflight) {
      return;
    }
    const signature = recommendationSignature(preflight.batch_and_lr_recommendation);
    if (signature === appliedRecommendationSignatureRef.current) {
      return;
    }
    appliedRecommendationSignatureRef.current = signature;
    setBatchRecommendation(preflight.batch_and_lr_recommendation);
  }, [preflight]);

  const runTrainingPreflight = useCallback(
    async (signal?: AbortSignal): Promise<TrainingPreflightResponse | null> => {
      if (preflightBlocker) {
        setPreflight(null);
        setPreflightError(preflightBlocker);
        return null;
      }
      if (!flow.projectId || !flow.tokenizerJobId || !trainingConfig || !dataloaderConfig) {
        return null;
      }

      const requestKey = preflightInputKey;
      setPreflightLoading(true);
      setPreflightError(null);
      try {
        const result = await validateTrainingPreflight(
          {
            project_id: flow.projectId,
            tokenizer_job_id: flow.tokenizerJobId,
            training_config: trainingConfig,
            dataloader_config: dataloaderConfig,
          },
          signal
        );
        if (!signal?.aborted) {
          startTransition(() => {
            lastValidatedPreflightKeyRef.current = requestKey;
            setPreflight(result);
            setPreflightError(null);
          });
        }
        return result;
      } catch (error) {
        if (!signal?.aborted) {
          setPreflight(null);
          setPreflightError(
            error instanceof Error ? error.message : "Preflight validation failed."
          );
        }
        return null;
      } finally {
        if (!signal?.aborted) {
          setPreflightLoading(false);
        }
      }
    },
    [
      dataloaderConfig,
      flow.projectId,
      flow.tokenizerJobId,
      preflightBlocker,
      preflightInputKey,
      trainingConfig,
    ]
  );

  useEffect(() => {
    if (preflightBlocker) {
      setPreflight(null);
      setPreflightLoading(false);
      setPreflightError(templatesError);
      return;
    }

    const manualRefreshRequested =
      handledManualPreflightIdRef.current !== manualPreflightRefreshId;
    if (!manualRefreshRequested && lastValidatedPreflightKeyRef.current === preflightInputKey) {
      return;
    }
    if (manualRefreshRequested) {
      handledManualPreflightIdRef.current = manualPreflightRefreshId;
    }

    let timeoutId: number | null = null;
    const controller = new AbortController();
    timeoutId = window.setTimeout(
      () => {
        void runTrainingPreflight(controller.signal);
      },
      manualRefreshRequested ? 0 : 350
    );

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
      controller.abort();
    };
  }, [
    manualPreflightRefreshId,
    preflightBlocker,
    preflightInputKey,
    runTrainingPreflight,
    templatesError,
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
    if (preflightBlocker) {
      setPreflightError(preflightBlocker);
      return;
    }
    setManualPreflightRefreshId((current) => current + 1);
  };

  const startTraining = async () => {
    if (preflightBlocker || !flow.projectId || !flow.tokenizerJobId || !trainingConfig || !dataloaderConfig) {
      setPreflightError(preflightBlocker ?? "Training settings are still being prepared.");
      return;
    }
    if (!preflight?.valid || lastValidatedPreflightKeyRef.current !== preflightInputKey) {
      setPreflightError("Fix preflight blockers before starting training.");
      setManualPreflightRefreshId((current) => current + 1);
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
