"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  startTransition,
  type ReactNode,
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  FiActivity,
  FiArchive,
  FiBarChart2,
  FiCheckCircle,
  FiCpu,
  FiLayers,
  FiPlus,
  FiRefreshCw,
  FiSearch,
  FiTrash2,
  FiX,
  FiXCircle,
} from "react-icons/fi";

import {
  fetchProject,
  fetchProjects,
  type ProjectDetail,
  type ProjectSummary,
} from "../../../lib/api";
import { useThemeMode } from "../../../lib/theme";
import {
  formatBytes,
  formatDate,
} from "../../../lib/workspaceAssets";
import {
  deleteTrainingJob,
  fetchTrainingCheckpoints,
  fetchTrainingConfigTemplates,
  fetchTrainingDataPreview,
  TrainingApiError,
  fetchTrainingJob,
  fetchTrainingJobs,
  fetchTrainingLogs,
  fetchTrainingMetrics,
  fetchTrainingSamples,
  stopTrainingJob,
  validateTrainingPreflight,
  type TrainingBatchLrRecommendationOption,
  type TrainingCheckpointEntry,
  type TrainingDataPreview,
  type TrainingFixSuggestion,
  type TrainingJob,
  type TrainingMetricPoint,
  type TrainingPreflightResponse,
  type TrainingSampleEntry,
  createTrainingJob,
} from "../../../lib/trainingApi";
import {
  fetchLocalTrainFileStats,
  fetchTrainingJob as fetchTokenizerJob,
  fetchTrainingJobs as fetchTokenizerJobs,
  type TrainingJob as TokenizerTrainingJob,
  uploadTrainFile,
} from "../../../lib/tokenizerLegacyApi";
import {
  LearningRateSchedulePlanner,
  fitSchedulersToMaxSteps,
} from "./LearningRateSchedulePlanner";
import { TrainingHeroSection } from "./TrainingHeroSection";
import { MetricChart } from "./MetricChart";
import { TrainingStudioNav } from "./TrainingStudioNav";
import { TrainingToastStack } from "./TrainingToastStack";
import {
  TrainingWorkflowSection,
  type TrainingWorkflowStep,
} from "./TrainingWorkflowSection";
import {
  ConfigNumberInput,
  OptionalConfigNumberInput,
} from "../../shared/components/ConfigNumberInput";
import {
  ACTIVE_RUN_STORAGE_KEY,
  DATALOADER_CONFIG_STORAGE_KEY,
  FILTER_OPERATORS,
  POLL_INTERVAL_MS,
  RECENT_RUNS_POLL_INTERVAL_MS,
  TRAINING_CONFIG_STORAGE_KEY,
  TRAINING_SELECTION_STORAGE_KEY,
  WORKFLOW_TARGET_HASH_MAP,
} from "../constants";
import {
  buildDatasetsFromUi,
  hydrateDatasetUiFromConfig,
  makeLocalTrainFileEntry,
  makeStreamingDatasetEntry,
  makeStreamingFilterEntry,
  normalizeLocalTrainFiles,
  normalizeStreamingDatasetWeights,
  sanitizeWeightInput,
} from "../lib/dataset";
import {
  canStopTrainingRun,
  defaultRunName,
  deriveTrainingStepProgress,
  formatDatasetScaleLabel,
  formatIssueLocation,
  formatTrainingElapsed,
  formatTrainingEta,
  issueTone,
  numbersRoughlyEqual,
  prettyJson,
  shouldPollTrainingRun,
  statusTone,
} from "../lib/display";
import {
  formatCharCount,
  stripGeneratedUploadPrefix,
} from "../lib/files";
import { formatInteger, formatMetric } from "../lib/metrics";
import {
  asNumber,
  asRecord,
  asRecordArray,
  asString,
  cloneRecord,
  deleteAtPath,
  readStoredJson,
  updateAtPath,
  writeStoredJson,
} from "../lib/object";
import {
  compactWorkflowMessage,
  formatLearningRate,
  formatStatusLabel,
  replaceRunInOrder,
  samplePromptSummary,
  splitGeneratedSampleText,
} from "../lib/run";
import type {
  AssetPickerKind,
  DatasetSourceMode,
  FilterOperator,
  LocalTrainFileFormState,
  StreamingDatasetFormState,
  StreamingFilterFormState,
  ToastLevel,
  ToastState,
  WorkflowTarget,
} from "../types";

type TrainingAdvisorInfoProps = {
  label: string;
  children: ReactNode;
};

function TrainingAdvisorInfo({ label, children }: TrainingAdvisorInfoProps) {
  return (
    <span className="trainingAdvisorInfo">
      <button
        type="button"
        className="trainingAdvisorInfoTrigger"
        aria-label={label}
        title={label}
      >
        <span aria-hidden="true">i</span>
      </button>
      <span className="trainingAdvisorTooltip" role="tooltip">
        {children}
      </span>
    </span>
  );
}

function formatRelativeDeltaPercent(value: number, baseline: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline <= 0) {
    return "0%";
  }
  return `${Math.max(1, Math.round((Math.abs(value - baseline) / baseline) * 100))}%`;
}

function describeBatchProfileShift(
  selectedBatchSize: number,
  baselineBatchSize: number,
  baselineLabel: string,
  isRecommended: boolean
): string {
  if (isRecommended) {
    return "This is the default step size because it best balances stability, accumulation depth, and throughput for the current run.";
  }
  if (selectedBatchSize < baselineBatchSize) {
    return `This profile keeps the optimizer step ${formatRelativeDeltaPercent(
      selectedBatchSize,
      baselineBatchSize
    )} smaller than ${baselineLabel} so each update is less aggressive and easier to stabilize.`;
  }
  if (selectedBatchSize > baselineBatchSize) {
    return `This profile makes the optimizer step ${formatRelativeDeltaPercent(
      selectedBatchSize,
      baselineBatchSize
    )} larger than ${baselineLabel} to push more tokens through each optimizer update.`;
  }
  return `This profile keeps the same optimizer-step size as ${baselineLabel}; the main difference is how aggressively it uses that step.`;
}

function describeLearningRateProfileShift(
  selectedLearningRate: number,
  baselineLearningRate: number,
  baselineLabel: string,
  isRecommended: boolean,
  peakLearningRate: number | null
): string {
  if (isRecommended) {
    return "This is the default base LR because it is the best fit for the current model scale, data regime, and scheduler.";
  }
  if (selectedLearningRate < baselineLearningRate) {
    return `This profile lowers the base LR by ${formatRelativeDeltaPercent(
      selectedLearningRate,
      baselineLearningRate
    )} versus ${baselineLabel}${
      peakLearningRate ? `, keeping the effective peak near ${formatLearningRate(peakLearningRate)}` : ""
    }.`;
  }
  if (selectedLearningRate > baselineLearningRate) {
    return `This profile raises the base LR by ${formatRelativeDeltaPercent(
      selectedLearningRate,
      baselineLearningRate
    )} versus ${baselineLabel}${
      peakLearningRate ? `, letting the schedule top out near ${formatLearningRate(peakLearningRate)}` : ""
    }.`;
  }
  return `This profile lands on the same canonical LR as ${baselineLabel}, so the main change comes from batch layout rather than LR.`;
}

function describeBatchHardwareContext(
  deviceType: string,
  maxMemoryMicroBatchSize: number,
  microBatchSize: number,
  gradAccumSteps: number
): string {
  return `Preflight fits up to ${formatInteger(
    maxMemoryMicroBatchSize
  )} sequences per micro-step on ${deviceType}; this profile uses micro batch ${formatInteger(
    microBatchSize
  )} with ${formatInteger(gradAccumSteps)} accumulation step${gradAccumSteps === 1 ? "" : "s"}.`;
}

function describeBatchDatasetContext(datasetScale: string): string {
  if (datasetScale === "streaming") {
    return "Streaming-scale data lets the advisor lean more on hardware fit and model scale than on corpus-size caps.";
  }
  if (datasetScale === "mixed") {
    return "Mixed local and streaming data keeps batch sizing more conservative so the local portion is not washed out in each update.";
  }
  if (datasetScale === "tiny_local" || datasetScale === "small_local") {
    return "A small local corpus favors smaller optimizer steps so repeated passes over the same data do not get too sharp.";
  }
  return `${formatDatasetScaleLabel(
    datasetScale
  )} data still benefits from measured step sizing so each update covers a useful slice of the corpus without overreaching.`;
}

function describeLearningRateScheduleContext(
  peakFactor: number,
  peakLearningRate: number | null
): string {
  if (peakFactor > 1.01 && peakLearningRate) {
    return `The current scheduler still lifts this base LR during the run, topping out near ${formatLearningRate(
      peakLearningRate
    )}.`;
  }
  return "The current scheduler keeps the effective LR close to the base value, so the chosen base LR needs to be safe on its own.";
}

function describeLearningRateDatasetContext(datasetScale: string): string {
  if (datasetScale === "streaming") {
    return "With streaming-scale data, LR can stay anchored more to model scale and schedule shape than to corpus-size limits.";
  }
  if (datasetScale === "mixed") {
    return "A mixed data regime usually benefits from a moderate LR so the local data is not overfit while streaming data still moves quickly.";
  }
  if (datasetScale === "tiny_local" || datasetScale === "small_local") {
    return "Smaller local corpora generally favor a lower LR so repeated exposure to the same samples stays stable.";
  }
  return `${formatDatasetScaleLabel(
    datasetScale
  )} data supports a measured LR that still respects repeated passes over the local corpus.`;
}

export function TrainingPageContent() {
  const searchParams = useSearchParams();
  const [theme, setTheme] = useThemeMode();
  const [trainingConfig, setTrainingConfig] = useState<Record<string, unknown> | null>(null);
  const [dataloaderConfig, setDataloaderConfig] = useState<Record<string, unknown> | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedTokenizerJobId, setSelectedTokenizerJobId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<ProjectDetail | null>(null);
  const [selectedTokenizer, setSelectedTokenizer] = useState<TokenizerTrainingJob | null>(null);
  const [runName, setRunName] = useState("");
  const [runNameDirty, setRunNameDirty] = useState(false);
  const [preflight, setPreflight] = useState<TrainingPreflightResponse | null>(null);
  const [selectedRecommendationOptionKey, setSelectedRecommendationOptionKey] = useState<string | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [isActiveRunOpen, setIsActiveRunOpen] = useState(true);
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
  const [launching, setLaunching] = useState(false);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const [pickerKind, setPickerKind] = useState<AssetPickerKind | null>(null);
  const [pickerQuery, setPickerQuery] = useState("");
  const [pickerLoading, setPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const [pickerProjects, setPickerProjects] = useState<ProjectSummary[]>([]);
  const [pickerTokenizerJobs, setPickerTokenizerJobs] = useState<TokenizerTrainingJob[]>([]);
  const [datasetSourceMode, setDatasetSourceMode] = useState<DatasetSourceMode>("streaming_hf");
  const [localTrainFiles, setLocalTrainFiles] = useState<LocalTrainFileFormState[]>([]);
  const [hfToken, setHfToken] = useState("");
  const [streamingDatasets, setStreamingDatasets] = useState<StreamingDatasetFormState[]>([
    makeStreamingDatasetEntry(),
  ]);
  const [isDraggingTrainFiles, setIsDraggingTrainFiles] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isLoadingDatasetTemplate, setIsLoadingDatasetTemplate] = useState(false);
  const [isResettingPrompts, setIsResettingPrompts] = useState(false);
  const [highlightedWorkflowTarget, setHighlightedWorkflowTarget] =
    useState<WorkflowTarget | null>(null);
  const initializedRef = useRef(false);
  const pickerRequestIdRef = useRef(0);
  const recentRunsRequestPendingRef = useRef(false);
  const datasetUiHydratedRef = useRef(false);
  const localFileDragDepthRef = useRef(0);
  const localTrainFileStatsPendingIdsRef = useRef(new Set<string>());
  const localTrainFileStatsFailedIdsRef = useRef(new Set<string>());
  const workflowHighlightTimeoutRef = useRef<number | null>(null);
  const trainingPlanPanelRef = useRef<HTMLDetailsElement | null>(null);
  const datasetPanelRef = useRef<HTMLDetailsElement | null>(null);
  const autoRunNameRef = useRef("");
  const modelSelectionRef = useRef<HTMLDivElement | null>(null);
  const tokenizerSelectionRef = useRef<HTMLDivElement | null>(null);
  const trainingSettingsRef = useRef<HTMLDivElement | null>(null);
  const datasetSettingsRef = useRef<HTMLDivElement | null>(null);
  const preflightSectionRef = useRef<HTMLElement | null>(null);

  const deferredTrainingConfig = useDeferredValue(trainingConfig);
  const deferredDataloaderConfig = useDeferredValue(dataloaderConfig);

  const notify = useCallback((level: ToastLevel, title: string, body: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts((current) => [...current, { id, level, title, body }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 3600);
  }, []);

  useEffect(() => {
    return () => {
      if (workflowHighlightTimeoutRef.current !== null) {
        window.clearTimeout(workflowHighlightTimeoutRef.current);
      }
    };
  }, []);

  const openWorkflowTarget = useCallback((target: WorkflowTarget) => {
    if (target === "training" && trainingPlanPanelRef.current) {
      trainingPlanPanelRef.current.open = true;
    }
    if (target === "dataset" && datasetPanelRef.current) {
      datasetPanelRef.current.open = true;
    }

    const targetRef =
      target === "model"
        ? modelSelectionRef
        : target === "tokenizer"
          ? tokenizerSelectionRef
          : target === "training"
            ? trainingSettingsRef
            : target === "dataset"
              ? datasetSettingsRef
              : preflightSectionRef;

    const hash = WORKFLOW_TARGET_HASH_MAP[target];
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }

    targetRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
      inline: "nearest",
    });

    setHighlightedWorkflowTarget(target);
    if (workflowHighlightTimeoutRef.current !== null) {
      window.clearTimeout(workflowHighlightTimeoutRef.current);
    }
    workflowHighlightTimeoutRef.current = window.setTimeout(() => {
      setHighlightedWorkflowTarget((previous) =>
        previous === target ? null : previous
      );
    }, 1800);
  }, []);

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
  }, [notify]);

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    const storedTraining = readStoredJson<Record<string, unknown> | null>(
      TRAINING_CONFIG_STORAGE_KEY,
      null
    );
    const storedDataloader = readStoredJson<Record<string, unknown> | null>(
      DATALOADER_CONFIG_STORAGE_KEY,
      null
    );
    const storedSelection = readStoredJson<{ projectId: string | null; tokenizerJobId: string | null }>(
      TRAINING_SELECTION_STORAGE_KEY,
      {
        projectId: null,
        tokenizerJobId: null,
      }
    );
    const storedActiveRun = readStoredJson<string | null>(ACTIVE_RUN_STORAGE_KEY, null);

    setActiveRunId(searchParams.get("run") ?? storedActiveRun);
    setSelectedProjectId(searchParams.get("project") ?? storedSelection.projectId);
    setSelectedTokenizerJobId(searchParams.get("tokenizerJob") ?? storedSelection.tokenizerJobId);

    void Promise.all([fetchTrainingConfigTemplates(), fetchTrainingJobs()])
      .then(([templates, jobs]) => {
        startTransition(() => {
          setTrainingConfig(storedTraining ?? templates.training_config_template);
          setDataloaderConfig(storedDataloader ?? templates.dataloader_config_template);
          setRecentRuns(jobs);
          if (!searchParams.get("run") && !storedActiveRun) {
            const nextRun = jobs[0] ?? null;
            if (nextRun) {
              setActiveRunId(nextRun.id);
            }
          }
        });
      })
      .catch((error) => {
        notify(
          "error",
          "Training config unavailable",
          error instanceof Error ? error.message : "Failed to load training defaults."
        );
      });
  }, [notify, searchParams]);

  useEffect(() => {
    writeStoredJson(TRAINING_SELECTION_STORAGE_KEY, {
      projectId: selectedProjectId,
      tokenizerJobId: selectedTokenizerJobId,
    });
  }, [selectedProjectId, selectedTokenizerJobId]);

  useEffect(() => {
    const hash = window.location.hash;
    if (hash === WORKFLOW_TARGET_HASH_MAP.model) {
      openWorkflowTarget("model");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.tokenizer) {
      openWorkflowTarget("tokenizer");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.training) {
      openWorkflowTarget("training");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.dataset) {
      openWorkflowTarget("dataset");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.preflight) {
      openWorkflowTarget("preflight");
    }
  }, [openWorkflowTarget]);

  useEffect(() => {
    writeStoredJson(ACTIVE_RUN_STORAGE_KEY, activeRunId);
  }, [activeRunId]);

  useEffect(() => {
    if (trainingConfig) {
      writeStoredJson(TRAINING_CONFIG_STORAGE_KEY, trainingConfig);
    }
  }, [trainingConfig]);

  useEffect(() => {
    if (dataloaderConfig) {
      writeStoredJson(DATALOADER_CONFIG_STORAGE_KEY, dataloaderConfig);
    }
  }, [dataloaderConfig]);

  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedProject(null);
      return;
    }
    const controller = new AbortController();
    void fetchProject(selectedProjectId, controller.signal)
      .then((project) => {
        startTransition(() => {
          setSelectedProject(project);
        });
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          notify("error", "Model config unavailable", error instanceof Error ? error.message : "Failed to load selected model config.");
        }
      });
    return () => controller.abort();
  }, [notify, selectedProjectId]);

  useEffect(() => {
    if (!selectedTokenizerJobId) {
      setSelectedTokenizer(null);
      return;
    }
    let cancelled = false;
    void fetchTokenizerJob(selectedTokenizerJobId)
      .then((job) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setSelectedTokenizer(job);
        });
      })
      .catch((error) => {
        if (!cancelled) {
          notify("error", "Tokenizer unavailable", error instanceof Error ? error.message : "Failed to load selected tokenizer.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [notify, selectedTokenizerJobId]);

  useEffect(() => {
    if (!selectedProject && !selectedTokenizer) {
      return;
    }
    const nextAutoName = defaultRunName(selectedProject, selectedTokenizer);
    setRunName((current) => {
      const shouldReplace = !runNameDirty || current === autoRunNameRef.current;
      autoRunNameRef.current = nextAutoName;
      return shouldReplace ? nextAutoName : current;
    });
  }, [runNameDirty, selectedProject, selectedTokenizer]);

  useEffect(() => {
    if (!selectedProjectId || !selectedTokenizerJobId || !deferredTrainingConfig || !deferredDataloaderConfig) {
      setPreflight(null);
      setPreflightError(null);
      return;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      setPreflightLoading(true);
      void validateTrainingPreflight(
        {
          project_id: selectedProjectId,
          tokenizer_job_id: selectedTokenizerJobId,
          training_config: deferredTrainingConfig,
          dataloader_config: deferredDataloaderConfig,
        },
        controller.signal
      )
        .then((result) => {
          startTransition(() => {
            setPreflight(result);
            setPreflightError(null);
          });
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            setPreflight(null);
            setPreflightError(error instanceof Error ? error.message : "Failed to validate preflight.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setPreflightLoading(false);
          }
        });
    }, 420);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [deferredDataloaderConfig, deferredTrainingConfig, selectedProjectId, selectedTokenizerJobId]);

  useEffect(() => {
    const recommendation = preflight?.batch_and_lr_recommendation;
    if (!recommendation || recommendation.options.length === 0) {
      setSelectedRecommendationOptionKey(null);
      return;
    }
    setSelectedRecommendationOptionKey((current) => {
      if (current && recommendation.options.some((option) => option.key === current)) {
        return current;
      }
      return recommendation.recommended_option_key;
    });
  }, [preflight]);

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
              setIsActiveRunOpen(false);
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
  }, [activeRunId, notify]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshRecentRuns();
    }, RECENT_RUNS_POLL_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, [refreshRecentRuns]);

  const datasetEntries = useMemo(
    () => asRecordArray(dataloaderConfig?.datasets),
    [dataloaderConfig]
  );

  const promptEntries = useMemo(() => {
    const sampler = asRecord(trainingConfig?.sampler);
    return asRecordArray(sampler.prompts);
  }, [trainingConfig]);

  const normalizedPickerQuery = pickerQuery.trim().toLowerCase();

  const visiblePickerProjects = useMemo(() => {
    return [...pickerProjects]
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .filter((project) => {
        if (normalizedPickerQuery === "") {
          return true;
        }
        return [project.name, project.id, project.artifact_file, project.artifact_path].some(
          (value) =>
            typeof value === "string" && value.toLowerCase().includes(normalizedPickerQuery)
        );
      });
  }, [normalizedPickerQuery, pickerProjects]);

  const visiblePickerTokenizerJobs = useMemo(() => {
    return [...pickerTokenizerJobs]
      .filter((job) => job.status === "completed")
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .filter((job) => {
        if (normalizedPickerQuery === "") {
          return true;
        }
        const tokenizerName = asString(job.tokenizer_config.name, job.id);
        return [tokenizerName, job.id, job.artifact_file, job.artifact_path].some(
          (value) =>
            typeof value === "string" && value.toLowerCase().includes(normalizedPickerQuery)
        );
      });
  }, [normalizedPickerQuery, pickerTokenizerJobs]);

  useEffect(() => {
    if (!dataloaderConfig || datasetUiHydratedRef.current) {
      return;
    }
    const hydrated = hydrateDatasetUiFromConfig(dataloaderConfig);
    setDatasetSourceMode(hydrated.sourceMode);
    setLocalTrainFiles(hydrated.localTrainFiles);
    setHfToken(hydrated.hfToken);
    setStreamingDatasets(hydrated.streamingDatasets);
    datasetUiHydratedRef.current = true;
  }, [dataloaderConfig]);

  useEffect(() => {
    if (!datasetUiHydratedRef.current) {
      return;
    }
    const nextDatasets = buildDatasetsFromUi(
      datasetSourceMode,
      localTrainFiles,
      hfToken,
      streamingDatasets
    );
    setDataloaderConfig((current) => {
      const next = cloneRecord(current ?? {});
      next.datasets = nextDatasets;
      return next;
    });
  }, [datasetSourceMode, hfToken, localTrainFiles, streamingDatasets]);

  useEffect(() => {
    if (!datasetUiHydratedRef.current) {
      return;
    }

    const currentIds = new Set(localTrainFiles.map((entry) => entry.id));
    for (const entryId of Array.from(localTrainFileStatsPendingIdsRef.current)) {
      if (!currentIds.has(entryId)) {
        localTrainFileStatsPendingIdsRef.current.delete(entryId);
      }
    }
    for (const entryId of Array.from(localTrainFileStatsFailedIdsRef.current)) {
      if (!currentIds.has(entryId)) {
        localTrainFileStatsFailedIdsRef.current.delete(entryId);
      }
    }

    const entriesNeedingStats = localTrainFiles.filter((entry) => {
      if (
        typeof entry.sizeBytes === "number" &&
        Number.isFinite(entry.sizeBytes) &&
        entry.sizeBytes >= 0 &&
        typeof entry.sizeChars === "number" &&
        Number.isFinite(entry.sizeChars) &&
        entry.sizeChars >= 0
      ) {
        return false;
      }
      if (entry.filePath.trim() === "") {
        return false;
      }
      if (localTrainFileStatsPendingIdsRef.current.has(entry.id)) {
        return false;
      }
      if (localTrainFileStatsFailedIdsRef.current.has(entry.id)) {
        return false;
      }
      return true;
    });

    if (entriesNeedingStats.length === 0) {
      return;
    }

    entriesNeedingStats.forEach((entry) => {
      localTrainFileStatsPendingIdsRef.current.add(entry.id);
    });

    let cancelled = false;

    void Promise.allSettled(
      entriesNeedingStats.map(async (entry) => ({
        entryId: entry.id,
        stats: await fetchLocalTrainFileStats(entry.filePath),
      }))
    ).then((results) => {
      const updatesById = new Map<string, { sizeBytes: number; sizeChars: number }>();

      results.forEach((result, index) => {
        const entry = entriesNeedingStats[index];
        localTrainFileStatsPendingIdsRef.current.delete(entry.id);

        if (result.status === "fulfilled") {
          localTrainFileStatsFailedIdsRef.current.delete(entry.id);
          updatesById.set(entry.id, {
            sizeBytes: result.value.stats.size_bytes,
            sizeChars: result.value.stats.size_chars,
          });
          return;
        }

        localTrainFileStatsFailedIdsRef.current.add(entry.id);
      });

      if (cancelled || updatesById.size === 0) {
        return;
      }

      setLocalTrainFiles((previous) =>
        normalizeLocalTrainFiles(
          previous.map((entry) => {
            const stats = updatesById.get(entry.id);
            return stats ? { ...entry, ...stats } : entry;
          })
        )
      );
    });

    return () => {
      cancelled = true;
    };
  }, [localTrainFiles]);

  const updateStreamingDataset = useCallback(
    (datasetId: string, updates: Partial<Omit<StreamingDatasetFormState, "id">>) => {
      setStreamingDatasets((previous) =>
        previous.map((entry) => (entry.id === datasetId ? { ...entry, ...updates } : entry))
      );
    },
    []
  );

  const updateStreamingWeight = useCallback((datasetId: string, rawWeight: string) => {
    const sanitizedWeight = sanitizeWeightInput(rawWeight);
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights(previous, datasetId, sanitizedWeight)
    );
  }, []);

  const addStreamingDataset = useCallback(() => {
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights([...previous, makeStreamingDatasetEntry()])
    );
  }, []);

  const removeStreamingDataset = useCallback((datasetId: string) => {
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights(
        previous.filter((entry) => entry.id !== datasetId).length > 0
          ? previous.filter((entry) => entry.id !== datasetId)
          : [makeStreamingDatasetEntry()]
      )
    );
  }, []);

  const updateStreamingFilter = useCallback(
    (
      datasetId: string,
      filterId: string,
      updates: Partial<Omit<StreamingFilterFormState, "id">>
    ) => {
      setStreamingDatasets((previous) =>
        previous.map((entry) => {
          if (entry.id !== datasetId) {
            return entry;
          }
          return {
            ...entry,
            filters: entry.filters.map((filter) =>
              filter.id === filterId ? { ...filter, ...updates } : filter
            ),
          };
        })
      );
    },
    []
  );

  const addStreamingFilter = useCallback((datasetId: string) => {
    setStreamingDatasets((previous) =>
      previous.map((entry) =>
        entry.id === datasetId
          ? { ...entry, filters: [...entry.filters, makeStreamingFilterEntry()] }
          : entry
      )
    );
  }, []);

  const removeStreamingFilter = useCallback((datasetId: string, filterId: string) => {
    setStreamingDatasets((previous) =>
      previous.map((entry) => {
        if (entry.id !== datasetId) {
          return entry;
        }
        return {
          ...entry,
          filters: entry.filters.filter((filter) => filter.id !== filterId),
        };
      })
    );
  }, []);

  const removeLocalTrainFile = useCallback((localFileId: string) => {
    setLocalTrainFiles((previous) => previous.filter((entry) => entry.id !== localFileId));
  }, []);

  const clearLocalTrainFiles = useCallback(() => {
    setLocalTrainFiles([]);
  }, []);

  const uploadLocalTrainFiles = useCallback(
    async (selectedFiles: File[]) => {
      if (selectedFiles.length === 0) {
        return;
      }

      notify(
        "info",
        "Uploading dataset files",
        selectedFiles.length === 1
          ? `Uploading ${selectedFiles[0].name}.`
          : `Uploading ${selectedFiles.length} local train files.`
      );
      setIsUploadingTrainFile(true);

      try {
        const uploadResults = await Promise.allSettled(
          selectedFiles.map((file) => uploadTrainFile(file))
        );

        const successfulUploads = uploadResults
          .filter(
            (
              result
            ): result is PromiseFulfilledResult<Awaited<ReturnType<typeof uploadTrainFile>>> =>
              result.status === "fulfilled"
          )
          .map((result) => result.value);

        if (successfulUploads.length > 0) {
          setLocalTrainFiles((previous) =>
            normalizeLocalTrainFiles([
              ...previous,
              ...successfulUploads.map((uploadedFile) =>
                makeLocalTrainFileEntry({
                  fileName: uploadedFile.file_name,
                  filePath: uploadedFile.file_path,
                  sizeBytes: uploadedFile.size_bytes,
                  sizeChars: uploadedFile.size_chars,
                })
              ),
            ])
          );
          notify(
            "success",
            "Dataset files added",
            successfulUploads.length === 1
              ? `Added ${stripGeneratedUploadPrefix(successfulUploads[0].file_name)}.`
              : `Added ${successfulUploads.length} local train files.`
          );
        }

        const failedUploads = uploadResults.filter(
          (result): result is PromiseRejectedResult => result.status === "rejected"
        );
        if (failedUploads.length > 0) {
          const firstFailure = failedUploads[0];
          const firstFailureMessage =
            firstFailure.reason instanceof Error ? firstFailure.reason.message : "Upload failed";
          notify(
            "error",
            "Dataset upload failed",
            `Failed to upload ${failedUploads.length} file(s). ${firstFailureMessage}`
          );
        }
      } catch (error) {
        notify(
          "error",
          "Dataset upload failed",
          error instanceof Error ? error.message : "Failed to upload local train files."
        );
      } finally {
        setIsUploadingTrainFile(false);
      }
    },
    [notify]
  );

  const handleTrainFilesSelected = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(event.target.files ?? []);
      event.target.value = "";
      await uploadLocalTrainFiles(selectedFiles);
    },
    [uploadLocalTrainFiles]
  );

  const handleLocalTrainFilesDragEnter = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!Array.from(event.dataTransfer.types).includes("Files")) {
      return;
    }
    localFileDragDepthRef.current += 1;
    setIsDraggingTrainFiles(true);
  }, []);

  const handleLocalTrainFilesDragOver = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setIsDraggingTrainFiles(true);
  }, []);

  const handleLocalTrainFilesDragLeave = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    localFileDragDepthRef.current = Math.max(0, localFileDragDepthRef.current - 1);
    if (localFileDragDepthRef.current === 0) {
      setIsDraggingTrainFiles(false);
    }
  }, []);

  const handleLocalTrainFilesDrop = useCallback(
    async (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      localFileDragDepthRef.current = 0;
      setIsDraggingTrainFiles(false);
      const droppedFiles = Array.from(event.dataTransfer.files ?? []);
      await uploadLocalTrainFiles(droppedFiles);
    },
    [uploadLocalTrainFiles]
  );

  const handleLoadStreamingTemplate = useCallback(async () => {
    notify("info", "Loading streaming template", "Refreshing dataset settings from the template.");
    setIsLoadingDatasetTemplate(true);
    try {
      const templates = await fetchTrainingConfigTemplates();
      const templateDataloaderConfig = cloneRecord(
        asRecord(templates.dataloader_config_template)
      );
      const hydrated = hydrateDatasetUiFromConfig(templateDataloaderConfig);
      setDatasetSourceMode("streaming_hf");
      setHfToken(hydrated.hfToken);
      setStreamingDatasets(hydrated.streamingDatasets);
      setDataloaderConfig(templateDataloaderConfig);
      notify("success", "Template loaded", "Loaded streaming dataset defaults.");
    } catch (error) {
      notify(
        "error",
        "Template unavailable",
        error instanceof Error ? error.message : "Failed to load the streaming dataset template."
      );
    } finally {
      setIsLoadingDatasetTemplate(false);
    }
  }, [notify]);

  const handleResetPrompts = useCallback(async () => {
    setIsResettingPrompts(true);
    try {
      const templates = await fetchTrainingConfigTemplates();
      const templateSampler = asRecord(asRecord(templates.training_config_template).sampler);
      const templatePrompts = asRecordArray(templateSampler.prompts).map(cloneRecord);

      setTrainingConfig((current) => {
        const next = cloneRecord(current ?? templates.training_config_template);
        const sampler = asRecord(next.sampler);
        sampler.prompts = templatePrompts;
        next.sampler = sampler;
        return next;
      });
      notify("success", "Prompts reset", "Loaded the template sampling prompts.");
    } catch (error) {
      notify(
        "error",
        "Prompt template unavailable",
        error instanceof Error ? error.message : "Failed to load the template sampling prompts."
      );
    } finally {
      setIsResettingPrompts(false);
    }
  }, [notify]);

  const handleAddPrompt = () => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    const prompts = asRecordArray(sampler.prompts);
    prompts.push({
      prompt: "Hello",
      max_tokens: 64,
      temperature: 0.7,
      top_k: 40,
    });
    sampler.prompts = prompts;
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handlePromptChange = (index: number, field: string, value: unknown) => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    const prompts = asRecordArray(sampler.prompts);
    const prompt = cloneRecord(prompts[index] ?? {});
    prompt[field] = value;
    prompts[index] = prompt;
    sampler.prompts = prompts;
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handleRemovePrompt = (index: number) => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    sampler.prompts = asRecordArray(sampler.prompts).filter((_, currentIndex) => currentIndex !== index);
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handleTrainingField = (path: string[], value: unknown) => {
    setTrainingConfig((current) => updateAtPath(current ?? {}, path, value));
  };

  const handleMaxStepsChange = (value: number) => {
    setTrainingConfig((current) => {
      const next = updateAtPath(current ?? {}, ["max_steps"], value);
      const lrScheduler = asRecord(next.lr_scheduler);
      const schedulers = asRecordArray(lrScheduler.schedulers);
      const baseLearningRate = asNumber(asRecord(next.optimizer).lr, 0.0003);
      next.lr_scheduler = {
        type: "sequential",
        schedulers: fitSchedulersToMaxSteps(schedulers, value, baseLearningRate),
      };
      return next;
    });
  };

  const handleLrSchedulersChange = (schedulers: Record<string, unknown>[]) => {
    handleTrainingField(["lr_scheduler"], {
      type: "sequential",
      schedulers,
    });
  };

  const handleOptionalTrainingField = (path: string[], value: unknown | null) => {
    setTrainingConfig((current) =>
      value === null
        ? deleteAtPath(current ?? {}, path)
        : updateAtPath(current ?? {}, path, value)
    );
  };

  const handleDataloaderField = (path: string[], value: unknown) => {
    setDataloaderConfig((current) => updateAtPath(current ?? {}, path, value));
  };

  const applyFix = (fix: TrainingFixSuggestion) => {
    if (fix.path.startsWith("training_config.")) {
      const path = fix.path.replace("training_config.", "").split(".");
      setTrainingConfig((current) =>
        fix.value === null
          ? deleteAtPath(current ?? {}, path)
          : updateAtPath(current ?? {}, path, cloneRecord(fix.value))
      );
      notify("success", fix.label, fix.description);
      return;
    }
    if (fix.path.startsWith("dataloader_config.")) {
      const path = fix.path.replace("dataloader_config.", "").split(".");
      setDataloaderConfig((current) =>
        fix.value === null
          ? deleteAtPath(current ?? {}, path)
          : updateAtPath(current ?? {}, path, cloneRecord(fix.value))
      );
      notify("success", fix.label, fix.description);
    }
  };

  const applyRecommendationOption = (
    option: TrainingBatchLrRecommendationOption,
    scope: "both" | "batch" | "lr" = "both"
  ) => {
    if (scope !== "lr") {
      handleTrainingField(["total_batch_size"], option.total_batch_size);
      if (option.clear_manual_micro_batch) {
        handleOptionalTrainingField(["micro_batch_size"], null);
      }
    }
    if (scope !== "batch") {
      handleTrainingField(["optimizer", "lr"], option.learning_rate);
    }

    const summary =
      scope === "batch"
        ? `Set total batch size to ${formatInteger(option.total_batch_size)} tokens.`
        : scope === "lr"
          ? `Set learning rate to ${formatLearningRate(option.learning_rate)}.`
          : `Set total batch size to ${formatInteger(option.total_batch_size)} tokens and learning rate to ${formatLearningRate(option.learning_rate)}.`;
    const microNote =
      scope !== "lr" && option.clear_manual_micro_batch
        ? " Cleared manual micro batch size so preflight can auto-select the best micro step."
        : "";
    notify("success", `${option.label} recommendation applied`, `${summary}${microNote}`);
  };

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

  const handleStartTraining = async () => {
    if (!selectedProjectId || !selectedTokenizerJobId || !trainingConfig || !dataloaderConfig || !preflight?.valid) {
      notify("error", "Training blocked", "Resolve the preflight issues before launching.");
      return;
    }
    setLaunching(true);
    try {
      const job = await createTrainingJob({
        name: runName.trim() || undefined,
        project_id: selectedProjectId,
        tokenizer_job_id: selectedTokenizerJobId,
        training_config: trainingConfig,
        dataloader_config: dataloaderConfig,
      });
      startTransition(() => {
        setActiveRunId(job.id);
        setIsActiveRunOpen(true);
        setActiveRun(job);
        setRecentRuns((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      });
      notify("success", "Training launched", `Run ${job.name} is now being tracked live.`);
    } catch (error) {
      notify("error", "Launch failed", error instanceof Error ? error.message : "Failed to start training.");
    } finally {
      setLaunching(false);
    }
  };

  const handleSelectRecentRun = useCallback((jobId: string) => {
    setActiveRunId(jobId);
    setIsActiveRunOpen(true);
  }, []);

  const handleStopTraining = async (jobId: string | null = activeRunId) => {
    if (!jobId) {
      return;
    }
    setStoppingRunId(jobId);
    try {
      const job = await stopTrainingJob(jobId);
      startTransition(() => {
        if (activeRunId === job.id) {
          setActiveRun(job);
        }
        setRecentRuns((current) => replaceRunInOrder(current, job));
      });
      notify("success", "Training stopped", `Run ${job.name} was cancelled.`);
    } catch (error) {
      notify("error", "Stop failed", error instanceof Error ? error.message : "Failed to stop the active run.");
    } finally {
      setStoppingRunId((current) => (current === jobId ? null : current));
    }
  };

  const handleDeleteRun = async (jobId: string) => {
    try {
      await deleteTrainingJob(jobId);
      startTransition(() => {
        setRecentRuns((current) => current.filter((job) => job.id !== jobId));
        if (activeRunId === jobId) {
          setActiveRunId(null);
        }
      });
      notify("success", "Run removed", "The training run was removed from the workspace.");
    } catch (error) {
      notify("error", "Delete failed", error instanceof Error ? error.message : "Failed to delete the training run.");
    }
  };

  const startReady = Boolean(preflight?.valid && selectedProjectId && selectedTokenizerJobId && !launching);
  const trainingRuntimeReady = Boolean(trainingConfig && dataloaderConfig);
  const hasTrainingInProgress =
    activeRun?.status === "running" || activeRun?.status === "pending";
  const activeRunCanBeStopped = canStopTrainingRun(activeRun);
  const stoppingActiveRun = activeRunCanBeStopped && stoppingRunId === activeRun.id;
  const activeRunStepProgress = useMemo(() => deriveTrainingStepProgress(activeRun), [activeRun]);
  const batchAndLrRecommendation = preflight?.batch_and_lr_recommendation ?? null;
  const selectedRecommendationOption = useMemo(() => {
    if (!batchAndLrRecommendation) {
      return null;
    }
    return (
      batchAndLrRecommendation.options.find((option) => option.key === selectedRecommendationOptionKey) ??
      batchAndLrRecommendation.options.find(
        (option) => option.key === batchAndLrRecommendation.recommended_option_key
      ) ??
      batchAndLrRecommendation.options[0] ??
      null
    );
  }, [batchAndLrRecommendation, selectedRecommendationOptionKey]);
  const recommendedRecommendationOption = useMemo(() => {
    if (!batchAndLrRecommendation) {
      return null;
    }
    return (
      batchAndLrRecommendation.options.find(
        (option) => option.key === batchAndLrRecommendation.recommended_option_key
      ) ??
      batchAndLrRecommendation.options[0] ??
      null
    );
  }, [batchAndLrRecommendation]);
  const recommendationConfidenceTone =
    batchAndLrRecommendation?.confidence === "high"
      ? "tone-good"
      : batchAndLrRecommendation?.confidence === "low"
        ? "tone-warn"
        : "tone-neutral";
  const recommendationConfidenceLabel = batchAndLrRecommendation?.confidence
    ? `${batchAndLrRecommendation.confidence.charAt(0).toUpperCase()}${batchAndLrRecommendation.confidence.slice(1)} confidence`
    : null;
  const selectedRecommendationIsRecommended =
    batchAndLrRecommendation !== null &&
    selectedRecommendationOption !== null &&
    selectedRecommendationOption.key === batchAndLrRecommendation.recommended_option_key;
  const selectedPeakLearningRate =
    batchAndLrRecommendation && selectedRecommendationOption
      ? selectedRecommendationOption.learning_rate *
        batchAndLrRecommendation.signals.schedule_peak_factor
      : null;
  const selectedBatchTooltipSummary = useMemo(() => {
    if (!selectedRecommendationOption || !recommendedRecommendationOption) {
      return "";
    }
    return describeBatchProfileShift(
      selectedRecommendationOption.total_batch_size,
      recommendedRecommendationOption.total_batch_size,
      recommendedRecommendationOption.label,
      selectedRecommendationOption.key === recommendedRecommendationOption.key
    );
  }, [recommendedRecommendationOption, selectedRecommendationOption]);
  const selectedLearningRateTooltipSummary = useMemo(() => {
    if (!selectedRecommendationOption || !recommendedRecommendationOption) {
      return "";
    }
    return describeLearningRateProfileShift(
      selectedRecommendationOption.learning_rate,
      recommendedRecommendationOption.learning_rate,
      recommendedRecommendationOption.label,
      selectedRecommendationOption.key === recommendedRecommendationOption.key,
      selectedPeakLearningRate
    );
  }, [recommendedRecommendationOption, selectedPeakLearningRate, selectedRecommendationOption]);
  const selectedBatchTooltipItems = useMemo(() => {
    if (!batchAndLrRecommendation || !selectedRecommendationOption) {
      return [];
    }
    return [
      {
        label: "Hardware fit",
        detail: describeBatchHardwareContext(
          batchAndLrRecommendation.signals.device_type,
          batchAndLrRecommendation.signals.max_memory_micro_batch_size,
          selectedRecommendationOption.micro_batch_size,
          selectedRecommendationOption.grad_accum_steps
        ),
      },
      {
        label: "Data regime",
        detail: describeBatchDatasetContext(batchAndLrRecommendation.signals.dataset_scale),
      },
    ];
  }, [batchAndLrRecommendation, selectedRecommendationOption]);
  const selectedLearningRateTooltipItems = useMemo(() => {
    if (!batchAndLrRecommendation || !selectedRecommendationOption) {
      return [];
    }
    return [
      {
        label: "Scheduler effect",
        detail: describeLearningRateScheduleContext(
          batchAndLrRecommendation.signals.schedule_peak_factor,
          selectedPeakLearningRate
        ),
      },
      {
        label: "Data regime",
        detail: describeLearningRateDatasetContext(batchAndLrRecommendation.signals.dataset_scale),
      },
    ];
  }, [batchAndLrRecommendation, selectedPeakLearningRate, selectedRecommendationOption]);
  const trainingCompleted = activeRun?.status === "completed";
  const sequenceLength = trainingConfig ? asNumber(trainingConfig.seq_len, 0) : 0;
  const maxSteps = trainingConfig ? asNumber(trainingConfig.max_steps, 0) : 0;
  const datasetSummary =
    datasetSourceMode === "local_file"
      ? `${localTrainFiles.length} local file${localTrainFiles.length === 1 ? "" : "s"}`
      : `${streamingDatasets.length} streaming dataset${
          streamingDatasets.length === 1 ? "" : "s"
        }`;
  const workflowSteps: TrainingWorkflowStep[] = [
    {
      title: "Step 1 - Choose saved model",
      state: selectedProject ? "ready" : "waiting",
      status: selectedProject ? "Ready" : "Waiting for configuration",
      body: selectedProject
        ? selectedProject.name ?? selectedProject.id
        : "Pick a saved model config from the home workspace or query parameters.",
      actionLabel: "Open model selection",
      onAction: () => openWorkflowTarget("model"),
    },
    {
      title: "Step 2 - Choose tokenizer artifact",
      state:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? "ready"
          : "waiting",
      status:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? "Ready"
          : "Waiting for configuration",
      body:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
          : "Select a completed tokenizer artifact to ensure vocabulary compatibility.",
      actionLabel: "Open tokenizer selection",
      onAction: () => openWorkflowTarget("tokenizer"),
    },
    {
      title: "Step 3 - Configure training run",
      state: trainingRuntimeReady ? "ready" : "waiting",
      status: trainingRuntimeReady ? "Ready" : "Waiting for configuration",
      body: trainingRuntimeReady
        ? `Sequence length ${formatInteger(sequenceLength)}, maximum training steps ${formatInteger(
            maxSteps
          )}, ${datasetSummary} configured.`
        : "Tune sequence length, batch size, save cadence, prompts, and dataset sources.",
      actionLabel: "Open training settings",
      onAction: () => openWorkflowTarget("training"),
    },
    {
      title: "Step 4 - Validate configurations",
      state: preflight?.valid ? "ready" : preflightLoading ? "inProgress" : "waiting",
      status: preflight?.valid
        ? "Ready"
        : preflightLoading
          ? "In progress"
          : "Waiting for configuration",
      body: preflight?.valid
        ? "Preflight passed for compatibility, runtime math, and memory checks."
        : preflightLoading
          ? "Validating the latest training and dataset configuration changes..."
          : compactWorkflowMessage(
              preflightError ??
                preflight?.errors[0]?.message ??
                "Complete steps 1-3 first. Preflight runs automatically."
            ),
      actionLabel: "Review preflight",
      onAction: () => openWorkflowTarget("preflight"),
    },
    {
      title: "Step 5 - Start training",
      state: trainingCompleted
        ? "ready"
        : hasTrainingInProgress
          ? "inProgress"
          : "waiting",
      status: trainingCompleted
        ? "Ready (trained)"
        : hasTrainingInProgress
          ? "In progress"
          : "Not ready",
      body: trainingCompleted
        ? "Latest training run completed. Artifacts and telemetry are ready."
        : hasTrainingInProgress
          ? `Current run is ${activeRun?.status ?? "running"}.`
          : startReady
            ? "Preflight passed. Start training to complete this step."
            : preflightLoading
              ? "Waiting for automatic preflight to finish."
              : "A passing preflight is required before launch.",
    },
  ];

  return (
    <main className="studioRoot trainingPage">
      <TrainingStudioNav
        theme={theme}
        onToggleTheme={() =>
          setTheme((current) => (current === "dark" ? "white" : "dark"))
        }
      />

      <TrainingHeroSection
        selectedProject={selectedProject}
        selectedProjectId={selectedProjectId}
        selectedTokenizer={selectedTokenizer}
        selectedTokenizerJobId={selectedTokenizerJobId}
        preflightValid={Boolean(preflight?.valid)}
        preflightLoading={preflightLoading}
        activeRun={activeRun}
        activeRunId={activeRunId}
        startReady={startReady}
        launching={launching}
        activeRunCanBeStopped={activeRunCanBeStopped}
        stoppingActiveRun={stoppingActiveRun}
        pickerKind={pickerKind}
        highlightedWorkflowTarget={highlightedWorkflowTarget}
        runName={runName}
        modelSelectionRef={modelSelectionRef}
        tokenizerSelectionRef={tokenizerSelectionRef}
        onStartTraining={handleStartTraining}
        onStopTraining={() => void handleStopTraining()}
        onOpenPicker={(kind) => {
          void openPicker(kind);
        }}
        onRunNameChange={(value) => {
          setRunNameDirty(true);
          setRunName(value);
        }}
      />

      <TrainingWorkflowSection
        steps={workflowSteps}
        startReady={startReady}
        launching={launching}
        hasTrainingInProgress={hasTrainingInProgress}
        onStartTraining={handleStartTraining}
      />

      <section className="trainingResultsGrid">
      {isActiveRunOpen ? (
        <section className="panelCard trainingActiveRunPanel">
          <div className="panelHead">
            <div>
              <h2>Active Run</h2>
              <p className="panelCopy">The monitor updates every {Math.round(POLL_INTERVAL_MS / 1000)} seconds with summary, metrics, samples, checkpoints, and logs.</p>
            </div>
            <div className="trainingActiveRunHeaderActions">
              {activeRun ? (
                <span className={`pillBadge ${statusTone(activeRun.status)}`}>{formatStatusLabel(activeRun.status)}</span>
              ) : null}
              {activeRunCanBeStopped ? (
                <button
                  type="button"
                  className="buttonDanger buttonSmall"
                  onClick={() => void handleStopTraining(activeRun.id)}
                  disabled={stoppingActiveRun}
                >
                  <FiXCircle aria-hidden="true" />
                  {stoppingActiveRun ? "Stopping..." : "Stop run"}
                </button>
              ) : null}
              <button
                type="button"
                className="trainingActiveRunCloseButton"
                onClick={() => setIsActiveRunOpen(false)}
                aria-label="Close active run"
                title="Close active run"
              >
                <FiX aria-hidden="true" />
              </button>
            </div>
          </div>

          {activeRun ? (
            <>
              <div className="trainingProgress">
                <div className="trainingSectionHeader">
                  <h3>{activeRun.name}</h3>
                </div>
                <div className="trainingProgressBar">
                  <span style={{ width: `${activeRunStepProgress.fraction * 100}%` }} />
                </div>
                <div className="trainingInlineMeta">
                  <span>Run identifier: {activeRun.id.slice(0, 8)}</span>
                  <span>Created {formatDate(activeRun.created_at)}</span>
                  <span>Started {activeRun.started_at ? formatDate(activeRun.started_at) : "waiting"}</span>
                  <span>Elapsed training time: {formatTrainingElapsed(activeRunStepProgress, activeRun.status)}</span>
                  <span>Estimated time remaining: {formatTrainingEta(activeRunStepProgress, activeRun.status)}</span>
                </div>
              </div>

              <div className="statusGrid">
                <div className="statusCard">
                  <div className="statusCardIcon"><FiActivity /></div>
                  <div>
                    <div className="statusCardTitle">Training step</div>
                    <div className="statusCardValue">
                      {formatInteger(activeRunStepProgress.completedSteps)} / {formatInteger(activeRunStepProgress.maxSteps)}
                    </div>
                    <div className="statusCardDetail">Training progress: {activeRunStepProgress.percentLabel} of steps</div>
                  </div>
                </div>
                <div className="statusCard">
                  <div className="statusCardIcon"><FiCheckCircle /></div>
                  <div>
                    <div className="statusCardTitle">Loss</div>
                    <div className="statusCardValue">{formatMetric(activeRun.latest_loss, 4)}</div>
                    <div className="statusCardDetail">Gradient norm: {formatMetric(activeRun.latest_grad_norm, 3)}</div>
                  </div>
                </div>
                <div className="statusCard">
                  <div className="statusCardIcon"><FiRefreshCw /></div>
                  <div>
                    <div className="statusCardTitle">Learning Rate</div>
                    <div className="statusCardValue">{formatLearningRate(activeRun.latest_lr)}</div>
                    <div className="statusCardDetail">Tokens per second: {formatInteger(activeRun.latest_tokens_per_sec)}</div>
                  </div>
                </div>
                <div className="statusCard">
                  <div className="statusCardIcon"><FiArchive /></div>
                  <div>
                    <div className="statusCardTitle">Saved artifacts</div>
                    <div className="statusCardValue">
                      {formatInteger(activeRun.checkpoint_count)} checkpoints
                    </div>
                    <div className="statusCardDetail">
                      {formatInteger(activeRun.sample_count)} sample groups • {formatBytes(activeRun.output_size_bytes)}
                    </div>
                  </div>
                </div>
              </div>

              <div className="trainingChartGrid">
                <MetricChart
                  title="Loss"
                  metricKey="loss"
                  metrics={metrics}
                  latestValue={formatMetric(activeRun.latest_loss, 4)}
                  stroke="var(--brand)"
                  digits={4}
                />
                <MetricChart
                  title="Learning Rate"
                  metricKey="lr"
                  metrics={metrics}
                  latestValue={formatLearningRate(activeRun.latest_lr)}
                  stroke="var(--ok)"
                  digits={3}
                />
                <MetricChart
                  title="Gradient Norm"
                  metricKey="norm"
                  metrics={metrics}
                  latestValue={formatMetric(activeRun.latest_grad_norm, 3)}
                  stroke="var(--warn)"
                  digits={3}
                />
                <MetricChart
                  title="Throughput"
                  metricKey="tok_per_sec"
                  metrics={metrics}
                  latestValue={formatInteger(activeRun.latest_tokens_per_sec)}
                  stroke="var(--danger)"
                  digits={1}
                />
              </div>

              <details className="sectionDisclosure">
                <summary className="sectionDisclosureSummary">Samples</summary>
                <div className="trainingSampleList">
                  {samples.length ? (
                    samples.slice().reverse().map((entry) => {
                      const sampleCount = entry.samples.length;
                      const totalChars = entry.samples.reduce(
                        (sum, sample) => sum + sample.text.length + (sample.prompt?.length ?? 0),
                        0
                      );

                      return (
                        <details
                          key={`sample-${entry.step}`}
                          className="trainingSampleCard trainingSampleStepDisclosure"
                        >
                          <summary className="trainingSampleStepSummary">
                            <span>
                              <span className="trainingSampleTitle">Step {entry.step}</span>
                              <span className="trainingSampleMeta">
                                {sampleCount} prompt{sampleCount === 1 ? "" : "s"} generated
                                {" - "}
                                {formatInteger(totalChars)} characters
                              </span>
                            </span>
                          </summary>

                          <div className="trainingSampleStepBody">
                            {entry.samples.map((sample) => {
                              const promptSummary = samplePromptSummary(sample.prompt, sample.index);
                              const splitSample = splitGeneratedSampleText(sample.text, sample.prompt);
                              const continuationLength = splitSample.continuation.length;

                              return (
                                <details
                                  key={`${entry.step}-${sample.index}`}
                                  className="trainingSampleTextDisclosure"
                                >
                                  <summary className="trainingSampleTextSummary">
                                    <span className="trainingSamplePromptSummary">{promptSummary}</span>
                                    <span className="trainingSampleMeta">
                                      {formatInteger(continuationLength)} continuation characters
                                    </span>
                                  </summary>
                                  <div className="trainingSampleGeneratedBlock">
                                    <div className="trainingSampleGeneratedHead">
                                      <span>Generated sample</span>
                                      <span>{formatInteger(sample.text.length)} total characters</span>
                                    </div>
                                    <pre className="trainingSampleGeneratedText">
                                      {splitSample.prefix ? (
                                        <>
                                          <span className="trainingSampleGeneratedPrompt">{splitSample.prefix}</span>
                                          <span className="trainingSampleGeneratedContinuation">{splitSample.continuation}</span>
                                        </>
                                      ) : (
                                        <span className="trainingSampleGeneratedContinuation">{splitSample.continuation}</span>
                                      )}
                                    </pre>
                                  </div>
                                </details>
                              );
                            })}
                          </div>
                        </details>
                      );
                    })
                  ) : (
                    <div className="trainingEmpty">No samples have been recorded yet.</div>
                  )}
                </div>
              </details>

              <details className="sectionDisclosure">
                <summary className="sectionDisclosureSummary">Checkpoints</summary>
                <div className="trainingCheckpointList">
                  {checkpoints.length ? (
                    checkpoints.map((checkpoint) => (
                      <div key={checkpoint.directory} className="trainingCheckpointCard">
                        <div className="trainingCheckpointTitle">Step {checkpoint.step}</div>
                        <div className="trainingCheckpointMeta">
                          {checkpoint.created_at ? formatDate(checkpoint.created_at) : "Created time unavailable"} • {formatBytes(checkpoint.size_bytes)}
                        </div>
                        <div className="trainingCheckpointMeta">{checkpoint.files.join(", ")}</div>
                      </div>
                    ))
                  ) : (
                    <div className="trainingEmpty">No checkpoints are available yet.</div>
                  )}
                </div>
              </details>

              <details className="sectionDisclosure">
                <summary className="sectionDisclosureSummary">Training Data Preview</summary>
                {dataPreview ? (
                  <div className="trainingJsonGrid">
                    <pre className="trainingCodeBlock">{prettyJson(dataPreview)}</pre>
                  </div>
                ) : (
                  <div className="trainingEmpty">
                    The trainer has not published a data preview for this run yet.
                  </div>
                )}
              </details>

              <details className="sectionDisclosure">
                <summary className="sectionDisclosureSummary">Logs</summary>
                <div className="trainingDualLog">
                  <div className="trainingLogBox">{logs.stdout.join("\n") || "stdout is quiet so far."}</div>
                  <div className="trainingLogBox">{logs.stderr.join("\n") || "stderr is clear."}</div>
                </div>
              </details>

              <details className="sectionDisclosure">
                <summary className="sectionDisclosureSummary">Resolved runtime and configurations</summary>
                <div className="trainingJsonGrid">
                  <pre className="trainingCodeBlock">{prettyJson(activeRun.resolved_runtime)}</pre>
                  <pre className="trainingCodeBlock">{prettyJson(activeRun.memory_estimate)}</pre>
                </div>
              </details>
            </>
          ) : (
            <div className="trainingEmpty">No active run selected. Launch a new run or choose one from the recent runs column.</div>
          )}
        </section>
      ) : null}

        <div className="trainingPanelStack">
          <section
            id="settings-preflight"
            ref={preflightSectionRef}
            className={`panelCard trainingPreflightPanel settingsCategoryAnchor ${
              highlightedWorkflowTarget === "preflight"
                ? "settingsCategoryAnchor-highlight"
                : ""
            }`}
          >
              <div className="panelHead">
                <div>
                  <h2>Preflight</h2>
                  <p className="panelCopy">
                    Compatibility, scheduler math, local dataset paths, special tokens, and runtime memory are validated here.
                  </p>
                </div>
                {preflight ? (
                  <span className={`pillBadge ${preflight.valid ? "tone-good" : "tone-error"}`}>
                    {preflight.valid ? "Ready to launch" : `${preflight.errors.length} blocking issues`}
                  </span>
                ) : null}
              </div>

              {preflightError ? <div className="trainingIssueCard tone-error">{preflightError}</div> : null}

              {preflight?.compatibility ? (
                <div className="statusGrid">
                  <div className={`statusCard ${preflight.valid ? "tone-good" : "tone-bad"}`}>
                    <div className="statusCardIcon">
                      <FiArchive />
                    </div>
                    <div>
                      <div className="statusCardTitle">Tokenizer vocabulary</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.tokenizer_vocab_size)}</div>
                      <div className="statusCardDetail">Model vocabulary size: {formatInteger(preflight.compatibility.model_vocab_size)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiLayers />
                    </div>
                    <div>
                      <div className="statusCardTitle">Sequence length</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.seq_len)}</div>
                      <div className="statusCardDetail">Model context limit: {formatInteger(preflight.compatibility.model_context_length)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiBarChart2 />
                    </div>
                    <div>
                      <div className="statusCardTitle">Micro batch size</div>
                      <div className="statusCardValue">
                        {formatInteger(preflight.derived_runtime?.micro_batch_size ?? null)}
                      </div>
                      <div className="statusCardDetail">
                        Gradient accumulation steps: {formatInteger(preflight.derived_runtime?.grad_accum_steps ?? null)}
                      </div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiCpu />
                    </div>
                    <div>
                      <div className="statusCardTitle">Device</div>
                      <div className="statusCardValue">{preflight.derived_runtime?.device_type ?? "n/a"}</div>
                      <div className="statusCardDetail">
                        Memory-estimated max batch size: {formatInteger((preflight.memory_estimate?.max_batch_size as number | undefined) ?? null)}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="trainingEmpty">
                  Select both assets and finish the configuration fields to run preflight.
                </div>
              )}

              {preflight?.errors.length ? (
                <div className="trainingIssueList">
                  {preflight.errors.map((item) => (
                    <div key={`${item.code}-${item.path}`} className={`trainingIssueCard tone-${issueTone(item)}`}>
                      <div className="trainingIssueTitle">{item.message}</div>
                      <div className="trainingIssueMeta" title={item.path}>{formatIssueLocation(item.path)}</div>
                    </div>
                  ))}
                </div>
              ) : null}

              {preflight?.warnings.length ? (
                <details className="sectionDisclosure" open>
                  <summary className="sectionDisclosureSummary">Warnings</summary>
                  <div className="trainingIssueList">
                    {preflight.warnings.map((item) => (
                      <div key={`${item.code}-${item.path}`} className="trainingIssueCard tone-warning">
                        <div className="trainingIssueTitle">{item.message}</div>
                        <div className="trainingIssueMeta" title={item.path}>{formatIssueLocation(item.path)}</div>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}

              {preflight?.recommended_fixes.length ? (
                <details className="sectionDisclosure" open>
                  <summary className="sectionDisclosureSummary">Suggested fixes</summary>
                  <div className="trainingFixList">
                    {preflight.recommended_fixes.map((fix) => (
                      <div key={fix.code} className="trainingFixCard">
                        <div className="trainingIssueTitle">{fix.label}</div>
                        <div className="trainingIssueMeta">{fix.description}</div>
                        <div className="trainingFixActions">
                          <button type="button" className="buttonGhost buttonSmall" onClick={() => applyFix(fix)}>
                            Apply
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </section>
        </div>

        <div className="trainingPanelStack">
          <section className="panelCard trainingRecentRunsPanel">
              <div className="panelHead">
                <div>
                  <h2>Recent Runs</h2>
                  <p className="panelCopy trainingRecentPanelCopy">Recent jobs stay navigable after refresh so you can jump between current and past runs quickly.</p>
                </div>
                <button type="button" className="buttonGhost buttonSmall" onClick={() => void refreshRecentRuns()}>
                  Refresh
                </button>
              </div>
              <div className="trainingRecentList">
                {recentRuns.length ? (
                  recentRuns.map((job) => (
                    <div key={job.id} className={`trainingRecentCard ${activeRunId === job.id ? "is-active" : ""}`}>
                      <button
                        type="button"
                        className="trainingRecentSelect"
                        onClick={() => handleSelectRecentRun(job.id)}
                      >
                        <div>
                          <strong className="trainingRecentTitle">{job.name}</strong>
                          <p>{job.project_name} / {job.tokenizer_name}</p>
                        </div>
                        <div className="trainingRecentRowMeta">
                          <span className={`pillBadge ${statusTone(job.status)}`}>{formatStatusLabel(job.status)}</span>
                          <p>{formatDate(job.created_at)}</p>
                        </div>
                      </button>
                      <div className="trainingRecentIconActions">
                        {canStopTrainingRun(job) ? (
                          <button
                            type="button"
                            className="trainingRecentIconButton trainingRecentIconButton-danger"
                            onClick={() => void handleStopTraining(job.id)}
                            disabled={stoppingRunId === job.id}
                            aria-label={`Stop ${job.name}`}
                            title={stoppingRunId === job.id ? "Stopping run" : "Stop run"}
                          >
                            <FiXCircle aria-hidden="true" />
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className="trainingRecentIconButton trainingRecentIconButton-danger"
                          onClick={() => void handleDeleteRun(job.id)}
                          disabled={job.status === "running" || job.status === "pending"}
                          aria-label={`Delete ${job.name}`}
                          title="Delete run"
                        >
                          <FiTrash2 aria-hidden="true" />
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="trainingEmpty">
                    No training runs yet.
                  </div>
                )}
              </div>
          </section>
        </div>
      </section>

      <section className="panelCard trainingSettingsStudio">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Settings</p>
            <h2>Configuration Studio</h2>
            <p className="panelCopy">
              Core launch fields stay visible. Dataset, prompts, and runtime controls live in
              foldable sections so the page stays compact like the tokenizer workflow.
            </p>
          </div>
          <div className="heroMetaPills">
            <span className="pillBadge tone-neutral">{datasetEntries.length} dataset source{datasetEntries.length === 1 ? "" : "s"}</span>
            <span className="pillBadge tone-neutral">{promptEntries.length} prompt{promptEntries.length === 1 ? "" : "s"}</span>
          </div>
        </div>

        {trainingConfig && dataloaderConfig ? (
          <div className="trainingSettingsStack">
            <details className="settingsPanel" open ref={trainingPlanPanelRef}>
              <summary>Training plan</summary>
              <div className="settingsGrid">
                <div
                  id="settings-training"
                  ref={trainingSettingsRef}
                  className={`settingsGroup settingsCategoryAnchor ${
                    highlightedWorkflowTarget === "training"
                      ? "settingsCategoryAnchor-highlight"
                      : ""
                  }`}
                >
                  <div className="settingsGroupHeader">
                    <h3>Core launch knobs</h3>
                    <p className="settingsGroupHint">
                      Tune the values you are most likely to touch between runs before opening the
                      deeper runtime controls.
                    </p>
                  </div>
                  <div className="fieldGrid trainingSettingsCompactGrid">
                    <label className="fieldLabel">
                      <span>Sequence length</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.seq_len, 128)}
                        onCommit={(value) => handleTrainingField(["seq_len"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Maximum training steps</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.max_steps, 0)}
                        onCommit={handleMaxStepsChange}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Total batch size (tokens)</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.total_batch_size, 0)}
                        onCommit={(value) => handleTrainingField(["total_batch_size"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Micro batch size <small>optional</small></span>
                      <OptionalConfigNumberInput
                        value={
                          typeof trainingConfig.micro_batch_size === "number"
                            ? trainingConfig.micro_batch_size
                            : null
                        }
                        onCommit={(value) =>
                          handleOptionalTrainingField(["micro_batch_size"], value)
                        }
                        placeholder="Auto"
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Learning rate</span>
                      <ConfigNumberInput
                        mode="scientific"
                        step="any"
                        value={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                        onCommit={(value) => handleTrainingField(["optimizer", "lr"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Weight decay</span>
                      <ConfigNumberInput
                        mode="decimal"
                        step="0.0001"
                        value={asNumber(asRecord(trainingConfig.optimizer).weight_decay, 0.1)}
                        onCommit={(value) =>
                          handleTrainingField(
                            ["optimizer", "weight_decay"],
                            value
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Save checkpoint every</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.save_every, 0)}
                        onCommit={(value) => handleTrainingField(["save_every"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Generate samples every</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.sample_every, 0)}
                        onCommit={(value) => handleTrainingField(["sample_every"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Dataset shuffle buffer</span>
                      <ConfigNumberInput
                        value={asNumber(asRecord(dataloaderConfig.shuffle).buffer_size, 1000)}
                        onCommit={(value) =>
                          handleDataloaderField(
                            ["shuffle", "buffer_size"],
                            value
                          )
                        }
                      />
                    </label>
                  </div>

                  <section className="trainingAdvisorCard">
                    <div className="trainingAdvisorHead">
                      <div>
                        <p className="panelEyebrow">Batch And LR Advisor</p>
                        <h3>Recommended optimizer step sizing</h3>
                        <p className="trainingAdvisorCopy">
                          {batchAndLrRecommendation
                            ? batchAndLrRecommendation.summary
                            : preflightLoading
                              ? "Preflight is recalculating the recommendation from the current model, dataset, runtime, and scheduler settings."
                              : "Select a model and tokenizer and let preflight run to see the recommended optimizer-step token batch and learning rate."}
                        </p>
                      </div>
                      {batchAndLrRecommendation ? (
                        <div className="trainingAdvisorMeta">
                          <span className={`pillBadge ${recommendationConfidenceTone}`}>
                            {recommendationConfidenceLabel}
                          </span>
                        </div>
                      ) : null}
                    </div>

                    {batchAndLrRecommendation && selectedRecommendationOption ? (
                      <>
                        <div className="trainingAdvisorToolbar">
                          <div className="trainingAdvisorProfileRow">
                            <span className="trainingAdvisorToolbarLabel">Profile</span>
                            <div
                              className="modeSwitch trainingAdvisorModeSwitch"
                              role="list"
                              aria-label="Batch and LR recommendation profiles"
                            >
                              {batchAndLrRecommendation.options.map((option) => {
                                const isSelected = option.key === selectedRecommendationOption.key;
                                return (
                                  <button
                                    key={option.key}
                                    type="button"
                                    className={`modeSwitchButton ${
                                      isSelected ? "modeSwitchButton-active" : ""
                                    }`}
                                    onClick={() => setSelectedRecommendationOptionKey(option.key)}
                                    aria-pressed={isSelected}
                                    title={option.description}
                                  >
                                    {option.label}
                                  </button>
                                );
                              })}
                            </div>
                            <span
                              className={`pillBadge ${
                                selectedRecommendationIsRecommended ? "tone-good" : "tone-neutral"
                              }`}
                            >
                              {selectedRecommendationIsRecommended ? "Recommended" : "Alternate"}
                            </span>
                          </div>
                          <button
                            type="button"
                            className="buttonPrimary"
                            onClick={() => applyRecommendationOption(selectedRecommendationOption)}
                          >
                            Apply recommendation
                          </button>
                        </div>

                        <div className="trainingAdvisorCompactGrid">
                          <article className="trainingAdvisorKeyStat">
                            <div className="trainingAdvisorStatLabel">
                              <span>Full batch size</span>
                              <TrainingAdvisorInfo label="Batch sizing details">
                                <strong>{selectedRecommendationOption.label}</strong>
                                <p>{selectedBatchTooltipSummary}</p>
                                <div className="trainingAdvisorTooltipList">
                                  {selectedBatchTooltipItems.map((item) => (
                                    <div key={item.label} className="trainingAdvisorTooltipItem">
                                      <span>{item.label}</span>
                                      <strong>{item.detail}</strong>
                                    </div>
                                  ))}
                                </div>
                              </TrainingAdvisorInfo>
                            </div>
                            <strong>{formatInteger(selectedRecommendationOption.total_batch_size)} tokens</strong>
                            <small>
                              Current {formatInteger(asNumber(trainingConfig.total_batch_size, 0))} tokens
                              {numbersRoughlyEqual(
                                selectedRecommendationOption.total_batch_size,
                                asNumber(trainingConfig.total_batch_size, 0)
                              )
                                ? " • already set"
                                : ""}
                            </small>
                          </article>

                          <article className="trainingAdvisorKeyStat">
                            <div className="trainingAdvisorStatLabel">
                              <span>Base learning rate</span>
                              <TrainingAdvisorInfo label="Learning rate details">
                                <strong>{selectedRecommendationOption.label}</strong>
                                <p>{selectedLearningRateTooltipSummary}</p>
                                <div className="trainingAdvisorTooltipList">
                                  {selectedLearningRateTooltipItems.map((item) => (
                                    <div key={item.label} className="trainingAdvisorTooltipItem">
                                      <span>{item.label}</span>
                                      <strong>{item.detail}</strong>
                                    </div>
                                  ))}
                                </div>
                              </TrainingAdvisorInfo>
                            </div>
                            <strong>{formatLearningRate(selectedRecommendationOption.learning_rate)}</strong>
                            <small>
                              Current{" "}
                              {formatLearningRate(asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003))}
                              {numbersRoughlyEqual(
                                selectedRecommendationOption.learning_rate,
                                asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003),
                                1e-9
                              )
                                ? " • already set"
                                : ""}
                            </small>
                          </article>
                        </div>

                        <div className="trainingAdvisorFoot">
                          <p className="trainingAdvisorNote">
                            {selectedRecommendationOption.clear_manual_micro_batch
                              ? "Applying this clears the manual micro batch so preflight can auto-size the step."
                              : "Applying this keeps the current micro-step behavior compatible with the selected profile."}
                          </p>
                          <div className="trainingAdvisorMeta">
                            <span className="pillBadge tone-neutral">
                              {formatInteger(selectedRecommendationOption.estimated_tokens_per_run)} run tokens
                            </span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="trainingAdvisorEmpty">
                        {preflightError
                          ? compactWorkflowMessage(preflightError)
                          : "The advisor appears here once preflight can evaluate the current runtime."}
                      </div>
                    )}
                  </section>

                  <LearningRateSchedulePlanner
                    baseLearningRate={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                    maxSteps={asNumber(trainingConfig.max_steps, 0)}
                    schedulerConfig={asRecord(trainingConfig.lr_scheduler)}
                    onSchedulersChange={handleLrSchedulersChange}
                  />
                </div>
              </div>
            </details>

            <details className="settingsPanel" open ref={datasetPanelRef}>
              <summary>Core dataset settings</summary>
              <div className="settingsGrid">
                <div
                  id="settings-dataset"
                  ref={datasetSettingsRef}
                  className={`settingsGroup settingsCategoryAnchor ${
                    highlightedWorkflowTarget === "dataset"
                      ? "settingsCategoryAnchor-highlight"
                      : ""
                  }`}
                >
                  <div className="settingsGroupHeader">
                    <h3>Dataset sources</h3>
                    <p className="settingsGroupHint">
                      Match the tokenizer trainer: choose one source mode and configure the full
                      dataset stack here.
                    </p>
                  </div>

                  <div className="sourceModeRow trainingTokenizerDatasetSection">
                    <span>Dataset source</span>
                    <div className="modeSwitch">
                      <button
                        type="button"
                        className={`modeSwitchButton ${
                          datasetSourceMode === "local_file" ? "modeSwitchButton-active" : ""
                        }`}
                        onClick={() => setDatasetSourceMode("local_file")}
                      >
                        Local files
                      </button>
                      <button
                        type="button"
                        className={`modeSwitchButton ${
                          datasetSourceMode === "streaming_hf" ? "modeSwitchButton-active" : ""
                        }`}
                        onClick={() => {
                          setDatasetSourceMode("streaming_hf");
                          setStreamingDatasets((previous) =>
                            normalizeStreamingDatasetWeights(previous)
                          );
                        }}
                      >
                        Streaming Hugging Face datasets
                      </button>
                    </div>
                  </div>

                  {datasetSourceMode === "local_file" ? (
                    <div className="datasetConfigurator trainingTokenizerDatasetSection">
                      <div
                        className={`localFileManager ${
                          isDraggingTrainFiles ? "localFileManager-dragging" : ""
                        }`}
                        onDragEnter={handleLocalTrainFilesDragEnter}
                        onDragOver={handleLocalTrainFilesDragOver}
                        onDragLeave={handleLocalTrainFilesDragLeave}
                        onDrop={handleLocalTrainFilesDrop}
                      >
                        <div className="localFileManagerHeader">
                          <div>
                            <strong>Local training files</strong>
                            <p>Training and evaluation use this same file set.</p>
                          </div>
                          <div className="localFileHeaderActions">
                            <div className="localFileHeaderButtons">
                              <label
                                className={`secondaryButton localFileUploadButton localFileHeaderButton ${
                                  isUploadingTrainFile ? "localFileUploadButton-disabled" : ""
                                }`}
                                aria-disabled={isUploadingTrainFile}
                              >
                                {isUploadingTrainFile ? "Uploading..." : "Add files"}
                                <input
                                  type="file"
                                  multiple
                                  onChange={handleTrainFilesSelected}
                                  disabled={isUploadingTrainFile}
                                />
                              </label>
                              <button
                                type="button"
                                className="textButton localFileHeaderButton"
                                onClick={clearLocalTrainFiles}
                                disabled={localTrainFiles.length === 0}
                              >
                                Remove all
                              </button>
                            </div>
                            <span className="localFileCount">
                              {localTrainFiles.length} file
                              {localTrainFiles.length === 1 ? "" : "s"}
                            </span>
                          </div>
                        </div>

                        {localTrainFiles.length === 0 ? (
                          <p className="filterEmpty">
                            No local files added yet. Add one or more files to train.
                          </p>
                        ) : (
                          <ul className="localFileList">
                            {localTrainFiles.map((entry) => {
                              const fileCharLabel = formatCharCount(entry.sizeChars);
                              return (
                                <li key={entry.id} className="localFileItem">
                                  <strong className="localFileName" title={entry.fileName}>
                                    {entry.fileName}
                                  </strong>
                                  <div className="localFileActions">
                                    <span className="localFileStat">
                                      {fileCharLabel
                                        ? `${fileCharLabel} characters`
                                        : "Character count pending"}
                                    </span>
                                    <button
                                      type="button"
                                      className="textButton localFileRemoveIconButton"
                                      onClick={() => removeLocalTrainFile(entry.id)}
                                      aria-label={`Remove ${entry.fileName}`}
                                      title={`Remove ${entry.fileName}`}
                                    >
                                      <FiX aria-hidden="true" />
                                    </button>
                                  </div>
                                </li>
                              );
                            })}
                          </ul>
                        )}

                        <span className="fieldNote">Files are deduplicated by stored path.</span>
                      </div>
                    </div>
                  ) : (
                    <div className="datasetConfigurator trainingTokenizerDatasetSection">
                      <label className="fieldLabel fullWidthField">
                        <span>Hugging Face access token <small>optional</small></span>
                        <input
                          type="password"
                          value={hfToken}
                          onChange={(event) => setHfToken(event.target.value)}
                          autoComplete="off"
                          placeholder="hf_..."
                        />
                        <span className="fieldNote">Required for gated/private datasets.</span>
                      </label>

                      <div className="actionRow">
                        <button
                          type="button"
                          className="secondaryButton"
                          onClick={addStreamingDataset}
                        >
                          Add dataset
                        </button>
                        <button
                          type="button"
                          className="secondaryButton"
                          onClick={() => {
                            void handleLoadStreamingTemplate();
                          }}
                          disabled={isLoadingDatasetTemplate}
                        >
                          {isLoadingDatasetTemplate
                            ? "Loading template..."
                            : "Load streaming template"}
                        </button>
                      </div>

                      <div className="datasetList">
                        {streamingDatasets.map((entry, index) => (
                          <div key={entry.id} className="datasetCard">
                            <div className="datasetCardHeader">
                              <strong>Streaming dataset {index + 1}</strong>
                              <button
                                type="button"
                                className="textButton datasetRemoveButton"
                                onClick={() => removeStreamingDataset(entry.id)}
                                disabled={streamingDatasets.length <= 1}
                                aria-label={`Remove streaming dataset ${index + 1}`}
                                title={`Remove streaming dataset ${index + 1}`}
                              >
                                <FiTrash2 aria-hidden="true" />
                              </button>
                            </div>

                            <div className="fieldGrid">
                              <label className="fieldLabel">
                                <span>Hugging Face dataset name</span>
                                <input
                                  value={entry.name}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      name: event.target.value,
                                    })
                                  }
                                  placeholder="HuggingFaceFW/fineweb-edu"
                                />
                              </label>

                              <label className="fieldLabel">
                                <span>Split</span>
                                <input
                                  value={entry.split}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      split: event.target.value,
                                    })
                                  }
                                  placeholder="train"
                                />
                              </label>

                              <label className="fieldLabel">
                                <span>Weight</span>
                                <input
                                  inputMode="decimal"
                                  pattern="[0-9]*[.]?[0-9]*"
                                  min="0"
                                  max="1"
                                  step="0.000001"
                                  value={entry.weight}
                                  onChange={(event) =>
                                    updateStreamingWeight(entry.id, event.target.value)
                                  }
                                  placeholder="1.0"
                                />
                              </label>

                              <label className="fieldLabel fullWidthField">
                                <span>Text columns</span>
                                <input
                                  value={entry.textColumns}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      textColumns: event.target.value,
                                    })
                                  }
                                  placeholder="text"
                                />
                              </label>
                            </div>

                            <details className="subPanel">
                              <summary>Advanced source options</summary>
                              <div className="fieldGrid">
                                <label className="fieldLabel">
                                  <span>Dataset config <small>optional</small></span>
                                  <input
                                    value={entry.config}
                                    onChange={(event) =>
                                      updateStreamingDataset(entry.id, {
                                        config: event.target.value,
                                      })
                                    }
                                  />
                                </label>

                                <div className="fullWidthField filterBuilder">
                                  <div className="filterBuilderHeader">
                                    <span className="filterBuilderTitle">Filters (optional)</span>
                                    <button
                                      type="button"
                                      className="secondaryButton"
                                      onClick={() => addStreamingFilter(entry.id)}
                                    >
                                      Add filter
                                    </button>
                                  </div>

                                  {entry.filters.length === 0 ? (
                                    <p className="filterEmpty">No filters yet.</p>
                                  ) : (
                                    <div className="filterList">
                                      {entry.filters.map((filter) => (
                                        <div key={filter.id} className="filterRow">
                                          <label className="fieldLabel">
                                            <span>Column</span>
                                            <input
                                              value={filter.column}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  column: event.target.value,
                                                })
                                              }
                                              placeholder="language_score"
                                            />
                                          </label>

                                          <label className="fieldLabel">
                                            <span>Operator</span>
                                            <select
                                              value={filter.operator}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  operator: event.target
                                                    .value as FilterOperator,
                                                })
                                              }
                                            >
                                              {FILTER_OPERATORS.map((operator) => (
                                                <option key={operator} value={operator}>
                                                  {operator}
                                                </option>
                                              ))}
                                            </select>
                                          </label>

                                          <label className="fieldLabel">
                                            <span>Value</span>
                                            <input
                                              value={filter.value}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  value: event.target.value,
                                                })
                                              }
                                              placeholder={
                                                filter.operator === "in" ||
                                                filter.operator === "not in"
                                                  ? '["en", "de"] or en,de'
                                                  : 'en, true, 0.95, {"k":1}'
                                              }
                                            />
                                          </label>

                                          <button
                                            type="button"
                                            className="textButton filterRemoveButton"
                                            onClick={() =>
                                              removeStreamingFilter(entry.id, filter.id)
                                            }
                                          >
                                            Remove
                                          </button>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  <p className="fieldNote">
                                    Values are inferred automatically. For `in`/`not in`, use a
                                    JSON array or comma-separated values.
                                  </p>
                                </div>
                              </div>
                            </details>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </details>

            <details className="settingsPanel" open>
              <summary>Sampling prompts</summary>
              <div className="settingsGrid">
                <div className="settingsGroup">
                  <div className="trainingSettingsPanelHead">
                    <div className="settingsGroupHeader">
                      <h3>Prompt presets</h3>
                      <p className="settingsGroupHint">
                        Short prefixes for checking raw pretraining continuations during the run.
                      </p>
                    </div>
                    <div className="trainingPromptToolbar">
                      <span className="pillBadge tone-neutral">
                        {promptEntries.length} preset{promptEntries.length === 1 ? "" : "s"}
                      </span>
                      <button
                        type="button"
                        className="buttonGhost buttonSmall"
                        onClick={() => void handleResetPrompts()}
                        disabled={isResettingPrompts}
                      >
                        <FiRefreshCw /> {isResettingPrompts ? "Resetting..." : "Reset prompts"}
                      </button>
                      <button
                        type="button"
                        className="buttonGhost buttonSmall"
                        onClick={handleAddPrompt}
                      >
                        <FiPlus /> Add prompt
                      </button>
                    </div>
                  </div>
                  <p className="trainingPromptHintLine">
                    Use autocomplete-style starts, not chat instructions or evaluation tasks.
                  </p>
                  <div className="trainingPromptGrid">
                    {promptEntries.map((prompt, index) => (
                      <article key={`prompt-${index}`} className="trainingPromptCard">
                        <div className="trainingPromptCardHead">
                          <div className="trainingPromptTitleGroup">
                            <div className="trainingPromptTitle">Prompt {index + 1}</div>
                            <p className="trainingPromptMeta">
                              {Math.max(0, asString(prompt.prompt).trim().length)} characters
                            </p>
                          </div>
                          <button
                            type="button"
                            className="textButton trainingPromptRemoveButton"
                            onClick={() => handleRemovePrompt(index)}
                            aria-label={`Remove prompt ${index + 1}`}
                            title={`Remove prompt ${index + 1}`}
                          >
                            <FiTrash2 aria-hidden="true" />
                          </button>
                        </div>

                        <label className="fieldLabel trainingPromptEditor">
                          <span>Prompt text</span>
                          <textarea
                            rows={4}
                            value={asString(prompt.prompt)}
                            onChange={(event) =>
                              handlePromptChange(index, "prompt", event.target.value)
                            }
                            placeholder="Hello"
                          />
                        </label>

                        <div className="trainingPromptFields">
                          <label className="fieldLabel">
                            <span>Max tokens</span>
                            <ConfigNumberInput
                              value={asNumber(prompt.max_tokens, 64)}
                              onCommit={(value) =>
                                handlePromptChange(
                                  index,
                                  "max_tokens",
                                  value
                                )
                              }
                            />
                          </label>
                          <label className="fieldLabel">
                            <span>Temperature</span>
                            <ConfigNumberInput
                              mode="decimal"
                              step="0.05"
                              value={asNumber(prompt.temperature, 0.7)}
                              onCommit={(value) =>
                                handlePromptChange(
                                  index,
                                  "temperature",
                                  value
                                )
                              }
                            />
                          </label>
                          <label className="fieldLabel">
                            <span>Top-k</span>
                            <ConfigNumberInput
                              value={asNumber(prompt.top_k, 40)}
                              onCommit={(value) => handlePromptChange(index, "top_k", value)}
                            />
                          </label>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            </details>

            <details className="settingsPanel">
              <summary>Advanced runtime controls</summary>
              <div className="settingsGrid">
                <div className="settingsGroup">
                  <div className="settingsGroupHeader">
                    <h3>Deeper runtime options</h3>
                    <p className="settingsGroupHint">
                      Token formatting, optimizer internals, and multi-node controls stay available
                      without taking over the main workflow.
                    </p>
                  </div>
                  <div className="fieldGrid trainingSettingsCompactGrid">
                    <label className="fieldLabel">
                      <span>Beginning-of-sequence token</span>
                      <input
                        value={asString(dataloaderConfig.bos_token)}
                        onChange={(event) =>
                          handleDataloaderField(["bos_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>End-of-sequence token</span>
                      <input
                        value={asString(dataloaderConfig.eos_token)}
                        onChange={(event) =>
                          handleDataloaderField(["eos_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Padding token</span>
                      <input
                        value={asString(dataloaderConfig.pad_token)}
                        onChange={(event) =>
                          handleDataloaderField(["pad_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Token data type</span>
                      <select
                        value={asString(dataloaderConfig.token_dtype, "int64")}
                        onChange={(event) =>
                          handleDataloaderField(["token_dtype"], event.target.value)
                        }
                      >
                        <option value="int64">int64</option>
                        <option value="int32">int32</option>
                        <option value="int16">int16</option>
                        <option value="uint8">uint8</option>
                      </select>
                    </label>
                    <label className="fieldLabel">
                      <span>Pretokenize batch size</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.pretokenize_batch_size, 1000)}
                        onCommit={(value) =>
                          handleDataloaderField(
                            ["pretokenize_batch_size"],
                            value
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Cache directory</span>
                      <input
                        value={asString(dataloaderConfig.cache_dir)}
                        onChange={(event) =>
                          handleDataloaderField(["cache_dir"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Optimizer betas</span>
                      <input
                        value={
                          Array.isArray(asRecord(trainingConfig.optimizer).betas)
                            ? (asRecord(trainingConfig.optimizer).betas as unknown[])
                                .map(String)
                                .join(", ")
                            : "0.9, 0.95"
                        }
                        onChange={(event) =>
                          handleTrainingField(
                            ["optimizer", "betas"],
                            event.target.value
                              .split(",")
                              .map((item) => Number(item.trim()))
                              .filter((value) => Number.isFinite(value))
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Optimizer epsilon</span>
                      <ConfigNumberInput
                        mode="decimal"
                        step="0.00000001"
                        value={asNumber(asRecord(trainingConfig.optimizer).eps, 1e-8)}
                        onCommit={(value) => handleTrainingField(["optimizer", "eps"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node split</span>
                      <select
                        value={String(Boolean(dataloaderConfig.node_split))}
                        onChange={(event) =>
                          handleDataloaderField(
                            ["node_split"],
                            event.target.value === "true"
                          )
                        }
                      >
                        <option value="false">Disabled</option>
                        <option value="true">Enabled</option>
                      </select>
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node rank</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.node_rank, 0)}
                        onCommit={(value) => handleDataloaderField(["node_rank"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node world size</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.node_world_size, 1)}
                        onCommit={(value) => handleDataloaderField(["node_world_size"], value)}
                      />
                    </label>
                  </div>
                </div>
              </div>
            </details>

            <details className="settingsPanel">
              <summary>Generated configuration JSON</summary>
              <div className="settingsGrid">
                <div className="trainingJsonGrid">
                  <pre className="trainingCodeBlock">{prettyJson(trainingConfig)}</pre>
                  <pre className="trainingCodeBlock">{prettyJson(dataloaderConfig)}</pre>
                </div>
              </div>
            </details>
          </div>
        ) : (
          <div className="trainingEmpty">Loading starter templates…</div>
        )}
      </section>

      {pickerKind ? (
        <div
          className="trainingAssetPickerOverlay"
          onClick={closePicker}
          role="presentation"
        >
          <section
            id="training-asset-picker"
            className="panelCard trainingAssetPickerDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="training-asset-picker-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="trainingAssetPickerHeader">
              <div>
                <h2 id="training-asset-picker-title">
                  {pickerKind === "project" ? "Choose model config" : "Choose tokenizer artifact"}
                </h2>
                <p className="panelCopy">
                  {pickerKind === "project"
                    ? "Select a saved model project from the workspace to pair with this run."
                    : "Select a completed tokenizer artifact. Only completed tokenizer jobs are shown here."}
                </p>
              </div>
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={closePicker}
                aria-label="Close asset picker"
              >
                <FiXCircle />
              </button>
            </div>

            <div className="trainingAssetPickerControls">
              <label className="trainingAssetPickerSearch">
                <FiSearch />
                <input
                  value={pickerQuery}
                  onChange={(event) => setPickerQuery(event.target.value)}
                  placeholder={
                    pickerKind === "project"
                      ? "Search model configurations by name, identifier, or file"
                      : "Search tokenizers by name, identifier, or artifact"
                  }
                />
              </label>
              <button
                type="button"
                className="buttonGhost"
                onClick={() => {
                  void openPicker(pickerKind);
                }}
              >
                <FiRefreshCw /> Refresh
              </button>
            </div>

            <div className="trainingAssetPickerResults">
              {pickerLoading ? <div className="trainingEmpty">Loading workspace assets…</div> : null}

              {!pickerLoading && pickerError ? (
                <div className="inlineNotice tone-info">{pickerError}</div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "project" && visiblePickerProjects.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No saved model configurations found.</h3>
                  <p className="panelCopy">
                    Create or save a model config from the Home workspace, then reopen the picker.
                  </p>
                  <Link className="buttonGhost" href="/">
                    <FiLayers /> Open Workspace Assets
                  </Link>
                </div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "tokenizer" && visiblePickerTokenizerJobs.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No completed tokenizer artifacts found.</h3>
                  <p className="panelCopy">
                    Finish a tokenizer training job first, then reopen the picker to pair it with this run.
                  </p>
                  <Link className="buttonGhost" href="/tokenizer">
                    <FiCpu /> Open Tokenizer Studio
                  </Link>
                </div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "project"
                ? visiblePickerProjects.map((project) => (
                    <button
                      key={project.id}
                      type="button"
                      className={`trainingAssetPickerOption ${
                        selectedProjectId === project.id ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedProjectId(project.id);
                        closePicker();
                      }}
                      >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">
                            {project.name ?? project.id}
                          </div>
                        </div>
                        <span
                          className={`pillBadge ${
                            selectedProjectId === project.id ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {selectedProjectId === project.id ? "Selected" : "Use model"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {formatDate(project.created_at)} • {formatBytes(project.size_bytes)}
                      </div>
                      <div className="trainingAssetPickerOptionMeta">{project.artifact_file}</div>
                    </button>
                  ))
                : null}

              {!pickerLoading && !pickerError && pickerKind === "tokenizer"
                ? visiblePickerTokenizerJobs.map((job) => (
                    <button
                      key={job.id}
                      type="button"
                      className={`trainingAssetPickerOption ${
                        selectedTokenizerJobId === job.id ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedTokenizerJobId(job.id);
                        closePicker();
                      }}
                      >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">
                            {asString(job.tokenizer_config.name, job.id)}
                          </div>
                        </div>
                        <span
                          className={`pillBadge ${
                            selectedTokenizerJobId === job.id ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {selectedTokenizerJobId === job.id ? "Selected" : "Use tokenizer"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {formatDate(job.created_at)}
                        {job.stats?.vocab_size ? ` • vocabulary size ${formatInteger(job.stats.vocab_size)}` : ""}
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {job.artifact_file ?? job.artifact_path ?? "Tokenizer artifact path unavailable"}
                      </div>
                    </button>
                  ))
                : null}
            </div>
          </section>
        </div>
      ) : null}

      <TrainingToastStack toasts={toasts} />
    </main>
  );
}
