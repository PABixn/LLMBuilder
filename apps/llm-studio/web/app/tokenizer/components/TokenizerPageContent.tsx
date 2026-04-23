"use client";

import { useSearchParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type MouseEvent,
} from "react";
import { FiTrash2, FiX } from "react-icons/fi";

import {
  artifactDownloadUrl,
  createTrainingJob,
  downloadJobArtifact,
  fetchConfigTemplates,
  fetchTrainingJob,
  fetchTrainingJobs,
  previewJobTokenizer,
  type TrainingJob,
  type TokenizerPreviewResult,
  uploadTrainFile,
  validateDataloaderConfig,
  validateTokenizerConfig,
} from "../../../lib/tokenizerLegacyApi";
import {
  defaultDataloaderConfig,
  defaultTokenizerConfig,
} from "../../../lib/tokenizerLegacyDefaults";
import { useThemeMode } from "../../../lib/theme";
import {
  FILTER_OPERATORS,
  LEGACY_TOKENIZER_THEME_STORAGE_KEY,
} from "../constants";
import {
  asRecord,
  buildDataloaderConfigFromForm,
  buildDatasetSelectionFromForm,
  buildTokenizerConfigFromForm,
  datasetFormFromConfig,
  formatCharCount,
  makeLocalTrainFileEntry,
  makeStreamingDatasetEntry,
  makeStreamingFilterEntry,
  normalizeLocalTrainFiles,
  normalizeStreamingDatasetWeights,
  parsePositiveInt,
  parseThresholds,
  sanitizeWeightInput,
  stripGeneratedUploadPrefix,
  tokenizerFormFromConfig,
  trainingFormFromConfig,
} from "../lib/dataset";
import { buildResult } from "../lib/config";
import {
  describeJobState,
  evaluationSourceLabel,
  formatDate,
  formatPercent,
  jobBadgeTone,
} from "../lib/display";
import {
  displayTokenLabel,
  makePreviewSegments,
  prettyJson,
  tokenHue,
} from "../lib/preview";
import {
  sanitizePositiveIntegerInput,
} from "../../shared/lib/configNumber";
import {
  getTauriInvoke,
  triggerBlobDownload,
} from "../lib/storage";
import { useTokenizerLocalFileStats } from "../hooks/useTokenizerLocalFileStats";
import { useTokenizerPersistence } from "../hooks/useTokenizerPersistence";
import { useTokenizerSettingsNavigation } from "../hooks/useTokenizerSettingsNavigation";
import { useTokenizerToasts } from "../hooks/useTokenizerToasts";
import { TokenizerStudioNav } from "./TokenizerStudioNav";
import { TokenizerToastViewport } from "./TokenizerToastViewport";
import { TokenizerWorkflowSection } from "./TokenizerWorkflowSection";
import type {
  BudgetBehavior,
  BudgetUnit,
  DatasetFormState,
  DecoderType,
  FilterOperator,
  PreTokenizerType,
  StreamingDatasetFormState,
  StreamingFilterFormState,
  TokenizerType,
  TokenizerFormState,
  TrainingFormState,
} from "../types";

function JobBadge({ job }: { job: TrainingJob }) {
  const tone = jobBadgeTone(job.state);
  const label = describeJobState(job.state);
  return <span className={`jobBadge jobBadge-${tone}`}>{label}</span>;
}

export function TokenizerPageContent() {
  const searchParams = useSearchParams();
  const [tokenizerForm, setTokenizerForm] = useState<TokenizerFormState>(() =>
    tokenizerFormFromConfig(defaultTokenizerConfig)
  );
  const [datasetForm, setDatasetForm] = useState<DatasetFormState>(() =>
    datasetFormFromConfig(defaultDataloaderConfig)
  );
  const [trainingForm, setTrainingForm] = useState<TrainingFormState>(() =>
    trainingFormFromConfig(defaultDataloaderConfig)
  );
  const [themeMode, setThemeMode] = useThemeMode({
    legacyStorageKeys: [LEGACY_TOKENIZER_THEME_STORAGE_KEY],
  });

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [hiddenRecentJobIds, setHiddenRecentJobIds] = useState<string[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [previewText, setPreviewText] = useState(
    "The quick brown fox jumps over the lazy dog."
  );
  const [previewResult, setPreviewResult] = useState<TokenizerPreviewResult | null>(
    null
  );
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isDownloadingArtifact, setIsDownloadingArtifact] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [hasValidationPassed, setHasValidationPassed] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isLoadingTemplate, setIsLoadingTemplate] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isDraggingTrainFiles, setIsDraggingTrainFiles] = useState(false);
  const jobNotificationKeysRef = useRef<Set<string>>(new Set());
  const locallyStartedJobIdsRef = useRef<Set<string>>(new Set());
  const validationRequestRef = useRef(0);
  const previewRequestRef = useRef(0);
  const localFileDragDepthRef = useRef(0);
  const controlsDisabled =
    isSubmitting ||
    isLoadingTemplate ||
    isUploadingTrainFile;
  const { toasts, notify, removeToast } = useTokenizerToasts();
  const { hasHydratedLocalState, hasHydratedLocalStateRef } = useTokenizerPersistence({
    tokenizerForm,
    setTokenizerForm,
    datasetForm,
    setDatasetForm,
    trainingForm,
    setTrainingForm,
    previewText,
    setPreviewText,
    activeJobId,
    setActiveJobId,
    hiddenRecentJobIds,
    setHiddenRecentJobIds,
  });
  const {
    highlightedSettingsCategory,
    handleSettingsCategoryNavigation,
    tokenizerAndTrainingPanelRef,
    datasetPanelRef,
    tokenizerCategoryRef,
    datasetCategoryRef,
    trainingCategoryRef,
  } = useTokenizerSettingsNavigation();

  useTokenizerLocalFileStats({
    hasHydratedLocalState,
    localTrainFiles: datasetForm.localTrainFiles,
    setDatasetForm,
  });

  useEffect(() => {
    if (!controlsDisabled) {
      return;
    }
    localFileDragDepthRef.current = 0;
    setIsDraggingTrainFiles(false);
  }, [controlsDisabled]);

  useEffect(() => {
    const jobIdFromUrl = searchParams.get("job");
    if (!jobIdFromUrl || jobIdFromUrl === activeJobId) {
      return;
    }

    async function loadJob() {
      try {
        const job = await fetchTrainingJob(jobIdFromUrl as string);
        setActiveJobId(job.id);
        setActiveJob(job);
        
        // Populate forms from job config
        const tokConfig = asRecord(job.tokenizer_config);
        const datConfig = asRecord(job.dataloader_config);
        
        if (Object.keys(tokConfig).length > 0) {
          setTokenizerForm(tokenizerFormFromConfig(tokConfig));
        }
        if (Object.keys(datConfig).length > 0) {
          setDatasetForm(datasetFormFromConfig(datConfig));
          setTrainingForm(trainingFormFromConfig(datConfig));
        }
        
        notify("info", `Loaded tokenizer job ${job.id.slice(0, 8)}`);
      } catch (err) {
        notify("error", `Failed to load job: ${err instanceof Error ? err.message : "Unknown error"}`);
      }
    }

    void loadJob();
  }, [searchParams, activeJobId, notify]);

  const hiddenRecentJobIdSet = useMemo(
    () => new Set(hiddenRecentJobIds),
    [hiddenRecentJobIds]
  );

  const tokenizerBuild = useMemo(
    () => buildResult(() => buildTokenizerConfigFromForm(tokenizerForm)),
    [tokenizerForm]
  );

  const dataloaderBuild = useMemo(
    () => buildResult(() => buildDataloaderConfigFromForm(datasetForm, trainingForm)),
    [datasetForm, trainingForm]
  );
  const datasetBuild = useMemo(
    () => buildResult(() => buildDatasetSelectionFromForm(datasetForm)),
    [datasetForm]
  );
  const trainingRuntimeBuild = useMemo(
    () =>
      buildResult(() => ({
        budget_limit: parsePositiveInt(trainingForm.budgetLimit, "Text budget limit"),
        thresholds: parseThresholds(trainingForm.evaluationThresholds),
      })),
    [trainingForm.budgetLimit, trainingForm.evaluationThresholds]
  );

  useEffect(() => {
    setHasValidationPassed(false);
    setValidationError(null);
  }, [tokenizerForm, datasetForm, trainingForm]);

  useEffect(() => {
    if (!hasHydratedLocalStateRef.current) {
      return;
    }
    if (
      tokenizerBuild.error !== null ||
      dataloaderBuild.error !== null ||
      !tokenizerBuild.value ||
      !dataloaderBuild.value
    ) {
      setIsValidating(false);
      return;
    }

    const requestId = ++validationRequestRef.current;
    setIsValidating(true);
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          await Promise.all([
            validateTokenizerConfig(tokenizerBuild.value as Record<string, unknown>),
            validateDataloaderConfig(dataloaderBuild.value as Record<string, unknown>),
          ]);
          if (validationRequestRef.current !== requestId) {
            return;
          }
          setHasValidationPassed(true);
          setValidationError(null);
        } catch (error) {
          if (validationRequestRef.current !== requestId) {
            return;
          }
          const message =
            error instanceof Error ? error.message : "Validation failed unexpectedly";
          setHasValidationPassed(false);
          setValidationError(message);
        } finally {
          if (validationRequestRef.current === requestId) {
            setIsValidating(false);
          }
        }
      })();
    }, 350);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    dataloaderBuild.error,
    dataloaderBuild.value,
    tokenizerBuild.error,
    tokenizerBuild.value,
  ]);

  const handleManualValidate = useCallback(async () => {
    if (!tokenizerBuild.value) {
      const message = tokenizerBuild.error ?? "Tokenizer config is invalid";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 5000);
      return;
    }
    if (!dataloaderBuild.value) {
      const message = dataloaderBuild.error ?? "Dataloader config is invalid";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 5000);
      return;
    }

    const requestId = ++validationRequestRef.current;
    setIsValidating(true);

    try {
      await Promise.all([
        validateTokenizerConfig(tokenizerBuild.value),
        validateDataloaderConfig(dataloaderBuild.value),
      ]);
      if (validationRequestRef.current !== requestId) {
        return;
      }
      setHasValidationPassed(true);
      setValidationError(null);
      notify("success", "Validation passed.", 3500);
    } catch (error) {
      if (validationRequestRef.current !== requestId) {
        return;
      }
      const message =
        error instanceof Error ? error.message : "Validation failed unexpectedly";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 6000);
    } finally {
      if (validationRequestRef.current === requestId) {
        setIsValidating(false);
      }
    }
  }, [
    dataloaderBuild.error,
    dataloaderBuild.value,
    notify,
    tokenizerBuild.error,
    tokenizerBuild.value,
  ]);

  const updateStreamingDataset = useCallback(
    (
      datasetId: string,
      updates: Partial<Omit<StreamingDatasetFormState, "id">>
    ) => {
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: previous.streamingDatasets.map((entry) =>
          entry.id === datasetId ? { ...entry, ...updates } : entry
        ),
      }));
    },
    []
  );

  const updateStreamingWeight = useCallback(
    (datasetId: string, rawWeight: string) => {
      const sanitizedWeight = sanitizeWeightInput(rawWeight);
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: normalizeStreamingDatasetWeights(
          previous.streamingDatasets,
          datasetId,
          sanitizedWeight
        ),
      }));
    },
    []
  );

  const addStreamingDataset = useCallback(() => {
    setDatasetForm((previous) => {
      const nextDatasets = normalizeStreamingDatasetWeights([
        ...previous.streamingDatasets,
        makeStreamingDatasetEntry(),
      ]);
      return {
        ...previous,
        streamingDatasets: nextDatasets,
      };
    });
  }, []);

  const removeStreamingDataset = useCallback((datasetId: string) => {
    setDatasetForm((previous) => {
      const nextDatasets = previous.streamingDatasets.filter(
        (entry) => entry.id !== datasetId
      );
      return {
        ...previous,
        streamingDatasets: normalizeStreamingDatasetWeights(
          nextDatasets.length > 0 ? nextDatasets : [makeStreamingDatasetEntry()]
        ),
      };
    });
  }, []);

  const updateStreamingFilter = useCallback(
    (
      datasetId: string,
      filterId: string,
      updates: Partial<Omit<StreamingFilterFormState, "id">>
    ) => {
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: previous.streamingDatasets.map((entry) => {
          if (entry.id !== datasetId) {
            return entry;
          }
          return {
            ...entry,
            filters: entry.filters.map((filter) =>
              filter.id === filterId ? { ...filter, ...updates } : filter
            ),
          };
        }),
      }));
    },
    []
  );

  const addStreamingFilter = useCallback((datasetId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      streamingDatasets: previous.streamingDatasets.map((entry) =>
        entry.id === datasetId
          ? { ...entry, filters: [...entry.filters, makeStreamingFilterEntry()] }
          : entry
      ),
    }));
  }, []);

  const removeStreamingFilter = useCallback((datasetId: string, filterId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      streamingDatasets: previous.streamingDatasets.map((entry) => {
        if (entry.id !== datasetId) {
          return entry;
        }
        return {
          ...entry,
          filters: entry.filters.filter((filter) => filter.id !== filterId),
        };
      }),
    }));
  }, []);

  const removeLocalTrainFile = useCallback((localFileId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      localTrainFiles: previous.localTrainFiles.filter(
        (entry) => entry.id !== localFileId
      ),
    }));
  }, []);

  const clearLocalTrainFiles = useCallback(() => {
    setDatasetForm((previous) => ({
      ...previous,
      localTrainFiles: [],
    }));
  }, []);

  const runPreview = useCallback(async (jobId: string, text: string) => {
    const requestId = ++previewRequestRef.current;
    setIsPreviewing(true);
    setPreviewError(null);
    setPreviewResult((previous) =>
      previous && previous.job_id === jobId && previous.text === text ? previous : null
    );

    try {
      const result = await previewJobTokenizer(jobId, { text });
      if (previewRequestRef.current !== requestId) {
        return;
      }
      setPreviewResult(result);
    } catch (error) {
      if (previewRequestRef.current !== requestId) {
        return;
      }
      const message =
        error instanceof Error ? error.message : "Failed to preview tokenizer output";
      setPreviewResult(null);
      setPreviewError(message);
    } finally {
      if (previewRequestRef.current === requestId) {
        setIsPreviewing(false);
      }
    }
  }, []);

  const handleDownloadArtifact = useCallback(
    async (event: MouseEvent<HTMLAnchorElement>) => {
      event.preventDefault();

      if (!activeJob?.artifact_file || isDownloadingArtifact) {
        return;
      }

      const fileName = activeJob.artifact_file;
      setIsDownloadingArtifact(true);

      try {
        const blob = await downloadJobArtifact(activeJob.id);
        const tauriInvoke = getTauriInvoke();

        if (tauriInvoke) {
          try {
            const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
            const result = await tauriInvoke("save_tokenizer_artifact", {
              file_name: fileName,
              bytes,
            });
            const savedPath =
              typeof result === "string" && result.trim() !== "" ? result : null;

            if (savedPath) {
              notify("success", `Saved ${fileName}`, 4500);
            }
            return;
          } catch {
            // Fall back to browser-style download if the desktop command is unavailable.
          }
        }

        triggerBlobDownload(blob, fileName);
        notify("success", `Downloaded ${fileName}`, 4500);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to download tokenizer";
        notify("error", message, 6500);
      } finally {
        setIsDownloadingArtifact(false);
      }
    },
    [activeJob, isDownloadingArtifact, notify]
  );

  const previewReadyJobId =
    activeJob && activeJob.status === "completed" && activeJob.artifact_file
      ? activeJob.id
      : null;

  const previewSegments = useMemo(() => {
    if (!previewResult) {
      return [];
    }
    return makePreviewSegments(previewResult.text, previewResult.tokens);
  }, [previewResult]);

  useEffect(() => {
    if (!previewReadyJobId) {
      setPreviewResult(null);
      setPreviewError(null);
      setIsPreviewing(false);
      return;
    }

    const timer = window.setTimeout(() => {
      void runPreview(previewReadyJobId, previewText);
    }, 280);

    return () => {
      window.clearTimeout(timer);
    };
  }, [previewReadyJobId, previewText, runPreview]);

  const refreshJobs = useCallback(async () => {
    if (!hasHydratedLocalStateRef.current) {
      return;
    }

    try {
      const latest = await fetchTrainingJobs();
      setJobs(latest);
      const visibleRecentJobs = latest.filter((job) => !hiddenRecentJobIdSet.has(job.id));

      if (visibleRecentJobs.length === 0) {
        setActiveJobId(null);
        setActiveJob(null);
        return;
      }

      if (!activeJobId || hiddenRecentJobIdSet.has(activeJobId)) {
        setActiveJobId(visibleRecentJobs[0].id);
        setActiveJob(visibleRecentJobs[0]);
        return;
      }

      const selected = visibleRecentJobs.find((job) => job.id === activeJobId);
      if (selected) {
        setActiveJob(selected);
      } else {
        setActiveJobId(visibleRecentJobs[0].id);
        setActiveJob(visibleRecentJobs[0]);
      }
    } catch {
      // Non-blocking background refresh failure.
    }
  }, [activeJobId, hiddenRecentJobIdSet]);

  const visibleRecentJobs = useMemo(
    () => jobs.filter((job) => !hiddenRecentJobIdSet.has(job.id)),
    [jobs, hiddenRecentJobIdSet]
  );

  const handleRemoveRecentJob = useCallback(
    (jobId: string) => {
      if (activeJobId === jobId) {
        const nextVisibleJob =
          jobs.find((job) => job.id !== jobId && !hiddenRecentJobIdSet.has(job.id)) ?? null;
        setActiveJobId(nextVisibleJob?.id ?? null);
        setActiveJob(nextVisibleJob);
      }
      setHiddenRecentJobIds((previous) =>
        previous.includes(jobId) ? previous : [...previous, jobId]
      );
    },
    [activeJobId, hiddenRecentJobIdSet, jobs]
  );

  useEffect(() => {
    void refreshJobs();
    const timer = window.setInterval(() => {
      void refreshJobs();
    }, 4000);

    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  useEffect(() => {
    if (!hasHydratedLocalStateRef.current || !activeJobId) {
      return;
    }

    let cancelled = false;
    let isPolling = false;
    let hasCapturedInitialSnapshot = false;

    const poll = async () => {
      if (isPolling) {
        return;
      }
      isPolling = true;

      try {
        const job = await fetchTrainingJob(activeJobId);
        if (cancelled) {
          return;
        }

        const isInitialSnapshot = !hasCapturedInitialSnapshot;
        hasCapturedInitialSnapshot = true;
        const suppressTerminalToast =
          isInitialSnapshot && !locallyStartedJobIdsRef.current.has(job.id);

        setActiveJob(job);

        if (job.status === "completed") {
          const completedKey = `${job.id}:completed`;
          if (!jobNotificationKeysRef.current.has(completedKey)) {
            jobNotificationKeysRef.current.add(completedKey);
            if (!suppressTerminalToast) {
              notify(
                "success",
                `Training job ${job.id.slice(0, 8)} completed. Artifact is ready.`,
                6000
              );
            }
          }
          void refreshJobs();
        }

        if (job.status === "failed") {
          const failedKey = `${job.id}:failed`;
          if (!jobNotificationKeysRef.current.has(failedKey)) {
            jobNotificationKeysRef.current.add(failedKey);
            if (!suppressTerminalToast) {
              notify("error", job.error ?? "Training job failed", 7000);
            }
          }
          void refreshJobs();
        }
      } catch (error) {
        if (!cancelled) {
          const message =
            error instanceof Error ? error.message : "Failed to poll job status";
          notify("error", message, 7000);
        }
      } finally {
        isPolling = false;
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobId, notify, refreshJobs]);

  const uploadLocalTrainFiles = useCallback(
    async (selectedFiles: File[]) => {
      if (selectedFiles.length === 0) {
        return;
      }

      notify(
        "info",
        selectedFiles.length === 1
          ? `Uploading local train file: ${selectedFiles[0].name}`
          : `Uploading ${selectedFiles.length} local train files...`,
        2500
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
          setDatasetForm((previous) => ({
            ...previous,
            localTrainFiles: normalizeLocalTrainFiles([
              ...previous.localTrainFiles,
              ...successfulUploads.map((uploadedFile) =>
                makeLocalTrainFileEntry({
                  fileName: uploadedFile.file_name,
                  filePath: uploadedFile.file_path,
                  sizeBytes: uploadedFile.size_bytes,
                  sizeChars: uploadedFile.size_chars,
                })
              ),
            ]),
          }));
        }

        if (successfulUploads.length > 0) {
          notify(
            "success",
            successfulUploads.length === 1
              ? `Added ${stripGeneratedUploadPrefix(successfulUploads[0].file_name)}.`
              : `Added ${successfulUploads.length} local train files.`,
            4500
          );
        }

        const failedUploads = uploadResults.filter(
          (result): result is PromiseRejectedResult => result.status === "rejected"
        );
        if (failedUploads.length > 0) {
          const firstFailure = failedUploads[0];
          const firstFailureMessage =
            firstFailure?.reason instanceof Error
              ? firstFailure.reason.message
              : "Upload failed";
          notify(
            "error",
            `Failed to upload ${failedUploads.length} file(s). ${firstFailureMessage}`,
            8000
          );
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to upload local train files";
        notify("error", `Could not upload local train files. ${message}`, 7000);
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

  const handleLocalTrainFilesDragEnter = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      if (!Array.from(event.dataTransfer.types).includes("Files")) {
        return;
      }
      localFileDragDepthRef.current += 1;
      setIsDraggingTrainFiles(true);
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDragOver = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      event.dataTransfer.dropEffect = "copy";
      setIsDraggingTrainFiles(true);
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDragLeave = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      localFileDragDepthRef.current = Math.max(0, localFileDragDepthRef.current - 1);
      if (localFileDragDepthRef.current === 0) {
        setIsDraggingTrainFiles(false);
      }
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDrop = useCallback(
    async (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      localFileDragDepthRef.current = 0;
      setIsDraggingTrainFiles(false);
      if (controlsDisabled) {
        return;
      }
      const droppedFiles = Array.from(event.dataTransfer.files ?? []);
      await uploadLocalTrainFiles(droppedFiles);
    },
    [controlsDisabled, uploadLocalTrainFiles]
  );

  const handleLoadStreamingTemplate = async () => {
    notify("info", "Loading streaming dataset template...", 2500);
    setIsLoadingTemplate(true);

    try {
      const templates = await fetchConfigTemplates();
      const templateConfig = asRecord(templates.dataloader_config_template);
      const templateDatasetForm = datasetFormFromConfig(templateConfig);
      const templateTrainingForm = trainingFormFromConfig(templateConfig);

      setDatasetForm((previous) => ({
        ...previous,
        sourceMode: "streaming_hf",
        streamingDatasets: normalizeStreamingDatasetWeights(
          templateDatasetForm.streamingDatasets
        ),
      }));
      setTrainingForm((previous) => ({
        ...previous,
        budgetLimit: templateTrainingForm.budgetLimit,
        budgetUnit: templateTrainingForm.budgetUnit,
        budgetBehavior: templateTrainingForm.budgetBehavior,
      }));
      notify("success", "Loaded streaming template datasets.");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to load streaming template";
      notify("error", `Could not load streaming template. ${message}`, 7000);
    } finally {
      setIsLoadingTemplate(false);
    }
  };

  const handleTrain = async () => {
    notify("info", "Submitting training job...", 2500);
    setIsSubmitting(true);

    try {
      if (!hasValidationPassed) {
        throw new Error("Automatic validation must pass before starting training");
      }
      if (!tokenizerBuild.value) {
        throw new Error(tokenizerBuild.error ?? "Tokenizer config is invalid");
      }
      if (!dataloaderBuild.value) {
        throw new Error(dataloaderBuild.error ?? "Dataloader config is invalid");
      }

      const thresholds = parseThresholds(trainingForm.evaluationThresholds);

      const job = await createTrainingJob({
        tokenizer_config: tokenizerBuild.value,
        dataloader_config: dataloaderBuild.value,
        evaluation_thresholds: thresholds,
      });

      locallyStartedJobIdsRef.current.add(job.id);
      setActiveJobId(job.id);
      setActiveJob(job);
      notify("success", `Training job ${job.id.slice(0, 8)} started.`);
      await refreshJobs();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to create training job";
      notify("error", `Could not start training. ${message}`, 7000);
    } finally {
      setIsSubmitting(false);
    }
  };

  const localTrainFileCount = datasetForm.localTrainFiles.length;
  const localTrainKnownCharCount = datasetForm.localTrainFiles.reduce(
    (sum, entry) => sum + (entry.sizeChars ?? 0),
    0
  );
  const localTrainKnownCharLabel =
    localTrainKnownCharCount > 0 ? formatCharCount(localTrainKnownCharCount) : null;
  const localTrainFilesHint =
    localTrainFileCount === 0
      ? "Add one or more local files, or switch to streaming datasets in settings."
      : datasetForm.sourceMode === "local_file"
        ? `${localTrainFileCount} file${localTrainFileCount === 1 ? "" : "s"} ready${
            localTrainKnownCharLabel ? ` (${localTrainKnownCharLabel} chars)` : ""
          }.`
        : `${localTrainFileCount} local file${
            localTrainFileCount === 1 ? "" : "s"
          } stored${
            localTrainKnownCharLabel ? ` (${localTrainKnownCharLabel} chars)` : ""
          }. Switch dataset source to Local files to use them.`;
  const tokenizerReady = tokenizerBuild.error === null;
  const datasetReady = datasetBuild.error === null;
  const trainingRuntimeReady = trainingRuntimeBuild.error === null;
  const preflightReady =
    tokenizerBuild.error === null &&
    dataloaderBuild.error === null &&
    trainingRuntimeBuild.error === null;
  const hasTrainingInProgress =
    activeJob !== null &&
    activeJob.status !== "completed" &&
    activeJob.status !== "failed";
  const trainingCompleted = activeJob?.status === "completed";
  const canStartTraining =
    hasValidationPassed &&
    !isValidating &&
    !controlsDisabled &&
    !hasTrainingInProgress;
  const activeThresholds =
    trainingRuntimeBuild.value && Array.isArray(trainingRuntimeBuild.value.thresholds)
      ? (trainingRuntimeBuild.value.thresholds as number[]).join(", ")
      : null;

  return (
    <main className="studioRoot tokenizerPage">
      <TokenizerStudioNav
        themeMode={themeMode}
        onToggleTheme={() =>
          setThemeMode((previous) =>
            previous === "white" ? "dark" : "white"
          )
        }
      />

      <TokenizerWorkflowSection
        tokenizerReady={tokenizerReady}
        tokenizerError={tokenizerBuild.error}
        tokenizerType={tokenizerForm.tokenizerType}
        datasetReady={datasetReady}
        datasetError={datasetBuild.error}
        datasetSourceMode={datasetForm.sourceMode}
        localTrainFilesHint={localTrainFilesHint}
        streamingDatasetCount={datasetForm.streamingDatasets.length}
        trainingRuntimeReady={trainingRuntimeReady}
        trainingRuntimeError={trainingRuntimeBuild.error}
        budgetLimit={trainingForm.budgetLimit}
        budgetUnit={trainingForm.budgetUnit}
        activeThresholds={activeThresholds}
        hasValidationPassed={hasValidationPassed}
        isValidating={isValidating}
        validationError={validationError}
        preflightReady={preflightReady}
        controlsDisabled={controlsDisabled}
        trainingCompleted={trainingCompleted}
        hasTrainingInProgress={hasTrainingInProgress}
        activeJobState={activeJob?.state ?? null}
        isSubmitting={isSubmitting}
        canStartTraining={canStartTraining}
        onNavigateSettings={handleSettingsCategoryNavigation}
        onValidate={handleManualValidate}
        onTrain={handleTrain}
      />

      <section id="results" className="workspaceGrid">
        <article className="panelCard activeJobCard">
          <div className="sectionHeader">
            <h2>Active Job and Evaluation</h2>
            {activeJob ? <JobBadge job={activeJob} /> : null}
          </div>

          {activeJob ? (
            <>
              <div className="metaList">
                <div className="metaItem">
                  <span className="metaLabel">Job ID</span>
                  <span className="metaValue">{activeJob.id}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Status</span>
                  <span className="metaValue">{describeJobState(activeJob.state)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Stage</span>
                  <span className="metaValue">{activeJob.stage}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Progress</span>
                  <span className="metaValue">{formatPercent(activeJob.progress)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Evaluation</span>
                  <span className="metaValue">
                    {evaluationSourceLabel(activeJob.evaluation_source)}
                  </span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Created</span>
                  <span className="metaValue">{formatDate(activeJob.created_at)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Started</span>
                  <span className="metaValue">{formatDate(activeJob.started_at)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Finished</span>
                  <span className="metaValue">{formatDate(activeJob.finished_at)}</span>
                </div>
              </div>

              <div className="progressTrack">
                <div
                  className="progressFill"
                  style={{
                    width: `${Math.max(0, Math.min(activeJob.progress * 100, 100))}%`,
                  }}
                />
              </div>

              {activeJob.status === "completed" && activeJob.stats ? (
                <>
                  <div className="statsGrid">
                    <div>
                      <span>Tokens / Char</span>
                      <strong>{activeJob.stats.token_per_char.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Chars / Token</span>
                      <strong>{activeJob.stats.chars_per_token.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Records evaluated</span>
                      <strong>{activeJob.stats.num_records}</strong>
                    </div>
                    <div>
                      <span>Avg chars / record</span>
                      <strong>{activeJob.stats.avg_chars_per_record.toFixed(2)}</strong>
                    </div>
                    <div>
                      <span>Avg tokens / record</span>
                      <strong>{activeJob.stats.avg_tokens_per_record.toFixed(2)}</strong>
                    </div>
                    <div>
                      <span>Vocab</span>
                      <strong>{activeJob.stats.vocab_size}</strong>
                    </div>
                    <div>
                      <span>Used tokens</span>
                      <strong>{activeJob.stats.num_used_tokens}</strong>
                    </div>
                    <div>
                      <span>Unused tokens</span>
                      <strong>{activeJob.stats.num_unused_tokens}</strong>
                    </div>
                  </div>

                  <div className="rareTokenCard">
                    <h3>Rare Token Coverage</h3>
                    <div className="rareTokenTable">
                      {Object.entries(activeJob.stats.rare_tokens)
                        .sort(([left], [right]) => Number(left) - Number(right))
                        .map(([threshold, count]) => {
                          const fraction =
                            activeJob.stats?.rare_token_fraction?.[threshold] ?? 0;
                          return (
                            <div
                              key={`rare-threshold-${threshold}`}
                              className="rareTokenRow"
                            >
                              <span>{`< ${threshold}`}</span>
                              <strong>{count}</strong>
                              <span>{`${(fraction * 100).toFixed(2)}%`}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </>
              ) : null}

              {activeJob.status === "completed" && activeJob.artifact_file ? (
                <section className="tokenizerPreview">
                  <div className="sectionHeader tokenizerPreviewHeader">
                    <h3>Tokenizer Test Bench</h3>
                    <p className="metaLine">
                      {isPreviewing
                        ? "Tokenizing..."
                        : previewResult
                          ? `${previewResult.num_tokens} token${
                              previewResult.num_tokens === 1 ? "" : "s"
                            }`
                          : "-"}
                    </p>
                  </div>

                  <label className="fullWidthField">
                    Text to tokenize
                    <textarea
                      className="previewInput"
                      value={previewText}
                      onChange={(event) => setPreviewText(event.target.value)}
                      placeholder="Type text to preview tokenization..."
                      rows={4}
                    />
                  </label>

                  {previewError ? <p className="hint hint-error">{previewError}</p> : null}

                  {previewResult && previewResult.text.length > 0 ? (
                    <>
                      <div className="tokenizedText" aria-live="polite">
                        {previewSegments.map((segment, segmentIndex) => {
                          if (segment.kind === "plain") {
                            return (
                              <span key={`plain-${segmentIndex}`} className="plainSegment">
                                {segment.text}
                              </span>
                            );
                          }

                          const token = segment.token;
                          if (!token) {
                            return (
                              <span
                                key={`plain-fallback-${segmentIndex}`}
                                className="plainSegment"
                              >
                                {segment.text}
                              </span>
                            );
                          }

                          const hue = tokenHue(token.index);
                          return (
                            <span
                              key={`token-${token.index}-${token.start}-${token.end}`}
                              className="tokenMark"
                              title={`Token ID: ${token.id}`}
                              style={{
                                backgroundColor: `hsla(${hue}, 95%, 55%, 0.2)`,
                                borderColor: `hsla(${hue}, 70%, 55%, 0.7)`,
                              }}
                            >
                              {segment.text}
                            </span>
                          );
                        })}
                      </div>

                      <div className="tokenChipList" aria-label="Tokenizer output tokens">
                        {previewResult.tokens.map((token) => {
                          const hue = tokenHue(token.index);
                          return (
                            <span
                              key={`chip-${token.index}-${token.id}`}
                              className="tokenChip"
                              title={`Token ID: ${token.id}`}
                              style={{
                                backgroundColor: `hsla(${hue}, 95%, 55%, 0.17)`,
                                borderColor: `hsla(${hue}, 70%, 55%, 0.62)`,
                              }}
                            >
                              <code>{displayTokenLabel(token.token)}</code>
                            </span>
                          );
                        })}
                      </div>
                    </>
                  ) : previewText.trim() === "" ? (
                    <p className="metaLine">Enter text above to preview tokenization.</p>
                  ) : null}
                </section>
              ) : null}

              {activeJob.artifact_file ? (
                <a
                  className="downloadLink"
                  href={artifactDownloadUrl(activeJob.id)}
                  download={activeJob.artifact_file}
                  onClick={(event) => void handleDownloadArtifact(event)}
                  aria-disabled={isDownloadingArtifact}
                  rel="noreferrer"
                >
                  {isDownloadingArtifact
                    ? `Downloading ${activeJob.artifact_file}...`
                    : `Download ${activeJob.artifact_file}`}
                </a>
              ) : null}

              {activeJob.error ? <p className="hint hint-error">{activeJob.error}</p> : null}
            </>
          ) : (
            <p className="metaLine">No job selected yet.</p>
          )}
        </article>

        <article className="panelCard recentJobsCard">
          <div className="sectionHeader">
            <h2>Recent Jobs</h2>
            <button type="button" className="textButton" onClick={() => void refreshJobs()}>
              Refresh
            </button>
          </div>

          {visibleRecentJobs.length === 0 ? (
            <p className="metaLine">No recent jobs.</p>
          ) : (
            <div className="jobsList">
              {visibleRecentJobs.map((job) => {
                const jobName = String(job.tokenizer_config.name ?? "tokenizer");
                return (
                  <div
                    key={job.id}
                    className={`jobRow ${activeJobId === job.id ? "jobRow-active" : ""}`}
                  >
                    <button
                      type="button"
                      className="jobRowSelect"
                      onClick={() => {
                        setActiveJobId(job.id);
                        setActiveJob(job);
                      }}
                    >
                      <div>
                        <strong>{jobName}</strong>
                        <p>{job.id.slice(0, 8)}</p>
                        <p>{job.stage}</p>
                      </div>
                      <div className="jobRowMeta">
                        <JobBadge job={job} />
                        {job.status === "completed" || job.status === "failed" ? null : (
                          <p>{formatPercent(job.progress)}</p>
                        )}
                      </div>
                    </button>
                    <button
                      type="button"
                      className="jobRowRemove"
                      onClick={() => handleRemoveRecentJob(job.id)}
                      aria-label={`Remove ${jobName} from recent jobs`}
                      title="Remove from recent jobs"
                    >
                      <FiTrash2 aria-hidden="true" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </article>
      </section>

      <section id="settings" className="panelCard settingsStudio">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Settings</p>
            <h2>Configuration Studio</h2>
            <p className="panelCopy">
              Core fields stay visible. Advanced options are collapsed so most
              users can train and test quickly.
            </p>
          </div>
        </div>

        <details className="settingsPanel" open ref={tokenizerAndTrainingPanelRef}>
          <summary>Tokenizer and training budget</summary>
          <div className="settingsGrid">
            <div
              id="settings-tokenizer"
              ref={tokenizerCategoryRef}
              className={`settingsGroup settingsCategoryAnchor ${
                highlightedSettingsCategory === "tokenizer"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <div className="settingsGroupHeader">
                <h3>Tokenizer core</h3>
                <p className="settingsGroupHint">
                  Set tokenizer identity, model type, and vocabulary behavior.
                </p>
              </div>

              <div className="fieldGrid">
                <label>
                  Name
                  <input
                    value={tokenizerForm.name}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        name: event.target.value,
                      }))
                    }
                  />
                </label>

                <label>
                  Type
                  <select
                    value={tokenizerForm.tokenizerType}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        tokenizerType: event.target.value as TokenizerType,
                      }))
                    }
                  >
                    <option value="bpe">BPE</option>
                    <option value="wordpiece">WordPiece</option>
                    <option value="unigram">Unigram</option>
                  </select>
                </label>

                <label>
                  Vocab size
                  <input
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={tokenizerForm.vocabSize}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        vocabSize: sanitizePositiveIntegerInput(event.target.value),
                      }))
                    }
                  />
                </label>

                <label>
                  Min frequency
                  <input
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={tokenizerForm.minFrequency}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        minFrequency: sanitizePositiveIntegerInput(event.target.value),
                      }))
                    }
                  />
                </label>

                <label className="fullWidthField">
                  Special tokens (comma or new line)
                  <textarea
                    rows={2}
                    value={tokenizerForm.specialTokens}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        specialTokens: event.target.value,
                      }))
                    }
                  />
                </label>
              </div>

              {tokenizerForm.tokenizerType === "bpe" ? (
                <div className="inlineRow">
                  <label className="checkLabel">
                    <input
                      type="checkbox"
                      checked={tokenizerForm.byteFallback}
                      onChange={(event) =>
                        setTokenizerForm((previous) => ({
                          ...previous,
                          byteFallback: event.target.checked,
                        }))
                      }
                    />
                    Byte fallback
                  </label>

                  {!tokenizerForm.byteFallback ? (
                    <label className="inlineField">
                      Unknown token
                      <input
                        value={tokenizerForm.unkToken}
                        onChange={(event) =>
                          setTokenizerForm((previous) => ({
                            ...previous,
                            unkToken: event.target.value,
                          }))
                        }
                      />
                    </label>
                  ) : null}
                </div>
              ) : null}

              {tokenizerForm.tokenizerType === "wordpiece" ? (
                <label className="inlineField">
                  Unknown token
                  <input
                    value={tokenizerForm.unkToken}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        unkToken: event.target.value,
                      }))
                    }
                  />
                </label>
              ) : null}

              <p className={`hint ${tokenizerBuild.error ? "hint-error" : "hint-ok"}`}>
                {tokenizerBuild.error ?? "Tokenizer config looks valid."}
              </p>
            </div>

            <div
              id="settings-training"
              ref={trainingCategoryRef}
              className={`settingsGroup settingsCategoryAnchor ${
                highlightedSettingsCategory === "training"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <div className="settingsGroupHeader">
                <h3>Training budget</h3>
                <p className="settingsGroupHint">
                  Control how much dataset text is consumed during tokenizer training.
                </p>
              </div>
              <div className="fieldGrid">
                <label>
                  Budget limit
                  <input
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={trainingForm.budgetLimit}
                    onChange={(event) =>
                      setTrainingForm((previous) => ({
                        ...previous,
                        budgetLimit: sanitizePositiveIntegerInput(event.target.value),
                      }))
                    }
                  />
                </label>

                <label>
                  Budget unit
                  <select
                    value={trainingForm.budgetUnit}
                    onChange={(event) =>
                      setTrainingForm((previous) => ({
                        ...previous,
                        budgetUnit: event.target.value as BudgetUnit,
                      }))
                    }
                  >
                    <option value="chars">chars</option>
                    <option value="bytes">bytes</option>
                  </select>
                </label>

                <label>
                  Budget behavior
                  <select
                    value={trainingForm.budgetBehavior}
                    onChange={(event) =>
                      setTrainingForm((previous) => ({
                        ...previous,
                        budgetBehavior: event.target.value as BudgetBehavior,
                      }))
                    }
                  >
                    <option value="truncate">truncate</option>
                    <option value="stop">stop</option>
                  </select>
                </label>
              </div>
            </div>
          </div>
        </details>

        <details className="settingsPanel" open ref={datasetPanelRef}>
          <summary>Core dataset settings</summary>
          <div className="settingsGrid">
            <div
              id="settings-dataset"
              ref={datasetCategoryRef}
              className={`settingsCategoryAnchor ${
                highlightedSettingsCategory === "dataset"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
            <div className="sourceModeRow">
              <span>Dataset source</span>
              <div className="modeSwitch">
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetForm.sourceMode === "local_file"
                      ? "modeSwitchButton-active"
                      : ""
                  }`}
                  disabled={controlsDisabled}
                  onClick={() =>
                    setDatasetForm((previous) => ({
                      ...previous,
                      sourceMode: "local_file",
                    }))
                  }
                >
                  Local files
                </button>
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetForm.sourceMode === "streaming_hf"
                      ? "modeSwitchButton-active"
                      : ""
                  }`}
                  disabled={controlsDisabled}
                  onClick={() =>
                    setDatasetForm((previous) => ({
                      ...previous,
                      sourceMode: "streaming_hf",
                      streamingDatasets: normalizeStreamingDatasetWeights(
                        previous.streamingDatasets
                      ),
                    }))
                  }
                >
                  Streaming HF datasets
                </button>
              </div>
            </div>

            {datasetForm.sourceMode === "local_file" ? (
              <div className="datasetConfigurator">
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
                            controlsDisabled ? "localFileUploadButton-disabled" : ""
                          }`}
                          aria-disabled={controlsDisabled}
                        >
                          {isUploadingTrainFile ? "Uploading..." : "Add files"}
                          <input
                            type="file"
                            multiple
                            onChange={handleTrainFilesSelected}
                            disabled={controlsDisabled}
                          />
                        </label>
                        <button
                          type="button"
                          className="textButton localFileHeaderButton"
                          onClick={clearLocalTrainFiles}
                          disabled={controlsDisabled || localTrainFileCount === 0}
                        >
                          Remove all
                        </button>
                      </div>
                      <span className="localFileCount">
                        {localTrainFileCount} file{localTrainFileCount === 1 ? "" : "s"}
                      </span>
                    </div>
                  </div>

                  {localTrainFileCount === 0 ? (
                    <p className="filterEmpty">
                      No local files added yet. Add one or more files to train.
                    </p>
                  ) : (
                    <ul className="localFileList">
                      {datasetForm.localTrainFiles.map((entry) => {
                        const fileCharLabel = formatCharCount(entry.sizeChars);
                        return (
                          <li key={entry.id} className="localFileItem">
                            <strong className="localFileName" title={entry.fileName}>
                              {entry.fileName}
                            </strong>
                            <div className="localFileActions">
                              <span className="localFileStat">
                                {fileCharLabel ? `${fileCharLabel} chars` : "char count pending"}
                              </span>
                              <button
                                type="button"
                                className="textButton localFileRemoveIconButton"
                                onClick={() => removeLocalTrainFile(entry.id)}
                                disabled={controlsDisabled}
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

                  <span className="fieldNote">
                    Files are deduplicated by stored path.
                  </span>
                </div>
              </div>
            ) : (
              <div className="datasetConfigurator">
                <label className="fullWidthField">
                  HF access token (optional)
                  <input
                    type="password"
                    value={datasetForm.hfToken}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        hfToken: event.target.value,
                      }))
                    }
                    disabled={controlsDisabled}
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
                    disabled={controlsDisabled}
                  >
                    Add dataset
                  </button>
                  <button
                    type="button"
                    className="secondaryButton"
                    onClick={handleLoadStreamingTemplate}
                    disabled={controlsDisabled}
                  >
                    {isLoadingTemplate
                      ? "Loading template..."
                      : "Load streaming template"}
                  </button>
                </div>

                <div className="datasetList">
                  {datasetForm.streamingDatasets.map((entry, index) => (
                    <div key={entry.id} className="datasetCard">
                      <div className="datasetCardHeader">
                        <strong>Streaming dataset {index + 1}</strong>
                        <button
                          type="button"
                          className="textButton datasetRemoveButton"
                          onClick={() => removeStreamingDataset(entry.id)}
                          disabled={
                            controlsDisabled || datasetForm.streamingDatasets.length <= 1
                          }
                          aria-label={`Remove streaming dataset ${index + 1}`}
                          title={`Remove streaming dataset ${index + 1}`}
                        >
                          <FiTrash2 aria-hidden="true" />
                        </button>
                      </div>

                      <div className="fieldGrid">
                        <label>
                          HF dataset name
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

                        <label>
                          Split
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

                        <label>
                          Weight
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

                        <label className="fullWidthField">
                          Text columns
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
                          <label>
                            Config (optional)
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
                                disabled={controlsDisabled}
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
                                    <label>
                                      Column
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

                                    <label>
                                      Operator
                                      <select
                                        value={filter.operator}
                                        onChange={(event) =>
                                          updateStreamingFilter(entry.id, filter.id, {
                                            operator: event.target.value as FilterOperator,
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

                                    <label>
                                      Value
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
                                            : "en, true, 0.95, {\"k\":1}"
                                        }
                                      />
                                    </label>

                                    <button
                                      type="button"
                                      className="textButton filterRemoveButton"
                                      onClick={() =>
                                        removeStreamingFilter(entry.id, filter.id)
                                      }
                                      disabled={controlsDisabled}
                                    >
                                      Remove
                                    </button>
                                  </div>
                                ))}
                              </div>
                            )}
                            <p className="fieldNote">
                              Values are inferred automatically. For `in`/`not in`, use
                              a JSON array or comma-separated values.
                            </p>
                          </div>
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p className={`hint ${dataloaderBuild.error ? "hint-error" : "hint-ok"}`}>
              {dataloaderBuild.error ?? "Dataloader config looks valid."}
            </p>
            </div>
          </div>
        </details>

        <details className="settingsPanel">
          <summary>Advanced tokenizer settings</summary>
          <div className="settingsGrid">
            <div className="fieldGrid">
              <label>
                Pre-tokenizer
                <select
                  value={tokenizerForm.preTokenizer}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      preTokenizer: event.target.value as PreTokenizerType,
                    }))
                  }
                >
                  <option value="byte_level">byte_level</option>
                  <option value="whitespace">whitespace</option>
                  <option value="metaspace">metaspace</option>
                </select>
              </label>

              <label>
                Decoder
                <select
                  value={tokenizerForm.decoder}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      decoder: event.target.value as DecoderType,
                    }))
                  }
                >
                  <option value="byte_level">byte_level</option>
                  <option value="wordpiece">wordpiece</option>
                  <option value="metaspace">metaspace</option>
                </select>
              </label>
            </div>
          </div>
        </details>

        <details className="settingsPanel">
          <summary>Generated config JSON</summary>
          <div className="settingsGrid">
            <div className="jsonGrid">
              <label>
                Tokenizer JSON
                <pre>
                  {tokenizerBuild.value
                    ? prettyJson(tokenizerBuild.value)
                    : "Invalid config"}
                </pre>
              </label>

              <label>
                Dataloader JSON
                <pre>
                  {dataloaderBuild.value
                    ? prettyJson(dataloaderBuild.value)
                    : "Invalid config"}
                </pre>
              </label>
            </div>
          </div>
        </details>
      </section>

      <TokenizerToastViewport toasts={toasts} onRemoveToast={removeToast} />
    </main>
  );
}
