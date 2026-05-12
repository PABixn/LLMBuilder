"use client";

import { useSearchParams } from "next/navigation";
import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useThemeMode } from "../../../lib/theme";
import {
  createTrainingJob,
  deleteTrainingJob,
  fetchTrainingConfigTemplates,
  fetchTrainingJobs,
  stopTrainingJob,
} from "../../../lib/training/jobs";
import {
  type TrainingBatchLrRecommendationOption,
  type TrainingFixSuggestion,
} from "../../../lib/training/types";
import { fitSchedulersToMaxSteps } from "./LearningRateSchedulePlanner";
import { ActiveRunPanel } from "./ActiveRunPanel";
import { AdvancedRuntimePanel } from "./AdvancedRuntimePanel";
import { AssetPickerDialog } from "./AssetPickerDialog";
import { DatasetSettingsPanel } from "./DatasetSettingsPanel";
import { ExecutionTargetPanel } from "./ExecutionTargetPanel";
import { GeneratedConfigPanel } from "./GeneratedConfigPanel";
import { PreflightPanel } from "./PreflightPanel";
import { TrainingHeroSection } from "./TrainingHeroSection";
import { RecentRunsPanel } from "./RecentRunsPanel";
import { SamplingPromptsPanel } from "./SamplingPromptsPanel";
import { TrainingStudioNav } from "./TrainingStudioNav";
import { TrainingPlanPanel } from "./TrainingPlanPanel";
import { TrainingToastStack } from "./TrainingToastStack";
import {
  TrainingWorkflowSection,
  type TrainingWorkflowStep,
} from "./TrainingWorkflowSection";
import {
  DATALOADER_CONFIG_STORAGE_KEY,
  POLL_INTERVAL_MS,
  TRAINING_CONFIG_STORAGE_KEY,
  WORKFLOW_TARGET_HASH_MAP,
} from "../constants";
import {
  canStopTrainingRun,
  defaultRunName,
} from "../lib/display";
import { formatInteger } from "../lib/metrics";
import {
  asNumber,
  asRecord,
  asRecordArray,
  cloneRecord,
  deleteAtPath,
  readStoredJson,
  updateAtPath,
  writeStoredJson,
} from "../lib/object";
import {
  formatLearningRate,
  replaceRunInOrder,
} from "../lib/run";
import { useRunPodSettings } from "../hooks/useRunPodSettings";
import { useAssetPicker } from "../hooks/useAssetPicker";
import { useDatasetSettings } from "../hooks/useDatasetSettings";
import { usePromptSettings } from "../hooks/usePromptSettings";
import { useTrainingPolling } from "../hooks/useTrainingPolling";
import {
  readInitialTrainingSelection,
  useTrainingSelection,
} from "../hooks/useTrainingSelection";
import { useTrainingPreflight } from "../hooks/useTrainingPreflight";
import { useTrainingToasts } from "../hooks/useTrainingToasts";
import type {
  WorkflowTarget,
} from "../types";

export function TrainingPageContent() {
  const searchParams = useSearchParams();
  const [theme, setTheme] = useThemeMode();
  const [trainingConfig, setTrainingConfig] = useState<Record<string, unknown> | null>(null);
  const [dataloaderConfig, setDataloaderConfig] = useState<Record<string, unknown> | null>(null);
  const [runName, setRunName] = useState("");
  const [runNameDirty, setRunNameDirty] = useState(false);
  const [isActiveRunOpen, setIsActiveRunOpen] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const {
    buildExecutionTarget,
    confirmLaunch: confirmRunPodLaunch,
    executionKind,
    handleValidateRunPodKey,
    runPodApiKey,
    runPodCleanupPod,
    runPodCloudType,
    runPodDataCenterId,
    runPodGpuCount,
    runPodGpuType,
    runPodInterruptible,
    runPodReady,
    runPodStatus,
    runPodValidationLoading,
    runPodValidationMessage,
    runPodVolumeSizeGb,
    setExecutionKind,
    setRunPodApiKey,
    setRunPodCleanupPod,
    setRunPodCloudType,
    setRunPodDataCenterId,
    setRunPodGpuCount,
    setRunPodGpuType,
    setRunPodInterruptible,
    setRunPodVolumeSizeGb,
  } = useRunPodSettings();
  const { notify, toasts } = useTrainingToasts();
  const {
    activeRunId,
    initializeTrainingSelection,
    selectedProject,
    selectedProjectId,
    selectedProjectRefreshId,
    selectedTokenizer,
    selectedTokenizerJobId,
    setActiveRunId,
    setSelectedProjectId,
    setSelectedTokenizerJobId,
  } = useTrainingSelection({ notify });
  const {
    activeRun,
    checkpoints,
    logs,
    metrics,
    recentRuns,
    refreshRecentRuns,
    samples,
    setActiveRun,
    setRecentRuns,
  } = useTrainingPolling({
    activeRunId,
    notify,
    setActiveRunOpen: setIsActiveRunOpen,
    setActiveRunId,
  });
  const {
    preflight,
    preflightError,
    preflightLoading,
    refreshPreflight,
    selectedRecommendationOptionKey,
    setSelectedRecommendationOptionKey,
  } = useTrainingPreflight({
    dataloaderConfig,
    selectedProjectRefreshId,
    selectedProjectId,
    selectedTokenizerJobId,
    trainingConfig,
  });
  const {
    closePicker,
    openPicker,
    pickerError,
    pickerKind,
    pickerLoading,
    pickerQuery,
    setPickerQuery,
    visiblePickerProjects,
    visiblePickerTokenizerJobs,
  } = useAssetPicker();
  const {
    addStreamingDataset,
    addStreamingFilter,
    clearLocalTrainFiles,
    datasetSourceMode,
    handleLoadStreamingTemplate,
    handleLocalTrainFilesDragEnter,
    handleLocalTrainFilesDragLeave,
    handleLocalTrainFilesDragOver,
    handleLocalTrainFilesDrop,
    handleTrainFilesSelected,
    hfToken,
    isDraggingTrainFiles,
    isLoadingDatasetTemplate,
    isUploadingTrainFile,
    localTrainFiles,
    removeLocalTrainFile,
    removeStreamingDataset,
    removeStreamingFilter,
    selectLocalDatasetSource,
    selectStreamingDatasetSource,
    setHfToken,
    streamingDatasets,
    updateStreamingDataset,
    updateStreamingFilter,
    updateStreamingWeight,
  } = useDatasetSettings({
    dataloaderConfig,
    notify,
    setDataloaderConfig,
  });
  const {
    handleAddPrompt,
    handlePromptChange,
    handleRemovePrompt,
    handleResetPrompts,
    isResettingPrompts,
    promptEntries,
  } = usePromptSettings({
    notify,
    setTrainingConfig,
    trainingConfig,
  });
  const [highlightedWorkflowTarget, setHighlightedWorkflowTarget] =
    useState<WorkflowTarget | null>(null);
  const initializedRef = useRef(false);
  const workflowHighlightTimeoutRef = useRef<number | null>(null);
  const trainingPlanPanelRef = useRef<HTMLDetailsElement | null>(null);
  const datasetPanelRef = useRef<HTMLDetailsElement | null>(null);
  const autoRunNameRef = useRef("");
  const modelSelectionRef = useRef<HTMLDivElement | null>(null);
  const tokenizerSelectionRef = useRef<HTMLDivElement | null>(null);
  const trainingSettingsRef = useRef<HTMLDivElement | null>(null);
  const datasetSettingsRef = useRef<HTMLDivElement | null>(null);
  const preflightSectionRef = useRef<HTMLElement | null>(null);

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
    const initialSelection = readInitialTrainingSelection(searchParams);

    initializeTrainingSelection(initialSelection);

    void Promise.all([fetchTrainingConfigTemplates(), fetchTrainingJobs()])
      .then(([templates, jobs]) => {
        startTransition(() => {
          setTrainingConfig(storedTraining ?? templates.training_config_template);
          setDataloaderConfig(storedDataloader ?? templates.dataloader_config_template);
          setRecentRuns(jobs);
          if (initialSelection.shouldSelectMostRecentRun) {
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
  }, [initializeTrainingSelection, notify, searchParams, setActiveRunId]);

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

  const datasetEntries = useMemo(
    () => asRecordArray(dataloaderConfig?.datasets),
    [dataloaderConfig]
  );

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
    if (scope === "both") {
      handleMaxStepsChange(option.recommended_max_steps);
    }

    const summary =
      scope === "batch"
        ? `Set total batch size to ${formatInteger(option.total_batch_size)} tokens.`
        : scope === "lr"
          ? `Set learning rate to ${formatLearningRate(option.learning_rate)}.`
          : `Set total batch size to ${formatInteger(option.total_batch_size)} tokens, learning rate to ${formatLearningRate(option.learning_rate)}, and max training steps to ${formatInteger(option.recommended_max_steps)}.`;
    const microNote =
      scope !== "lr" && option.clear_manual_micro_batch
        ? " Cleared manual micro batch size so preflight can auto-select the best micro step."
        : "";
    const schedulerNote =
      scope === "both"
        ? " Refit the scheduler phases to match the recommended training length."
        : "";
    notify("success", `${option.label} recommendation applied`, `${summary}${microNote}${schedulerNote}`);
  };

  const handleStartTraining = async () => {
    if (!selectedProjectId || !selectedTokenizerJobId || !trainingConfig || !dataloaderConfig || !preflight?.valid) {
      notify("error", "Training blocked", "Resolve the preflight issues before launching.");
      return;
    }
    if (executionKind === "runpod_pod" && !runPodStatus?.configured && runPodApiKey.trim() === "") {
      notify("error", "RunPod key required", "Paste and validate a RunPod API key before launching a pod.");
      return;
    }
    if (!confirmRunPodLaunch()) {
      return;
    }
    setLaunching(true);
    try {
      const executionTarget = buildExecutionTarget();
      const job = await createTrainingJob({
        name: runName.trim() || undefined,
        project_id: selectedProjectId,
        tokenizer_job_id: selectedTokenizerJobId,
        training_config: trainingConfig,
        dataloader_config: dataloaderConfig,
        execution_target: executionTarget,
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

  const startReady = Boolean(preflight?.valid && selectedProjectId && selectedTokenizerJobId && !launching && runPodReady);
  const trainingRuntimeReady = Boolean(trainingConfig && dataloaderConfig);
  const hasTrainingInProgress =
    activeRun?.status === "running" || activeRun?.status === "pending";
  const activeRunCanBeStopped = canStopTrainingRun(activeRun);
  const stoppingActiveRun = activeRunCanBeStopped && stoppingRunId === activeRun.id;
  const batchAndLrRecommendation = preflight?.batch_and_lr_recommendation ?? null;
  const trainingCompleted = activeRun?.status === "completed";
  const workflowSteps: TrainingWorkflowStep[] = [
    {
      title: "Step 1 - Choose saved model",
      state: selectedProject ? "ready" : "waiting",
      status: selectedProject ? "Ready" : "Waiting for configuration",
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
      actionLabel: "Open tokenizer selection",
      onAction: () => openWorkflowTarget("tokenizer"),
    },
    {
      title: "Step 3 - Configure training run",
      state: trainingRuntimeReady ? "ready" : "waiting",
      status: trainingRuntimeReady ? "Ready" : "Waiting for configuration",
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
        <ActiveRunPanel
          activeRun={activeRun}
          checkpoints={checkpoints}
          logs={logs}
          metrics={metrics}
          onClose={() => setIsActiveRunOpen(false)}
          onStopRun={(jobId) => void handleStopTraining(jobId)}
          pollIntervalSeconds={Math.round(POLL_INTERVAL_MS / 1000)}
          samples={samples}
          stoppingRunId={stoppingRunId}
        />
      ) : null}

        <div className="trainingPanelStack">
          <PreflightPanel
            ref={preflightSectionRef}
            highlighted={highlightedWorkflowTarget === "preflight"}
            onApplyFix={applyFix}
            preflight={preflight}
            preflightError={preflightError}
          />
        </div>

        <div className="trainingPanelStack">
          <RecentRunsPanel
            activeRunId={activeRunId}
            onDeleteRun={(jobId) => void handleDeleteRun(jobId)}
            onRefresh={() => void refreshRecentRuns()}
            onSelectRun={handleSelectRecentRun}
            onStopRun={(jobId) => void handleStopTraining(jobId)}
            recentRuns={recentRuns}
            stoppingRunId={stoppingRunId}
          />
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
            <ExecutionTargetPanel
              executionKind={executionKind}
              onExecutionKindChange={setExecutionKind}
              onValidateRunPodKey={() => void handleValidateRunPodKey()}
              runPodApiKey={runPodApiKey}
              runPodCleanupPod={runPodCleanupPod}
              runPodCloudType={runPodCloudType}
              runPodDataCenterId={runPodDataCenterId}
              runPodGpuCount={runPodGpuCount}
              runPodGpuType={runPodGpuType}
              runPodInterruptible={runPodInterruptible}
              runPodStatus={runPodStatus}
              runPodValidationLoading={runPodValidationLoading}
              runPodValidationMessage={runPodValidationMessage}
              runPodVolumeSizeGb={runPodVolumeSizeGb}
              setRunPodApiKey={setRunPodApiKey}
              setRunPodCleanupPod={setRunPodCleanupPod}
              setRunPodCloudType={setRunPodCloudType}
              setRunPodDataCenterId={setRunPodDataCenterId}
              setRunPodGpuCount={setRunPodGpuCount}
              setRunPodGpuType={setRunPodGpuType}
              setRunPodInterruptible={setRunPodInterruptible}
              setRunPodVolumeSizeGb={setRunPodVolumeSizeGb}
            />

            <TrainingPlanPanel
              ref={trainingPlanPanelRef}
              dataloaderConfig={dataloaderConfig}
              handleDataloaderField={handleDataloaderField}
              handleLrSchedulersChange={handleLrSchedulersChange}
              handleMaxStepsChange={handleMaxStepsChange}
              handleOptionalTrainingField={handleOptionalTrainingField}
              handleTrainingField={handleTrainingField}
              highlighted={highlightedWorkflowTarget === "training"}
              onApplyRecommendation={applyRecommendationOption}
              onRefreshRecommendation={refreshPreflight}
              preflightError={preflightError}
              preflightLoading={preflightLoading}
              recommendation={batchAndLrRecommendation}
              selectedRecommendationOptionKey={selectedRecommendationOptionKey}
              setSelectedRecommendationOptionKey={setSelectedRecommendationOptionKey}
              trainingConfig={trainingConfig}
              trainingSettingsRef={trainingSettingsRef}
            />

            <DatasetSettingsPanel
              ref={datasetPanelRef}
              addStreamingDataset={addStreamingDataset}
              addStreamingFilter={addStreamingFilter}
              clearLocalTrainFiles={clearLocalTrainFiles}
              datasetSettingsRef={datasetSettingsRef}
              datasetSourceMode={datasetSourceMode}
              handleLoadStreamingTemplate={() => void handleLoadStreamingTemplate()}
              handleLocalTrainFilesDragEnter={handleLocalTrainFilesDragEnter}
              handleLocalTrainFilesDragLeave={handleLocalTrainFilesDragLeave}
              handleLocalTrainFilesDragOver={handleLocalTrainFilesDragOver}
              handleLocalTrainFilesDrop={handleLocalTrainFilesDrop}
              handleTrainFilesSelected={handleTrainFilesSelected}
              hfToken={hfToken}
              highlighted={highlightedWorkflowTarget === "dataset"}
              isDraggingTrainFiles={isDraggingTrainFiles}
              isLoadingDatasetTemplate={isLoadingDatasetTemplate}
              isUploadingTrainFile={isUploadingTrainFile}
              localTrainFiles={localTrainFiles}
              removeLocalTrainFile={removeLocalTrainFile}
              removeStreamingDataset={removeStreamingDataset}
              removeStreamingFilter={removeStreamingFilter}
              selectLocalDatasetSource={selectLocalDatasetSource}
              selectStreamingDatasetSource={selectStreamingDatasetSource}
              setHfToken={setHfToken}
              streamingDatasets={streamingDatasets}
              updateStreamingDataset={updateStreamingDataset}
              updateStreamingFilter={updateStreamingFilter}
              updateStreamingWeight={updateStreamingWeight}
            />

            <SamplingPromptsPanel
              handleAddPrompt={handleAddPrompt}
              handlePromptChange={handlePromptChange}
              handleRemovePrompt={handleRemovePrompt}
              handleResetPrompts={() => void handleResetPrompts()}
              isResettingPrompts={isResettingPrompts}
              promptEntries={promptEntries}
            />

            <AdvancedRuntimePanel
              dataloaderConfig={dataloaderConfig}
              handleDataloaderField={handleDataloaderField}
              handleTrainingField={handleTrainingField}
              trainingConfig={trainingConfig}
            />

            <GeneratedConfigPanel
              dataloaderConfig={dataloaderConfig}
              trainingConfig={trainingConfig}
            />
          </div>
        ) : (
          <div className="trainingEmpty">Loading starter templates…</div>
        )}
      </section>

      <AssetPickerDialog
        onClose={closePicker}
        onOpenPicker={(kind) => void openPicker(kind)}
        onProjectSelected={setSelectedProjectId}
        onQueryChange={setPickerQuery}
        onTokenizerSelected={setSelectedTokenizerJobId}
        pickerError={pickerError}
        pickerKind={pickerKind}
        pickerLoading={pickerLoading}
        pickerQuery={pickerQuery}
        selectedProjectId={selectedProjectId}
        selectedTokenizerJobId={selectedTokenizerJobId}
        visiblePickerProjects={visiblePickerProjects}
        visiblePickerTokenizerJobs={visiblePickerTokenizerJobs}
      />

      <TrainingToastStack toasts={toasts} />
    </main>
  );
}
