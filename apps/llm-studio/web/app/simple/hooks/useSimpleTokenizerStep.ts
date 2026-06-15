"use client";

import { startTransition, useEffect, useMemo, useRef, useState } from "react";

import {
  createTrainingJob,
  fetchTrainingJob,
  previewJobTokenizer,
  uploadTrainFile,
  validateDataloaderConfig,
  validateTokenizerConfig,
  type TrainingJob,
  type TokenizerPreviewResult,
} from "../../../lib/tokenizerLegacyApi";
import { invalidateWorkspaceAssetInventory } from "../../../lib/workspaceAssets";
import { SIMPLE_DEFAULT_PROMPT, SIMPLE_POLL_INTERVAL_MS } from "../constants";
import {
  buildSimpleTokenizerConfig,
  buildSimpleTokenizerDataloaderConfig,
  simpleDatasetBlocker,
  tokenizerBudgetForDataset,
} from "../lib/tokenizerDefaults";
import { targetVocabForPresetDataset } from "../lib/modelPresets";
import { readTokenizerVocabSize } from "../lib/vocabularySync";
import type {
  SimpleFlowState,
  SimpleLocalTrainFile,
  SimpleTokenizerStepState,
} from "../types";

interface UseSimpleTokenizerStepOptions {
  flow: SimpleFlowState;
  updateFlow: (updater: (current: SimpleFlowState) => SimpleFlowState) => void;
  syncProjectVocab: (vocabSize: number) => Promise<unknown>;
}

function fileNameFromPath(filePath: string): string {
  return filePath.replaceAll("\\", "/").split("/").pop() || "Training file";
}

function normalizeFiles(files: SimpleLocalTrainFile[]): SimpleLocalTrainFile[] {
  const byPath = new Map<string, SimpleLocalTrainFile>();
  for (const file of files) {
    const filePath = file.filePath.trim();
    if (!filePath) {
      continue;
    }
    byPath.set(filePath, {
      ...file,
      filePath,
      fileName: file.fileName.trim() || fileNameFromPath(filePath),
      sizeBytes:
        typeof file.sizeBytes === "number" && Number.isFinite(file.sizeBytes) && file.sizeBytes >= 0
          ? file.sizeBytes
          : null,
      sizeChars:
        typeof file.sizeChars === "number" && Number.isFinite(file.sizeChars) && file.sizeChars >= 0
          ? file.sizeChars
          : null,
    });
  }
  return Array.from(byPath.values());
}

function tokenizerNameForModel(modelName: string): string {
  const slug = modelName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `${slug || "simple"}_bpe_bytelevel`;
}

export function useSimpleTokenizerStep({
  flow,
  syncProjectVocab,
  updateFlow,
}: UseSimpleTokenizerStepOptions): SimpleTokenizerStepState {
  const [tokenizerJob, setTokenizerJob] = useState<TrainingJob | null>(null);
  const [tokenizerError, setTokenizerError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [previewText, setPreviewText] = useState(SIMPLE_DEFAULT_PROMPT);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewResult, setPreviewResult] = useState<TokenizerPreviewResult | null>(null);
  const syncedTokenizerJobIdRef = useRef<string | null>(null);

  const budgetLimit = tokenizerBudgetForDataset(flow.datasetSource, flow.targetVocabSize);
  const tokenizerConfig = useMemo(
    () =>
      buildSimpleTokenizerConfig({
        name: tokenizerNameForModel(flow.modelName),
        vocabSize: flow.targetVocabSize,
      }),
    [flow.modelName, flow.targetVocabSize]
  );
  const dataloaderConfig = useMemo(
    () =>
      buildSimpleTokenizerDataloaderConfig({
        budgetLimit,
        datasetSource: flow.datasetSource,
        localTrainFiles: flow.localTrainFiles,
        streamingAdditionalDatasetIds: flow.streamingAdditionalDatasetIds,
        streamingPrimaryDatasetId: flow.streamingPrimaryDatasetId,
      }),
    [
      budgetLimit,
      flow.datasetSource,
      flow.localTrainFiles,
      flow.streamingAdditionalDatasetIds,
      flow.streamingPrimaryDatasetId,
    ]
  );
  const datasetBlocker = simpleDatasetBlocker(flow.datasetSource, flow.localTrainFiles);
  const datasetReady = datasetBlocker === null;

  useEffect(() => {
    const recommendedVocab = targetVocabForPresetDataset(flow.presetId, flow.datasetSource);
    if (flow.projectId || flow.tokenizerJobId || flow.targetVocabSize === recommendedVocab) {
      return;
    }
    updateFlow((current) => ({
      ...current,
      targetVocabSize: recommendedVocab,
    }));
  }, [
    flow.datasetSource,
    flow.presetId,
    flow.projectId,
    flow.targetVocabSize,
    flow.tokenizerJobId,
    updateFlow,
  ]);

  useEffect(() => {
    if (!flow.tokenizerJobId) {
      setTokenizerJob(null);
      setTokenizerError(null);
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const poll = async () => {
      try {
        const job = await fetchTrainingJob(flow.tokenizerJobId as string);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setTokenizerJob(job);
          setTokenizerError(job.error);
        });
        if (job.status === "pending" || job.status === "running") {
          timeoutId = window.setTimeout(() => {
            void poll();
          }, SIMPLE_POLL_INTERVAL_MS);
        }
      } catch (error) {
        if (!cancelled) {
          startTransition(() => {
            setTokenizerJob(null);
            setTokenizerError(
              error instanceof Error ? error.message : "Tokenizer job could not be loaded."
            );
          });
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
  }, [flow.tokenizerJobId]);

  useEffect(() => {
    if (!tokenizerJob || tokenizerJob.status !== "completed") {
      return;
    }
    const vocabSize = readTokenizerVocabSize(tokenizerJob);
    if (!vocabSize || syncedTokenizerJobIdRef.current === tokenizerJob.id) {
      return;
    }
    syncedTokenizerJobIdRef.current = tokenizerJob.id;
    void syncProjectVocab(vocabSize).finally(() => {
      updateFlow((current) => ({
        ...current,
        tokenizerJobId: tokenizerJob.id,
        targetVocabSize: vocabSize,
        lastCompletedStep: "tokenizer",
      }));
      invalidateWorkspaceAssetInventory();
    });
  }, [syncProjectVocab, tokenizerJob, updateFlow]);

  useEffect(() => {
    if (!tokenizerJob || tokenizerJob.status !== "completed" || !tokenizerJob.artifact_file) {
      setPreviewResult(null);
      setPreviewError(null);
      setPreviewing(false);
      return;
    }

    let cancelled = false;
    const timeoutId = window.setTimeout(() => {
      setPreviewing(true);
      setPreviewError(null);
      void previewJobTokenizer(tokenizerJob.id, { text: previewText })
        .then((result) => {
          if (!cancelled) {
            setPreviewResult(result);
          }
        })
        .catch((error) => {
          if (!cancelled) {
            setPreviewResult(null);
            setPreviewError(error instanceof Error ? error.message : "Tokenizer preview failed.");
          }
        })
        .finally(() => {
          if (!cancelled) {
            setPreviewing(false);
          }
        });
    }, 280);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [previewText, tokenizerJob]);

  const updatePreviewText = (value: string) => {
    setPreviewText(value.slice(0, 50_000));
  };

  const validateTokenizer = async (): Promise<boolean> => {
    if (!datasetReady) {
      setValidationError(datasetBlocker ?? "Choose a dataset.");
      return false;
    }
    setValidating(true);
    setValidationError(null);
    setValidationMessage(null);
    try {
      await Promise.all([
        validateTokenizerConfig(tokenizerConfig),
        validateDataloaderConfig(dataloaderConfig),
      ]);
      setValidationMessage("Tokenizer and data settings are valid.");
      return true;
    } catch (error) {
      setValidationError(
        error instanceof Error ? error.message : "Tokenizer validation failed."
      );
      return false;
    } finally {
      setValidating(false);
    }
  };

  const startTokenizerTraining = async () => {
    const valid = await validateTokenizer();
    if (!valid) {
      return;
    }
    setTraining(true);
    setTokenizerError(null);
    setValidationMessage(null);
    try {
      const job = await createTrainingJob({
        tokenizer_config: tokenizerConfig,
        dataloader_config: dataloaderConfig,
        evaluation_thresholds: [5, 10, 25],
      });
      setTokenizerJob(job);
      updateFlow((current) => ({
        ...current,
        tokenizerJobId: job.id,
      }));
      invalidateWorkspaceAssetInventory();
    } catch (error) {
      setTokenizerError(
        error instanceof Error ? error.message : "Could not start tokenizer training."
      );
    } finally {
      setTraining(false);
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }
    setUploading(true);
    setValidationError(null);
    try {
      const results = await Promise.all(files.map((file) => uploadTrainFile(file)));
      const uploadedFiles = results.map<SimpleLocalTrainFile>((file) => ({
        id: `local-file-${file.file_path}`,
        fileName: file.file_name,
        filePath: file.file_path,
        sizeBytes: file.size_bytes,
        sizeChars: file.size_chars,
      }));
      updateFlow((current) => ({
        ...current,
        datasetSource: "upload",
        localTrainFiles: normalizeFiles([...current.localTrainFiles, ...uploadedFiles]),
      }));
      setValidationMessage(
        results.length === 1 ? `Added ${results[0].file_name}.` : `Added ${results.length} files.`
      );
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "File upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const removeLocalFile = (fileId: string) => {
    updateFlow((current) => ({
      ...current,
      localTrainFiles: current.localTrainFiles.filter((file) => file.id !== fileId),
    }));
  };

  const clearLocalFiles = () => {
    updateFlow((current) => ({
      ...current,
      localTrainFiles: [],
    }));
  };

  return {
    tokenizerJob,
    tokenizerError,
    validationError,
    validationMessage,
    validating,
    uploading,
    training,
    previewing,
    previewText,
    previewError,
    previewResult,
    tokenizerConfig,
    dataloaderConfig,
    datasetReady,
    datasetBlocker,
    setPreviewText: updatePreviewText,
    uploadFiles,
    validateTokenizer,
    startTokenizerTraining,
    removeLocalFile,
    clearLocalFiles,
  };
}
