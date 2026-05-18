"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type Dispatch,
  type DragEvent,
  type SetStateAction,
} from "react";

import { fetchTrainingConfigTemplates } from "../../../lib/training/jobs";
import {
  fetchLocalTrainFileStats,
  uploadTrainFile,
} from "../../../lib/tokenizerLegacyApi";
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
import { stripGeneratedUploadPrefix } from "../lib/files";
import {
  asRecord,
  cloneRecord,
} from "../lib/object";
import type {
  DatasetSourceMode,
  LocalTrainFileFormState,
  StreamingDatasetFormState,
  StreamingFilterFormState,
  ToastLevel,
} from "../types";

type NotifyDatasetSettings = (level: ToastLevel, title: string, body: string) => void;

interface UseDatasetSettingsOptions {
  dataloaderConfig: Record<string, unknown> | null;
  notify: NotifyDatasetSettings;
  setDataloaderConfig: Dispatch<SetStateAction<Record<string, unknown> | null>>;
}

export function useDatasetSettings({
  dataloaderConfig,
  notify,
  setDataloaderConfig,
}: UseDatasetSettingsOptions) {
  const [datasetSourceMode, setDatasetSourceMode] =
    useState<DatasetSourceMode>("streaming_hf");
  const [localTrainFiles, setLocalTrainFiles] = useState<LocalTrainFileFormState[]>([]);
  const [hfToken, setHfToken] = useState("");
  const [streamingDatasets, setStreamingDatasets] = useState<StreamingDatasetFormState[]>([
    makeStreamingDatasetEntry(),
  ]);
  const [isDraggingTrainFiles, setIsDraggingTrainFiles] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isLoadingDatasetTemplate, setIsLoadingDatasetTemplate] = useState(false);
  const datasetUiHydratedRef = useRef(false);
  const localFileDragDepthRef = useRef(0);
  const localTrainFileStatsPendingIdsRef = useRef(new Set<string>());
  const localTrainFileStatsFailedIdsRef = useRef(new Set<string>());

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
  }, [datasetSourceMode, hfToken, localTrainFiles, setDataloaderConfig, streamingDatasets]);

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

  const selectLocalDatasetSource = useCallback(() => {
    setDatasetSourceMode("local_file");
  }, []);

  const selectStreamingDatasetSource = useCallback(() => {
    setDatasetSourceMode("streaming_hf");
    setStreamingDatasets((previous) => normalizeStreamingDatasetWeights(previous));
  }, []);

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
        "Uploading files",
        selectedFiles.length === 1
          ? `Uploading ${selectedFiles[0].name}.`
          : `Uploading ${selectedFiles.length} files.`
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
            "Files added",
            successfulUploads.length === 1
              ? `Added ${stripGeneratedUploadPrefix(successfulUploads[0].file_name)}.`
              : `Added ${successfulUploads.length} files.`
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
            "Upload failed",
            `Could not upload ${failedUploads.length} file(s). ${firstFailureMessage}`
          );
        }
      } catch (error) {
        notify(
          "error",
          "Upload failed",
          error instanceof Error ? error.message : "Could not upload files."
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
    notify("info", "Loading template", "Refreshing dataset settings.");
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
      notify("success", "Template loaded", "Loaded dataset defaults.");
    } catch (error) {
      notify(
        "error",
        "Template unavailable",
        error instanceof Error ? error.message : "Could not load the dataset template."
      );
    } finally {
      setIsLoadingDatasetTemplate(false);
    }
  }, [notify, setDataloaderConfig]);

  return {
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
  };
}
