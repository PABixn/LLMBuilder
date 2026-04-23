"use client";

import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";

import { fetchLocalTrainFileStats } from "../../../lib/tokenizerLegacyApi";
import { normalizeLocalTrainFiles } from "../lib/dataset";
import type { DatasetFormState, LocalTrainFileFormState } from "../types";

type UseTokenizerLocalFileStatsArgs = {
  hasHydratedLocalState: boolean;
  localTrainFiles: LocalTrainFileFormState[];
  setDatasetForm: Dispatch<SetStateAction<DatasetFormState>>;
};

export function useTokenizerLocalFileStats({
  hasHydratedLocalState,
  localTrainFiles,
  setDatasetForm,
}: UseTokenizerLocalFileStatsArgs) {
  const localTrainFileStatsPendingIdsRef = useRef<Set<string>>(new Set());
  const localTrainFileStatsFailedIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!hasHydratedLocalState) {
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

      setDatasetForm((previous) => {
        let changed = false;
        const nextLocalTrainFiles = previous.localTrainFiles.map((entry) => {
          const stats = updatesById.get(entry.id);
          if (!stats) {
            return entry;
          }
          if (entry.sizeBytes === stats.sizeBytes && entry.sizeChars === stats.sizeChars) {
            return entry;
          }
          changed = true;
          return {
            ...entry,
            sizeBytes: stats.sizeBytes,
            sizeChars: stats.sizeChars,
          };
        });

        if (!changed) {
          return previous;
        }

        return {
          ...previous,
          localTrainFiles: normalizeLocalTrainFiles(nextLocalTrainFiles),
        };
      });
    });

    return () => {
      cancelled = true;
    };
  }, [hasHydratedLocalState, localTrainFiles, setDatasetForm]);
}
