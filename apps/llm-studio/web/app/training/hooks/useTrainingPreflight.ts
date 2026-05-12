"use client";

import {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useState,
} from "react";

import { validateTrainingPreflight } from "../../../lib/training/jobs";
import type {
  TrainingBatchLrRecommendation,
  TrainingPreflightResponse,
} from "../../../lib/training/types";

interface UseTrainingPreflightOptions {
  dataloaderConfig: Record<string, unknown> | null;
  selectedProjectRefreshId?: number;
  selectedProjectId: string | null;
  selectedTokenizerJobId: string | null;
  trainingConfig: Record<string, unknown> | null;
}

export function selectRecommendationOptionKey(
  current: string | null,
  recommendation: TrainingBatchLrRecommendation | null | undefined
): string | null {
  if (!recommendation || recommendation.options.length === 0) {
    return null;
  }
  if (current && recommendation.options.some((option) => option.key === current)) {
    return current;
  }
  return recommendation.recommended_option_key;
}

export function useTrainingPreflight({
  dataloaderConfig,
  selectedProjectRefreshId = 0,
  selectedProjectId,
  selectedTokenizerJobId,
  trainingConfig,
}: UseTrainingPreflightOptions) {
  const [preflight, setPreflight] = useState<TrainingPreflightResponse | null>(null);
  const [selectedRecommendationOptionKey, setSelectedRecommendationOptionKey] =
    useState<string | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [manualPreflightRefreshId, setManualPreflightRefreshId] = useState(0);

  const deferredTrainingConfig = useDeferredValue(trainingConfig);
  const deferredDataloaderConfig = useDeferredValue(dataloaderConfig);
  const refreshPreflight = useCallback(() => {
    setPreflightLoading(true);
    setManualPreflightRefreshId((current) => current + 1);
  }, []);

  useEffect(() => {
    if (!selectedProjectId || !selectedTokenizerJobId || !deferredTrainingConfig || !deferredDataloaderConfig) {
      setPreflight(null);
      setPreflightError(null);
      setPreflightLoading(false);
      return;
    }

    const controller = new AbortController();
    const delayMs = manualPreflightRefreshId > 0 ? 0 : 420;
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
    }, delayMs);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [
    deferredDataloaderConfig,
    deferredTrainingConfig,
    manualPreflightRefreshId,
    selectedProjectId,
    selectedProjectRefreshId,
    selectedTokenizerJobId,
  ]);

  useEffect(() => {
    setSelectedRecommendationOptionKey((current) =>
      selectRecommendationOptionKey(current, preflight?.batch_and_lr_recommendation)
    );
  }, [preflight]);

  return {
    preflight,
    preflightError,
    preflightLoading,
    refreshPreflight,
    selectedRecommendationOptionKey,
    setSelectedRecommendationOptionKey,
  };
}
