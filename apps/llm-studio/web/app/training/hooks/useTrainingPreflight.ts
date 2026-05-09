"use client";

import {
  startTransition,
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
  selectedProjectId,
  selectedTokenizerJobId,
  trainingConfig,
}: UseTrainingPreflightOptions) {
  const [preflight, setPreflight] = useState<TrainingPreflightResponse | null>(null);
  const [selectedRecommendationOptionKey, setSelectedRecommendationOptionKey] =
    useState<string | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightError, setPreflightError] = useState<string | null>(null);

  const deferredTrainingConfig = useDeferredValue(trainingConfig);
  const deferredDataloaderConfig = useDeferredValue(dataloaderConfig);

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
    setSelectedRecommendationOptionKey((current) =>
      selectRecommendationOptionKey(current, preflight?.batch_and_lr_recommendation)
    );
  }, [preflight]);

  return {
    preflight,
    preflightError,
    preflightLoading,
    selectedRecommendationOptionKey,
    setSelectedRecommendationOptionKey,
  };
}
