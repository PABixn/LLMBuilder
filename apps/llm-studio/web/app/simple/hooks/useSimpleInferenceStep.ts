"use client";

import { startTransition, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchTrainingCheckpoints,
  fetchTrainingJobs,
} from "../../../lib/training/jobs";
import { streamTrainingCompletion } from "../../../lib/training/generation";
import type {
  GenerateTrainingCompletionResponse,
  TrainingCheckpointEntry,
  TrainingJob,
} from "../../../lib/training/types";
import {
  SIMPLE_DEFAULT_PROMPT,
  SIMPLE_LATEST_CHECKPOINT_VALUE,
  SIMPLE_RECENT_RUNS_POLL_INTERVAL_MS,
} from "../constants";
import { buildInferenceSettings } from "../lib/inferencePresets";
import type {
  SimpleFlowState,
  SimpleInferenceCreativity,
  SimpleInferenceLength,
  SimpleInferenceStepState,
} from "../types";

interface UseSimpleInferenceStepOptions {
  flow: SimpleFlowState;
  updateFlow: (updater: (current: SimpleFlowState) => SimpleFlowState) => void;
}

export function useSimpleInferenceStep({
  flow,
  updateFlow,
}: UseSimpleInferenceStepOptions): SimpleInferenceStepState {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);
  const [checkpointsLoading, setCheckpointsLoading] = useState(false);
  const [lengthPreset, setLengthPreset] = useState<SimpleInferenceLength>("medium");
  const [creativityPreset, setCreativityPreset] =
    useState<SimpleInferenceCreativity>("balanced");
  const [prompt, setPrompt] = useState(SIMPLE_DEFAULT_PROMPT);
  const [seed, setSeed] = useState(42);
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateTrainingCompletionResponse | null>(null);
  const checkpointRequestIdRef = useRef(0);

  const completedRuns = useMemo(
    () => jobs.filter((job) => job.status === "completed" && job.checkpoint_count > 0),
    [jobs]
  );

  const selectedRun = useMemo(() => {
    if (flow.trainingJobId) {
      return completedRuns.find((job) => job.id === flow.trainingJobId) ?? null;
    }
    return null;
  }, [completedRuns, flow.trainingJobId]);

  const latestCheckpoint = useMemo(
    () =>
      checkpoints.reduce<TrainingCheckpointEntry | null>(
        (latest, checkpoint) =>
          latest === null || checkpoint.step > latest.step ? checkpoint : latest,
        null
      ),
    [checkpoints]
  );

  useEffect(() => {
    let cancelled = false;
    let intervalId: number | null = null;

    const refresh = async () => {
      try {
        const nextJobs = await fetchTrainingJobs();
        if (!cancelled) {
          startTransition(() => setJobs(nextJobs));
        }
      } catch {
        // The inference step can still render its blocker from an empty run list.
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
  }, []);

  useEffect(() => {
    setCheckpoints([]);
    setCheckpointError(null);
    if (!selectedRun) {
      setCheckpointsLoading(false);
      return;
    }

    const requestId = checkpointRequestIdRef.current + 1;
    checkpointRequestIdRef.current = requestId;
    setCheckpointsLoading(true);

    void fetchTrainingCheckpoints(selectedRun.id)
      .then((nextCheckpoints) => {
        if (checkpointRequestIdRef.current !== requestId) {
          return;
        }
        startTransition(() => {
          setCheckpoints(nextCheckpoints);
          setCheckpointError(null);
        });
      })
      .catch((error) => {
        if (checkpointRequestIdRef.current !== requestId) {
          return;
        }
        setCheckpointError(
          error instanceof Error ? error.message : "Checkpoints could not be loaded."
        );
      })
      .finally(() => {
        if (checkpointRequestIdRef.current === requestId) {
          setCheckpointsLoading(false);
        }
      });
  }, [selectedRun]);

  const generateWithSeed = async (nextSeed: number) => {
    if (!selectedRun || !latestCheckpoint || prompt.trim() === "") {
      setGenerationError("Choose a completed training run and enter a prompt.");
      return;
    }

    setGenerating(true);
    setGenerationError(null);
    setResult(null);

    const settings = buildInferenceSettings(lengthPreset, creativityPreset);
    const checkpointStep =
      flow.checkpointValue === SIMPLE_LATEST_CHECKPOINT_VALUE
        ? null
        : Number(flow.checkpointValue);
    const payload = {
      prompt,
      checkpoint_step: Number.isFinite(checkpointStep) ? checkpointStep : null,
      max_tokens: settings.max_tokens,
      temperature: settings.temperature,
      top_k: settings.top_k,
      seed: nextSeed,
      repetition_penalty: settings.repetition_penalty,
    };

    let currentResult: GenerateTrainingCompletionResponse | null = null;
    let completion = "";
    let generatedTokenIds: number[] = [];

    try {
      await streamTrainingCompletion(selectedRun.id, payload, (event) => {
        if (event.type === "start") {
          currentResult = {
            job_id: event.job_id,
            checkpoint_step: event.checkpoint_step,
            checkpoint_path: event.checkpoint_path,
            tokenizer_job_id: event.tokenizer_job_id,
            prompt: event.prompt,
            completion: "",
            text: event.prompt,
            prompt_token_count: event.prompt_token_count,
            generated_token_count: 0,
            generated_token_ids: [],
          };
          setResult(currentResult);
          return;
        }

        if (event.type === "token" && currentResult) {
          completion += event.token_text;
          generatedTokenIds = [...generatedTokenIds, event.token_id];
          currentResult = {
            ...currentResult,
            completion,
            text: `${currentResult.prompt}${completion}`,
            generated_token_count: generatedTokenIds.length,
            generated_token_ids: generatedTokenIds,
          };
          setResult(currentResult);
          return;
        }

        if (event.type === "done" && currentResult) {
          currentResult = {
            ...currentResult,
            completion: event.completion,
            text: event.text,
            generated_token_count: event.generated_token_count,
            generated_token_ids: event.generated_token_ids,
          };
          setResult(currentResult);
        }
      });
      updateFlow((current) => ({
        ...current,
        checkpointValue: flow.checkpointValue || SIMPLE_LATEST_CHECKPOINT_VALUE,
        lastCompletedStep: "inference",
      }));
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : "Generation failed.");
    } finally {
      setGenerating(false);
    }
  };

  const generate = async () => {
    await generateWithSeed(seed);
  };

  const tryAnother = async () => {
    const nextSeed = seed + 1;
    setSeed(nextSeed);
    await generateWithSeed(nextSeed);
  };

  return {
    selectedRun,
    completedRuns,
    checkpoints,
    checkpointError,
    checkpointsLoading,
    latestCheckpoint,
    lengthPreset,
    creativityPreset,
    prompt,
    generating,
    generationError,
    result,
    setLengthPreset,
    setCreativityPreset,
    setPrompt,
    generate,
    tryAnother,
  };
}
