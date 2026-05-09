"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useThemeMode } from "../../../lib/theme";
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
  CHECKPOINT_SEARCH_PLACEHOLDER,
  DEFAULT_PROMPT,
  LATEST_CHECKPOINT_VALUE,
  PICKER_SEARCH_PLACEHOLDER,
} from "../constants";
import {
  checkpointOptionValue,
  matchesCheckpointQuery,
  matchesJobQuery,
} from "../lib/formatters";

export function useInferenceController() {
  const [theme, setTheme] = useThemeMode();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [seed, setSeed] = useState(42);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateTrainingCompletionResponse | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerQuery, setPickerQuery] = useState("");
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [checkpointValue, setCheckpointValue] = useState(LATEST_CHECKPOINT_VALUE);
  const [checkpointPickerOpen, setCheckpointPickerOpen] = useState(false);
  const [checkpointPickerQuery, setCheckpointPickerQuery] = useState("");
  const [checkpointsLoading, setCheckpointsLoading] = useState(false);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);
  const checkpointRequestIdRef = useRef(0);

  const completedJobs = useMemo(
    () => jobs.filter((job) => job.status === "completed" && job.checkpoint_count > 0),
    [jobs]
  );
  const selectedJob = useMemo(
    () => completedJobs.find((job) => job.id === selectedJobId) ?? null,
    [completedJobs, selectedJobId]
  );
  const visiblePickerJobs = useMemo(() => {
    const normalizedQuery = pickerQuery.trim().toLowerCase();
    return completedJobs.filter((job) => matchesJobQuery(job, normalizedQuery));
  }, [completedJobs, pickerQuery]);
  const latestCheckpoint = useMemo(
    () =>
      checkpoints.reduce<TrainingCheckpointEntry | null>(
        (latest, checkpoint) =>
          latest === null || checkpoint.step > latest.step ? checkpoint : latest,
        null
      ),
    [checkpoints]
  );
  const selectedCheckpoint = useMemo(() => {
    if (checkpointValue === LATEST_CHECKPOINT_VALUE) {
      return latestCheckpoint;
    }
    const selectedStep = Number(checkpointValue);
    return checkpoints.find((checkpoint) => checkpoint.step === selectedStep) ?? null;
  }, [checkpointValue, checkpoints, latestCheckpoint]);
  const visibleCheckpointOptions = useMemo(() => {
    const normalizedQuery = checkpointPickerQuery.trim().toLowerCase();
    return checkpoints.filter((checkpoint) =>
      matchesCheckpointQuery(checkpoint, normalizedQuery)
    );
  }, [checkpointPickerQuery, checkpoints]);
  const showLatestCheckpointOption = useMemo(() => {
    const normalizedQuery = checkpointPickerQuery.trim().toLowerCase();
    return (
      normalizedQuery === "" ||
      "latest".includes(normalizedQuery) ||
      "newest".includes(normalizedQuery) ||
      "automatic".includes(normalizedQuery)
    );
  }, [checkpointPickerQuery]);

  const generationCheckpointStep =
    checkpointValue === LATEST_CHECKPOINT_VALUE ? null : selectedCheckpoint?.step ?? null;

  const closePicker = useCallback(() => {
    setPickerOpen(false);
    setPickerQuery("");
  }, []);
  const closeCheckpointPicker = useCallback(() => {
    setCheckpointPickerOpen(false);
    setCheckpointPickerQuery("");
  }, []);

  const refreshJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextJobs = await fetchTrainingJobs();
      setJobs(nextJobs);
      const nextCompletedJobs = nextJobs.filter(
        (job) => job.status === "completed" && job.checkpoint_count > 0
      );
      setSelectedJobId((current) => {
        if (current && nextCompletedJobs.some((job) => job.id === current)) {
          return current;
        }
        return nextCompletedJobs[0]?.id ?? "";
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to load training artifacts.");
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshCheckpoints = useCallback(async (jobId: string) => {
    const requestId = checkpointRequestIdRef.current + 1;
    checkpointRequestIdRef.current = requestId;
    setCheckpointsLoading(true);
    setCheckpointError(null);

    try {
      const nextCheckpoints = await fetchTrainingCheckpoints(jobId);
      if (checkpointRequestIdRef.current !== requestId) {
        return;
      }
      setCheckpoints(nextCheckpoints);
    } catch (caught) {
      if (checkpointRequestIdRef.current !== requestId) {
        return;
      }
      setCheckpointError(caught instanceof Error ? caught.message : "Failed to load checkpoints.");
    } finally {
      if (checkpointRequestIdRef.current === requestId) {
        setCheckpointsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshJobs();
  }, [refreshJobs]);

  useEffect(() => {
    setCheckpointValue(LATEST_CHECKPOINT_VALUE);
    setCheckpoints([]);
    setCheckpointError(null);

    if (!selectedJobId) {
      setCheckpointsLoading(false);
      return;
    }

    void refreshCheckpoints(selectedJobId);
  }, [refreshCheckpoints, selectedJobId]);

  useEffect(() => {
    if (!pickerOpen && !checkpointPickerOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closePicker();
        closeCheckpointPicker();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [checkpointPickerOpen, closeCheckpointPicker, closePicker, pickerOpen]);

  const handleGenerate = useCallback(() => {
    if (!selectedJob || prompt.trim() === "") {
      return;
    }

    setError(null);
    setResult(null);
    setGenerating(true);
    const payload = {
      prompt,
      checkpoint_step: generationCheckpointStep,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      seed,
      repetition_penalty: repetitionPenalty,
    };

    void (async () => {
      let currentResult: GenerateTrainingCompletionResponse | null = null;
      let completion = "";
      let generatedTokenIds: number[] = [];

      try {
        await streamTrainingCompletion(selectedJob.id, payload, (event) => {
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
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Generation failed.");
      } finally {
        setGenerating(false);
      }
    })();
  }, [
    generationCheckpointStep,
    maxTokens,
    prompt,
    repetitionPenalty,
    seed,
    selectedJob,
    temperature,
    topK,
  ]);

  return {
    theme,
    setTheme,
    loading,
    error,
    generating,
    selectedJobId,
    setSelectedJobId,
    selectedJob,
    completedJobs,
    prompt,
    setPrompt,
    maxTokens,
    setMaxTokens,
    temperature,
    setTemperature,
    topK,
    setTopK,
    seed,
    setSeed,
    repetitionPenalty,
    setRepetitionPenalty,
    result,
    pickerOpen,
    setPickerOpen,
    pickerQuery,
    setPickerQuery,
    visiblePickerJobs,
    checkpoints,
    checkpointValue,
    setCheckpointValue,
    checkpointPickerOpen,
    setCheckpointPickerOpen,
    checkpointPickerQuery,
    setCheckpointPickerQuery,
    checkpointsLoading,
    checkpointError,
    latestCheckpoint,
    selectedCheckpoint,
    visibleCheckpointOptions,
    showLatestCheckpointOption,
    closePicker,
    closeCheckpointPicker,
    refreshJobs,
    refreshCheckpoints,
    handleGenerate,
    generationCheckpointStep,
    pickerSearchPlaceholder: PICKER_SEARCH_PLACEHOLDER,
    checkpointSearchPlaceholder: CHECKPOINT_SEARCH_PLACEHOLDER,
    latestCheckpointValue: LATEST_CHECKPOINT_VALUE,
    checkpointOptionValue,
  };
}
