"use client";

import {
  useCallback,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import { fetchTrainingConfigTemplates } from "../../../lib/training/jobs";
import {
  asRecord,
  asRecordArray,
  cloneRecord,
} from "../lib/object";
import type { ToastLevel } from "../types";

type NotifyPromptSettings = (level: ToastLevel, title: string, body: string) => void;

interface UsePromptSettingsOptions {
  notify: NotifyPromptSettings;
  setTrainingConfig: Dispatch<SetStateAction<Record<string, unknown> | null>>;
  trainingConfig: Record<string, unknown> | null;
}

export function usePromptSettings({
  notify,
  setTrainingConfig,
  trainingConfig,
}: UsePromptSettingsOptions) {
  const [isResettingPrompts, setIsResettingPrompts] = useState(false);

  const promptEntries = useMemo(() => {
    const sampler = asRecord(trainingConfig?.sampler);
    return asRecordArray(sampler.prompts);
  }, [trainingConfig]);

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
      notify("success", "Prompts reset", "Loaded template prompts.");
    } catch (error) {
      notify(
        "error",
        "Prompt template unavailable",
        error instanceof Error ? error.message : "Could not load template prompts."
      );
    } finally {
      setIsResettingPrompts(false);
    }
  }, [notify, setTrainingConfig]);

  const handleAddPrompt = useCallback(() => {
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
  }, [setTrainingConfig, trainingConfig]);

  const handlePromptChange = useCallback(
    (index: number, field: string, value: unknown) => {
      const next = cloneRecord(trainingConfig ?? {});
      const sampler = asRecord(next.sampler);
      const prompts = asRecordArray(sampler.prompts);
      const prompt = cloneRecord(prompts[index] ?? {});
      prompt[field] = value;
      prompts[index] = prompt;
      sampler.prompts = prompts;
      next.sampler = sampler;
      setTrainingConfig(next);
    },
    [setTrainingConfig, trainingConfig]
  );

  const handleRemovePrompt = useCallback(
    (index: number) => {
      const next = cloneRecord(trainingConfig ?? {});
      const sampler = asRecord(next.sampler);
      sampler.prompts = asRecordArray(sampler.prompts).filter(
        (_, currentIndex) => currentIndex !== index
      );
      next.sampler = sampler;
      setTrainingConfig(next);
    },
    [setTrainingConfig, trainingConfig]
  );

  return {
    handleAddPrompt,
    handlePromptChange,
    handleRemovePrompt,
    handleResetPrompts,
    isResettingPrompts,
    promptEntries,
  };
}
