"use client";

import { startTransition, useCallback, useEffect, useState } from "react";

import {
  fetchProject,
  type ProjectDetail,
} from "../../../lib/api";
import {
  fetchTrainingJob as fetchTokenizerJob,
  type TrainingJob as TokenizerTrainingJob,
} from "../../../lib/tokenizerLegacyApi";
import {
  ACTIVE_RUN_STORAGE_KEY,
  TRAINING_SELECTION_STORAGE_KEY,
} from "../constants";
import {
  readStoredJson,
  writeStoredJson,
} from "../lib/object";
import type { ToastLevel } from "../types";

type NotifyTrainingSelection = (level: ToastLevel, title: string, body: string) => void;

interface SearchParamReader {
  get(name: string): string | null;
}

interface StoredTrainingSelection {
  projectId: string | null;
  tokenizerJobId: string | null;
}

export interface InitialTrainingSelection {
  activeRunId: string | null;
  projectId: string | null;
  shouldSelectMostRecentRun: boolean;
  tokenizerJobId: string | null;
}

interface UseTrainingSelectionOptions {
  notify: NotifyTrainingSelection;
}

export function readInitialTrainingSelection(
  searchParams: SearchParamReader
): InitialTrainingSelection {
  const storedSelection = readStoredJson<StoredTrainingSelection>(
    TRAINING_SELECTION_STORAGE_KEY,
    {
      projectId: null,
      tokenizerJobId: null,
    }
  );
  const storedActiveRun = readStoredJson<string | null>(ACTIVE_RUN_STORAGE_KEY, null);
  const requestedRunId = searchParams.get("run");

  return {
    activeRunId: requestedRunId ?? storedActiveRun,
    projectId: searchParams.get("project") ?? storedSelection.projectId,
    shouldSelectMostRecentRun: !requestedRunId && !storedActiveRun,
    tokenizerJobId: searchParams.get("tokenizerJob") ?? storedSelection.tokenizerJobId,
  };
}

export function useTrainingSelection({ notify }: UseTrainingSelectionOptions) {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedTokenizerJobId, setSelectedTokenizerJobId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<ProjectDetail | null>(null);
  const [selectedTokenizer, setSelectedTokenizer] = useState<TokenizerTrainingJob | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  const initializeTrainingSelection = useCallback((selection: InitialTrainingSelection) => {
    setActiveRunId(selection.activeRunId);
    setSelectedProjectId(selection.projectId);
    setSelectedTokenizerJobId(selection.tokenizerJobId);
  }, []);

  useEffect(() => {
    writeStoredJson(TRAINING_SELECTION_STORAGE_KEY, {
      projectId: selectedProjectId,
      tokenizerJobId: selectedTokenizerJobId,
    });
  }, [selectedProjectId, selectedTokenizerJobId]);

  useEffect(() => {
    writeStoredJson(ACTIVE_RUN_STORAGE_KEY, activeRunId);
  }, [activeRunId]);

  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedProject(null);
      return;
    }
    const controller = new AbortController();
    void fetchProject(selectedProjectId, controller.signal)
      .then((project) => {
        startTransition(() => {
          setSelectedProject(project);
        });
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          notify("error", "Model config unavailable", error instanceof Error ? error.message : "Failed to load selected model config.");
        }
      });
    return () => controller.abort();
  }, [notify, selectedProjectId]);

  useEffect(() => {
    if (!selectedTokenizerJobId) {
      setSelectedTokenizer(null);
      return;
    }
    let cancelled = false;
    void fetchTokenizerJob(selectedTokenizerJobId)
      .then((job) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setSelectedTokenizer(job);
        });
      })
      .catch((error) => {
        if (!cancelled) {
          notify("error", "Tokenizer unavailable", error instanceof Error ? error.message : "Failed to load selected tokenizer.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [notify, selectedTokenizerJobId]);

  return {
    activeRunId,
    initializeTrainingSelection,
    selectedProject,
    selectedProjectId,
    selectedTokenizer,
    selectedTokenizerJobId,
    setActiveRunId,
    setSelectedProjectId,
    setSelectedTokenizerJobId,
  };
}
