"use client";

import { startTransition, useEffect, useMemo, useState } from "react";

import {
  analyzeModelConfig,
  createProject,
  fetchProject,
  updateProject,
  validateModelConfig,
  type ModelAnalysisResponse,
  type ProjectDetail,
} from "../../../lib/api";
import { upsertCachedWorkspaceProject } from "../../../lib/workspaceAssets";
import {
  backendAnalysisSkipReason,
  buildPresetModelConfig,
  getSimpleModelPreset,
  shouldAnalyzePresetWithBackend,
} from "../lib/modelPresets";
import { buildModelConfigWithSyncedVocab } from "../lib/vocabularySync";
import type { SimpleFlowState, SimpleModelStepState } from "../types";

interface UseSimpleModelStepOptions {
  flow: SimpleFlowState;
  updateFlow: (updater: (current: SimpleFlowState) => SimpleFlowState) => void;
}

export function useSimpleModelStep({
  flow,
  updateFlow,
}: UseSimpleModelStepOptions): SimpleModelStepState {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [creating, setCreating] = useState(false);
  const [analysisByPresetId, setAnalysisByPresetId] = useState<Record<string, ModelAnalysisResponse | null>>({});
  const [analysisErrorsByPresetId, setAnalysisErrorsByPresetId] = useState<Record<string, string>>({});

  const selectedPreset = getSimpleModelPreset(flow.presetId);
  const selectedConfig = useMemo(
    () =>
      buildPresetModelConfig(selectedPreset.id, {
        vocabSize: flow.targetVocabSize,
        contextLength: flow.targetContextLength,
      }),
    [flow.targetContextLength, flow.targetVocabSize, selectedPreset.id]
  );

  const selectedAnalysis = analysisByPresetId[selectedPreset.id] ?? null;

  const refreshProject = async () => {
    if (!flow.projectId) {
      setProject(null);
      setProjectError(null);
      return;
    }

    try {
      const nextProject = await fetchProject(flow.projectId);
      startTransition(() => {
        setProject(nextProject);
        setProjectError(null);
      });
    } catch (error) {
      startTransition(() => {
        setProject(null);
        setProjectError(
          error instanceof Error ? error.message : "Saved model project could not be loaded."
        );
      });
    }
  };

  useEffect(() => {
    let cancelled = false;
    if (!flow.projectId) {
      setProject(null);
      setProjectError(null);
      return;
    }

    void fetchProject(flow.projectId)
      .then((nextProject) => {
        if (!cancelled) {
          startTransition(() => {
            setProject(nextProject);
            setProjectError(null);
          });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          startTransition(() => {
            setProject(null);
            setProjectError(
              error instanceof Error ? error.message : "Saved model project could not be loaded."
            );
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [flow.projectId]);

  useEffect(() => {
    const controller = new AbortController();
    const analysisOptions = {
      vocabSize: flow.targetVocabSize,
      contextLength: flow.targetContextLength,
    };

    if (!shouldAnalyzePresetWithBackend(selectedPreset, analysisOptions)) {
      const skippedReason = backendAnalysisSkipReason(selectedPreset, analysisOptions);
      setAnalyzing(false);
      startTransition(() => {
        setAnalysisByPresetId((current) => ({
          ...current,
          [selectedPreset.id]: null,
        }));
        setAnalysisErrorsByPresetId((current) => ({
          ...current,
          [selectedPreset.id]: skippedReason ?? "Backend analysis skipped.",
        }));
      });
      return () => controller.abort();
    }

    setAnalyzing(true);

    void analyzeModelConfig(selectedConfig, controller.signal)
      .then((analysis) => {
        if (controller.signal.aborted) {
          return;
        }
        startTransition(() => {
          setAnalysisByPresetId((current) => ({
            ...current,
            [selectedPreset.id]: analysis,
          }));
          setAnalysisErrorsByPresetId((current) => {
            const { [selectedPreset.id]: _removed, ...rest } = current;
            void _removed;
            return rest;
          });
        });
      })
      .catch((error) => {
        if (controller.signal.aborted) {
          return;
        }
        startTransition(() => {
          setAnalysisByPresetId((current) => ({
            ...current,
            [selectedPreset.id]: null,
          }));
          setAnalysisErrorsByPresetId((current) => ({
            ...current,
            [selectedPreset.id]:
              error instanceof Error ? error.message : "Analysis failed.",
          }));
        });
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setAnalyzing(false);
        }
      });

    return () => controller.abort();
  }, [
    flow.targetContextLength,
    flow.targetVocabSize,
    selectedPreset.id,
    selectedPreset.trainingTarget,
    selectedConfig,
  ]);

  const createArchitecture = async () => {
    const name = flow.modelName.trim();
    if (!name) {
      setProjectError("Enter a model name.");
      return;
    }

    setCreating(true);
    setProjectError(null);
    try {
      const validation = await validateModelConfig(selectedConfig);
      if (!validation.valid) {
        const firstError = validation.errors[0]?.message ?? "Preset did not pass validation.";
        throw new Error(firstError);
      }

      const projectDetail = flow.projectId
        ? await updateProject(flow.projectId, name, validation.normalized_config)
        : await createProject(name, validation.normalized_config);
      upsertCachedWorkspaceProject(projectDetail);
      startTransition(() => {
        setProject(projectDetail);
        setProjectError(null);
        updateFlow((current) => ({
          ...current,
          projectId: projectDetail.id,
          modelName: name,
          presetId: selectedPreset.id,
          targetContextLength: validation.normalized_config.context_length,
          targetVocabSize: validation.normalized_config.vocab_size,
          lastCompletedStep: "architecture",
        }));
      });
    } catch (error) {
      setProjectError(
        error instanceof Error ? error.message : "Could not create the architecture."
      );
    } finally {
      setCreating(false);
    }
  };

  const syncProjectVocab = async (vocabSize: number): Promise<ProjectDetail | null> => {
    if (!project && !flow.projectId) {
      return null;
    }

    const sourceProject = project ?? (flow.projectId ? await fetchProject(flow.projectId) : null);
    if (!sourceProject || sourceProject.model_config.vocab_size === vocabSize) {
      return sourceProject;
    }

    const syncedConfig = buildModelConfigWithSyncedVocab(sourceProject.model_config, vocabSize);
    const updatedProject = await updateProject(
      sourceProject.id,
      sourceProject.name,
      syncedConfig
    );
    upsertCachedWorkspaceProject(updatedProject);
    startTransition(() => {
      setProject(updatedProject);
      updateFlow((current) => ({
        ...current,
        targetVocabSize: updatedProject.model_config.vocab_size,
      }));
    });
    return updatedProject;
  };

  return {
    project,
    projectError,
    analyzing,
    creating,
    analysisByPresetId,
    analysisErrorsByPresetId,
    selectedAnalysis,
    createArchitecture,
    refreshProject,
    syncProjectVocab,
  };
}
