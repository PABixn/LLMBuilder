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
  buildPresetModelConfig,
  getSimpleModelPreset,
  SIMPLE_MODEL_PRESETS,
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
    setAnalyzing(true);

    void Promise.allSettled(
      SIMPLE_MODEL_PRESETS.map(async (preset) => ({
        presetId: preset.id,
        analysis: await analyzeModelConfig(
          buildPresetModelConfig(preset.id, {
            vocabSize:
              preset.id === selectedPreset.id ? flow.targetVocabSize : preset.defaultVocabSize,
            contextLength:
              preset.id === selectedPreset.id
                ? flow.targetContextLength
                : preset.defaultContextLength,
          }),
          controller.signal
        ),
      }))
    )
      .then((results) => {
        if (controller.signal.aborted) {
          return;
        }
        const nextAnalyses: Record<string, ModelAnalysisResponse | null> = {};
        const nextErrors: Record<string, string> = {};
        results.forEach((result, index) => {
          const presetId = SIMPLE_MODEL_PRESETS[index].id;
          if (result.status === "fulfilled") {
            nextAnalyses[presetId] = result.value.analysis;
          } else {
            nextAnalyses[presetId] = null;
            nextErrors[presetId] =
              result.reason instanceof Error ? result.reason.message : "Analysis failed.";
          }
        });
        startTransition(() => {
          setAnalysisByPresetId(nextAnalyses);
          setAnalysisErrorsByPresetId(nextErrors);
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
    selectedPreset.defaultContextLength,
    selectedPreset.id,
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
