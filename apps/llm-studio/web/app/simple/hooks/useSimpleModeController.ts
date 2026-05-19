"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { useUiMode } from "../../shared/hooks/useUiMode";
import {
  deriveSimpleStepStatuses,
  isTokenizerCompleted,
} from "../lib/stepStatus";
import type { SimpleModeController, SimpleStepId } from "../types";
import { useSimpleFlowPersistence } from "./useSimpleFlowPersistence";
import { useSimpleInferenceStep } from "./useSimpleInferenceStep";
import { useSimpleModelStep } from "./useSimpleModelStep";
import { useSimpleTokenizerStep } from "./useSimpleTokenizerStep";
import { useSimpleTrainingStep } from "./useSimpleTrainingStep";

function nextStepAfter(step: SimpleStepId | null): SimpleStepId {
  if (step === "architecture") {
    return "tokenizer";
  }
  if (step === "tokenizer") {
    return "training";
  }
  if (step === "training") {
    return "inference";
  }
  if (step === "inference") {
    return "inference";
  }
  return "architecture";
}

export function useSimpleModeController(): SimpleModeController {
  const [, setUiMode] = useUiMode();
  const { flow, hydrated, updateFlow, resetFlow } = useSimpleFlowPersistence();
  const [activeStep, setActiveStep] = useState<SimpleStepId>("architecture");
  const initializedActiveStepRef = useRef(false);

  useEffect(() => {
    setUiMode("simple");
  }, [setUiMode]);

  const modelStep = useSimpleModelStep({
    flow,
    updateFlow,
  });
  const tokenizerStep = useSimpleTokenizerStep({
    flow,
    syncProjectVocab: modelStep.syncProjectVocab,
    updateFlow,
  });
  const tokenizerReady = isTokenizerCompleted(tokenizerStep.tokenizerJob);
  const trainingStep = useSimpleTrainingStep({
    flow,
    projectReady: Boolean(modelStep.project?.valid),
    tokenizerReady,
    updateFlow,
  });
  const inferenceStep = useSimpleInferenceStep({
    flow,
    updateFlow,
  });

  const steps = useMemo(
    () =>
      deriveSimpleStepStatuses({
        flow,
        project: modelStep.project,
        projectLoading: modelStep.creating,
        projectError: modelStep.projectError,
        tokenizerJob: tokenizerStep.tokenizerJob,
        tokenizerError: tokenizerStep.tokenizerError,
        datasetReady: tokenizerStep.datasetReady,
        datasetBlocker: tokenizerStep.datasetBlocker,
        tokenizerValidationError: tokenizerStep.validationError,
        trainingRun: trainingStep.trainingRun,
        trainingCheckpoints: trainingStep.checkpoints,
        preflightValid: Boolean(trainingStep.preflight?.valid),
        preflightError: trainingStep.preflightError,
        trainingLaunching: trainingStep.launching,
        inferenceGenerating: inferenceStep.generating,
        generationSucceeded: Boolean(inferenceStep.result && !inferenceStep.generationError),
        checkpointError: inferenceStep.checkpointError,
      }),
    [
      flow,
      inferenceStep.checkpointError,
      inferenceStep.generating,
      inferenceStep.generationError,
      inferenceStep.result,
      modelStep.creating,
      modelStep.project,
      modelStep.projectError,
      tokenizerStep.datasetBlocker,
      tokenizerStep.datasetReady,
      tokenizerStep.tokenizerError,
      tokenizerStep.tokenizerJob,
      tokenizerStep.validationError,
      trainingStep.checkpoints,
      trainingStep.launching,
      trainingStep.preflight,
      trainingStep.preflightError,
      trainingStep.trainingRun,
    ]
  );

  useEffect(() => {
    if (!hydrated || initializedActiveStepRef.current) {
      return;
    }
    initializedActiveStepRef.current = true;
    const firstIncomplete = steps.find((step) => step.state !== "completed");
    setActiveStep(firstIncomplete?.id ?? "inference");
  }, [hydrated, steps]);

  useEffect(() => {
    if (!hydrated || !flow.lastCompletedStep) {
      return;
    }
    setActiveStep(nextStepAfter(flow.lastCompletedStep));
  }, [flow.lastCompletedStep, hydrated]);

  return {
    flow,
    updateFlow,
    resetFlow,
    activeStep,
    setActiveStep,
    steps,
    modelStep,
    tokenizerStep,
    trainingStep,
    inferenceStep,
  };
}
