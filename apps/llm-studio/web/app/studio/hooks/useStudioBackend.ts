import { useEffect, useRef, useState } from "react";

import {
  analyzeModelConfig,
  validateModelConfig,
  type ValidationIssue,
} from "../../../lib/api";
import type { ModelConfig } from "../../../lib/defaults";

import type {
  BackendAnalysisState,
  BackendValidationState,
  Diagnostic,
} from "../types";
import { VALIDATION_DEBOUNCE_MS } from "../types";
import { pushDiagnostic } from "../utils/validation";

type SetNoticeMessage = (tone: "info" | "success" | "error", message: string) => void;

type UseBackendValidationArgs = {
  modelConfig: ModelConfig;
  compactJson: string;
  localErrorCount: number;
};

type UseBackendAnalysisArgs = {
  modelConfig: ModelConfig;
  compactJson: string;
  localErrorCount: number;
  setNoticeMessage: SetNoticeMessage;
};

function createInitialBackendValidationState(): BackendValidationState {
  return {
    phase: "idle",
    message: "Waiting for edits",
    lastValidatedAt: null,
    warnings: [],
    errors: [],
    normalizedChanged: false,
  };
}

function createInitialBackendAnalysisState(): BackendAnalysisState {
  return {
    phase: "idle",
    message: "Run backend analysis to instantiate ConfigurableGPT and inspect parameter counts.",
    lastAnalyzedAt: null,
    configSignature: null,
    summary: null,
    warnings: [],
    errors: [],
    instantiationError: null,
  };
}

export function useStudioBackendValidation({
  modelConfig,
  compactJson,
  localErrorCount,
}: UseBackendValidationArgs): BackendValidationState {
  const validationRunRef = useRef(0);
  const [backendValidation, setBackendValidation] = useState<BackendValidationState>(
    createInitialBackendValidationState
  );

  useEffect(() => {
    if (localErrorCount > 0) {
      setBackendValidation((current) => ({
        ...current,
        phase: "skipped",
        message: "Backend validation paused until local errors are fixed.",
        warnings: [],
        errors: [],
        normalizedChanged: false,
      }));
      return;
    }

    const runId = validationRunRef.current + 1;
    validationRunRef.current = runId;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      setBackendValidation((current) => ({
        ...current,
        phase: "validating",
        message: "Validating with backend…",
      }));

      try {
        const result = await validateModelConfig(modelConfig, controller.signal);
        if (validationRunRef.current !== runId) {
          return;
        }

        const normalizedChanged =
          JSON.stringify(result.normalized_config) !== JSON.stringify(modelConfig);
        const issueSummary = [
          result.errors.length > 0
            ? `${result.errors.length} backend error${result.errors.length === 1 ? "" : "s"}`
            : null,
          result.warnings.length > 0
            ? `${result.warnings.length} backend warning${result.warnings.length === 1 ? "" : "s"}`
            : null,
        ]
          .filter((part): part is string => part !== null)
          .join(" · ");

        setBackendValidation({
          phase: "success",
          message:
            result.errors.length > 0
              ? `Backend validation found issues${issueSummary ? ` (${issueSummary})` : ""}.`
              : normalizedChanged
                ? `Backend validation passed (normalized config differs)${issueSummary ? ` · ${issueSummary}` : ""}.`
                : issueSummary
                  ? `Backend validation passed · ${issueSummary}.`
                  : "Backend validation passed.",
          lastValidatedAt: Date.now(),
          warnings: result.warnings,
          errors: result.errors,
          normalizedChanged,
        });
      } catch (error) {
        if (controller.signal.aborted || validationRunRef.current !== runId) {
          return;
        }
        setBackendValidation({
          phase: "fallback",
          message:
            error instanceof Error
              ? `Backend validation unavailable: ${error.message}`
              : "Backend validation unavailable; using local checks.",
          lastValidatedAt: Date.now(),
          warnings: [],
          errors: [],
          normalizedChanged: false,
        });
      }
    }, VALIDATION_DEBOUNCE_MS);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
    // Intentionally key off the JSON signature instead of `modelConfig` object identity.
    // `studioDocumentToConfig(...)` creates a new object on each render, which would
    // otherwise retrigger validation in a loop and spam backend requests.
  }, [compactJson, localErrorCount]);

  return backendValidation;
}

export function buildBackendDiagnostics(backendValidation: BackendValidationState): Diagnostic[] {
  const backendDiagnostics: Diagnostic[] = [];

  if (backendValidation.phase === "fallback") {
    pushDiagnostic(
      backendDiagnostics,
      "warning",
      "backend",
      "/validate/model",
      backendValidation.message
    );
  }

  if (backendValidation.phase === "success" && backendValidation.normalizedChanged) {
    pushDiagnostic(
      backendDiagnostics,
      "info",
      "backend",
      "/validate/model",
      "Backend returned a normalized config that differs from the current draft."
    );
  }

  backendValidation.errors.forEach((issue) => {
    pushDiagnostic(backendDiagnostics, "error", "backend", issue.path, issue.message);
  });
  backendValidation.warnings.forEach((issue) => {
    pushDiagnostic(backendDiagnostics, "warning", "backend", issue.path, issue.message);
  });

  return backendDiagnostics;
}

export function useStudioBackendAnalysis({
  modelConfig,
  compactJson,
  localErrorCount,
  setNoticeMessage,
}: UseBackendAnalysisArgs): {
  backendAnalysis: BackendAnalysisState;
  runBackendAnalysis: () => Promise<void>;
} {
  const [backendAnalysis, setBackendAnalysis] = useState<BackendAnalysisState>(
    createInitialBackendAnalysisState
  );

  async function runBackendAnalysis(): Promise<void> {
    if (localErrorCount > 0) {
      setNoticeMessage("error", "Resolve local errors before running backend model analysis.");
      return;
    }

    const signature = compactJson;
    setBackendAnalysis((current) => ({
      ...current,
      phase: "running",
      message: "Instantiating ConfigurableGPT on backend…",
      warnings: [],
      errors: [],
      instantiationError: null,
    }));

    try {
      const result = await analyzeModelConfig(modelConfig);
      const analysis = result.analysis;
      const analysisReady = result.instantiated && analysis !== null;
      const hasIssues = result.errors.length > 0;

      setBackendAnalysis({
        phase: analysisReady ? "success" : "error",
        message: analysisReady
          ? `Backend model analysis ready (${analysis.instantiation_time_ms.toFixed(1)} ms).`
          : result.instantiation_error ??
            (hasIssues
              ? "Backend analysis blocked by validation issues."
              : "Backend analysis did not return metrics."),
        lastAnalyzedAt: Date.now(),
        configSignature: signature,
        summary: analysis,
        warnings: result.warnings,
        errors: result.errors,
        instantiationError: result.instantiation_error,
      });

      if (!analysisReady) {
        setNoticeMessage("error", result.instantiation_error ?? "Backend model analysis failed.");
      }
    } catch (error) {
      setBackendAnalysis({
        phase: "error",
        message:
          error instanceof Error
            ? `Backend analysis unavailable: ${error.message}`
            : "Backend analysis unavailable.",
        lastAnalyzedAt: Date.now(),
        configSignature: signature,
        summary: null,
        warnings: [],
        errors: [],
        instantiationError: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }

  return { backendAnalysis, runBackendAnalysis };
}

export function flattenValidationIssues(...issueLists: ValidationIssue[][]): ValidationIssue[] {
  return issueLists.flat();
}
