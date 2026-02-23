"use client";

import {
  Fragment,
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
} from "react";
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiChevronDown,
  FiChevronRight,
  FiCopy,
  FiDownload,
  FiHardDrive,
  FiLayers,
  FiMoon,
  FiMove,
  FiPlus,
  FiRefreshCw,
  FiServer,
  FiSun,
  FiTrash2,
  FiUpload,
  FiXCircle,
} from "react-icons/fi";

import {
  analyzeModelConfig,
  apiBaseUrl,
  validateModelConfig,
  type ModelAnalysisSummary,
  type ValidationIssue as ApiValidationIssue,
} from "../lib/api";
import {
  ACTIVATION_TYPES,
  createDefaultBlockConfig,
  createDefaultModelConfig,
  type ActivationComponentConfig,
  type ActivationConfig,
  type ActivationType,
  type AttentionComponentConfig,
  type AttentionConfig,
  type BlockComponent,
  type LayerNormConfig,
  type LinearConfig,
  type LinearStepConfig,
  type MlpComponentConfig,
  type MlpConfig,
  type MlpStep,
  type ModelBlock,
  type ModelConfig,
  type NormComponentConfig,
  type NormConfig,
  type RMSNormConfig,
} from "../lib/defaults";

type ThemeMode = "white" | "dark";
type StudioComponentKind = "attention" | "mlp" | "norm" | "activation";
type MlpStepKind = "linear" | "norm" | "activation";
type DiagnosticLevel = "error" | "warning" | "info";
type DiagnosticSource = "local" | "backend";
type NoticeTone = "info" | "success" | "error";
type BackendValidationPhase =
  | "idle"
  | "skipped"
  | "validating"
  | "success"
  | "fallback";
type BackendAnalysisPhase = "idle" | "running" | "success" | "error";

const THEME_STORAGE_KEY = "llm-studio-theme";
const DOCUMENT_STORAGE_KEY = "llm-studio-document";
const IMPORT_DRAFT_STORAGE_KEY = "llm-studio-import-draft";
const VALIDATION_DEBOUNCE_MS = 420;
const DND_MIME = "application/x-llm-studio-dnd";

type StudioMlpStep =
  | {
      id: string;
      kind: "linear";
      linear: LinearConfig;
    }
  | {
      id: string;
      kind: "norm";
      norm: NormConfig;
    }
  | {
      id: string;
      kind: "activation";
      activation: ActivationConfig;
    };

type StudioComponent =
  | {
      id: string;
      kind: "attention";
      attention: AttentionConfig;
    }
  | {
      id: string;
      kind: "mlp";
      mlp: {
        multiplier: number;
        sequence: StudioMlpStep[];
      };
    }
  | {
      id: string;
      kind: "norm";
      norm: NormConfig;
    }
  | {
      id: string;
      kind: "activation";
      activation: ActivationConfig;
    };

interface StudioBlock {
  id: string;
  components: StudioComponent[];
}

interface StudioDocument {
  context_length: number;
  vocab_size: number;
  n_embd: number;
  weight_tying: boolean;
  blocks: StudioBlock[];
}

interface Diagnostic {
  id: string;
  level: DiagnosticLevel;
  source: DiagnosticSource;
  path: string;
  message: string;
}

interface BackendValidationState {
  phase: BackendValidationPhase;
  message: string;
  lastValidatedAt: number | null;
  warnings: ApiValidationIssue[];
  errors: ApiValidationIssue[];
  normalizedChanged: boolean;
}

interface BackendAnalysisState {
  phase: BackendAnalysisPhase;
  message: string;
  lastAnalyzedAt: number | null;
  configSignature: string | null;
  summary: ModelAnalysisSummary | null;
  warnings: ApiValidationIssue[];
  errors: ApiValidationIssue[];
  instantiationError: string | null;
}

interface NoticeState {
  tone: NoticeTone;
  message: string;
  at: number;
}

interface BuilderMetrics {
  blockCount: number;
  componentCount: number;
  attentionCount: number;
  mlpCount: number;
  normCount: number;
  activationCount: number;
  mlpStepCount: number;
}

type DragPayload =
  | {
      kind: "palette-component";
      componentKind: StudioComponentKind;
    }
  | {
      kind: "block";
      blockId: string;
    }
  | {
      kind: "block-component";
      fromBlockId: string;
      componentId: string;
    }
  | {
      kind: "palette-mlp-step";
      stepKind: MlpStepKind;
    }
  | {
      kind: "mlp-step";
      fromBlockId: string;
      fromComponentId: string;
      stepId: string;
    };

function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function clone<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function moveItem<T>(items: T[], fromIndex: number, toIndex: number): T[] {
  if (
    fromIndex < 0 ||
    fromIndex >= items.length ||
    toIndex < 0 ||
    toIndex > items.length ||
    (fromIndex === toIndex || fromIndex + 1 === toIndex)
  ) {
    return items.slice();
  }
  const next = items.slice();
  const [item] = next.splice(fromIndex, 1);
  const adjustedIndex = fromIndex < toIndex ? toIndex - 1 : toIndex;
  next.splice(adjustedIndex, 0, item);
  return next;
}

function integerInputValue(value: number): number | "" {
  return Number.isFinite(value) ? Math.trunc(value) : "";
}

function numberInputValue(value: number): number | "" {
  return Number.isFinite(value) ? value : "";
}

function parseIntegerInput(value: string, fallback: number): number {
  if (value.trim() === "") {
    return 0;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseNumberInput(value: string, fallback: number): number {
  if (value.trim() === "") {
    return 0;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatTimeAgo(timestamp: number | null): string {
  if (!timestamp) {
    return "Not yet";
  }
  const deltaMs = Date.now() - timestamp;
  if (deltaMs < 3_000) {
    return "just now";
  }
  const seconds = Math.round(deltaMs / 1_000);
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

function formatCompactCount(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  const digits = amount >= 100 || unitIndex === 0 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(digits)} ${units[unitIndex]}`;
}

function componentDomIdPrefix(blockIndex: number, componentIndex: number): string {
  return `block-${blockIndex}-component-${componentIndex}`;
}

function mlpStepDomIdPrefix(
  blockIndex: number,
  componentIndex: number,
  stepIndex: number
): string {
  return `block-${blockIndex}-component-${componentIndex}-mlp-step-${stepIndex}`;
}

function downloadTextFile(fileName: string, text: string): void {
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.rel = "noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

function createDefaultStudioNorm(): NormConfig {
  return { type: "layernorm" };
}

function createDefaultStudioMlpStep(kind: MlpStepKind): StudioMlpStep {
  if (kind === "linear") {
    return { id: createId("mlp-step"), kind: "linear", linear: { bias: true } };
  }
  if (kind === "norm") {
    return { id: createId("mlp-step"), kind: "norm", norm: createDefaultStudioNorm() };
  }
  return {
    id: createId("mlp-step"),
    kind: "activation",
    activation: { type: "relu" },
  };
}

function createDefaultStudioComponent(kind: StudioComponentKind): StudioComponent {
  if (kind === "attention") {
    return {
      id: createId("component"),
      kind: "attention",
      attention: { n_head: 12, n_kv_head: 12 },
    };
  }
  if (kind === "mlp") {
    return {
      id: createId("component"),
      kind: "mlp",
      mlp: {
        multiplier: 4,
        sequence: [
          createDefaultStudioMlpStep("linear"),
          createDefaultStudioMlpStep("activation"),
          createDefaultStudioMlpStep("linear"),
        ],
      },
    };
  }
  if (kind === "norm") {
    return { id: createId("component"), kind: "norm", norm: createDefaultStudioNorm() };
  }
  return {
    id: createId("component"),
    kind: "activation",
    activation: { type: "relu" },
  };
}

function studioMlpStepFromConfig(step: MlpStep): StudioMlpStep {
  if ("linear" in step) {
    return {
      id: createId("mlp-step"),
      kind: "linear",
      linear: clone(step.linear),
    };
  }
  if ("norm" in step) {
    return {
      id: createId("mlp-step"),
      kind: "norm",
      norm: clone(step.norm),
    };
  }
  return {
    id: createId("mlp-step"),
    kind: "activation",
    activation: clone(step.activation),
  };
}

function studioComponentFromConfig(component: BlockComponent): StudioComponent {
  if ("attention" in component) {
    return {
      id: createId("component"),
      kind: "attention",
      attention: clone(component.attention),
    };
  }
  if ("mlp" in component) {
    return {
      id: createId("component"),
      kind: "mlp",
      mlp: {
        multiplier: component.mlp.multiplier,
        sequence: component.mlp.sequence.map(studioMlpStepFromConfig),
      },
    };
  }
  if ("norm" in component) {
    return {
      id: createId("component"),
      kind: "norm",
      norm: clone(component.norm),
    };
  }
  return {
    id: createId("component"),
    kind: "activation",
    activation: clone(component.activation),
  };
}

function studioBlockFromConfig(block: ModelBlock): StudioBlock {
  return {
    id: createId("block"),
    components: block.components.map(studioComponentFromConfig),
  };
}

function studioDocumentFromConfig(config: ModelConfig): StudioDocument {
  return {
    context_length: config.context_length,
    vocab_size: config.vocab_size,
    n_embd: config.n_embd,
    weight_tying: config.weight_tying,
    blocks: config.blocks.map(studioBlockFromConfig),
  };
}

function mlpStepToConfig(step: StudioMlpStep): MlpStep {
  if (step.kind === "linear") {
    const out: LinearStepConfig = { linear: clone(step.linear) };
    return out;
  }
  if (step.kind === "norm") {
    const out: NormComponentConfig = { norm: clone(step.norm) };
    return out;
  }
  const out: ActivationComponentConfig = { activation: clone(step.activation) };
  return out;
}

function studioComponentToConfig(component: StudioComponent): BlockComponent {
  if (component.kind === "attention") {
    const out: AttentionComponentConfig = { attention: clone(component.attention) };
    return out;
  }
  if (component.kind === "mlp") {
    const out: MlpComponentConfig = {
      mlp: {
        multiplier: component.mlp.multiplier,
        sequence: component.mlp.sequence.map(mlpStepToConfig),
      },
    };
    return out;
  }
  if (component.kind === "norm") {
    const out: NormComponentConfig = { norm: clone(component.norm) };
    return out;
  }
  const out: ActivationComponentConfig = { activation: clone(component.activation) };
  return out;
}

function studioDocumentToConfig(document: StudioDocument): ModelConfig {
  return {
    context_length: document.context_length,
    vocab_size: document.vocab_size,
    n_embd: document.n_embd,
    weight_tying: document.weight_tying,
    blocks: document.blocks.map((block) => ({
      components: block.components.map(studioComponentToConfig),
    })),
  };
}

function cloneBlockWithNewIds(block: StudioBlock): StudioBlock {
  return {
    id: createId("block"),
    components: block.components.map((component) => {
      if (component.kind === "mlp") {
        return {
          ...clone(component),
          id: createId("component"),
          mlp: {
            multiplier: component.mlp.multiplier,
            sequence: component.mlp.sequence.map((step) => ({
              ...clone(step),
              id: createId("mlp-step"),
            })),
          },
        } satisfies StudioComponent;
      }
      return {
        ...clone(component),
        id: createId("component"),
      } satisfies StudioComponent;
    }),
  };
}

function findBlockIndex(document: StudioDocument, blockId: string): number {
  return document.blocks.findIndex((block) => block.id === blockId);
}

function findComponentIndex(block: StudioBlock, componentId: string): number {
  return block.components.findIndex((component) => component.id === componentId);
}

function getMlpComponent(
  document: StudioDocument,
  blockId: string,
  componentId: string
): { blockIndex: number; componentIndex: number; component: Extract<StudioComponent, { kind: "mlp" }> } | null {
  const blockIndex = findBlockIndex(document, blockId);
  if (blockIndex < 0) {
    return null;
  }
  const componentIndex = findComponentIndex(document.blocks[blockIndex], componentId);
  if (componentIndex < 0) {
    return null;
  }
  const component = document.blocks[blockIndex].components[componentIndex];
  if (component.kind !== "mlp") {
    return null;
  }
  return { blockIndex, componentIndex, component };
}

function labelForComponentKind(kind: StudioComponentKind): string {
  if (kind === "attention") {
    return "Attention";
  }
  if (kind === "mlp") {
    return "MLP";
  }
  if (kind === "norm") {
    return "Norm";
  }
  return "Activation";
}

function labelForMlpStepKind(kind: MlpStepKind): string {
  if (kind === "linear") {
    return "Linear";
  }
  if (kind === "norm") {
    return "Norm";
  }
  return "Activation";
}

function summarizeComponent(component: StudioComponent): string {
  if (component.kind === "attention") {
    return `${component.attention.n_head} heads / ${component.attention.n_kv_head} kv`;
  }
  if (component.kind === "mlp") {
    return `${component.mlp.sequence.length} steps, x${component.mlp.multiplier}`;
  }
  if (component.kind === "norm") {
    if (component.norm.type === "layernorm") {
      return "LayerNorm";
    }
    return component.norm.learnable_gamma ? "RMSNorm (learnable)" : "RMSNorm (fixed)";
  }
  return component.activation.type.toUpperCase();
}

function summarizeMlpStep(step: StudioMlpStep): string {
  if (step.kind === "linear") {
    return step.linear.bias ? "bias on" : "bias off";
  }
  if (step.kind === "norm") {
    if (step.norm.type === "layernorm") {
      return "LayerNorm";
    }
    return step.norm.learnable_gamma ? "RMSNorm learnable" : "RMSNorm fixed";
  }
  return step.activation.type;
}

function collectBuilderMetrics(document: StudioDocument): BuilderMetrics {
  const metrics: BuilderMetrics = {
    blockCount: document.blocks.length,
    componentCount: 0,
    attentionCount: 0,
    mlpCount: 0,
    normCount: 0,
    activationCount: 0,
    mlpStepCount: 0,
  };

  for (const block of document.blocks) {
    metrics.componentCount += block.components.length;
    for (const component of block.components) {
      if (component.kind === "attention") {
        metrics.attentionCount += 1;
      } else if (component.kind === "mlp") {
        metrics.mlpCount += 1;
        metrics.mlpStepCount += component.mlp.sequence.length;
      } else if (component.kind === "norm") {
        metrics.normCount += 1;
      } else if (component.kind === "activation") {
        metrics.activationCount += 1;
      }
    }
  }

  return metrics;
}

function collectAllComponentIds(document: StudioDocument): string[] {
  const ids: string[] = [];
  for (const block of document.blocks) {
    for (const component of block.components) {
      ids.push(component.id);
    }
  }
  return ids;
}

function collectAllMlpStepIds(document: StudioDocument): string[] {
  const ids: string[] = [];
  for (const block of document.blocks) {
    for (const component of block.components) {
      if (component.kind !== "mlp") {
        continue;
      }
      for (const step of component.mlp.sequence) {
        ids.push(step.id);
      }
    }
  }
  return ids;
}

function pushDiagnostic(
  diagnostics: Diagnostic[],
  level: DiagnosticLevel,
  source: DiagnosticSource,
  path: string,
  message: string
): void {
  diagnostics.push({
    id: createId("diag"),
    level,
    source,
    path,
    message,
  });
}

function validateLocalConfig(config: ModelConfig): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];

  const integerFields: Array<keyof Pick<ModelConfig, "context_length" | "vocab_size" | "n_embd">> = [
    "context_length",
    "vocab_size",
    "n_embd",
  ];

  for (const field of integerFields) {
    const value = config[field];
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
      pushDiagnostic(
        diagnostics,
        "error",
        "local",
        field,
        "Must be an integer greater than 0."
      );
    }
  }

  if (!Array.isArray(config.blocks) || config.blocks.length === 0) {
    pushDiagnostic(
      diagnostics,
      "error",
      "local",
      "blocks",
      "At least one block is required."
    );
    return diagnostics;
  }

  config.blocks.forEach((block, blockIndex) => {
    const blockPath = `blocks[${blockIndex}]`;
    if (!Array.isArray(block.components) || block.components.length === 0) {
      pushDiagnostic(
        diagnostics,
        "error",
        "local",
        `${blockPath}.components`,
        "Block must contain at least one component."
      );
      return;
    }

    const hasAttention = block.components.some((component) => "attention" in component);
    const hasMlp = block.components.some((component) => "mlp" in component);
    if (!hasAttention) {
      pushDiagnostic(
        diagnostics,
        "warning",
        "local",
        blockPath,
        "Block has no attention component. This is allowed but unusual for transformer blocks."
      );
    }
    if (!hasMlp) {
      pushDiagnostic(
        diagnostics,
        "warning",
        "local",
        blockPath,
        "Block has no MLP component. Residual capacity may be limited."
      );
    }

    block.components.forEach((component, componentIndex) => {
      const componentPath = `${blockPath}.components[${componentIndex}]`;

      if ("attention" in component) {
        const { n_head, n_kv_head } = component.attention;
        if (!Number.isInteger(n_head) || n_head < 1) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${componentPath}.attention.n_head`,
            "n_head must be an integer greater than 0."
          );
        }
        if (!Number.isInteger(n_kv_head) || n_kv_head < 1) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${componentPath}.attention.n_kv_head`,
            "n_kv_head must be an integer greater than 0."
          );
        }
        if (Number.isInteger(n_head) && Number.isInteger(n_kv_head) && n_head > 0 && n_kv_head > 0) {
          if (n_kv_head > n_head) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention.n_kv_head`,
              "n_kv_head cannot exceed n_head."
            );
          }
          if (n_head % n_kv_head !== 0) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention`,
              "n_head must be divisible by n_kv_head (GQA grouping constraint)."
            );
          }
          if (Number.isInteger(config.n_embd) && config.n_embd > 0 && config.n_embd % n_head !== 0) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention.n_head`,
              "n_embd must be divisible by n_head."
            );
          }
          if (
            Number.isInteger(config.n_embd) &&
            config.n_embd > 0 &&
            config.n_embd % n_head === 0 &&
            (config.n_embd / n_head) % 2 !== 0
          ) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention`,
              "Rotary embeddings require an even head_dim (n_embd / n_head)."
            );
          }
        }
      }

      if ("mlp" in component) {
        const { multiplier, sequence } = component.mlp;
        const mlpPath = `${componentPath}.mlp`;

        if (!Number.isFinite(multiplier) || multiplier <= 0) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${mlpPath}.multiplier`,
            "MLP multiplier must be a finite number greater than 0."
          );
        }

        if (!Array.isArray(sequence) || sequence.length === 0) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${mlpPath}.sequence`,
            "MLP sequence must contain at least one step."
          );
          return;
        }

        const linearIndices: number[] = [];
        let activationCount = 0;

        sequence.forEach((step, stepIndex) => {
          if ("linear" in step) {
            linearIndices.push(stepIndex);
          }
          if ("activation" in step) {
            activationCount += 1;
          }
          if ("norm" in step && step.norm.type === "rmsnorm") {
            if (typeof step.norm.learnable_gamma !== "boolean") {
              pushDiagnostic(
                diagnostics,
                "error",
                "local",
                `${mlpPath}.sequence[${stepIndex}].norm.learnable_gamma`,
                "RMSNorm requires learnable_gamma boolean."
              );
            }
          }
        });

        if (linearIndices.length === 0) {
          pushDiagnostic(
            diagnostics,
            "warning",
            "local",
            mlpPath,
            "MLP sequence has no linear steps. This behaves like a nonlinear pass-through."
          );
        } else {
          const firstLinearIndex = linearIndices[0];
          const lastLinearIndex = linearIndices[linearIndices.length - 1];
          const multiplierIsIdentity = Math.abs(multiplier - 1) < 1e-9;

          if (firstLinearIndex !== 0 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "When multiplier != 1, the first MLP step should be linear to preserve input dimensions."
            );
          }
          if (lastLinearIndex !== sequence.length - 1 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "When multiplier != 1, the last MLP step should be linear to project back to n_embd."
            );
          }
          if (linearIndices.length === 1 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "A single linear step with multiplier != 1 will leave the MLP output dimension mismatched."
            );
          }
        }

        if (activationCount === 0) {
          pushDiagnostic(
            diagnostics,
            "warning",
            "local",
            mlpPath,
            "MLP sequence has no activation step."
          );
        }
      }
    });
  });

  return diagnostics;
}

function parseActivationType(value: unknown, path: string, errors: string[]): ActivationType {
  if (typeof value === "string" && (ACTIVATION_TYPES as readonly string[]).includes(value)) {
    return value as ActivationType;
  }
  errors.push(`${path} must be one of: ${ACTIVATION_TYPES.join(", ")}.`);
  return "relu";
}

function parseIntegerField(value: unknown, path: string, errors: string[]): number {
  if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)) {
    return value;
  }
  errors.push(`${path} must be an integer.`);
  return 0;
}

function parseNumberField(value: unknown, path: string, errors: string[]): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  errors.push(`${path} must be a number.`);
  return 0;
}

function parseBooleanField(value: unknown, path: string, errors: string[]): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  errors.push(`${path} must be a boolean.`);
  return false;
}

function parseNormConfig(value: unknown, path: string, errors: string[]): NormConfig {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { type: "layernorm" };
  }
  if (value.type === "layernorm") {
    const out: LayerNormConfig = { type: "layernorm" };
    return out;
  }
  if (value.type === "rmsnorm") {
    const learnable_gamma = parseBooleanField(
      value.learnable_gamma,
      `${path}.learnable_gamma`,
      errors
    );
    const out: RMSNormConfig = { type: "rmsnorm", learnable_gamma };
    return out;
  }
  errors.push(`${path}.type must be 'layernorm' or 'rmsnorm'.`);
  return { type: "layernorm" };
}

function parseActivationComponent(
  value: unknown,
  path: string,
  errors: string[]
): ActivationComponentConfig {
  if (!isRecord(value) || !isRecord(value.activation)) {
    errors.push(`${path}.activation must be an object.`);
    return { activation: { type: "relu" } };
  }
  return {
    activation: {
      type: parseActivationType(value.activation.type, `${path}.activation.type`, errors),
    },
  };
}

function parseNormComponent(value: unknown, path: string, errors: string[]): NormComponentConfig {
  if (!isRecord(value) || !("norm" in value)) {
    errors.push(`${path}.norm must be present.`);
    return { norm: { type: "layernorm" } };
  }
  return { norm: parseNormConfig(value.norm, `${path}.norm`, errors) };
}

function parseMlpStep(value: unknown, path: string, errors: string[]): MlpStep {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { activation: { type: "relu" } };
  }
  const keys = ["linear", "norm", "activation"].filter((key) => key in value);
  if (keys.length !== 1) {
    errors.push(`${path} must contain exactly one of linear/norm/activation.`);
    return { activation: { type: "relu" } };
  }

  if ("linear" in value) {
    if (!isRecord(value.linear)) {
      errors.push(`${path}.linear must be an object.`);
      return { linear: { bias: true } };
    }
    return {
      linear: {
        bias: parseBooleanField(value.linear.bias, `${path}.linear.bias`, errors),
      },
    };
  }
  if ("norm" in value) {
    return { norm: parseNormConfig(value.norm, `${path}.norm`, errors) };
  }
  return {
    activation: {
      type: parseActivationType(
        isRecord(value.activation) ? value.activation.type : undefined,
        `${path}.activation.type`,
        errors
      ),
    },
  };
}

function parseBlockComponent(value: unknown, path: string, errors: string[]): BlockComponent {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { activation: { type: "relu" } };
  }

  const keys = ["attention", "mlp", "norm", "activation"].filter((key) => key in value);
  if (keys.length !== 1) {
    errors.push(`${path} must contain exactly one of attention/mlp/norm/activation.`);
    return { activation: { type: "relu" } };
  }

  if ("attention" in value) {
    if (!isRecord(value.attention)) {
      errors.push(`${path}.attention must be an object.`);
      return { attention: { n_head: 12, n_kv_head: 12 } };
    }
    return {
      attention: {
        n_head: parseIntegerField(value.attention.n_head, `${path}.attention.n_head`, errors),
        n_kv_head: parseIntegerField(
          value.attention.n_kv_head,
          `${path}.attention.n_kv_head`,
          errors
        ),
      },
    };
  }

  if ("mlp" in value) {
    if (!isRecord(value.mlp)) {
      errors.push(`${path}.mlp must be an object.`);
      return { mlp: { multiplier: 4, sequence: [] } };
    }
    const sequenceValue = value.mlp.sequence;
    const sequence: MlpStep[] = [];
    if (!Array.isArray(sequenceValue)) {
      errors.push(`${path}.mlp.sequence must be an array.`);
    } else {
      sequenceValue.forEach((step, stepIndex) => {
        sequence.push(parseMlpStep(step, `${path}.mlp.sequence[${stepIndex}]`, errors));
      });
    }
    return {
      mlp: {
        multiplier: parseNumberField(value.mlp.multiplier, `${path}.mlp.multiplier`, errors),
        sequence,
      },
    };
  }

  if ("norm" in value) {
    return parseNormComponent(value, path, errors);
  }

  return parseActivationComponent(value, path, errors);
}

function parseImportedModelConfig(value: unknown): { config: ModelConfig | null; errors: string[] } {
  const errors: string[] = [];
  if (!isRecord(value)) {
    return { config: null, errors: ["Root JSON value must be an object."] };
  }

  const blocksValue = value.blocks;
  const blocks: ModelBlock[] = [];
  if (!Array.isArray(blocksValue)) {
    errors.push("blocks must be an array.");
  } else {
    blocksValue.forEach((blockValue, blockIndex) => {
      const blockPath = `blocks[${blockIndex}]`;
      if (!isRecord(blockValue)) {
        errors.push(`${blockPath} must be an object.`);
        return;
      }
      const componentsValue = blockValue.components;
      if (!Array.isArray(componentsValue)) {
        errors.push(`${blockPath}.components must be an array.`);
        blocks.push({ components: [] });
        return;
      }
      const components = componentsValue.map((component, componentIndex) =>
        parseBlockComponent(component, `${blockPath}.components[${componentIndex}]`, errors)
      );
      blocks.push({ components });
    });
  }

  const config: ModelConfig = {
    context_length: parseIntegerField(value.context_length, "context_length", errors),
    vocab_size: parseIntegerField(value.vocab_size, "vocab_size", errors),
    n_embd: parseIntegerField(value.n_embd, "n_embd", errors),
    weight_tying: parseBooleanField(value.weight_tying, "weight_tying", errors),
    blocks,
  };

  return {
    config: errors.length === 0 ? config : null,
    errors,
  };
}

function DropSlot({
  active,
  compact,
  label,
  onDragOver,
  onDrop,
}: {
  active: boolean;
  compact?: boolean;
  label: string;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      className={`dropSlot${compact ? " isCompact" : ""}${active ? " isActive" : ""}`}
      onDragOver={onDragOver}
      onDrop={onDrop}
      aria-label={label}
    >
      <span>{label}</span>
    </div>
  );
}

function PaletteTile({
  title,
  subtitle,
  colorClass,
  draggable,
  onDragStart,
  onDragEnd,
}: {
  title: string;
  subtitle: string;
  colorClass: string;
  draggable: boolean;
  onDragStart: (event: DragEvent<HTMLDivElement>) => void;
  onDragEnd: () => void;
}) {
  return (
    <div
      className={`paletteTile ${colorClass}`}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
    >
      <div className="paletteTileTitle">{title}</div>
      <div className="paletteTileSubtitle">{subtitle}</div>
      <div className="paletteTileHint">Drag to canvas</div>
    </div>
  );
}

function StatusCard({
  title,
  value,
  detail,
  tone,
  icon,
}: {
  title: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  icon: ReactNode;
}) {
  return (
    <div className={`statusCard${tone ? ` tone-${tone}` : ""}`}>
      <div className="statusCardIcon" aria-hidden>
        {icon}
      </div>
      <div>
        <div className="statusCardTitle">{title}</div>
        <div className="statusCardValue">{value}</div>
        <div className="statusCardDetail">{detail}</div>
      </div>
    </div>
  );
}

export default function Page() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<DragPayload | null>(null);
  const validationRunRef = useRef(0);

  const [theme, setTheme] = useState<ThemeMode>("white");
  const [documentState, setDocumentState] = useState<StudioDocument>(() =>
    studioDocumentFromConfig(createDefaultModelConfig())
  );
  const [importDraft, setImportDraft] = useState<string>("");
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);
  const [expandedComponentIds, setExpandedComponentIds] = useState<Set<string>>(() => new Set());
  const [expandedMlpStepIds, setExpandedMlpStepIds] = useState<Set<string>>(() => new Set());
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);
  const [backendValidation, setBackendValidation] = useState<BackendValidationState>({
    phase: "idle",
    message: "Waiting for edits",
    lastValidatedAt: null,
    warnings: [],
    errors: [],
    normalizedChanged: false,
  });
  const [backendAnalysis, setBackendAnalysis] = useState<BackendAnalysisState>({
    phase: "idle",
    message: "Run backend analysis to instantiate ConfigurableGPT and inspect parameter counts.",
    lastAnalyzedAt: null,
    configSignature: null,
    summary: null,
    warnings: [],
    errors: [],
    instantiationError: null,
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
      if (savedTheme === "dark" || savedTheme === "white") {
        setTheme(savedTheme);
      }

      const savedImportDraft = window.localStorage.getItem(IMPORT_DRAFT_STORAGE_KEY);
      if (typeof savedImportDraft === "string") {
        setImportDraft(savedImportDraft);
      }

      const raw = window.localStorage.getItem(DOCUMENT_STORAGE_KEY);
      if (!raw) {
        return;
      }

      const parsed = JSON.parse(raw) as unknown;
      const imported = parseImportedModelConfig(parsed);
      if (imported.config) {
        setDocumentState(studioDocumentFromConfig(imported.config));
      }
    } catch {
      // ignore corrupted local storage and continue with defaults
    }
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.dataset.theme = theme;
    }
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const config = studioDocumentToConfig(documentState);
      window.localStorage.setItem(DOCUMENT_STORAGE_KEY, JSON.stringify(config));
      setLastSavedAt(Date.now());
    } catch {
      // no-op
    }
  }, [documentState]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(IMPORT_DRAFT_STORAGE_KEY, importDraft);
  }, [importDraft]);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setNotice((current) => (current?.at === notice.at ? null : current));
    }, 2800);
    return () => window.clearTimeout(timeoutId);
  }, [notice]);

  useEffect(() => {
    const validComponentIds = new Set(collectAllComponentIds(documentState));
    setExpandedComponentIds((current) => {
      let changed = false;
      const next = new Set<string>();
      current.forEach((id) => {
        if (validComponentIds.has(id)) {
          next.add(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : current;
    });

    const validMlpStepIds = new Set(collectAllMlpStepIds(documentState));
    setExpandedMlpStepIds((current) => {
      let changed = false;
      const next = new Set<string>();
      current.forEach((id) => {
        if (validMlpStepIds.has(id)) {
          next.add(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : current;
    });
  }, [documentState]);

  const modelConfig = studioDocumentToConfig(documentState);
  const localDiagnostics = validateLocalConfig(modelConfig);
  const localErrors = localDiagnostics.filter((item) => item.level === "error");
  const localWarnings = localDiagnostics.filter((item) => item.level === "warning");
  const metrics = collectBuilderMetrics(documentState);

  const compactJson = JSON.stringify(modelConfig);
  const deferredJsonSignature = useDeferredValue(compactJson);
  const previewJson = JSON.stringify(JSON.parse(deferredJsonSignature), null, 2);

  useEffect(() => {
    const hasLocalErrors = localErrors.length > 0;
    if (hasLocalErrors) {
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
  }, [compactJson, localErrors.length]);

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

  const diagnostics = [...localDiagnostics, ...backendDiagnostics];
  const totalErrors = diagnostics.filter((item) => item.level === "error").length;
  const totalWarnings = diagnostics.filter((item) => item.level === "warning").length;
  const backendAnalysisStale =
    backendAnalysis.configSignature !== null && backendAnalysis.configSignature !== compactJson;

  const validationStatusLabel =
    totalErrors > 0
      ? `${totalErrors} error${totalErrors === 1 ? "" : "s"}`
      : totalWarnings > 0
        ? `${totalWarnings} warning${totalWarnings === 1 ? "" : "s"}`
        : "Clean";

  function setNoticeMessage(tone: NoticeTone, message: string): void {
    setNotice({ tone, message, at: Date.now() });
  }

  function toggleExpandedComponent(componentId: string): void {
    setExpandedComponentIds((current) => {
      const next = new Set(current);
      if (next.has(componentId)) {
        next.delete(componentId);
      } else {
        next.add(componentId);
      }
      return next;
    });
  }

  function toggleExpandedMlpStep(stepId: string): void {
    setExpandedMlpStepIds((current) => {
      const next = new Set(current);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  }

  function expandAllCanvasNodes(): void {
    setExpandedComponentIds(new Set(collectAllComponentIds(documentState)));
    setExpandedMlpStepIds(new Set(collectAllMlpStepIds(documentState)));
  }

  function collapseAllCanvasNodes(): void {
    setExpandedComponentIds(new Set());
    setExpandedMlpStepIds(new Set());
  }

  function writeDragPayload(event: DragEvent, payload: DragPayload): void {
    dragRef.current = payload;
    event.dataTransfer.setData(DND_MIME, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "move";
  }

  function readDragPayload(event: DragEvent): DragPayload | null {
    if (dragRef.current) {
      return dragRef.current;
    }
    const raw = event.dataTransfer.getData(DND_MIME);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw) as DragPayload;
    } catch {
      return null;
    }
  }

  function clearDragState(): void {
    dragRef.current = null;
    setDragOverKey(null);
  }

  function beginDragBlock(event: DragEvent<HTMLDivElement>, blockId: string): void {
    writeDragPayload(event, { kind: "block", blockId });
  }

  function beginDragPaletteComponent(
    event: DragEvent<HTMLDivElement>,
    componentKind: StudioComponentKind
  ): void {
    writeDragPayload(event, { kind: "palette-component", componentKind });
  }

  function beginDragComponent(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    componentId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "block-component", fromBlockId, componentId });
  }

  function beginDragPaletteMlpStep(
    event: DragEvent<HTMLDivElement>,
    stepKind: MlpStepKind
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "palette-mlp-step", stepKind });
  }

  function beginDragMlpStep(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    fromComponentId: string,
    stepId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "mlp-step", fromBlockId, fromComponentId, stepId });
  }

  function markDropTarget(event: DragEvent<HTMLDivElement>, key: string): void {
    event.preventDefault();
    setDragOverKey(key);
    event.dataTransfer.dropEffect = "move";
  }

  function handleDropBlock(event: DragEvent<HTMLDivElement>, insertIndex: number): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload || payload.kind !== "block") {
      return;
    }
    setDocumentState((current) => {
      const fromIndex = current.blocks.findIndex((block) => block.id === payload.blockId);
      if (fromIndex < 0) {
        return current;
      }
      return {
        ...current,
        blocks: moveItem(current.blocks, fromIndex, insertIndex),
      };
    });
  }

  function handleDropComponent(
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }
    const createdComponent =
      payload.kind === "palette-component"
        ? createDefaultStudioComponent(payload.componentKind)
        : null;

    setDocumentState((current) => {
      const next = clone(current);
      const targetBlockIndex = findBlockIndex(next, targetBlockId);
      if (targetBlockIndex < 0) {
        return current;
      }
      const targetBlock = next.blocks[targetBlockIndex];
      const targetInsertIndex = clamp(insertIndex, 0, targetBlock.components.length);

      if (payload.kind === "palette-component") {
        targetBlock.components.splice(targetInsertIndex, 0, createdComponent!);
        return next;
      }

      if (payload.kind !== "block-component") {
        return current;
      }

      const sourceBlockIndex = findBlockIndex(next, payload.fromBlockId);
      if (sourceBlockIndex < 0) {
        return current;
      }
      const sourceBlock = next.blocks[sourceBlockIndex];
      const sourceComponentIndex = findComponentIndex(sourceBlock, payload.componentId);
      if (sourceComponentIndex < 0) {
        return current;
      }
      const [moved] = sourceBlock.components.splice(sourceComponentIndex, 1);
      let adjustedInsertIndex = targetInsertIndex;
      if (
        payload.fromBlockId === targetBlockId &&
        sourceComponentIndex < adjustedInsertIndex
      ) {
        adjustedInsertIndex -= 1;
      }
      adjustedInsertIndex = clamp(adjustedInsertIndex, 0, targetBlock.components.length);
      targetBlock.components.splice(adjustedInsertIndex, 0, moved);
      return next;
    });

    if (createdComponent) {
      setExpandedComponentIds((current) => new Set([...current, createdComponent.id]));
      if (createdComponent.kind === "mlp") {
        setExpandedMlpStepIds(
          (current) => new Set([...current, ...createdComponent.mlp.sequence.map((step) => step.id)])
        );
      }
    }
  }

  function handleDropMlpStep(
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }
    const createdStep =
      payload.kind === "palette-mlp-step" ? createDefaultStudioMlpStep(payload.stepKind) : null;

    setDocumentState((current) => {
      const next = clone(current);
      const targetRef = getMlpComponent(next, targetBlockId, targetComponentId);
      if (!targetRef) {
        return current;
      }
      const targetSequence = targetRef.component.mlp.sequence;
      let targetInsertIndex = clamp(insertIndex, 0, targetSequence.length);

      if (payload.kind === "palette-mlp-step") {
        targetSequence.splice(targetInsertIndex, 0, createdStep!);
        return next;
      }

      if (payload.kind !== "mlp-step") {
        return current;
      }

      const sourceRef = getMlpComponent(next, payload.fromBlockId, payload.fromComponentId);
      if (!sourceRef) {
        return current;
      }
      const sourceIndex = sourceRef.component.mlp.sequence.findIndex((step) => step.id === payload.stepId);
      if (sourceIndex < 0) {
        return current;
      }

      const [moved] = sourceRef.component.mlp.sequence.splice(sourceIndex, 1);
      if (
        payload.fromBlockId === targetBlockId &&
        payload.fromComponentId === targetComponentId &&
        sourceIndex < targetInsertIndex
      ) {
        targetInsertIndex -= 1;
      }
      targetInsertIndex = clamp(targetInsertIndex, 0, targetRef.component.mlp.sequence.length);
      targetRef.component.mlp.sequence.splice(targetInsertIndex, 0, moved);
      return next;
    });

    if (createdStep) {
      setExpandedMlpStepIds((current) => new Set([...current, createdStep.id]));
    }
  }

  function updateBaseField<K extends keyof Pick<StudioDocument, "context_length" | "vocab_size" | "n_embd">>(
    key: K,
    value: number
  ): void {
    setDocumentState((current) => ({ ...current, [key]: value }));
  }

  function addBlock(): void {
    setDocumentState((current) => ({
      ...current,
      blocks: [...current.blocks, studioBlockFromConfig(createDefaultBlockConfig())],
    }));
  }

  function duplicateBlock(blockId: string): void {
    setDocumentState((current) => {
      const index = current.blocks.findIndex((block) => block.id === blockId);
      if (index < 0) {
        return current;
      }
      const duplicate = cloneBlockWithNewIds(current.blocks[index]);
      const nextBlocks = current.blocks.slice();
      nextBlocks.splice(index + 1, 0, duplicate);
      return { ...current, blocks: nextBlocks };
    });
  }

  function deleteBlock(blockId: string): void {
    setDocumentState((current) => {
      if (current.blocks.length <= 1) {
        return current;
      }
      return {
        ...current,
        blocks: current.blocks.filter((block) => block.id !== blockId),
      };
    });
  }

  function resetDefaults(): void {
    setDocumentState(studioDocumentFromConfig(createDefaultModelConfig()));
    setNoticeMessage("info", "Reset to default LLM config template.");
  }

  function removeComponent(blockId: string, componentId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components = next.blocks[blockIndex].components.filter(
        (component) => component.id !== componentId
      );
      return next;
    });
  }

  function updateComponent(
    blockId: string,
    componentId: string,
    updater: (component: StudioComponent) => StudioComponent
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      const componentIndex = findComponentIndex(next.blocks[blockIndex], componentId);
      if (componentIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components[componentIndex] = updater(
        next.blocks[blockIndex].components[componentIndex]
      );
      return next;
    });
  }

  function updateMlpStep(
    blockId: string,
    componentId: string,
    stepId: string,
    updater: (step: StudioMlpStep) => StudioMlpStep
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      const stepIndex = mlpRef.component.mlp.sequence.findIndex((step) => step.id === stepId);
      if (stepIndex < 0) {
        return current;
      }
      mlpRef.component.mlp.sequence[stepIndex] = updater(mlpRef.component.mlp.sequence[stepIndex]);
      return next;
    });
  }

  function removeMlpStep(blockId: string, componentId: string, stepId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      mlpRef.component.mlp.sequence = mlpRef.component.mlp.sequence.filter((step) => step.id !== stepId);
      return next;
    });
  }

  async function importFromFile(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      setImportDraft(text);
      applyImportText(text);
    } catch {
      setNoticeMessage("error", "Failed to read selected JSON file.");
    }
  }

  function applyImportText(text: string): void {
    try {
      const parsedJson = JSON.parse(text) as unknown;
      const imported = parseImportedModelConfig(parsedJson);
      if (!imported.config) {
        setNoticeMessage(
          "error",
          `Import failed: ${imported.errors.slice(0, 3).join(" ")}`
        );
        return;
      }
      startTransition(() => {
        setDocumentState(studioDocumentFromConfig(imported.config as ModelConfig));
      });
      setNoticeMessage("success", "Imported model config JSON into visual builder.");
    } catch (error) {
      setNoticeMessage(
        "error",
        error instanceof Error ? `Import failed: ${error.message}` : "Import failed."
      );
    }
  }

  function exportJson(): void {
    downloadTextFile("model_config.json", JSON.stringify(modelConfig, null, 2));
    setNoticeMessage("success", "Exported model config JSON.");
  }

  async function copyJson(): Promise<void> {
    try {
      await navigator.clipboard.writeText(JSON.stringify(modelConfig, null, 2));
      setNoticeMessage("success", "Copied JSON to clipboard.");
    } catch {
      setNoticeMessage("error", "Clipboard write failed in this environment.");
    }
  }

  async function runBackendAnalysis(): Promise<void> {
    if (localErrors.length > 0) {
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
        setNoticeMessage(
          "error",
          result.instantiation_error ?? "Backend model analysis failed."
        );
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

  function renderNormFields(
    norm: NormConfig,
    onChange: (next: NormConfig) => void,
    idPrefix: string
  ): ReactNode {
    return (
      <div className="fieldGrid compact">
        <label className="fieldLabel" htmlFor={`${idPrefix}-norm-type`}>
          <span>Norm type</span>
          <select
            id={`${idPrefix}-norm-type`}
            value={norm.type}
            onChange={(event) => {
              if (event.target.value === "rmsnorm") {
                onChange({ type: "rmsnorm", learnable_gamma: true });
              } else {
                onChange({ type: "layernorm" });
              }
            }}
          >
            <option value="layernorm">layernorm</option>
            <option value="rmsnorm">rmsnorm</option>
          </select>
        </label>
        {norm.type === "rmsnorm" ? (
          <label className="toggleField" htmlFor={`${idPrefix}-learnable-gamma`}>
            <input
              id={`${idPrefix}-learnable-gamma`}
              type="checkbox"
              checked={norm.learnable_gamma}
              onChange={(event) =>
                onChange({ type: "rmsnorm", learnable_gamma: event.target.checked })
              }
            />
            <span>learnable_gamma</span>
          </label>
        ) : null}
      </div>
    );
  }

  return (
    <main className="studioRoot">
      <nav className="studioNav" aria-label="LLM Studio navigation">
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Studio</span>
        </div>
        <div className="studioNavLinks">
          <a className="studioNavLink" href="#base-model">
            Base Model
          </a>
          <a className="studioNavLink" href="#block-builder">
            Builder
          </a>
          <a className="studioNavLink" href="#diagnostics">
            Diagnostics
          </a>
          <a className="studioNavLink" href="#model-analysis">
            Analysis
          </a>
          <a className="studioNavLink" href="#json-preview">
            JSON
          </a>
        </div>
        <button
          type="button"
          className="themeToggle"
          onClick={() => setTheme((current) => (current === "dark" ? "white" : "dark"))}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </nav>

      <section className="panelCard heroCard">
        <div className="panelHead heroHead">
          <div>
            <p className="panelEyebrow">Visual /model Designer</p>
            <h1>Compose transformer blocks, validate live, export clean JSON.</h1>
            <p className="panelCopy">
              Build `{"/model"}` configs using draggable block components and nested MLP step editors.
              LLM Studio keeps a local semantic validator active even when the backend validator is offline.
            </p>
          </div>
          <div className="heroActions">
            <button type="button" className="buttonGhost" onClick={addBlock}>
              <FiPlus /> Add Block
            </button>
            <button type="button" className="buttonGhost" onClick={() => fileInputRef.current?.click()}>
              <FiUpload /> Import File
            </button>
            <button type="button" className="buttonGhost" onClick={exportJson}>
              <FiDownload /> Export JSON
            </button>
            <button type="button" className="buttonGhost" onClick={resetDefaults}>
              <FiRefreshCw /> Reset
            </button>
          </div>
        </div>

        {notice ? (
          <div className={`inlineNotice tone-${notice.tone}`} role="status" aria-live="polite">
            {notice.message}
          </div>
        ) : null}

        <div className="statusGrid">
          <StatusCard
            title="Validation"
            value={validationStatusLabel}
            detail={backendValidation.message}
            tone={
              totalErrors > 0
                ? "bad"
                : totalWarnings > 0 || backendValidation.phase === "fallback"
                  ? "warn"
                  : "good"
            }
            icon={totalErrors > 0 ? <FiXCircle /> : totalWarnings > 0 ? <FiAlertTriangle /> : <FiCheckCircle />}
          />
          <StatusCard
            title="Blocks"
            value={`${metrics.blockCount}`}
            detail={`${metrics.componentCount} components · ${metrics.mlpStepCount} MLP steps`}
            tone="neutral"
            icon={<FiLayers />}
          />
          <StatusCard
            title="Backend"
            value={backendValidation.phase === "success" ? "Connected" : backendValidation.phase === "validating" ? "Validating" : backendValidation.phase === "fallback" ? "Fallback" : "Idle"}
            detail={`${apiBaseUrl()} · ${formatTimeAgo(backendValidation.lastValidatedAt)}`}
            tone={backendValidation.phase === "success" ? "good" : backendValidation.phase === "fallback" ? "warn" : "neutral"}
            icon={<FiServer />}
          />
          <StatusCard
            title="Local Cache"
            value="Auto-saved"
            detail={`Last save ${formatTimeAgo(lastSavedAt)}`}
            tone="neutral"
            icon={<FiHardDrive />}
          />
        </div>
      </section>

      <div className="twoColLayout">
        <section id="base-model" className="panelCard">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Base Model</p>
              <h2>Core config controls</h2>
              <p className="panelCopy">
                These values shape the shared model dimensions and drive semantic checks for attention head sizing.
              </p>
            </div>
          </div>
          <div className="fieldGrid">
            <label className="fieldLabel" htmlFor="context_length">
              <span>context_length</span>
              <input
                id="context_length"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.context_length)}
                onChange={(event) =>
                  updateBaseField(
                    "context_length",
                    parseIntegerInput(event.target.value, documentState.context_length)
                  )
                }
              />
            </label>
            <label className="fieldLabel" htmlFor="vocab_size">
              <span>vocab_size</span>
              <input
                id="vocab_size"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.vocab_size)}
                onChange={(event) =>
                  updateBaseField(
                    "vocab_size",
                    parseIntegerInput(event.target.value, documentState.vocab_size)
                  )
                }
              />
            </label>
            <label className="fieldLabel" htmlFor="n_embd">
              <span>n_embd</span>
              <input
                id="n_embd"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.n_embd)}
                onChange={(event) =>
                  updateBaseField("n_embd", parseIntegerInput(event.target.value, documentState.n_embd))
                }
              />
            </label>
            <label className="toggleField tall" htmlFor="weight_tying">
              <input
                id="weight_tying"
                type="checkbox"
                checked={documentState.weight_tying}
                onChange={(event) =>
                  setDocumentState((current) => ({ ...current, weight_tying: event.target.checked }))
                }
              />
              <span>weight_tying</span>
            </label>
          </div>
        </section>

        <section className="panelCard">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Palette</p>
              <h2>Drag components into blocks</h2>
              <p className="panelCopy">
                Use native drag-and-drop to attach components to any block row or reorder existing components.
              </p>
            </div>
          </div>
          <div className="paletteGrid">
            {([
              {
                kind: "attention",
                subtitle: "Self-attention with n_head / n_kv_head",
                colorClass: "tone-attention",
              },
              {
                kind: "mlp",
                subtitle: "Configurable MLP with nested sequence editor",
                colorClass: "tone-mlp",
              },
              {
                kind: "norm",
                subtitle: "LayerNorm or RMSNorm (optional learnable gamma)",
                colorClass: "tone-norm",
              },
              {
                kind: "activation",
                subtitle: "Standalone activation block component",
                colorClass: "tone-activation",
              },
            ] as const).map((entry) => (
              <PaletteTile
                key={entry.kind}
                title={labelForComponentKind(entry.kind)}
                subtitle={entry.subtitle}
                colorClass={entry.colorClass}
                draggable
                onDragStart={(event) => beginDragPaletteComponent(event, entry.kind)}
                onDragEnd={clearDragState}
              />
            ))}
          </div>
        </section>
      </div>

      <section id="block-builder" className="panelCard builderPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Visual Builder</p>
            <h2>Horizontal block canvas</h2>
            <p className="panelCopy">
              Scroll horizontally for model depth and vertically for details. Drag blocks to reorder, then expand only the nodes you want to edit.
            </p>
          </div>
          <div className="actionCluster">
            <button type="button" className="buttonGhost" onClick={collapseAllCanvasNodes}>
              <FiChevronRight /> Collapse all
            </button>
            <button type="button" className="buttonGhost" onClick={expandAllCanvasNodes}>
              <FiChevronDown /> Expand all
            </button>
            <button type="button" className="buttonGhost" onClick={addBlock}>
              <FiPlus /> Add block
            </button>
          </div>
        </div>

        <div className="builderCanvasToolbar">
          <div className="builderCanvasHint">
            <span className="builderCanvasHintDot" aria-hidden />
            Canvas scrolls in both directions. Blocks are columns; components stay attached inside each block.
          </div>
          <div className="builderCanvasStats" aria-label="Builder canvas statistics">
            <span>{metrics.blockCount} blocks</span>
            <span>{metrics.componentCount} components</span>
            <span>{metrics.mlpStepCount} MLP steps</span>
          </div>
        </div>

        <div className="blockCanvasViewport" role="region" aria-label="Horizontal model block canvas">
          <div className="blockCanvas">
          <DropSlot
            active={dragOverKey === "block-slot-0"}
            label="Insert block"
            onDragOver={(event) => markDropTarget(event, "block-slot-0")}
            onDrop={(event) => handleDropBlock(event, 0)}
          />

          {documentState.blocks.map((block, blockIndex) => (
            <Fragment key={block.id}>
              <article className="blockCard">
                <div className="blockCardHead">
                  <div className="blockCardTitleWrap">
                    <div
                      className="dragBadge"
                      draggable
                      onDragStart={(event) => beginDragBlock(event, block.id)}
                      onDragEnd={clearDragState}
                      title={`Drag block ${blockIndex + 1}`}
                      aria-hidden
                    >
                      <FiMove />
                    </div>
                    <div>
                      <h3>Block {blockIndex + 1}</h3>
                      <p>{block.components.length} component{block.components.length === 1 ? "" : "s"}</p>
                    </div>
                  </div>
                  <div className="blockCardActions">
                    <button
                      type="button"
                      className="iconButton"
                      onClick={() => duplicateBlock(block.id)}
                      title="Duplicate block"
                      aria-label={`Duplicate block ${blockIndex + 1}`}
                    >
                      <FiCopy />
                    </button>
                    <button
                      type="button"
                      className="iconButton danger"
                      onClick={() => deleteBlock(block.id)}
                      title={documentState.blocks.length <= 1 ? "Keep at least one block" : "Delete block"}
                      aria-label={`Delete block ${blockIndex + 1}`}
                      disabled={documentState.blocks.length <= 1}
                    >
                      <FiTrash2 />
                    </button>
                  </div>
                </div>
                <div className="blockQuickMap" aria-label={`Block ${blockIndex + 1} sequence`}>
                  {block.components.length === 0 ? (
                    <span className="flowMiniChip isEmpty">Empty</span>
                  ) : (
                    block.components.map((component, componentIndex) => (
                      <span
                        key={`${block.id}-${component.id}-chip`}
                        className={`flowMiniChip kind-${component.kind}`}
                        title={`${componentIndex + 1}. ${labelForComponentKind(component.kind)} · ${summarizeComponent(component)}`}
                      >
                        {componentIndex + 1}. {labelForComponentKind(component.kind)}
                      </span>
                    ))
                  )}
                </div>

                <div className="componentLane">
                  <DropSlot
                    compact
                    active={dragOverKey === `component-slot-${block.id}-0`}
                    label="Insert"
                    onDragOver={(event) => markDropTarget(event, `component-slot-${block.id}-0`)}
                    onDrop={(event) => handleDropComponent(event, block.id, 0)}
                  />

                  {block.components.length === 0 ? (
                    <div className="emptyLaneHint">Drop a palette component here to start this block.</div>
                  ) : null}

                  {block.components.map((component, componentIndex) => (
                    <Fragment key={component.id}>
                      <section
                        className={`componentCard kind-${component.kind}${expandedComponentIds.has(component.id) ? "" : " isCollapsed"}`}
                      >
                        <div className="componentCardHead">
                          <div
                            className="dragBadge"
                            draggable
                            onDragStart={(event) => beginDragComponent(event, block.id, component.id)}
                            onDragEnd={clearDragState}
                            title={`Drag ${labelForComponentKind(component.kind)} component`}
                            aria-hidden
                          >
                            <FiMove />
                          </div>
                          <div className="componentMeta">
                            <span className="componentTag">{labelForComponentKind(component.kind)}</span>
                            <span className="componentSummary">{summarizeComponent(component)}</span>
                          </div>
                          <div className="componentHeadActions">
                            <button
                              type="button"
                              className="iconButton"
                              onClick={() => toggleExpandedComponent(component.id)}
                              aria-label={`${expandedComponentIds.has(component.id) ? "Collapse" : "Expand"} ${labelForComponentKind(component.kind)} component settings`}
                              title={expandedComponentIds.has(component.id) ? "Collapse settings" : "Expand settings"}
                              aria-expanded={expandedComponentIds.has(component.id)}
                            >
                              {expandedComponentIds.has(component.id) ? <FiChevronDown /> : <FiChevronRight />}
                            </button>
                            <button
                              type="button"
                              className="iconButton danger"
                              onClick={() => removeComponent(block.id, component.id)}
                              aria-label="Remove component"
                              title="Remove component"
                            >
                              <FiTrash2 />
                            </button>
                          </div>
                        </div>

                        {!expandedComponentIds.has(component.id) && component.kind === "mlp" ? (
                          <div className="componentCollapsedTrail" aria-label="MLP sequence summary">
                            {component.mlp.sequence.map((step, stepIndex) => (
                              <span
                                key={`${component.id}-${step.id}-trail`}
                                className={`miniStepBadge kind-${step.kind}`}
                                title={`${stepIndex + 1}. ${labelForMlpStepKind(step.kind)} · ${summarizeMlpStep(step)}`}
                              >
                                {stepIndex + 1}. {labelForMlpStepKind(step.kind)}
                              </span>
                            ))}
                          </div>
                        ) : null}

                        {expandedComponentIds.has(component.id) ? (
                          <div className="componentBody">
                            {component.kind === "attention" ? (
                              <div className="fieldGrid compact">
                                <label
                                  className="fieldLabel"
                                  htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-head`}
                                >
                                  <span>n_head</span>
                                  <input
                                    id={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-head`}
                                    type="number"
                                    min={1}
                                    step={1}
                                    value={integerInputValue(component.attention.n_head)}
                                    onChange={(event) =>
                                      updateComponent(block.id, component.id, (current) =>
                                        current.kind !== "attention"
                                          ? current
                                          : {
                                              ...current,
                                              attention: {
                                                ...current.attention,
                                                n_head: parseIntegerInput(
                                                  event.target.value,
                                                  current.attention.n_head
                                                ),
                                              },
                                            }
                                      )
                                    }
                                  />
                                </label>
                                <label
                                  className="fieldLabel"
                                  htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-kv-head`}
                                >
                                  <span>n_kv_head</span>
                                  <input
                                    id={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-kv-head`}
                                    type="number"
                                    min={1}
                                    step={1}
                                    value={integerInputValue(component.attention.n_kv_head)}
                                    onChange={(event) =>
                                      updateComponent(block.id, component.id, (current) =>
                                        current.kind !== "attention"
                                          ? current
                                          : {
                                              ...current,
                                              attention: {
                                                ...current.attention,
                                                n_kv_head: parseIntegerInput(
                                                  event.target.value,
                                                  current.attention.n_kv_head
                                                ),
                                              },
                                            }
                                      )
                                    }
                                  />
                                </label>
                              </div>
                            ) : null}

                            {component.kind === "norm"
                              ? renderNormFields(component.norm, (nextNorm) => {
                                  updateComponent(block.id, component.id, (current) =>
                                    current.kind !== "norm" ? current : { ...current, norm: nextNorm }
                                  );
                                }, componentDomIdPrefix(blockIndex, componentIndex))
                              : null}

                            {component.kind === "activation" ? (
                              <div className="fieldGrid compact">
                                <label
                                  className="fieldLabel"
                                  htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-activation-type`}
                                >
                                  <span>Activation</span>
                                  <select
                                    id={`${componentDomIdPrefix(blockIndex, componentIndex)}-activation-type`}
                                    value={component.activation.type}
                                    onChange={(event) =>
                                      updateComponent(block.id, component.id, (current) =>
                                        current.kind !== "activation"
                                          ? current
                                          : {
                                              ...current,
                                              activation: {
                                                type: event.target.value as ActivationType,
                                              },
                                            }
                                      )
                                    }
                                  >
                                    {ACTIVATION_TYPES.map((type) => (
                                      <option key={type} value={type}>
                                        {type}
                                      </option>
                                    ))}
                                  </select>
                                </label>
                              </div>
                            ) : null}

                            {component.kind === "mlp" ? (
                              <div className="mlpEditor">
                                <div className="fieldGrid compact">
                                  <label
                                    className="fieldLabel"
                                    htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-multiplier`}
                                  >
                                    <span>multiplier</span>
                                    <input
                                      id={`${componentDomIdPrefix(blockIndex, componentIndex)}-multiplier`}
                                      type="number"
                                      min={0.001}
                                      step="any"
                                      value={numberInputValue(component.mlp.multiplier)}
                                      onChange={(event) =>
                                        updateComponent(block.id, component.id, (current) =>
                                          current.kind !== "mlp"
                                            ? current
                                            : {
                                                ...current,
                                                mlp: {
                                                  ...current.mlp,
                                                  multiplier: parseNumberInput(
                                                    event.target.value,
                                                    current.mlp.multiplier
                                                  ),
                                                },
                                              }
                                        )
                                      }
                                    />
                                  </label>
                                </div>

                                <div className="mlpPaletteRow">
                                  <div className="miniLabel">MLP Step Palette</div>
                                  <div className="miniPaletteGrid">
                                    {([
                                      {
                                        kind: "linear",
                                        subtitle: "Linear layer",
                                        colorClass: "tone-linear",
                                      },
                                      {
                                        kind: "norm",
                                        subtitle: "Norm step",
                                        colorClass: "tone-norm",
                                      },
                                      {
                                        kind: "activation",
                                        subtitle: "Activation step",
                                        colorClass: "tone-activation",
                                      },
                                    ] as const).map((entry) => (
                                      <PaletteTile
                                        key={`${component.id}-${entry.kind}`}
                                        title={labelForMlpStepKind(entry.kind)}
                                        subtitle={entry.subtitle}
                                        colorClass={entry.colorClass}
                                        draggable
                                        onDragStart={(event) => beginDragPaletteMlpStep(event, entry.kind)}
                                        onDragEnd={clearDragState}
                                      />
                                    ))}
                                  </div>
                                </div>

                                <div className="mlpSequenceShell">
                                  <div className="miniLabel">Sequence</div>
                                  <div className="mlpSequenceList">
                                    <DropSlot
                                      compact
                                      active={dragOverKey === `mlp-slot-${block.id}-${component.id}-0`}
                                      label="Insert step"
                                      onDragOver={(event) =>
                                        markDropTarget(event, `mlp-slot-${block.id}-${component.id}-0`)
                                      }
                                      onDrop={(event) => handleDropMlpStep(event, block.id, component.id, 0)}
                                    />
                                    {component.mlp.sequence.length === 0 ? (
                                      <div className="emptyLaneHint compact">Drop a linear/norm/activation step here.</div>
                                    ) : null}
                                    {component.mlp.sequence.map((step, stepIndex) => (
                                      <Fragment key={step.id}>
                                        <div
                                          className={`mlpStepCard kind-${step.kind}${expandedMlpStepIds.has(step.id) ? "" : " isCollapsed"}`}
                                        >
                                          <div className="componentCardHead">
                                            <div
                                              className="dragBadge"
                                              draggable
                                              onDragStart={(event) =>
                                                beginDragMlpStep(event, block.id, component.id, step.id)
                                              }
                                              onDragEnd={clearDragState}
                                              title={`Drag ${labelForMlpStepKind(step.kind)} step`}
                                              aria-hidden
                                            >
                                              <FiMove />
                                            </div>
                                            <div className="componentMeta">
                                              <span className="componentTag">{stepIndex + 1}. {labelForMlpStepKind(step.kind)}</span>
                                              <span className="componentSummary">{summarizeMlpStep(step)}</span>
                                            </div>
                                            <div className="componentHeadActions">
                                              <button
                                                type="button"
                                                className="iconButton"
                                                onClick={() => toggleExpandedMlpStep(step.id)}
                                                aria-label={`${expandedMlpStepIds.has(step.id) ? "Collapse" : "Expand"} MLP step settings`}
                                                title={expandedMlpStepIds.has(step.id) ? "Collapse settings" : "Expand settings"}
                                                aria-expanded={expandedMlpStepIds.has(step.id)}
                                              >
                                                {expandedMlpStepIds.has(step.id) ? <FiChevronDown /> : <FiChevronRight />}
                                              </button>
                                              <button
                                                type="button"
                                                className="iconButton danger"
                                                onClick={() => removeMlpStep(block.id, component.id, step.id)}
                                                aria-label="Remove MLP step"
                                                title="Remove MLP step"
                                              >
                                                <FiTrash2 />
                                              </button>
                                            </div>
                                          </div>
                                          {expandedMlpStepIds.has(step.id) ? (
                                            <div className="componentBody">
                                              {step.kind === "linear" ? (
                                                <label
                                                  className="toggleField"
                                                  htmlFor={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-linear-bias`}
                                                >
                                                  <input
                                                    id={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-linear-bias`}
                                                    type="checkbox"
                                                    checked={step.linear.bias}
                                                    onChange={(event) =>
                                                      updateMlpStep(block.id, component.id, step.id, (current) =>
                                                        current.kind !== "linear"
                                                          ? current
                                                          : {
                                                              ...current,
                                                              linear: { bias: event.target.checked },
                                                            }
                                                      )
                                                    }
                                                  />
                                                  <span>bias</span>
                                                </label>
                                              ) : null}

                                              {step.kind === "norm"
                                                ? renderNormFields(step.norm, (nextNorm) => {
                                                    updateMlpStep(
                                                      block.id,
                                                      component.id,
                                                      step.id,
                                                      (current) =>
                                                        current.kind !== "norm"
                                                          ? current
                                                          : { ...current, norm: nextNorm }
                                                    );
                                                  }, mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex))
                                                : null}

                                              {step.kind === "activation" ? (
                                                <div className="fieldGrid compact">
                                                  <label
                                                    className="fieldLabel"
                                                    htmlFor={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-activation-type`}
                                                  >
                                                    <span>Activation</span>
                                                    <select
                                                      id={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-activation-type`}
                                                      value={step.activation.type}
                                                      onChange={(event) =>
                                                        updateMlpStep(
                                                          block.id,
                                                          component.id,
                                                          step.id,
                                                          (current) =>
                                                            current.kind !== "activation"
                                                              ? current
                                                              : {
                                                                  ...current,
                                                                  activation: {
                                                                    type: event.target.value as ActivationType,
                                                                  },
                                                                }
                                                        )
                                                      }
                                                    >
                                                      {ACTIVATION_TYPES.map((type) => (
                                                        <option key={type} value={type}>
                                                          {type}
                                                        </option>
                                                      ))}
                                                    </select>
                                                  </label>
                                                </div>
                                              ) : null}
                                            </div>
                                          ) : null}
                                        </div>
                                        <DropSlot
                                          compact
                                          active={
                                            dragOverKey ===
                                            `mlp-slot-${block.id}-${component.id}-${stepIndex + 1}`
                                          }
                                          label="Insert step"
                                          onDragOver={(event) =>
                                            markDropTarget(
                                              event,
                                              `mlp-slot-${block.id}-${component.id}-${stepIndex + 1}`
                                            )
                                          }
                                          onDrop={(event) =>
                                            handleDropMlpStep(event, block.id, component.id, stepIndex + 1)
                                          }
                                        />
                                      </Fragment>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </section>
                      <DropSlot
                        compact
                        active={dragOverKey === `component-slot-${block.id}-${componentIndex + 1}`}
                        label="Insert"
                        onDragOver={(event) =>
                          markDropTarget(event, `component-slot-${block.id}-${componentIndex + 1}`)
                        }
                        onDrop={(event) =>
                          handleDropComponent(event, block.id, componentIndex + 1)
                        }
                      />
                    </Fragment>
                  ))}
                </div>
              </article>
              <DropSlot
                active={dragOverKey === `block-slot-${blockIndex + 1}`}
                label="Insert block"
                onDragOver={(event) => markDropTarget(event, `block-slot-${blockIndex + 1}`)}
                onDrop={(event) => handleDropBlock(event, blockIndex + 1)}
              />
            </Fragment>
          ))}
          </div>
        </div>
      </section>

      <section id="diagnostics" className="panelCard diagnosticsPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Diagnostics</p>
            <h2>Validation and warnings</h2>
            <p className="panelCopy">
              Local semantic checks run on every edit. Backend `{"/validate/model"}` validation is attempted when local errors are clear.
            </p>
          </div>
        </div>

        <div className="diagnosticSummaryRow">
          <div className="pillBadge tone-error">{localErrors.length} local errors</div>
          <div className="pillBadge tone-warn">{localWarnings.length} local warnings</div>
          <div className="pillBadge tone-error">{backendValidation.errors.length} backend errors</div>
          <div className="pillBadge tone-warn">{backendValidation.warnings.length} backend warnings</div>
          <div className={`pillBadge ${backendValidation.phase === "success" ? "tone-good" : backendValidation.phase === "fallback" ? "tone-warn" : "tone-neutral"}`}>
            backend: {backendValidation.phase}
          </div>
        </div>

        <div className="diagnosticList" role="list">
          {diagnostics.length === 0 ? (
            <div className="diagnosticItem tone-good" role="listitem">
              <div className="diagnosticIcon">
                <FiCheckCircle />
              </div>
              <div>
                <div className="diagnosticTitle">No local or backend warnings.</div>
                <div className="diagnosticMeta">Configuration looks ready to export.</div>
              </div>
            </div>
          ) : (
            diagnostics.map((diagnostic) => (
              <div
                key={diagnostic.id}
                className={`diagnosticItem tone-${diagnostic.level}`}
                role="listitem"
              >
                <div className="diagnosticIcon">
                  {diagnostic.level === "error" ? (
                    <FiXCircle />
                  ) : diagnostic.level === "warning" ? (
                    <FiAlertTriangle />
                  ) : (
                    <FiCheckCircle />
                  )}
                </div>
                <div>
                  <div className="diagnosticTitle">{diagnostic.message}</div>
                  <div className="diagnosticMeta">
                    <code>{diagnostic.path}</code> · {diagnostic.source}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section id="model-analysis" className="panelCard analysisPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Backend Model Analysis</p>
            <h2>Instantiate `ConfigurableGPT` and inspect runtime-facing metrics</h2>
            <p className="panelCopy">
              Runs on the backend using `{"/model/model.py"}` to confirm the config constructs a real model and to estimate parameter and KV-cache memory.
            </p>
          </div>
          <div className="actionCluster">
            <button
              type="button"
              className="buttonGhost"
              onClick={() => {
                void runBackendAnalysis();
              }}
              disabled={backendAnalysis.phase === "running" || localErrors.length > 0}
            >
              <FiServer /> {backendAnalysis.phase === "running" ? "Analyzing..." : "Run Analysis"}
            </button>
          </div>
        </div>

        <div className="diagnosticSummaryRow">
          <div
            className={`pillBadge ${
              backendAnalysis.phase === "success"
                ? "tone-good"
                : backendAnalysis.phase === "error"
                  ? "tone-error"
                  : backendAnalysis.phase === "running"
                    ? "tone-warn"
                    : "tone-neutral"
            }`}
          >
            analysis: {backendAnalysis.phase}
          </div>
          <div className="pillBadge tone-neutral">
            last run: {formatTimeAgo(backendAnalysis.lastAnalyzedAt)}
          </div>
          {backendAnalysisStale ? (
            <div className="pillBadge tone-warn">stale vs current draft</div>
          ) : null}
        </div>

        <p className="analysisMessage">{backendAnalysis.message}</p>

        {backendAnalysis.summary ? (
          <>
            <div className="statusGrid analysisStatsGrid">
              <StatusCard
                title="Parameters"
                value={formatCompactCount(backendAnalysis.summary.total_parameters)}
                detail={`${formatBytes(backendAnalysis.summary.parameter_memory_bytes_fp32)} fp32 · ${formatBytes(backendAnalysis.summary.parameter_memory_bytes_bf16)} bf16`}
                tone="good"
                icon={<FiLayers />}
              />
              <StatusCard
                title="KV Cache / Token"
                value={formatBytes(
                  backendAnalysis.summary.estimated_kv_cache_bytes_per_token_fp16
                )}
                detail={`${formatBytes(backendAnalysis.summary.estimated_kv_cache_bytes_for_context_fp16)} @ context_length`}
                tone="neutral"
                icon={<FiHardDrive />}
              />
              <StatusCard
                title="Head Dim"
                value={
                  backendAnalysis.summary.min_head_dim === null
                    ? "n/a"
                    : backendAnalysis.summary.min_head_dim ===
                        backendAnalysis.summary.max_head_dim
                      ? `${backendAnalysis.summary.min_head_dim}`
                      : `${backendAnalysis.summary.min_head_dim}-${backendAnalysis.summary.max_head_dim}`
                }
                detail={`${backendAnalysis.summary.attention_component_count} attention components`}
                tone="neutral"
                icon={<FiServer />}
              />
              <StatusCard
                title="Instantiation"
                value={`${backendAnalysis.summary.instantiation_time_ms.toFixed(1)} ms`}
                detail={`${formatCompactCount(backendAnalysis.summary.trainable_parameters)} trainable`}
                tone="neutral"
                icon={<FiRefreshCw />}
              />
            </div>

            <div className="twoColLayout analysisLayout">
              <div className="workflowItem">
                <div className="workflowTitle">Component counts</div>
                <div className="analysisChipRow">
                  <span>{backendAnalysis.summary.block_count} blocks</span>
                  <span>{backendAnalysis.summary.component_count} components</span>
                  <span>{backendAnalysis.summary.attention_component_count} attention</span>
                  <span>{backendAnalysis.summary.mlp_component_count} mlp</span>
                  <span>{backendAnalysis.summary.norm_component_count} norm</span>
                  <span>{backendAnalysis.summary.activation_component_count} activation</span>
                </div>
              </div>

              <div className="workflowItem">
                <div className="workflowTitle">Module inventory</div>
                <div className="analysisModuleList">
                  {Object.entries(backendAnalysis.summary.module_counts).map(([name, count]) => (
                    <div key={name} className="analysisModuleRow">
                      <code>{name}</code>
                      <span>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : null}

        {backendAnalysis.instantiationError ? (
          <div className="diagnosticList">
            <div className="diagnosticItem tone-error" role="listitem">
              <div className="diagnosticIcon">
                <FiXCircle />
              </div>
              <div>
                <div className="diagnosticTitle">Model instantiation failed</div>
                <div className="diagnosticMeta">{backendAnalysis.instantiationError}</div>
              </div>
            </div>
          </div>
        ) : null}
      </section>

      <div id="json-preview" className="twoColLayout previewLayout">
        <section className="panelCard previewPanel">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">JSON Preview</p>
              <h2>Exportable model config</h2>
              <p className="panelCopy">
                Live JSON mirrors the visual state. Use copy/export for downstream training or backend validation calls.
              </p>
            </div>
            <div className="actionCluster">
              <button type="button" className="buttonGhost" onClick={copyJson}>
                <FiCopy /> Copy
              </button>
              <button type="button" className="buttonGhost" onClick={exportJson}>
                <FiDownload /> Export
              </button>
            </div>
          </div>
          <pre className="jsonPreview"><code>{previewJson}</code></pre>
        </section>

        <section className="panelCard importPanel">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Import / Workflow</p>
              <h2>Paste or load JSON</h2>
              <p className="panelCopy">
                Import a `/model` JSON document to regenerate blocks and nested MLP sequences. Unsupported shapes are rejected before state changes.
              </p>
            </div>
          </div>

          <div className="actionRowWrap">
            <button type="button" className="buttonGhost" onClick={() => fileInputRef.current?.click()}>
              <FiUpload /> Choose JSON File
            </button>
            <button type="button" className="buttonGhost" onClick={() => applyImportText(importDraft)}>
              <FiRefreshCw /> Apply Import Text
            </button>
            <button
              type="button"
              className="buttonGhost"
              onClick={() => setImportDraft(JSON.stringify(modelConfig, null, 2))}
            >
              <FiCopy /> Load Current Into Editor
            </button>
          </div>

          <label className="fieldLabel" htmlFor="import-draft">
            <span>Import JSON</span>
            <textarea
              id="import-draft"
              value={importDraft}
              onChange={(event) => setImportDraft(event.target.value)}
              placeholder="Paste /model JSON here..."
              rows={16}
            />
          </label>

          <div className="workflowList">
            <div className="workflowItem">
              <div className="workflowTitle">Suggested workflow</div>
              <ol>
                <li>Set base dimensions (`n_embd`, `context_length`, `vocab_size`).</li>
                <li>Compose one reference block, then duplicate and tune variants.</li>
                <li>Resolve diagnostics, then export JSON or submit to backend validator.</li>
              </ol>
            </div>
            <div className="workflowItem">
              <div className="workflowTitle">Counts</div>
              <div className="workflowStats">
                <span>{metrics.attentionCount} attention</span>
                <span>{metrics.mlpCount} mlp</span>
                <span>{metrics.normCount} norm</span>
                <span>{metrics.activationCount} activation</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          void importFromFile(event);
        }}
      />
    </main>
  );
}
