import type { ModelAnalysisSummary, ValidationIssue as ApiValidationIssue } from "../../lib/api";
import type {
  ActivationType,
  ActivationConfig,
  AttentionConfig,
  LinearConfig,
  ModelConfig,
  NormConfig,
} from "../../lib/defaults";

export type ThemeMode = "white" | "dark";
export type StudioComponentKind = "attention" | "mlp" | "norm" | "activation";
export type MlpStepKind = "linear" | "norm" | "activation";
export type BlockInsertPreset = "default" | "empty";
export type DiagnosticLevel = "error" | "warning" | "info";
export type DiagnosticSource = "local" | "backend";
export type NoticeTone = "info" | "success" | "error";
export type BackendValidationPhase =
  | "idle"
  | "skipped"
  | "validating"
  | "success"
  | "fallback";
export type BackendAnalysisPhase = "idle" | "running" | "success" | "error";

export const THEME_STORAGE_KEY = "llm-studio-theme";
export const DOCUMENT_STORAGE_KEY = "llm-studio-document";
export const IMPORT_DRAFT_STORAGE_KEY = "llm-studio-import-draft";
export const VALIDATION_DEBOUNCE_MS = 420;
export const DND_MIME = "application/x-llm-studio-dnd";

export type StudioMlpStep =
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

export type StudioComponent =
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

export interface StudioBlock {
  id: string;
  components: StudioComponent[];
}

export interface StudioDocument {
  context_length: number;
  vocab_size: number;
  n_embd: number;
  weight_tying: boolean;
  blocks: StudioBlock[];
}

export interface Diagnostic {
  id: string;
  level: DiagnosticLevel;
  source: DiagnosticSource;
  path: string;
  message: string;
}

export type SuggestionPriority = "critical" | "high" | "medium" | "low";
export type SuggestionCategory =
  | "correctness"
  | "architecture"
  | "efficiency"
  | "consistency"
  | "workflow";
export type SuggestionSource = "local" | "backend" | "analysis" | "combined";
export type SuggestionApplyAction =
  | { kind: "run_backend_analysis" }
  | { kind: "set_all_mlp_activations"; activation: ActivationType }
  | { kind: "set_all_norm_family"; normType: NormConfig["type"]; learnableGamma?: boolean }
  | { kind: "set_all_mlp_multipliers"; multiplier: number }
  | { kind: "set_all_attention_kv_heads"; strategy: "half" | "one" };

export interface SuggestionApplyOption {
  id: string;
  label: string;
  action: SuggestionApplyAction;
}

export interface DesignSuggestion {
  id: string;
  priority: SuggestionPriority;
  category: SuggestionCategory;
  source: SuggestionSource;
  title: string;
  summary: string;
  action: string;
  path: string | null;
  applyOptions: SuggestionApplyOption[];
  score: number;
}

export interface BackendValidationState {
  phase: BackendValidationPhase;
  message: string;
  lastValidatedAt: number | null;
  warnings: ApiValidationIssue[];
  errors: ApiValidationIssue[];
  normalizedChanged: boolean;
}

export interface BackendAnalysisState {
  phase: BackendAnalysisPhase;
  message: string;
  lastAnalyzedAt: number | null;
  configSignature: string | null;
  summary: ModelAnalysisSummary | null;
  warnings: ApiValidationIssue[];
  errors: ApiValidationIssue[];
  instantiationError: string | null;
}

export interface NoticeState {
  tone: NoticeTone;
  message: string;
  at: number;
}

export interface BuilderMetrics {
  blockCount: number;
  componentCount: number;
  attentionCount: number;
  mlpCount: number;
  normCount: number;
  activationCount: number;
  mlpStepCount: number;
  mlpActivationStepCount: number;
}

export interface ConsecutiveBlockGroup {
  key: string;
  startIndex: number;
  endIndex: number;
  count: number;
}

export type DragPayload =
  | {
      kind: "palette-component";
      componentKind: StudioComponentKind;
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

export type ImportedModelConfigParseResult = {
  config: ModelConfig | null;
  errors: string[];
};

export type MlpComponentRef = {
  blockIndex: number;
  componentIndex: number;
  component: Extract<StudioComponent, { kind: "mlp" }>;
};

export type StudioDocumentNumericField = keyof Pick<
  StudioDocument,
  "context_length" | "vocab_size" | "n_embd"
>;

export type BaseIntegerModelField = keyof Pick<ModelConfig, "context_length" | "vocab_size" | "n_embd">;
