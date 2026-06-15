import type {
  ActivationType,
  BlockComponent,
  MlpStep,
  ModelBlock,
  ModelConfig,
  NormConfig,
} from "../../../lib/defaults";
import type {
  SimpleDatasetSource,
  SimpleExecutionKind,
  SimpleTrainingProfile,
} from "../types";

export type SimplePresetSize = "micro" | "tiny" | "small" | "medium" | "large";
export type SimplePresetArchitectureType = "dense" | "gqa" | "mqa";
export type SimplePresetTrainingTarget = "local" | "runpod";
export type SimplePresetSupport = "inspired" | "size-style" | "family-style" | "app-native";

export interface SimplePresetSizeGroup {
  id: SimplePresetSize;
  label: string;
  description: string;
  target: string;
}

export interface SimplePresetArchitectureTypeOption {
  id: SimplePresetArchitectureType;
  label: string;
  description: string;
}

export interface SimplePresetTrainingTargetOption {
  id: SimplePresetTrainingTarget;
  label: string;
  description: string;
}

export interface SimpleModelPreset {
  id: string;
  name: string;
  intent: string;
  bestUse: string;
  relativeSize: SimplePresetSize;
  architectureType: SimplePresetArchitectureType;
  trainingTarget: SimplePresetTrainingTarget;
  defaultDatasetSource: SimpleDatasetSource;
  defaultTrainingProfile: SimpleTrainingProfile;
  defaultExecutionKind: SimpleExecutionKind;
  defaultVocabSize: number;
  starterVocabSize: number;
  uploadVocabSize: number;
  streamingVocabSize: number;
  defaultContextLength: number;
  contextLengthOptions: number[];
  blockCount: number;
  nEmbeddings: number;
  nHead: number;
  nKvHead: number;
  norm: "layernorm" | "rmsnorm";
  activation: Extract<ActivationType, "gelu" | "silu" | "relu">;
  mlpMultiplier: number;
  weightTying: boolean;
  headLayout: string;
  normActivationLabel: string;
  hardwareTier: string;
  honestyNote: string;
  hardwareWarning: string | null;
  support: SimplePresetSupport;
}

export interface BuildPresetModelConfigOptions {
  vocabSize?: number;
  contextLength?: number;
}

export const SIMPLE_PRESET_SIZE_GROUPS: readonly SimplePresetSizeGroup[] = [
  {
    id: "micro",
    label: "Micro",
    description: "Smoke tests and CPU-friendly local experiments.",
    target: "1M-15M params",
  },
  {
    id: "tiny",
    label: "Tiny",
    description: "Fast local GPU or high-end CPU iteration.",
    target: "15M-45M params",
  },
  {
    id: "small",
    label: "Small",
    description: "Local GPU training with practical toy corpora.",
    target: "45M-100M params",
  },
  {
    id: "medium",
    label: "Medium",
    description: "RunPod-first templates for real GPU-backed runs.",
    target: "100M-250M params",
  },
  {
    id: "large",
    label: "Large",
    description: "RunPod-only starts for ambitious from-scratch training.",
    target: "250M+ params",
  },
];

export const SIMPLE_PRESET_ARCHITECTURE_TYPES: readonly SimplePresetArchitectureTypeOption[] = [
  {
    id: "dense",
    label: "Dense",
    description: "Full key/value heads. Familiar baseline, more KV memory.",
  },
  {
    id: "gqa",
    label: "GQA",
    description: "Grouped query attention. Strong default for GPU efficiency.",
  },
  {
    id: "mqa",
    label: "MQA",
    description: "Single KV head. Memory-saving layout for longer contexts.",
  },
];

export const SIMPLE_PRESET_TRAINING_TARGETS: readonly SimplePresetTrainingTargetOption[] = [
  {
    id: "local",
    label: "Local-ready",
    description: "Designed to train on this machine or a local GPU.",
  },
  {
    id: "runpod",
    label: "RunPod recommended",
    description: "Defaults to cloud GPU training and skips local model instantiation in the picker.",
  },
];

export const SIMPLE_PRESET_BACKEND_ANALYSIS_PARAMETER_LIMIT = 100_000_000;

const PRESETS: SimpleModelPreset[] = [
  {
    id: "nano-gpt-quick",
    name: "Local quickstart GQA",
    intent: "Small decoder-only transformer with efficient modern defaults.",
    bestUse: "First local run, tiny corpora, and full workflow checks.",
    relativeSize: "micro",
    architectureType: "gqa",
    trainingTarget: "local",
    defaultDatasetSource: "starter",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 1000,
    starterVocabSize: 1000,
    uploadVocabSize: 8000,
    streamingVocabSize: 16000,
    defaultContextLength: 512,
    contextLengthOptions: [256, 512, 1024],
    blockCount: 4,
    nEmbeddings: 256,
    nHead: 4,
    nKvHead: 2,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "4 query heads, 2 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "CPU or entry local GPU",
    honestyNote:
      "App-native quickstart template. It uses decoder-only transformer defaults, not a pretrained checkpoint.",
    hardwareWarning: null,
    support: "app-native",
  },
  {
    id: "micro-cpu-dense",
    name: "Micro dense CPU baseline",
    intent: "A compact dense transformer for comparing against GQA templates.",
    bestUse: "CPU-safe architecture validation and small local text runs.",
    relativeSize: "micro",
    architectureType: "dense",
    trainingTarget: "local",
    defaultDatasetSource: "starter",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 8000,
    starterVocabSize: 1000,
    uploadVocabSize: 8000,
    streamingVocabSize: 16000,
    defaultContextLength: 512,
    contextLengthOptions: [256, 512, 1024],
    blockCount: 6,
    nEmbeddings: 256,
    nHead: 4,
    nKvHead: 4,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "4 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "CPU or entry local GPU",
    honestyNote: "Dense app-native baseline for small-model comparisons.",
    hardwareWarning: null,
    support: "app-native",
  },
  {
    id: "micro-mqa-scout",
    name: "Micro MQA scout",
    intent: "A small single-KV-head model for testing memory-saving attention.",
    bestUse: "Longer-context local experiments where KV cache size matters.",
    relativeSize: "micro",
    architectureType: "mqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 12000,
    starterVocabSize: 1000,
    uploadVocabSize: 12000,
    streamingVocabSize: 16000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 6,
    nEmbeddings: 320,
    nHead: 5,
    nKvHead: 1,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "5 query heads, 1 KV head",
    normActivationLabel: "RMSNorm + SiLU + MQA",
    hardwareTier: "CPU or entry local GPU",
    honestyNote: "App-native MQA template; useful for measuring local KV-cache behavior.",
    hardwareWarning: null,
    support: "app-native",
  },
  {
    id: "micro-wide-silu",
    name: "Micro wide SiLU GQA",
    intent: "Fewer blocks with a wider hidden state for quick quality checks.",
    bestUse: "Testing width-vs-depth tradeoffs on small local datasets.",
    relativeSize: "micro",
    architectureType: "gqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 12000,
    starterVocabSize: 1000,
    uploadVocabSize: 12000,
    streamingVocabSize: 16000,
    defaultContextLength: 512,
    contextLengthOptions: [512, 1024],
    blockCount: 4,
    nEmbeddings: 384,
    nHead: 6,
    nKvHead: 3,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "6 query heads, 3 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "CPU or entry local GPU",
    honestyNote: "App-native width-biased template for small local sweeps.",
    hardwareWarning: null,
    support: "app-native",
  },
  {
    id: "tiny-local-dense",
    name: "Tiny dense local",
    intent: "A straightforward dense tiny model with a familiar block layout.",
    bestUse: "Local GPU experiments and apples-to-apples dense comparisons.",
    relativeSize: "tiny",
    architectureType: "dense",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 16000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 24000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 8,
    nEmbeddings: 384,
    nHead: 6,
    nKvHead: 6,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "6 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "Local GPU preferred",
    honestyNote: "Size/style baseline with this app's RoPE attention implementation.",
    hardwareWarning: "CPU training is possible but slow beyond short checks.",
    support: "size-style",
  },
  {
    id: "tiny-local-gqa",
    name: "Tiny local GQA",
    intent: "A compact GQA model with better KV efficiency than dense tiny baselines.",
    bestUse: "General local training when you want a small but capable template.",
    relativeSize: "tiny",
    architectureType: "gqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "quick",
    defaultExecutionKind: "local",
    defaultVocabSize: 16000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 24000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 8,
    nEmbeddings: 384,
    nHead: 6,
    nKvHead: 2,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "6 query heads, 2 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "Local GPU preferred",
    honestyNote: "App-native tiny GQA template for local iteration.",
    hardwareWarning: "Local CPU runs should stay short; use a local GPU for useful training.",
    support: "app-native",
  },
  {
    id: "tiny-mqa-2k",
    name: "Tiny MQA 2K context",
    intent: "Single-KV-head tiny model tuned for inexpensive longer-context tests.",
    bestUse: "Local context-length checks before moving to cloud-sized models.",
    relativeSize: "tiny",
    architectureType: "mqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "local",
    defaultVocabSize: 16000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 24000,
    defaultContextLength: 2048,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 10,
    nEmbeddings: 512,
    nHead: 8,
    nKvHead: 1,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "8 query heads, 1 KV head",
    normActivationLabel: "RMSNorm + SiLU + MQA",
    hardwareTier: "Local GPU preferred",
    honestyNote: "App-native MQA layout; not a named external checkpoint.",
    hardwareWarning: "Prefer a local GPU for the default 2K context length.",
    support: "app-native",
  },
  {
    id: "tiny-neo-style",
    name: "Tiny GPT-Neo style",
    intent: "A research-style dense decoder with GELU and LayerNorm.",
    bestUse: "Dense research baselines that are still feasible locally.",
    relativeSize: "tiny",
    architectureType: "dense",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "local",
    defaultVocabSize: 16000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 24000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 10,
    nEmbeddings: 512,
    nHead: 8,
    nKvHead: 8,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "8 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "Local GPU preferred",
    honestyNote: "Family-style GPT-Neo-like baseline, adapted to this app runtime.",
    hardwareWarning: "CPU runs are mainly useful for smoke tests.",
    support: "family-style",
  },
  {
    id: "small-local-gqa",
    name: "Small local GQA",
    intent: "A stronger local template with grouped-query attention.",
    bestUse: "Local GPU training on uploaded corpora.",
    relativeSize: "small",
    architectureType: "gqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "local",
    defaultVocabSize: 24000,
    starterVocabSize: 1000,
    uploadVocabSize: 24000,
    streamingVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 10,
    nEmbeddings: 512,
    nHead: 8,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "8 query heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "Local GPU",
    honestyNote: "App-native local GPU template with modern attention defaults.",
    hardwareWarning: "Recommended for a local GPU; CPU training will be very slow.",
    support: "app-native",
  },
  {
    id: "small-dense-60m",
    name: "Small dense baseline",
    intent: "A dense small model for controlled comparisons against GQA.",
    bestUse: "Local GPU baselines where dense KV heads are acceptable.",
    relativeSize: "small",
    architectureType: "dense",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "local",
    defaultVocabSize: 24000,
    starterVocabSize: 1000,
    uploadVocabSize: 24000,
    streamingVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 12,
    nEmbeddings: 512,
    nHead: 8,
    nKvHead: 8,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "8 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "Local GPU",
    honestyNote: "Size/style dense baseline, not tied to an external checkpoint.",
    hardwareWarning: "Use a local GPU for anything beyond validation or short runs.",
    support: "size-style",
  },
  {
    id: "small-context-gqa-2k",
    name: "Small GQA 2K context",
    intent: "A small model with a wider hidden state and 2K default context.",
    bestUse: "Local GPU runs that need more context before scaling up.",
    relativeSize: "small",
    architectureType: "gqa",
    trainingTarget: "local",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "local",
    defaultVocabSize: 24000,
    starterVocabSize: 1000,
    uploadVocabSize: 24000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 12,
    nEmbeddings: 640,
    nHead: 10,
    nKvHead: 5,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "10 query heads, 5 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "Local GPU",
    honestyNote: "App-native small template with a context-forward shape.",
    hardwareWarning: "A local GPU is strongly recommended at the default context length.",
    support: "app-native",
  },
  {
    id: "small-runpod-mqa-stream",
    name: "Small MQA streaming seed",
    intent: "Small enough to reason about, but defaults to cloud for streaming datasets.",
    bestUse: "First RunPod job with streaming data and low KV memory.",
    relativeSize: "small",
    architectureType: "mqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 24000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 14,
    nEmbeddings: 640,
    nHead: 10,
    nKvHead: 1,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "10 query heads, 1 KV head",
    normActivationLabel: "RMSNorm + SiLU + MQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "Cloud-first app-native MQA seed for streaming data tests.",
    hardwareWarning: "Defaults to RunPod because streaming data and 2K context make local runs impractical.",
    support: "app-native",
  },
  {
    id: "gpt2-small-style",
    name: "GPT-2 Small size baseline",
    intent: "A familiar dense transformer baseline.",
    bestUse: "Compare against a well-known dense GPT size class.",
    relativeSize: "medium",
    architectureType: "dense",
    trainingTarget: "runpod",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 12,
    nEmbeddings: 768,
    nHead: 12,
    nKvHead: 12,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "12 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "RunPod GPU",
    honestyNote:
      "Size/layout style only; this runtime uses this app's positional and norm implementation.",
    hardwareWarning: "RunPod is recommended for useful training; local use is mainly for GPU workstations.",
    support: "size-style",
  },
  {
    id: "pythia-160m-style",
    name: "Pythia/GPT-NeoX-style baseline",
    intent: "Efficient research-style dense baseline.",
    bestUse: "Run a modern-style dense decoder with a longer context.",
    relativeSize: "medium",
    architectureType: "dense",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 12,
    nEmbeddings: 768,
    nHead: 12,
    nKvHead: 12,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "12 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "RunPod GPU",
    honestyNote: "Family-style preset, not exact Pythia architecture.",
    hardwareWarning: "Use a GPU for practical training at the default context length.",
    support: "family-style",
  },
  {
    id: "llama-tiny-gqa",
    name: "LLaMA-family tiny GQA model",
    intent: "A compact grouped-query run for local GPUs.",
    bestUse: "Try an efficient LLaMA-like layout when GQA matters.",
    relativeSize: "medium",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048],
    blockCount: 16,
    nEmbeddings: 1024,
    nHead: 16,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "16 attention heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU",
    hardwareTier: "RunPod GPU",
    honestyNote:
      "LLaMA-like because this runtime supports RoPE, RMSNorm, GQA, and SiLU; not exact LLaMA until gated MLP support exists.",
    hardwareWarning: "Expect GPU training; local CPU runs are mainly for compatibility checks.",
    support: "family-style",
  },
  {
    id: "gqa-balanced",
    name: "Efficient GQA balanced model",
    intent: "A balanced app-native model for GPU users.",
    bestUse: "Default choice for a moderate GPU-backed run.",
    relativeSize: "medium",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "upload",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024, 2048],
    blockCount: 12,
    nEmbeddings: 768,
    nHead: 12,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "12 attention heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native efficient template rather than a named external architecture.",
    hardwareWarning: "Recommended for GPU-backed runs, not long CPU experiments.",
    support: "app-native",
  },
  {
    id: "medium-mqa-efficient",
    name: "Medium MQA efficient",
    intent: "A medium model that trades KV capacity for lower cache cost.",
    bestUse: "RunPod training when longer contexts matter more than dense KV capacity.",
    relativeSize: "medium",
    architectureType: "mqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "balanced",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 16,
    nEmbeddings: 896,
    nHead: 14,
    nKvHead: 1,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "14 query heads, 1 KV head",
    normActivationLabel: "RMSNorm + SiLU + MQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native MQA preset for memory-conscious cloud experiments.",
    hardwareWarning: "RunPod is recommended; backend picker analysis uses estimates to avoid local memory pressure.",
    support: "app-native",
  },
  {
    id: "medium-long-context-gqa",
    name: "Medium GQA 4K context",
    intent: "A moderate GQA shape with a 4K default context.",
    bestUse: "Cloud runs for longer documents and streaming corpora.",
    relativeSize: "medium",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 4096,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 14,
    nEmbeddings: 768,
    nHead: 12,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "12 query heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native long-context template within the current block capabilities.",
    hardwareWarning: "Use RunPod for practical training at 4K context.",
    support: "app-native",
  },
  {
    id: "runpod-gqa-250m",
    name: "RunPod GQA 250M class",
    intent: "A larger GQA template for serious cloud-backed experiments.",
    bestUse: "RunPod jobs where a compact but capable architecture is the goal.",
    relativeSize: "large",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 20,
    nEmbeddings: 1024,
    nHead: 16,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "16 query heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native large GQA preset; not a pretrained model family clone.",
    hardwareWarning: "RunPod required for practical training. Keep local analysis disabled for this class.",
    support: "app-native",
  },
  {
    id: "runpod-dense-300m",
    name: "RunPod dense 300M class",
    intent: "A larger dense baseline for measuring the cost of full KV heads.",
    bestUse: "Cloud baseline runs against similarly sized GQA/MQA templates.",
    relativeSize: "large",
    architectureType: "dense",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048, 4096],
    blockCount: 20,
    nEmbeddings: 1024,
    nHead: 16,
    nKvHead: 16,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "16 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    hardwareTier: "RunPod GPU",
    honestyNote: "Size/style dense baseline for cloud comparisons.",
    hardwareWarning: "RunPod required; dense KV heads make local experiments expensive.",
    support: "size-style",
  },
  {
    id: "runpod-long-context-gqa",
    name: "RunPod GQA 4K context",
    intent: "A large GQA shape biased toward longer sequences.",
    bestUse: "Cloud training on document-style corpora at 4K context.",
    relativeSize: "large",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 4096,
    contextLengthOptions: [2048, 4096],
    blockCount: 18,
    nEmbeddings: 1024,
    nHead: 16,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "16 query heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native long-context large preset using supported GQA blocks.",
    hardwareWarning: "RunPod required for 4K context and this parameter class.",
    support: "app-native",
  },
  {
    id: "runpod-wide-480m",
    name: "RunPod wide GQA 480M class",
    intent: "A wider cloud-first GQA model with strong hidden capacity.",
    bestUse: "Ambitious RunPod experiments after the medium presets are stable.",
    relativeSize: "large",
    architectureType: "gqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [2048, 4096],
    blockCount: 24,
    nEmbeddings: 1280,
    nHead: 20,
    nKvHead: 4,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "20 query heads, 4 KV heads",
    normActivationLabel: "RMSNorm + SiLU + GQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "App-native wide GQA template for cloud-only training.",
    hardwareWarning: "RunPod required; this template intentionally avoids local picker instantiation.",
    support: "app-native",
  },
  {
    id: "runpod-mqa-memory-saver",
    name: "RunPod MQA memory saver",
    intent: "A large MQA model that preserves width while reducing KV cache pressure.",
    bestUse: "Cloud runs where sequence length and memory headroom are the priority.",
    relativeSize: "large",
    architectureType: "mqa",
    trainingTarget: "runpod",
    defaultDatasetSource: "streaming",
    defaultTrainingProfile: "longer",
    defaultExecutionKind: "runpod_pod",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 16000,
    streamingVocabSize: 32000,
    defaultContextLength: 4096,
    contextLengthOptions: [2048, 4096],
    blockCount: 24,
    nEmbeddings: 1152,
    nHead: 18,
    nKvHead: 1,
    norm: "rmsnorm",
    activation: "silu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "18 query heads, 1 KV head",
    normActivationLabel: "RMSNorm + SiLU + MQA",
    hardwareTier: "RunPod GPU",
    honestyNote: "Cloud-first MQA template using supported app-native primitives.",
    hardwareWarning: "RunPod required; local execution is not recommended for this class.",
    support: "app-native",
  },
];

export const SIMPLE_MODEL_PRESETS: readonly SimpleModelPreset[] = PRESETS;

export function getSimpleModelPreset(presetId: string): SimpleModelPreset {
  return PRESETS.find((preset) => preset.id === presetId) ?? PRESETS[0];
}

export function isSimpleModelPresetId(value: unknown): value is string {
  return typeof value === "string" && PRESETS.some((preset) => preset.id === value);
}

function createNorm(kind: SimpleModelPreset["norm"]): NormConfig {
  return kind === "rmsnorm"
    ? { type: "rmsnorm", learnable_gamma: true }
    : { type: "layernorm" };
}

function createMlpSequence(activation: SimpleModelPreset["activation"]): MlpStep[] {
  return [
    { linear: { bias: true } },
    { activation: { type: activation } },
    { linear: { bias: true } },
  ];
}

function createPresetBlock(preset: SimpleModelPreset): ModelBlock {
  return {
    components: [
      { norm: createNorm(preset.norm) },
      { attention: { n_head: preset.nHead, n_kv_head: preset.nKvHead } },
      { norm: createNorm(preset.norm) },
      {
        mlp: {
          multiplier: preset.mlpMultiplier,
          sequence: createMlpSequence(preset.activation),
        },
      },
    ],
  };
}

export function buildPresetModelConfig(
  presetId: string,
  options: BuildPresetModelConfigOptions = {}
): ModelConfig {
  const preset = getSimpleModelPreset(presetId);
  const config: ModelConfig = {
    context_length: options.contextLength ?? preset.defaultContextLength,
    vocab_size: options.vocabSize ?? preset.defaultVocabSize,
    n_embd: preset.nEmbeddings,
    weight_tying: preset.weightTying,
    blocks: Array.from({ length: preset.blockCount }, () => createPresetBlock(preset)),
  };

  assertPresetModelConfig(config);
  return config;
}

function normParameterCount(kind: SimpleModelPreset["norm"], width: number): number {
  return kind === "layernorm" ? width * 2 : width;
}

export function estimatePresetParameterCount(
  presetOrId: SimpleModelPreset | string,
  options: BuildPresetModelConfigOptions = {}
): number {
  const preset =
    typeof presetOrId === "string" ? getSimpleModelPreset(presetOrId) : presetOrId;
  const vocabSize = options.vocabSize ?? preset.defaultVocabSize;
  const width = preset.nEmbeddings;
  const headDim = width / preset.nHead;
  const kvWidth = preset.nKvHead * headDim;
  const mlpHidden = width * preset.mlpMultiplier;

  const embeddingParameters = vocabSize * width;
  const outputParameters = preset.weightTying ? 0 : vocabSize * width;
  const modelNormParameters = width * 2;
  const blockNormParameters = normParameterCount(preset.norm, width) * 2;
  const attentionParameters =
    width * width + width * kvWidth * 2 + width * width + headDim * 2;
  const mlpParameters = width * mlpHidden + mlpHidden + mlpHidden * width + width;

  return Math.round(
    embeddingParameters +
      outputParameters +
      modelNormParameters +
      preset.blockCount * (blockNormParameters + attentionParameters + mlpParameters)
  );
}

export function estimatePresetBf16MemoryBytes(
  presetOrId: SimpleModelPreset | string,
  options: BuildPresetModelConfigOptions = {}
): number {
  return estimatePresetParameterCount(presetOrId, options) * 2;
}

export function shouldAnalyzePresetWithBackend(
  presetOrId: SimpleModelPreset | string,
  options: BuildPresetModelConfigOptions = {}
): boolean {
  const preset =
    typeof presetOrId === "string" ? getSimpleModelPreset(presetOrId) : presetOrId;
  if (preset.trainingTarget !== "local") {
    return false;
  }
  return (
    estimatePresetParameterCount(preset, options) <=
    SIMPLE_PRESET_BACKEND_ANALYSIS_PARAMETER_LIMIT
  );
}

export function backendAnalysisSkipReason(
  presetOrId: SimpleModelPreset | string,
  options: BuildPresetModelConfigOptions = {}
): string | null {
  const preset =
    typeof presetOrId === "string" ? getSimpleModelPreset(presetOrId) : presetOrId;
  if (shouldAnalyzePresetWithBackend(preset, options)) {
    return null;
  }
  if (preset.trainingTarget === "runpod") {
    return "Backend analysis is estimated for RunPod templates to avoid local memory pressure.";
  }
  return "Backend analysis is estimated because this preset is above the local analysis limit.";
}

function hasAttention(
  component: BlockComponent
): component is Extract<BlockComponent, { attention: unknown }> {
  return "attention" in component;
}

function hasMlp(component: BlockComponent): component is Extract<BlockComponent, { mlp: unknown }> {
  return "mlp" in component;
}

export function assertPresetModelConfig(config: ModelConfig): void {
  if (!Number.isInteger(config.n_embd) || config.n_embd <= 0) {
    throw new Error("Embedding size must be a positive integer.");
  }

  for (const [index, block] of config.blocks.entries()) {
    const attention = block.components.find(hasAttention)?.attention;
    const mlp = block.components.find(hasMlp)?.mlp;
    if (!attention) {
      throw new Error(`Preset block ${index + 1} is missing attention.`);
    }
    if (!mlp) {
      throw new Error(`Preset block ${index + 1} is missing an MLP.`);
    }
    if (config.n_embd % attention.n_head !== 0) {
      throw new Error(`Preset block ${index + 1} has an invalid head layout.`);
    }
    if ((config.n_embd / attention.n_head) % 2 !== 0) {
      throw new Error(`Preset block ${index + 1} has an odd RoPE head dimension.`);
    }
    if (attention.n_kv_head > attention.n_head) {
      throw new Error(`Preset block ${index + 1} has too many KV heads.`);
    }
    if (attention.n_head % attention.n_kv_head !== 0) {
      throw new Error(`Preset block ${index + 1} has non-divisible KV heads.`);
    }
    if (mlp.multiplier !== 1) {
      const firstStep = mlp.sequence[0];
      const lastStep = mlp.sequence[mlp.sequence.length - 1];
      if (!firstStep || !("linear" in firstStep) || !lastStep || !("linear" in lastStep)) {
        throw new Error(`Preset block ${index + 1} MLP must start and end with linear steps.`);
      }
    }
  }
}

export function targetVocabForPresetDataset(
  presetId: string,
  datasetSource: "starter" | "upload" | "streaming"
): number {
  const preset = getSimpleModelPreset(presetId);
  if (datasetSource === "starter") {
    return preset.starterVocabSize;
  }
  if (datasetSource === "streaming") {
    return preset.streamingVocabSize;
  }
  return preset.uploadVocabSize;
}
