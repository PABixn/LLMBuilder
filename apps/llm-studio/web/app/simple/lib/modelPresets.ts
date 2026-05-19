import type {
  ActivationType,
  BlockComponent,
  MlpStep,
  ModelBlock,
  ModelConfig,
  NormConfig,
} from "../../../lib/defaults";

export type SimplePresetSize = "tiny" | "small" | "medium";
export type SimplePresetSupport = "inspired" | "size-style" | "family-style" | "app-native";

export interface SimpleModelPreset {
  id: string;
  name: string;
  intent: string;
  bestUse: string;
  relativeSize: SimplePresetSize;
  defaultVocabSize: number;
  starterVocabSize: number;
  uploadVocabSize: number;
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
  honestyNote: string;
  hardwareWarning: string | null;
  support: SimplePresetSupport;
}

export interface BuildPresetModelConfigOptions {
  vocabSize?: number;
  contextLength?: number;
}

const PRESETS: SimpleModelPreset[] = [
  {
    id: "nano-gpt-quick",
    name: "NanoGPT-style quick model",
    intent: "Fast local smoke tests and tiny datasets.",
    bestUse: "Verify the full workflow quickly on CPU or a small GPU.",
    relativeSize: "tiny",
    defaultVocabSize: 1000,
    starterVocabSize: 1000,
    uploadVocabSize: 8000,
    defaultContextLength: 512,
    contextLengthOptions: [256, 512, 1024],
    blockCount: 4,
    nEmbeddings: 256,
    nHead: 4,
    nKvHead: 4,
    norm: "layernorm",
    activation: "gelu",
    mlpMultiplier: 4,
    weightTying: true,
    headLayout: "4 attention heads, dense KV",
    normActivationLabel: "LayerNorm + GELU",
    honestyNote:
      "Inspired by small decoder-only GPT examples; not a pretrained nanoGPT checkpoint.",
    hardwareWarning: null,
    support: "inspired",
  },
  {
    id: "gpt2-small-style",
    name: "GPT-2 Small size baseline",
    intent: "A familiar dense transformer baseline.",
    bestUse: "Compare against a well-known dense size class.",
    relativeSize: "medium",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [512, 1024],
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
    honestyNote:
      "Size/layout style only; this runtime uses this app's positional and norm implementation.",
    hardwareWarning: "Likely slow on CPU and may need a GPU for useful runs.",
    support: "size-style",
  },
  {
    id: "pythia-160m-style",
    name: "Pythia/GPT-NeoX-style baseline",
    intent: "Efficient research-style dense baseline.",
    bestUse: "Run a modern-style dense decoder with a longer context.",
    relativeSize: "medium",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 32000,
    defaultContextLength: 2048,
    contextLengthOptions: [1024, 2048],
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
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 32000,
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
    honestyNote:
      "LLaMA-like because this runtime supports RoPE, RMSNorm, GQA, and SiLU; not exact LLaMA until gated MLP support exists.",
    hardwareWarning: "Expect GPU training; local CPU runs are mainly for compatibility checks.",
    support: "family-style",
  },
  {
    id: "gqa-balanced",
    name: "Efficient GQA balanced model",
    intent: "A balanced app-native model for GPU users.",
    bestUse: "Default choice when a GPU can fit a moderate local run.",
    relativeSize: "medium",
    defaultVocabSize: 32000,
    starterVocabSize: 1000,
    uploadVocabSize: 32000,
    defaultContextLength: 1024,
    contextLengthOptions: [1024, 2048],
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
    honestyNote: "App-native efficient template rather than a named external architecture.",
    hardwareWarning: "Recommended for GPU-backed runs, not long CPU experiments.",
    support: "app-native",
  },
];

export const SIMPLE_MODEL_PRESETS: readonly SimpleModelPreset[] = PRESETS;

export function getSimpleModelPreset(presetId: string): SimpleModelPreset {
  return PRESETS.find((preset) => preset.id === presetId) ?? PRESETS[0];
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
  return preset.uploadVocabSize;
}
