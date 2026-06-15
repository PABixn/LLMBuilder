import {
  estimatePresetParameterCount,
  type SimpleModelPreset,
  type SimplePresetArchitectureType,
  type SimplePresetSize,
  type SimplePresetTrainingTarget,
} from "./modelPresets";

export interface ArchitectureTemplateGuidance {
  title: string;
  summary: string;
  highlights: string[];
}

interface SizeGuidance {
  label: string;
  expectation: string;
}

interface ArchitectureGuidance {
  label: string;
  tradeoff: string;
}

interface TrainingTargetGuidance {
  label: string;
  nextStep: string;
}

const SIZE_GUIDANCE: Record<SimplePresetSize, SizeGuidance> = {
  micro: {
    label: "micro",
    expectation:
      "can imitate short patterns, names, and formatting; expect fragile toy output",
  },
  tiny: {
    label: "tiny",
    expectation:
      "can learn a narrow voice or simple syntax from clean examples; demo quality",
  },
  small: {
    label: "small",
    expectation:
      "can become useful for narrow autocomplete, style imitation, or domain phrasing",
  },
  medium: {
    label: "medium",
    expectation:
      "can learn broader document style and domain vocabulary with a large clean corpus",
  },
  large: {
    label: "large",
    expectation:
      "has the best ceiling here, but still needs serious data and will not reason like frontier models",
  },
};

const ARCHITECTURE_GUIDANCE: Record<SimplePresetArchitectureType, ArchitectureGuidance> = {
  dense: {
    label: "Dense",
    tradeoff:
      "plain full-attention baseline; easiest to compare, but costs more memory at long context",
  },
  gqa: {
    label: "GQA",
    tradeoff:
      "balanced modern attention; usually the safest first GPU choice for speed, memory, and quality",
  },
  mqa: {
    label: "MQA",
    tradeoff:
      "memory-saving attention for longer prompts; useful when fitting context matters most",
  },
};

const TRAINING_TARGET_GUIDANCE: Record<SimplePresetTrainingTarget, TrainingTargetGuidance> = {
  local: {
    label: "local runs",
    nextStep: "run Quick check first, then train on clean uploaded text",
  },
  runpod: {
    label: "RunPod",
    nextStep: "run preflight before spending GPU time",
  },
};

const DATASET_ADVICE: Record<SimpleModelPreset["defaultDatasetSource"], string> = {
  starter: "Starter data only proves the pipeline; upload real examples for useful behavior.",
  upload: "Clean uploaded text matters more than tiny setting changes.",
  streaming: "Streaming helps scale, but tokenization and training take longer.",
};

const PRACTICAL_USE_BY_PRESET_ID: Record<string, string> = {
  "nano-gpt-quick":
    "checking that model creation, tokenizer training, model training, and generation all work",
  "micro-cpu-dense":
    "CPU-safe smoke tests and dense-vs-GQA comparisons before you scale up",
  "micro-mqa-scout":
    "testing longer prompts or memory-saving attention without paying for a bigger run",
  "micro-wide-silu":
    "quick local text experiments where you want a little more capacity than the smallest micro model",
  "tiny-local-dense":
    "a simple local-GPU baseline for notes, snippets, or small writing-style experiments",
  "tiny-local-gqa":
    "your first practical local-GPU run on a small uploaded corpus",
  "tiny-mqa-2k":
    "testing 2K-token prompts, outlines, or document chunks on a small local model",
  "tiny-neo-style":
    "research-style dense experiments that should resemble older GPT-Neo-like baselines",
  "small-local-gqa":
    "local-GPU autocomplete, style imitation, or domain phrasing on a focused corpus",
  "small-dense-60m":
    "measuring whether dense attention improves your task enough to justify extra memory",
  "small-context-gqa-2k":
    "local document-style training where seeing more surrounding text matters",
  "small-runpod-mqa-stream":
    "a first cloud streaming-data run with low memory pressure and modest cost",
  "gpt2-small-style":
    "cloud comparisons against a familiar GPT-2-sized dense baseline",
  "pythia-160m-style":
    "modern research-baseline experiments on public or streaming text",
  "llama-tiny-gqa":
    "trying a LLaMA-like attention and normalization shape without expecting real LLaMA behavior",
  "gqa-balanced":
    "a general cloud-GPU starting point for useful narrow-domain experiments",
  "medium-mqa-efficient":
    "longer-context cloud runs where memory headroom matters more than dense attention",
  "medium-long-context-gqa":
    "training on longer documents, transcripts, or code-like chunks with 4K context",
  "runpod-gqa-250m":
    "serious narrow-domain cloud experiments after smaller presets look promising",
  "runpod-dense-300m":
    "a larger dense baseline to compare quality against GQA or MQA runs",
  "runpod-long-context-gqa":
    "cloud training on long documents where context continuity matters",
  "runpod-wide-480m":
    "ambitious cloud experiments that need stronger hidden capacity and plenty of data",
  "runpod-mqa-memory-saver":
    "large long-context cloud training when fitting memory is the main constraint",
};

export function buildArchitectureTemplateGuidance(
  preset: SimpleModelPreset,
  parameterCount = estimatePresetParameterCount(preset)
): ArchitectureTemplateGuidance {
  const size = SIZE_GUIDANCE[preset.relativeSize];
  const architecture = ARCHITECTURE_GUIDANCE[preset.architectureType];
  const target = TRAINING_TARGET_GUIDANCE[preset.trainingTarget];
  const parameterLabel = formatGuidanceParameterCount(parameterCount);
  const practicalUse = PRACTICAL_USE_BY_PRESET_ID[preset.id] ?? trimSentence(preset.bestUse);

  return {
    title: preset.name,
    summary: `${parameterLabel} ${size.label} ${architecture.label} template for ${target.label}. Starts blank; quality comes from your data.`,
    highlights: [
      `Best for: ${practicalUse}.`,
      `Expect: ${size.expectation}.`,
      `Why this shape: ${architecture.tradeoff}.`,
      `Next step: ${target.nextStep}. ${DATASET_ADVICE[preset.defaultDatasetSource]}`,
      `Watch: ${buildWatchLine(preset)}`,
    ],
  };
}

function buildWatchLine(preset: SimpleModelPreset): string {
  if (preset.hardwareWarning) {
    return `${trimSentence(preset.hardwareWarning)}; it still knows nothing until trained.`;
  }

  if (preset.support === "app-native") {
    return "it can memorize more easily than it generalizes; keep test prompts separate.";
  }

  return "family or size style only; this is not a pretrained clone or assistant.";
}

function formatGuidanceParameterCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "Estimated";
  }
  if (value >= 1_000_000_000) {
    return `~${(value / 1_000_000_000).toFixed(2)}B-param`;
  }
  if (value >= 1_000_000) {
    return `~${(value / 1_000_000).toFixed(0)}M-param`;
  }
  return `~${Math.max(1, Math.round(value / 1_000))}K-param`;
}

function trimSentence(value: string): string {
  return value.trim().replace(/[.!?]+$/u, "");
}
