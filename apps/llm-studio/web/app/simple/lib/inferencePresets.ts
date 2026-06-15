import type {
  SimpleInferenceCreativity,
  SimpleInferenceLength,
} from "../types";

export interface SimpleInferenceSettings {
  max_tokens: number;
  temperature: number;
  top_k: number;
  repetition_penalty: number;
}

export const SIMPLE_LENGTH_PRESETS: Record<SimpleInferenceLength, { label: string; maxTokens: number }> = {
  short: { label: "Short", maxTokens: 48 },
  medium: { label: "Medium", maxTokens: 96 },
  long: { label: "Long", maxTokens: 160 },
};

export const SIMPLE_CREATIVITY_PRESETS: Record<
  SimpleInferenceCreativity,
  { label: string; temperature: number; topK: number; repetitionPenalty: number }
> = {
  precise: {
    label: "Precise",
    temperature: 0.2,
    topK: 20,
    repetitionPenalty: 1.12,
  },
  balanced: {
    label: "Balanced",
    temperature: 0.7,
    topK: 40,
    repetitionPenalty: 1.08,
  },
  creative: {
    label: "Creative",
    temperature: 0.95,
    topK: 80,
    repetitionPenalty: 1.04,
  },
};

export function buildInferenceSettings(
  length: SimpleInferenceLength,
  creativity: SimpleInferenceCreativity
): SimpleInferenceSettings {
  const lengthPreset = SIMPLE_LENGTH_PRESETS[length];
  const creativityPreset = SIMPLE_CREATIVITY_PRESETS[creativity];
  return {
    max_tokens: lengthPreset.maxTokens,
    temperature: creativityPreset.temperature,
    top_k: creativityPreset.topK,
    repetition_penalty: creativityPreset.repetitionPenalty,
  };
}
