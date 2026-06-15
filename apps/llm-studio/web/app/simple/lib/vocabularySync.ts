import type { ModelConfig } from "../../../lib/defaults";
import type { ProjectDetail } from "../../../lib/api";
import type { TrainingJob as TokenizerJob } from "../../../lib/tokenizerLegacyApi";

export function readTokenizerVocabSize(job: TokenizerJob | null): number | null {
  const statsVocab = job?.stats?.vocab_size;
  if (typeof statsVocab === "number" && Number.isFinite(statsVocab) && statsVocab > 0) {
    return Math.trunc(statsVocab);
  }

  const configVocab = job?.tokenizer_config.vocab_size;
  if (typeof configVocab === "number" && Number.isFinite(configVocab) && configVocab > 0) {
    return Math.trunc(configVocab);
  }

  return null;
}

export function buildModelConfigWithSyncedVocab(
  config: ModelConfig,
  tokenizerVocabSize: number
): ModelConfig {
  return {
    ...config,
    vocab_size: Math.max(1, Math.trunc(tokenizerVocabSize)),
  };
}

export function modelNeedsTokenizerVocabSync(
  project: ProjectDetail | null,
  tokenizerJob: TokenizerJob | null
): boolean {
  const tokenizerVocabSize = readTokenizerVocabSize(tokenizerJob);
  return Boolean(
    project &&
      tokenizerVocabSize !== null &&
      project.model_config.vocab_size !== tokenizerVocabSize
  );
}
