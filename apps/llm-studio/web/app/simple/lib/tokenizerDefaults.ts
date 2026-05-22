import {
  SIMPLE_STARTER_DATASET_PATH,
} from "../constants";
import type {
  SimpleDatasetSource,
  SimpleLocalTrainFile,
  SimpleStreamingDatasetId,
} from "../types";
import { buildSimpleStreamingDatasetSpecs } from "./streamingDatasets";

export interface BuildSimpleTokenizerConfigOptions {
  name: string;
  vocabSize: number;
}

export interface BuildSimpleDataloaderConfigOptions {
  datasetSource: SimpleDatasetSource;
  localTrainFiles: SimpleLocalTrainFile[];
  streamingPrimaryDatasetId: SimpleStreamingDatasetId;
  streamingAdditionalDatasetIds: SimpleStreamingDatasetId[];
  budgetLimit: number;
}

export function buildSimpleTokenizerConfig({
  name,
  vocabSize,
}: BuildSimpleTokenizerConfigOptions): Record<string, unknown> {
  return {
    name: name.trim() || "simple_bpe_bytelevel",
    tokenizer_type: "bpe",
    byte_fallback: true,
    vocab_size: Math.max(1, Math.trunc(vocabSize)),
    min_frequency: 2,
    special_tokens: ["<|endoftext|>", "<|pad|>"],
    pre_tokenizer: "byte_level",
    decoder: "byte_level",
  };
}

export function tokenizerBudgetForDataset(
  datasetSource: SimpleDatasetSource,
  targetVocabSize: number
): number {
  if (datasetSource === "starter") {
    return targetVocabSize <= 1000 ? 250_000 : 500_000;
  }
  if (datasetSource === "upload") {
    if (targetVocabSize <= 8000) {
      return 2_000_000;
    }
    if (targetVocabSize <= 16000) {
      return 6_000_000;
    }
    return 12_000_000;
  }
  return targetVocabSize <= 16000 ? 8_000_000 : 16_000_000;
}

export function buildSimpleTokenizerDataloaderConfig({
  datasetSource,
  localTrainFiles,
  streamingPrimaryDatasetId,
  streamingAdditionalDatasetIds,
  budgetLimit,
}: BuildSimpleDataloaderConfigOptions): Record<string, unknown> {
  const budget = {
    limit: Math.max(1, Math.trunc(budgetLimit)),
    unit: "chars",
    behavior: "truncate",
  };

  if (datasetSource === "starter") {
    return {
      datasets: [
        {
          name: "text",
          data_files: { train: SIMPLE_STARTER_DATASET_PATH },
          split: "train",
          text_columns: ["text"],
          weight: 1,
        },
      ],
      budget,
    };
  }

  if (datasetSource === "upload") {
    const paths = localTrainFiles
      .map((file) => file.filePath.trim())
      .filter((filePath) => filePath !== "");
    return {
      datasets: [
        {
          name: "text",
          data_files: { train: paths.length <= 1 ? (paths[0] ?? "") : paths },
          split: "train",
          text_columns: ["text"],
          weight: 1,
        },
      ],
      budget,
    };
  }

  return {
    datasets: buildSimpleStreamingDatasetSpecs(
      streamingPrimaryDatasetId,
      streamingAdditionalDatasetIds,
      { includeStreamingFlag: false }
    ),
    budget,
  };
}

export function simpleDatasetBlocker(
  datasetSource: SimpleDatasetSource,
  localTrainFiles: SimpleLocalTrainFile[]
): string | null {
  if (datasetSource === "upload" && localTrainFiles.length === 0) {
    return "Upload at least one text file.";
  }
  return null;
}
