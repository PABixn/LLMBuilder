import assert from "node:assert/strict";
import test from "node:test";

import {
  buildDataloaderConfigFromForm,
  normalizeStreamingDatasetWeights,
} from "./dataset";
import type { DatasetFormState, TrainingFormState } from "../types";

test("tokenizer streaming weights normalize without changing storage/API shape", () => {
  const normalized = normalizeStreamingDatasetWeights([
    { id: "a", name: "dataset-a", config: "", split: "train", textColumns: "text", weight: "2", filters: [] },
    { id: "b", name: "dataset-b", config: "", split: "train", textColumns: "text", weight: "1", filters: [] },
  ]);
  const sum = normalized.reduce((total, entry) => total + Number(entry.weight), 0);
  assert.equal(Math.round(sum * 1_000_000) / 1_000_000, 1);
});

test("tokenizer dataloader filters serialize as legacy tuple filters", () => {
  const dataset: DatasetFormState = {
    sourceMode: "streaming_hf",
    localTrainFiles: [],
    hfToken: "",
    streamingDatasets: [
      {
        id: "a",
        name: "dataset-a",
        config: "",
        split: "train",
        textColumns: "text",
        weight: "1",
        filters: [{ id: "f", column: "score", operator: ">=", value: "0.9" }],
      },
    ],
  };
  const training: TrainingFormState = {
    budgetLimit: "1000",
    budgetUnit: "chars",
    budgetBehavior: "truncate",
    evaluationThresholds: "5,10",
  };

  const config = buildDataloaderConfigFromForm(dataset, training);
  const datasets = config.datasets as Array<Record<string, unknown>>;
  assert.deepEqual(datasets[0].filters, [["score", ">=", 0.9]]);
});

test("tokenizer dataloader config never embeds the UI HF token", () => {
  const dataset: DatasetFormState = {
    sourceMode: "streaming_hf",
    localTrainFiles: [],
    hfToken: "hf_private_secret",
    streamingDatasets: [
      {
        id: "a",
        name: "private-dataset",
        config: "",
        split: "train",
        textColumns: "text",
        weight: "1",
        filters: [],
      },
    ],
  };
  const training: TrainingFormState = {
    budgetLimit: "1000",
    budgetUnit: "chars",
    budgetBehavior: "truncate",
    evaluationThresholds: "5,10",
  };

  assert.equal(JSON.stringify(buildDataloaderConfigFromForm(dataset, training)).includes("hf_private_secret"), false);
});
