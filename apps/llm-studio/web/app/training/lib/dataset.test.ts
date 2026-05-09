import assert from "node:assert/strict";
import test from "node:test";

import {
  buildDatasetsFromUi,
  hydrateDatasetUiFromConfig,
  makeLocalTrainFileEntry,
  makeStreamingDatasetEntry,
  normalizeStreamingDatasetWeights,
} from "./dataset";

test("training dataset UI builds a local file dataloader with normalized file paths", () => {
  assert.deepEqual(
    buildDatasetsFromUi(
      "local_file",
      [
        makeLocalTrainFileEntry({
          filePath: " /data/train-a.txt ",
        }),
        makeLocalTrainFileEntry({
          filePath: "/data/train-b.txt",
        }),
      ],
      "",
      []
    ),
    [
      {
        name: "text",
        split: "train",
        text_columns: ["text"],
        weight: 1,
        streaming: true,
        data_files: {
          train: ["/data/train-a.txt", "/data/train-b.txt"],
        },
      },
    ]
  );
});

test("training dataset UI hydrates streaming datasets with filters and token", () => {
  const hydrated = hydrateDatasetUiFromConfig({
    datasets: [
      {
        name: "roneneldan/TinyStories",
        split: "train",
        text_columns: ["text", "summary"],
        weight: 2,
        streaming: true,
        hf_token: " hf_test ",
        filters: [
          {
            column: "language",
            operator: "==",
            value: "en",
          },
        ],
      },
    ],
  });

  assert.equal(hydrated.sourceMode, "streaming_hf");
  assert.equal(hydrated.hfToken, "hf_test");
  assert.equal(hydrated.streamingDatasets[0].name, "roneneldan/TinyStories");
  assert.equal(hydrated.streamingDatasets[0].textColumns, "text, summary");
  assert.equal(hydrated.streamingDatasets[0].filters[0].value, "en");
});

test("training dataset weight normalization keeps a valid locked weight", () => {
  const datasets = normalizeStreamingDatasetWeights(
    [
      makeStreamingDatasetEntry({ id: "first", weight: "0.2" }),
      makeStreamingDatasetEntry({ id: "second", weight: "0.8" }),
      makeStreamingDatasetEntry({ id: "third", weight: "0.5" }),
    ],
    "first",
    "0.5"
  );

  assert.deepEqual(
    datasets.map((entry) => entry.weight),
    ["0.5", "0.307692", "0.192308"]
  );
});
