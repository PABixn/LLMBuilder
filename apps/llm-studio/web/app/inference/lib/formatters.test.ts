import assert from "node:assert/strict";
import test from "node:test";

import {
  formatCheckpointMeta,
  formatJobMeta,
  matchesCheckpointQuery,
  matchesJobQuery,
} from "./formatters";
import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";

test("inference job and checkpoint search covers identifiers, metadata, and files", () => {
  const job = {
    id: "run_nvda_123",
    name: "Demo model",
    project_name: "Transformer",
    tokenizer_name: "byte-tokenizer",
    artifact_bundle_file: "artifact.tar.gz",
    artifact_dir: "runs/demo",
    stage: "completed",
    last_step: 42,
    checkpoint_count: 2,
    finished_at: "2026-04-23T12:00:00Z",
    status: "completed",
  } as TrainingJob;
  const checkpoint = {
    step: 42,
    directory: "checkpoint-42",
    created_at: "2026-04-23T12:00:00Z",
    size_bytes: 1024,
    files: ["model.pt", "config.json"],
  } as TrainingCheckpointEntry;

  assert.equal(matchesJobQuery(job, "byte-tokenizer"), true);
  assert.equal(matchesJobQuery(job, "missing"), false);
  assert.equal(matchesCheckpointQuery(checkpoint, "config.json"), true);
  assert.match(formatJobMeta(job), /Step 43/);
  assert.match(formatCheckpointMeta(checkpoint), /files/);
});

test("inference model artifact step is displayed as a completed step count", () => {
  const job = {
    id: "run_3000",
    name: "Demo model",
    project_name: "Transformer",
    tokenizer_name: "byte-tokenizer",
    artifact_bundle_file: "artifact.tar.gz",
    artifact_dir: "runs/demo",
    stage: "completed",
    last_step: 2999,
    checkpoint_count: 1,
    finished_at: "2026-04-23T12:00:00Z",
    status: "completed",
  } as TrainingJob;

  assert.match(formatJobMeta(job), /Step 3,000/);
});
