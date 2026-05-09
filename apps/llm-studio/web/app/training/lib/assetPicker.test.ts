import assert from "node:assert/strict";
import test from "node:test";

import type { ProjectSummary } from "../../../lib/api";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import {
  filterPickerProjects,
  filterPickerTokenizerJobs,
  normalizeAssetPickerQuery,
} from "./assetPicker";

test("asset picker query normalization trims and lowercases search text", () => {
  assert.equal(normalizeAssetPickerQuery("  Byte TOKENIZER  "), "byte tokenizer");
});

test("asset picker projects are newest first and match identifiers or artifact metadata", () => {
  const projects = [
    {
      id: "project-old",
      name: "Old model",
      artifact_file: "old.pt",
      artifact_path: "/models/old.pt",
      created_at: "2026-04-01T00:00:00Z",
    },
    {
      id: "project-new",
      name: "New model",
      artifact_file: "new.pt",
      artifact_path: "/models/new.pt",
      created_at: "2026-04-02T00:00:00Z",
    },
  ] as ProjectSummary[];

  assert.deepEqual(
    filterPickerProjects(projects, "").map((project) => project.id),
    ["project-new", "project-old"]
  );
  assert.deepEqual(
    filterPickerProjects(projects, "old.pt").map((project) => project.id),
    ["project-old"]
  );
});

test("asset picker tokenizers only show completed jobs and match tokenizer names", () => {
  const jobs = [
    {
      id: "tok-running",
      status: "running",
      tokenizer_config: { name: "Byte tokenizer running" },
      artifact_file: "running.json",
      artifact_path: "/tokenizers/running.json",
      created_at: "2026-04-03T00:00:00Z",
    },
    {
      id: "tok-old",
      status: "completed",
      tokenizer_config: { name: "Wordpiece tokenizer" },
      artifact_file: "old.json",
      artifact_path: "/tokenizers/old.json",
      created_at: "2026-04-01T00:00:00Z",
    },
    {
      id: "tok-new",
      status: "completed",
      tokenizer_config: { name: "Byte tokenizer" },
      artifact_file: "new.json",
      artifact_path: "/tokenizers/new.json",
      created_at: "2026-04-02T00:00:00Z",
    },
  ] as unknown as TokenizerTrainingJob[];

  assert.deepEqual(
    filterPickerTokenizerJobs(jobs, "").map((job) => job.id),
    ["tok-new", "tok-old"]
  );
  assert.deepEqual(
    filterPickerTokenizerJobs(jobs, "byte").map((job) => job.id),
    ["tok-new"]
  );
});
