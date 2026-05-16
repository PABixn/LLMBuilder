import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingJob } from "../../../lib/training/types";
import { deriveTrainingStepProgress } from "./display";

function trainingJob(overrides: Partial<TrainingJob>): TrainingJob {
  return {
    status: "running",
    last_step: 0,
    max_steps: 0,
    progress: 0,
    ...overrides,
  } as TrainingJob;
}

test("completed training progress displays the configured max step count", () => {
  const progress = deriveTrainingStepProgress(
    trainingJob({
      status: "completed",
      last_step: 2999,
      max_steps: 3000,
      progress: 1,
    })
  );

  assert.equal(progress.completedSteps, 3000);
  assert.equal(progress.maxSteps, 3000);
  assert.equal(progress.fraction, 1);
  assert.equal(progress.percentLabel, "100%");
});

test("running training progress preserves the reported runtime step", () => {
  const progress = deriveTrainingStepProgress(
    trainingJob({
      status: "running",
      last_step: 2999,
      max_steps: 3000,
      progress: 0.999,
    })
  );

  assert.equal(progress.completedSteps, 2999);
  assert.equal(progress.maxSteps, 3000);
});
