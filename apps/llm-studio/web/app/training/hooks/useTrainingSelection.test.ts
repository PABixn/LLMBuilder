import assert from "node:assert/strict";
import test from "node:test";

import {
  ACTIVE_RUN_STORAGE_KEY,
  TRAINING_SELECTION_STORAGE_KEY,
} from "../constants";
import { readInitialTrainingSelection } from "./useTrainingSelection";

function searchParams(values: Record<string, string | null>) {
  return {
    get(name: string): string | null {
      return values[name] ?? null;
    },
  };
}

function withStoredValues(values: Record<string, string>, run: () => void) {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem(key: string): string | null {
          return values[key] ?? null;
        },
      },
    },
  });

  try {
    run();
  } finally {
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
}

test("initial training selection prefers URL parameters over stored values", () => {
  withStoredValues(
    {
      [ACTIVE_RUN_STORAGE_KEY]: JSON.stringify("stored-run"),
      [TRAINING_SELECTION_STORAGE_KEY]: JSON.stringify({
        projectId: "stored-project",
        tokenizerJobId: "stored-tokenizer",
      }),
    },
    () => {
      assert.deepEqual(
        readInitialTrainingSelection(
          searchParams({
            project: "url-project",
            run: "url-run",
            tokenizerJob: "url-tokenizer",
          })
        ),
        {
          activeRunId: "url-run",
          projectId: "url-project",
          shouldSelectMostRecentRun: false,
          tokenizerJobId: "url-tokenizer",
        }
      );
    }
  );
});

test("initial training selection uses stored IDs and selects the latest run when no run is stored", () => {
  withStoredValues(
    {
      [TRAINING_SELECTION_STORAGE_KEY]: JSON.stringify({
        projectId: "stored-project",
        tokenizerJobId: "stored-tokenizer",
      }),
    },
    () => {
      assert.deepEqual(
        readInitialTrainingSelection(searchParams({})),
        {
          activeRunId: null,
          projectId: "stored-project",
          shouldSelectMostRecentRun: true,
          tokenizerJobId: "stored-tokenizer",
        }
      );
    }
  );
});
