import type { BuildResult } from "../types";

export function buildResult(factory: () => Record<string, unknown>): BuildResult {
  try {
    return {
      value: factory(),
      error: null,
    };
  } catch (error) {
    return {
      value: null,
      error: error instanceof Error ? error.message : "Unknown form error",
    };
  }
}
