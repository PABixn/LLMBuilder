import type { TrainingJob } from "../../../lib/training/types";
import { formatExponentialValue } from "../../shared/lib/configNumber";
import { fileNameFromPath } from "./files";

export function compactWorkflowMessage(value: string): string {
  return value.replace(/(?:[A-Za-z]:)?[\\/][^\s]+/g, (path) => {
    const fileName = fileNameFromPath(path);
    return fileName || path;
  });
}

export function samplePromptSummary(prompt: string | null | undefined, index: number): string {
  const normalized = prompt?.replace(/\s+/g, " ").trim();
  return normalized && normalized.length > 0 ? normalized : `Prompt ${index + 1}`;
}

export function splitGeneratedSampleText(
  text: string,
  prompt: string | null | undefined
): { prefix: string | null; continuation: string } {
  if (!prompt || !text.startsWith(prompt)) {
    return { prefix: null, continuation: text };
  }
  return {
    prefix: prompt,
    continuation: text.slice(prompt.length),
  };
}

export function formatStatusLabel(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatLearningRate(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return formatExponentialValue(value, 3);
}

export function replaceRunInOrder(runs: TrainingJob[], job: TrainingJob): TrainingJob[] {
  let replaced = false;
  const next = runs.map((item) => {
    if (item.id !== job.id) {
      return item;
    }
    replaced = true;
    return job;
  });
  return replaced ? next : [...next, job];
}
