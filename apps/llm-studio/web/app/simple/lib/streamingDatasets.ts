import {
  SIMPLE_DEFAULT_STREAMING_PRIMARY_DATASET_ID,
  SIMPLE_STREAMING_DATASET_FILTERS,
} from "../constants";
import type { SimpleStreamingDatasetId } from "../types";

interface SimpleStreamingDatasetPreset {
  id: SimpleStreamingDatasetId;
  label: string;
  description: string;
  name: string;
  split: string;
  textColumns: string[];
  filters?: readonly (readonly [string, string, string | number])[];
}

interface NormalizedStreamingSelection {
  primaryId: SimpleStreamingDatasetId;
  additionalIds: SimpleStreamingDatasetId[];
}

export const SIMPLE_STREAMING_DATASETS: readonly SimpleStreamingDatasetPreset[] = [
  {
    id: "fineweb-edu",
    label: "FineWeb-Edu",
    description: "Filtered educational web text",
    name: "HuggingFaceFW/fineweb-edu",
    split: "train",
    textColumns: ["text"],
    filters: SIMPLE_STREAMING_DATASET_FILTERS,
  },
  {
    id: "tinystories",
    label: "TinyStories",
    description: "Short synthetic stories",
    name: "roneneldan/TinyStories",
    split: "train",
    textColumns: ["text"],
  },
  {
    id: "the-stack",
    label: "The Stack",
    description: "Code-heavy source mix",
    name: "bigcode/the-stack",
    split: "train",
    textColumns: ["content"],
  },
];

export function isSimpleStreamingDatasetId(value: unknown): value is SimpleStreamingDatasetId {
  return typeof value === "string" && SIMPLE_STREAMING_DATASETS.some((item) => item.id === value);
}

export function getSimpleStreamingDataset(
  datasetId: SimpleStreamingDatasetId
): SimpleStreamingDatasetPreset {
  return (
    SIMPLE_STREAMING_DATASETS.find((item) => item.id === datasetId) ??
    SIMPLE_STREAMING_DATASETS[0]
  );
}

export function normalizeSimpleStreamingSelection(
  primaryId: unknown,
  additionalIds: unknown
): NormalizedStreamingSelection {
  const primary = isSimpleStreamingDatasetId(primaryId)
    ? primaryId
    : SIMPLE_DEFAULT_STREAMING_PRIMARY_DATASET_ID;
  const additions = Array.isArray(additionalIds) ? additionalIds : [];
  const uniqueAdditionalIds = additions.filter(
    (value, index): value is SimpleStreamingDatasetId =>
      isSimpleStreamingDatasetId(value) &&
      value !== primary &&
      additions.indexOf(value) === index
  );

  return {
    primaryId: primary,
    additionalIds: uniqueAdditionalIds,
  };
}

export function buildSimpleStreamingDatasetSpecs(
  primaryId: SimpleStreamingDatasetId,
  additionalIds: SimpleStreamingDatasetId[],
  options: { includeStreamingFlag: boolean }
): Record<string, unknown>[] {
  const selection = normalizeSimpleStreamingSelection(primaryId, additionalIds);
  const orderedIds = [selection.primaryId, ...selection.additionalIds];
  const additionalWeight =
    selection.additionalIds.length > 0 ? 0.2 / selection.additionalIds.length : 0;

  return orderedIds.map((datasetId, index) => {
    const dataset = getSimpleStreamingDataset(datasetId);
    const spec: Record<string, unknown> = {
      name: dataset.name,
      split: dataset.split,
      text_columns: dataset.textColumns,
      weight: index === 0 ? (orderedIds.length > 1 ? 0.8 : 1) : additionalWeight,
    };
    if (dataset.filters) {
      spec.filters = dataset.filters.map((filter) => [...filter]);
    }
    if (options.includeStreamingFlag) {
      spec.streaming = true;
    }
    return spec;
  });
}
