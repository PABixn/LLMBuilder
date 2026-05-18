import { formatExponentialValue } from "../../shared/lib/configNumber";
import type { TrainingMetricPoint } from "../../../lib/training/types";
import type { MetricChartKey, MetricValueNotation } from "../types";

export type MetricChartDatum = {
  step: number;
  value: number;
  plotValue: number;
};

export type MetricChartRange = {
  startIndex: number;
  endIndex: number;
};

export function metricChartData(
  metrics: TrainingMetricPoint[],
  metricKey: MetricChartKey
): MetricChartDatum[] {
  return metrics
    .map((item) => {
      const value = item[metricKey];
      if (typeof value !== "number" || !Number.isFinite(value)) {
        return null;
      }
      return {
        step: item.step,
        value,
        plotValue: value,
      };
    })
    .filter((item): item is MetricChartDatum => item !== null);
}

export function metricChartStats(data: Array<{ value: number }>) {
  if (!data.length) {
    return null;
  }

  let min = data[0].value;
  let max = data[0].value;
  let total = 0;
  data.forEach((item) => {
    min = Math.min(min, item.value);
    max = Math.max(max, item.value);
    total += item.value;
  });

  return {
    latest: data[data.length - 1].value,
    min,
    max,
    average: total / data.length,
  };
}

export function metricAxisDomain(
  data: Array<{ plotValue: number }>
): [number | string | ((value: number) => number), number | string | ((value: number) => number)] {
  if (!data.length) {
    return [0, 1];
  }

  const values = data.map((item) => item.plotValue);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min;
  const padding = spread === 0 ? Math.max(Math.abs(max) * 0.08, 1) : spread * 0.08;

  return [min - padding, max + padding];
}

export function formatMetricValue(
  value: number | null | undefined,
  digits: number,
  notation: MetricValueNotation = "standard"
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  if (notation === "exponential") {
    return formatExponentialValue(value, digits);
  }
  if (Math.abs(value) > 0 && Math.abs(value) < 0.0001) {
    return value.toExponential(2);
  }
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

export function formatMetricAxis(
  value: number,
  digits: number,
  notation: MetricValueNotation = "standard"
): string {
  return formatMetricValue(value, digits, notation);
}

export function clampMetricRange(
  range: MetricChartRange | null,
  pointCount: number
): MetricChartRange | null {
  if (pointCount <= 0) {
    return null;
  }
  const lastIndex = pointCount - 1;
  if (!range) {
    return { startIndex: 0, endIndex: lastIndex };
  }
  const startIndex = Math.max(0, Math.min(range.startIndex, lastIndex));
  const endIndex = Math.max(startIndex, Math.min(range.endIndex, lastIndex));
  return { startIndex, endIndex };
}

export function chartBrushHandlePosition(percent: number): string {
  const handleHalfWidth = 5;
  const edgeOffset = (1 - 2 * (percent / 100)) * handleHalfWidth;
  return `calc(${percent}% + ${edgeOffset.toFixed(2)}px)`;
}

export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toFixed(digits);
}

export function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toLocaleString();
}
