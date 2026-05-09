import assert from "node:assert/strict";
import test from "node:test";

import {
  clampMetricRange,
  formatMetricValue,
  metricAxisDomain,
  metricChartData,
  metricChartStats,
} from "./metrics";
import type { TrainingMetricPoint } from "../../../lib/training/types";

test("training metric helpers filter invalid values and preserve chart domains", () => {
  const metrics = [
    { step: 1, loss: 2 },
    { step: 2, loss: Number.NaN },
    { step: 3, loss: 1 },
  ] as TrainingMetricPoint[];

  const data = metricChartData(metrics, "loss");
  assert.deepEqual(data.map((item) => item.step), [1, 3]);
  assert.equal(metricChartStats(data)?.latest, 1);
  assert.deepEqual(clampMetricRange({ startIndex: -5, endIndex: 99 }, data.length), {
    startIndex: 0,
    endIndex: 1,
  });
  assert.equal(formatMetricValue(0.000001, 3), "1.00e-6");
  assert.equal(metricAxisDomain([])[0], 0);
});
