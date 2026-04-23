import assert from "node:assert/strict";
import test from "node:test";

import {
  formatNumberInputValue,
  parseConfigNumberInput,
  sanitizePositiveDecimalInput,
  sanitizePositiveScientificInput,
} from "./configNumber";

test("config number helpers preserve decimal and scientific editing behavior", () => {
  assert.equal(sanitizePositiveDecimalInput("1.2.3abc"), "1.23");
  assert.equal(sanitizePositiveScientificInput(" 1.2E-03 "), "1.2e-03");
  assert.equal(parseConfigNumberInput("3.5", "integer"), null);
  assert.equal(parseConfigNumberInput("3.5", "decimal"), 3.5);
  assert.equal(parseConfigNumberInput("1e-", "scientific"), null);
  assert.equal(formatNumberInputValue(0.0003, "scientific"), "3e-4");
});
