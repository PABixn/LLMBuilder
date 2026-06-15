import assert from "node:assert/strict";
import test from "node:test";

import { normalizeUiMode } from "./useUiMode";

test("ui mode parser accepts only supported modes", () => {
  assert.equal(normalizeUiMode("simple"), "simple");
  assert.equal(normalizeUiMode("expert"), "expert");
  assert.equal(normalizeUiMode("invalid"), "expert");
  assert.equal(normalizeUiMode(null, "simple"), "simple");
});
