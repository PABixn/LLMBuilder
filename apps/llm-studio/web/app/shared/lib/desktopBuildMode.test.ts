import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isDesktopBuildEnvironment } from "../../../desktop-build-mode";

test("desktop build mode accepts an explicit environment opt-in", () => {
  assert.equal(isDesktopBuildEnvironment({ LLM_STUDIO_DESKTOP_BUILD: "1" }), true);
  assert.equal(isDesktopBuildEnvironment({ LLM_STUDIO_DESKTOP_BUILD: "0" }), false);
});

test("desktop build mode recognizes the cross-platform npm lifecycle", () => {
  assert.equal(isDesktopBuildEnvironment({ npm_lifecycle_event: "build:desktop" }), true);
  assert.equal(isDesktopBuildEnvironment({ npm_lifecycle_event: "build" }), false);
  assert.equal(isDesktopBuildEnvironment({}), false);
});

test("desktop build command is portable across npm host shells", () => {
  const packageJson = JSON.parse(
    readFileSync(new URL("../../../package.json", import.meta.url), "utf8"),
  ) as { scripts: Record<string, string> };
  const command = packageJson.scripts["build:desktop"];

  assert.match(command, /\bnext build\b/);
  assert.match(command, /\bnpm run validate:desktop-output\b/);
  assert.doesNotMatch(command, /(?:^|\s)[A-Za-z_][A-Za-z0-9_]*=/);
});
