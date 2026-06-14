import assert from "node:assert/strict";
import test from "node:test";

import {
  ACTIVE_JOB_STORAGE_KEY,
  DATASET_FORM_STORAGE_KEY,
  HIDDEN_RECENT_JOB_IDS_STORAGE_KEY,
  PREVIEW_TEXT_STORAGE_KEY,
  TOKENIZER_FORM_STORAGE_KEY,
  TOKENIZER_STORAGE_KEY_MIGRATIONS,
  TRAINING_FORM_STORAGE_KEY,
} from "../constants";
import { migrateStoredValues } from "./storage";

class MemoryStorage {
  readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

function withStorage(storage: MemoryStorage, run: () => void): void {
  const previousWindow = globalThis.window;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { localStorage: storage },
  });
  try {
    run();
  } finally {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: previousWindow,
    });
  }
}

test("active tokenizer persistence keys use the LLM Studio namespace", () => {
  const activeKeys = [
    TOKENIZER_FORM_STORAGE_KEY,
    DATASET_FORM_STORAGE_KEY,
    TRAINING_FORM_STORAGE_KEY,
    ACTIVE_JOB_STORAGE_KEY,
    PREVIEW_TEXT_STORAGE_KEY,
    HIDDEN_RECENT_JOB_IDS_STORAGE_KEY,
  ];

  assert.equal(activeKeys.every((key) => key.startsWith("llm-studio-tokenizer-")), true);
  assert.equal(activeKeys.some((key) => key.startsWith("tokenizer-studio-")), false);
});

test("tokenizer persistence migration copies legacy values before removing them", () => {
  const storage = new MemoryStorage();
  for (const [index, migration] of TOKENIZER_STORAGE_KEY_MIGRATIONS.entries()) {
    storage.setItem(migration.legacyKey, `legacy-${index}`);
  }

  withStorage(storage, () => migrateStoredValues(TOKENIZER_STORAGE_KEY_MIGRATIONS));

  for (const [index, migration] of TOKENIZER_STORAGE_KEY_MIGRATIONS.entries()) {
    assert.equal(storage.getItem(migration.currentKey), `legacy-${index}`);
    assert.equal(storage.getItem(migration.legacyKey), null);
  }
});

test("tokenizer persistence migration never overwrites current values", () => {
  const storage = new MemoryStorage();
  const migration = TOKENIZER_STORAGE_KEY_MIGRATIONS[0];
  storage.setItem(migration.currentKey, "current");
  storage.setItem(migration.legacyKey, "legacy");

  withStorage(storage, () => migrateStoredValues([migration]));

  assert.equal(storage.getItem(migration.currentKey), "current");
  assert.equal(storage.getItem(migration.legacyKey), null);
});
