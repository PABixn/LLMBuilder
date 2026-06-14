import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";

import {
  LEGACY_THEME_STORAGE_KEYS,
  THEME_BOOTSTRAP_SCRIPT,
  THEME_STORAGE_KEY,
  migrateStoredTheme,
  writeStoredTheme,
} from "../../../lib/themeStorage";

class MemoryStorage {
  readonly values = new Map<string, string>();
  failSet = false;

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    if (this.failSet) {
      throw new Error("storage is read-only");
    }
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

test("theme migration copies the Tokenizer Studio preference into the shared key", () => {
  const storage = new MemoryStorage();
  storage.setItem(LEGACY_THEME_STORAGE_KEYS[0], "dark");

  assert.equal(migrateStoredTheme(storage), "dark");
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "dark");
  assert.equal(storage.getItem(LEGACY_THEME_STORAGE_KEYS[0]), null);
});

test("theme migration keeps a current preference and removes stale legacy state", () => {
  const storage = new MemoryStorage();
  storage.setItem(THEME_STORAGE_KEY, "white");
  storage.setItem(LEGACY_THEME_STORAGE_KEYS[0], "dark");

  assert.equal(migrateStoredTheme(storage), "white");
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "white");
  assert.equal(storage.getItem(LEGACY_THEME_STORAGE_KEYS[0]), null);
});

test("theme migration preserves and applies legacy state when current storage cannot be written", () => {
  const storage = new MemoryStorage();
  storage.setItem(LEGACY_THEME_STORAGE_KEYS[0], "dark");
  storage.failSet = true;

  assert.equal(migrateStoredTheme(storage), "dark");
  assert.equal(storage.getItem(THEME_STORAGE_KEY), null);
  assert.equal(storage.getItem(LEGACY_THEME_STORAGE_KEYS[0]), "dark");
  assert.equal(writeStoredTheme(storage, "white"), false);
});

test("theme bootstrap migrates the preference before first paint", () => {
  const storage = new MemoryStorage();
  storage.setItem(LEGACY_THEME_STORAGE_KEYS[0], "dark");
  const document = { documentElement: { dataset: {} as Record<string, string> } };

  vm.runInNewContext(THEME_BOOTSTRAP_SCRIPT, { document, localStorage: storage });

  assert.equal(document.documentElement.dataset.theme, "dark");
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "dark");
  assert.equal(storage.getItem(LEGACY_THEME_STORAGE_KEYS[0]), null);
});
