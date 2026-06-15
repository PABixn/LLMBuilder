import assert from "node:assert/strict";
import test from "node:test";

import {
  __resetRuntimeConfigForTests,
  __setRuntimeConfigForTests,
  getRuntimeConfig,
  initializeRuntimeConfig,
  RuntimeHttpError,
  RuntimeRequestAbortedError,
  RuntimeUnavailableError,
  runtimeApiUrl,
  runtimeHeaders,
  runtimeJsonRequest,
  runtimeRequest,
} from "../../../lib/runtimeConfig";

test("runtime config injects desktop token through headers only", () => {
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-secret",
    capabilities: {
      native_save: true,
      open_logs: true,
      open_data: true,
      reveal_artifact: true,
      diagnostics_export: true,
    },
    versions: { runtime: "test" },
  });

  const url = runtimeApiUrl("/projects");
  const headers = runtimeHeaders();

  assert.equal(url, "http://127.0.0.1:43123/api/v1/projects");
  assert.equal(headers.get("X-LLM-Studio-Token"), "memory-only-secret");
  assert.equal(url.includes("memory-only-secret"), false);
  assert.equal(getRuntimeConfig().runtimeToken, "memory-only-secret");

  __resetRuntimeConfigForTests();
});

test("runtime config preserves explicit caller headers", () => {
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "runtime-token",
    capabilities: {
      native_save: false,
      open_logs: false,
      open_data: false,
      reveal_artifact: false,
      diagnostics_export: false,
    },
    versions: {},
  });

  const headers = runtimeHeaders({
    "Content-Type": "application/json",
    "X-LLM-Studio-Token": "explicit-token",
  });

  assert.equal(headers.get("Content-Type"), "application/json");
  assert.equal(headers.get("X-LLM-Studio-Token"), "explicit-token");

  __resetRuntimeConfigForTests();
});

test("runtime request normalizes validation errors and preserves domain status", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        detail: [{ loc: ["body", "name"], msg: "Field required" }],
      }),
      {
        status: 422,
        headers: { "Content-Type": "application/json" },
      }
    );
  try {
    await assert.rejects(
      runtimeJsonRequest("/projects", { method: "POST", body: "{}" }),
      (error: unknown) =>
        error instanceof RuntimeHttpError &&
        error.status === 422 &&
        error.message === "body.name: Field required"
    );
  } finally {
    globalThis.fetch = originalFetch;
    __resetRuntimeConfigForTests();
  }
});

test("runtime request normalizes abort and network failures", async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => {
      const error = new Error("cancelled");
      error.name = "AbortError";
      throw error;
    };
    await assert.rejects(runtimeRequest("/health"), RuntimeRequestAbortedError);

    globalThis.fetch = async () => {
      throw new Error("secret-bearing low-level network failure");
    };
    await assert.rejects(
      runtimeRequest("/health"),
      (error: unknown) =>
        error instanceof RuntimeUnavailableError &&
        !error.message.includes("secret-bearing")
    );
  } finally {
    globalThis.fetch = originalFetch;
    __resetRuntimeConfigForTests();
  }
});

test("desktop runtime bootstrap and retry use narrow commands without persistence", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const calls: string[] = [];
  const persisted: Array<[string, string]> = [];
  let token = "first-memory-only-token";
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        setItem: (key: string, value: string) => persisted.push([key, value]),
      },
      __TAURI__: {
        core: {
          invoke: async (command: string) => {
            calls.push(command);
            return {
              environment: "desktop",
              api_base_url: "http://127.0.0.1:43123/api/v1/",
              runtime_token: token,
              capabilities: {
                native_save: true,
                open_logs: true,
                open_data: true,
                reveal_artifact: true,
                diagnostics_export: true,
              },
              versions: { runtime: "test" },
            };
          },
        },
      },
    },
  });

  try {
    const initial = await initializeRuntimeConfig();
    assert.equal(initial.apiBaseUrl, "http://127.0.0.1:43123/api/v1");
    assert.equal(initial.runtimeToken, "first-memory-only-token");

    token = "second-memory-only-token";
    const retried = await initializeRuntimeConfig({ retry: true });
    assert.equal(retried.runtimeToken, "second-memory-only-token");
    assert.deepEqual(calls, ["runtime_bootstrap", "retry_runtime"]);
    assert.deepEqual(persisted, []);
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("desktop runtime requests use the native bridge without exposing the token", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async (command: string, args?: Record<string, unknown>) => {
            calls.push({ command, args });
            return {
              status: 200,
              headers: { "content-type": "application/json" },
              body: Array.from(new TextEncoder().encode('{"projects":[]}')),
            };
          },
        },
      },
    },
  });
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-secret",
    capabilities: {
      native_save: false,
      open_logs: false,
      open_data: false,
      reveal_artifact: false,
      diagnostics_export: false,
    },
    versions: {},
  });

  try {
    const response = await runtimeRequest("/projects", {
      headers: { "X-LLM-Studio-Token": "must-not-cross-ipc" },
    });
    assert.deepEqual(await response.json(), { projects: [] });
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.command, "runtime_request");
    const request = calls[0]?.args?.request as {
      path: string;
      headers: Record<string, string>;
    };
    assert.equal(request.path, "/projects");
    assert.equal(request.headers["x-llm-studio-token"], undefined);
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("desktop native transport preserves no-content responses", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async () => ({ status: 204, headers: {}, body: [] }),
        },
      },
    },
  });
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-secret",
    capabilities: {
      native_save: false,
      open_logs: false,
      open_data: false,
      reveal_artifact: false,
      diagnostics_export: false,
    },
    versions: {},
  });

  try {
    const response = await runtimeRequest("/projects/project_123", { method: "DELETE" });
    assert.equal(response.status, 204);
    assert.equal(await response.text(), "");
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("failed desktop retry clears stale runtime connectivity and can recover", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  let retryFails = true;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async (command: string) => {
            if (command === "retry_runtime" && retryFails) {
              throw new Error("backend restart failed");
            }
            return {
              environment: "desktop",
              api_base_url: "http://127.0.0.1:43123/api/v1",
              runtime_token: retryFails ? "stale-token" : "recovered-token",
              capabilities: {
                native_save: false,
                open_logs: true,
                open_data: true,
                reveal_artifact: false,
                diagnostics_export: true,
              },
              versions: {},
            };
          },
        },
      },
    },
  });

  try {
    await initializeRuntimeConfig();
    assert.equal(getRuntimeConfig().runtimeToken, "stale-token");

    await assert.rejects(
      initializeRuntimeConfig({ retry: true }),
      (error: unknown) =>
        error instanceof RuntimeUnavailableError &&
        error.message === "backend restart failed"
    );
    assert.equal(getRuntimeConfig().environment, "desktop");
    assert.equal(getRuntimeConfig().runtimeToken, null);
    await assert.rejects(runtimeRequest("/health"), RuntimeUnavailableError);

    retryFails = false;
    const recovered = await initializeRuntimeConfig({ retry: true });
    assert.equal(recovered.runtimeToken, "recovered-token");
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});
