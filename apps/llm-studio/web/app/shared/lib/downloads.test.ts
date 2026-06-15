import assert from "node:assert/strict";
import test from "node:test";

import {
  downloadApiArtifact,
  downloadTextFile,
  revealApiArtifact,
} from "../../../lib/downloads";
import {
  __resetRuntimeConfigForTests,
  __setRuntimeConfigForTests,
} from "../../../lib/runtimeConfig";

test("desktop artifact downloads use the narrow native streaming command", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const previousFetch = globalThis.fetch;
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async (command: string, args?: Record<string, unknown>) => {
            calls.push({ command, args });
            return "/tmp/model-config.json";
          },
        },
      },
    },
  });
  globalThis.fetch = async () => {
    throw new Error("desktop artifact flow must not use browser fetch");
  };
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-token",
    capabilities: {
      native_save: true,
      open_logs: true,
      open_data: true,
      reveal_artifact: true,
      diagnostics_export: true,
    },
    versions: {},
  });

  try {
    const result = await downloadApiArtifact(
      "/projects/project_123/artifact",
      "model-config.json"
    );

    assert.equal(result, "native");
    assert.deepEqual(calls, [
      {
        command: "save_api_artifact",
        args: {
          api_path: "/projects/project_123/artifact",
          suggested_name: "model-config.json",
        },
      },
    ]);
  } finally {
    __resetRuntimeConfigForTests();
    globalThis.fetch = previousFetch;
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("desktop artifact reveal uses the narrow native metadata command", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async (command: string, args?: Record<string, unknown>) => {
            calls.push({ command, args });
          },
        },
      },
    },
  });
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-token",
    capabilities: {
      native_save: true,
      open_logs: true,
      open_data: true,
      reveal_artifact: true,
      diagnostics_export: true,
    },
    versions: {},
  });

  try {
    await revealApiArtifact("/projects/project_123/artifact");
    assert.deepEqual(calls, [
      {
        command: "reveal_api_artifact",
        args: { api_path: "/projects/project_123/artifact" },
      },
    ]);
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("desktop text save reports native cancellation without browser fallback", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI__: {
        core: {
          invoke: async (command: string, args?: Record<string, unknown>) => {
            calls.push({ command, args });
            return null;
          },
        },
      },
    },
  });
  __setRuntimeConfigForTests({
    environment: "desktop",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "memory-only-token",
    capabilities: {
      native_save: true,
      open_logs: true,
      open_data: true,
      reveal_artifact: true,
      diagnostics_export: true,
    },
    versions: {},
  });

  try {
    const result = await downloadTextFile("config.json", "{}");
    assert.equal(result, "cancelled");
    assert.deepEqual(calls, [
      {
        command: "save_file",
        args: {
          suggested_name: "config.json",
          bytes: [123, 125],
        },
      },
    ]);
  } finally {
    __resetRuntimeConfigForTests();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("web artifact download preserves authenticated browser fallback", async () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const previousDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const previousFetch = globalThis.fetch;
  const previousCreateObjectUrl = URL.createObjectURL;
  const previousRevokeObjectUrl = URL.revokeObjectURL;
  const actions: string[] = [];
  const anchor = {
    href: "",
    download: "",
    rel: "",
    click: () => actions.push("click"),
    remove: () => actions.push("remove"),
  };
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      setTimeout: (callback: () => void) => {
        callback();
        return 1;
      },
    },
  });
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: {
      createElement: (tag: string) => {
        assert.equal(tag, "a");
        return anchor;
      },
      body: {
        appendChild: (value: unknown) => {
          assert.equal(value, anchor);
          actions.push("append");
        },
      },
    },
  });
  URL.createObjectURL = () => "blob:artifact";
  URL.revokeObjectURL = (url: string) => actions.push(`revoke:${url}`);
  globalThis.fetch = async (input, init) => {
    assert.equal(input, "http://127.0.0.1:43123/api/v1/projects/project_123/artifact");
    assert.equal(new Headers(init?.headers).get("X-LLM-Studio-Token"), "web-token");
    return new Response("artifact-data", { status: 200 });
  };
  __setRuntimeConfigForTests({
    environment: "web",
    apiBaseUrl: "http://127.0.0.1:43123/api/v1",
    runtimeToken: "web-token",
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
    const result = await downloadApiArtifact(
      "/projects/project_123/artifact",
      "model-config.json"
    );
    assert.equal(result, "browser");
    assert.equal(anchor.href, "blob:artifact");
    assert.equal(anchor.download, "model-config.json");
    assert.deepEqual(actions, ["append", "click", "remove", "revoke:blob:artifact"]);
  } finally {
    __resetRuntimeConfigForTests();
    globalThis.fetch = previousFetch;
    URL.createObjectURL = previousCreateObjectUrl;
    URL.revokeObjectURL = previousRevokeObjectUrl;
    if (previousDocument) {
      Object.defineProperty(globalThis, "document", previousDocument);
    } else {
      Reflect.deleteProperty(globalThis, "document");
    }
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});
