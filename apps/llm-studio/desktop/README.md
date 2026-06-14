# LLM Studio Desktop

Tauri v2 shell for the static LLM Studio frontend and supervised Python runtime.

The shell never navigates the main window to the backend. It keeps the bundled
asset origin stable and provides the frontend with a loopback API URL plus a
memory-only per-launch token through `runtime_bootstrap`.

Development:

```bash
make -C apps/llm-studio desktop-check
make -C apps/llm-studio desktop-dev
```

Set `LLM_STUDIO_RUNTIME_DIR` to a validated runtime built by
`scripts/desktop/build_runtime.py`. Production bundles receive that runtime as a
Tauri resource.

