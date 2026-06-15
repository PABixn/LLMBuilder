# LLM Studio Desktop

Tauri v2 shell for the static LLM Studio frontend and supervised Python runtime.

The shell never navigates the main window to the backend. It keeps the bundled
asset origin stable and provides the frontend with a loopback API URL plus a
memory-only per-launch token through `runtime_bootstrap`.

Development:

```bash
make -C apps/llm-studio install-desktop
make -C apps/llm-studio desktop-check
make -C apps/llm-studio desktop-build-runtime
make -C apps/llm-studio desktop-dev
```

`desktop-dev` uses the target-native linked development runtime produced by
`desktop-build-runtime`. Production bundles receive a reviewed portable runtime
as a Tauri resource.
