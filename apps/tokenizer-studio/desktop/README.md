# Tokenizer Studio Desktop Shell

Tauri shell responsible for:

- Starting and supervising local FastAPI sidecar.
- Waiting for sidecar health readiness.
- Routing users into `http://127.0.0.1:<dynamic_port>/`.
- Rotating backend logs in app-data.
- Gracefully terminating sidecar on app exit.

## Commands

```bash
npm install
npm run icons
npm run dev
npm run build
npm run tauri dev
npm run tauri build
```

## Runtime Contract

The shell expects a runtime bundle with:

```text
runtime/
  python/
  app/
  web/
  VERSION
```

By default runtime lookup order is:

1. `TOKENIZER_STUDIO_RUNTIME_DIR`
2. `<app_data>/runtime/current`
3. `<app_data>/runtime/default`
4. Development fallback directories near current working directory

If no runtime is found and `TOKENIZER_STUDIO_RUNTIME_MANIFEST_URL` is set, the shell downloads and installs runtime from that manifest.

## Key Tauri Commands

- `start_backend`: starts sidecar or returns existing healthy instance.
- `backend_status`: returns current healthy backend metadata if available.
- `stop_backend`: shuts down sidecar process.

## Startup UX

The shell launches a local splash UI first. It shows:

- startup status while backend initializes
- explicit error detail when startup fails
- retry action for user recovery
