# Tokenizer Studio

Tokenizer Studio is a local-first tokenizer training product:

- `web/`: Next.js frontend (configuration UI + job monitoring).
- `api/`: FastAPI backend (validation + training sidecar service).
- `desktop/`: Tauri shell that launches the local backend sidecar for non-technical desktop use.

## Developer Local Run

### 1. Backend (FastAPI)

```bash
cd apps/tokenizer-studio/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API base URL: `http://127.0.0.1:8000/api/v1`

### 2. Frontend (Next.js)

```bash
cd apps/tokenizer-studio/web
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:3000`

## Production/Desktop Runtime Notes

- Backend defaults to localhost-only (`127.0.0.1`) with configurable port.
- API exposes both `/health` and `/api/v1/health`.
- Built frontend assets can be served directly by backend (`TOKENIZER_STUDIO_WEB_DIST_DIR`).
- Storage defaults are OS app-data directories:
  - macOS: `~/Library/Application Support/TokenizerStudio/`
  - Windows: `%AppData%\TokenizerStudio\`
  - Linux: `~/.local/share/TokenizerStudio/`
- Default subfolders:
  - `db/`
  - `uploads/`
  - `artifacts/tokenizers/`
  - `logs/`
  - `cache/huggingface/`

Environment overrides are documented in `apps/tokenizer-studio/api/.env.example`.

## Desktop Build Pipeline

Build scripts live in `apps/tokenizer-studio/desktop/scripts/`.

Example (macOS arm64):

```bash
apps/tokenizer-studio/desktop/scripts/build_web.sh
apps/tokenizer-studio/desktop/scripts/build_runtime_macos.sh macos-arm64
apps/tokenizer-studio/desktop/scripts/smoke_test_runtime.sh macos-arm64
cd apps/tokenizer-studio/desktop
npm install
npm run tauri build
```

One-command pipeline:

```bash
apps/tokenizer-studio/desktop/scripts/build_desktop.sh macos-arm64
```

## Typical Workflow In App

1. Configure tokenizer + dataset settings.
2. Validate configs.
3. Start training.
4. Monitor progress and tokenizer stats.
5. Download tokenizer JSON artifact.
