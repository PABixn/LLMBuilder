# Tokenizer Studio (Local)

A local full-stack app for training the repository's fully configurable tokenizer:

- `web/`: Next.js frontend (configuration UI + job monitoring)
- `api/`: FastAPI backend (validation + training job execution)

## 1. Backend (FastAPI)

```bash
cd apps/tokenizer-studio/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

API base URL: `http://127.0.0.1:8000/api/v1`

## 2. Frontend (Next.js)

```bash
cd apps/tokenizer-studio/web
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:3000`

## 3. Typical Workflow

1. Open the frontend.
2. Configure tokenizer + dataloader in Form Builder mode.
3. Click **Validate Configs**.
4. Click **Start Training**.
5. Monitor progress and download the tokenizer JSON artifact when complete.

## Notes

- Backend jobs train tokenizers with the existing modules under `tokenizer/`.
- Tokenizer evaluation always runs on the same dataset configuration used for training.
- API config templates and schemas are served from `api/templates/`.
- Artifacts are saved by default to `apps/tokenizer-studio/api/artifacts/tokenizers/`.
- Training jobs and upload metadata are persisted in SQLite at `apps/tokenizer-studio/api/data/tokenizer_studio.db` by default.
- Override output/storage behavior with environment variables in `api/.env.example` (relative paths are resolved from `api/`).
- If you see `ImportError: cannot import name 'load_dataset' from 'datasets'`, you are likely using a different Python than your project venv. Use `python -m uvicorn ...` from the activated environment.
