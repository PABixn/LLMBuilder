A local-first studio for LLM architecture design and tokenizer training.

Designed specifically for simple and analytics-friendly experiments involving creating, pretraining, fine-tuning and evaluating deeply customized LLMs.

See `docs/dataloader.md` for the streaming tokenizer dataloader (HF streaming datasets, weighted mixing, and text-budgeted runs).
See `docs/training-dataloader.md` for the training dataloader (tokenization, packing, and training batches).

Primary merged app:
- `apps/llm-studio/web`: unified Next.js frontend with model + tokenizer workspaces.
- `apps/llm-studio/api`: unified FastAPI backend with model and `/api/v1/tokenizer/*` routes.
