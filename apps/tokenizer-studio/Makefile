.PHONY: api web

api:
	cd api && python -m uvicorn app.main:app --reload --port 8000

web:
	cd web && npm run dev
