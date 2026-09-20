# RAG Faithfulness Lab — Frontend

Phase 1 static demo UI for comparing mT5 and Qwen on retrieval-grounded QA.

## Stack

- Vite + React + TypeScript
- Tailwind CSS v4 (`@tailwindcss/vite`)
- Recharts + Lucide React

## Run locally

```bash
# from repo root — start the mock API first
pip install fastapi "uvicorn[standard]"
uvicorn backend.api:app --reload --port 8000

# then the UI
cd frontend
npm install
npm run dev
```

- UI: http://localhost:5173
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

Vite proxies `/api` to `http://localhost:8000`. The Compare button currently calls `http://localhost:8000/api/compare` directly (CORS allows localhost:5173). If the API is down, the UI falls back to built-in demo metrics.

## Build

```bash
npm run build
```

Architecture and phases: [`docs/WEBSITE_GUIDE.md`](../docs/WEBSITE_GUIDE.md).
