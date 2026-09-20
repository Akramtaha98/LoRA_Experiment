# RAG Faithfulness Lab — Frontend

Interactive UI for comparing **mT5** and **Qwen** on retrieval-grounded QA.

## Features

- **Dataset** tab: curated Arabic + Malay logged examples (100+) with side-by-side answers, metrics, and best-model highlight (faithfulness → EM → F1).
- **Demos** tab: English scenarios with offline mock results (updates without FastAPI).
- Compact lab layout + dark/light theme.

## Stack

- Vite + React + TypeScript
- Tailwind CSS v4 (`@tailwindcss/vite`)
- Recharts + Lucide React

## Run locally

```bash
cd frontend
npm install
npm run dev
```

Optional mock API (Compare tab only):

```bash
# from repo root
pip install fastapi "uvicorn[standard]"
uvicorn backend.api:app --reload --port 8000
```

- UI: http://localhost:5173
- Dataset JSON: http://localhost:5173/data/dataset-examples.json
- Vite proxies `/api` → `http://localhost:8000`

## Build / Vercel

```bash
npm run build
```

Set Vercel **Root Directory** to `frontend`. Output directory: `dist`. SPA rewrites are in `vercel.json`. Dataset compare works without a backend.

Architecture: [`docs/WEBSITE_GUIDE.md`](../docs/WEBSITE_GUIDE.md).
