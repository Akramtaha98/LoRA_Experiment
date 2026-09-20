# mT5 vs Qwen RAG Comparison Website

## 1. Purpose

This website provides an interactive interface for comparing **mT5** and **Qwen** on retrieval-grounded question answering.

The website is designed to show:

- Generated answers from mT5 and Qwen side by side.
- Context faithfulness.
- Exact Match.
- Token F1.
- Inference latency.
- Answer length.
- Differences between model behavior.
- Hallucination examples.
- Knowledge-conflict examples.
- Existing experiment results.
- LoRA-family configurations such as QLoRA, AdaLoRA, DoRA, and VeRA.

The website UI is written in **English**.

The existing Arabic and Malay experimental results remain unchanged. English examples are treated as interactive demonstrations unless a separate English benchmark is evaluated.

---

## 2. Recommended Architecture

The application should use two layers:

```text
Browser
   |
   v
React + TypeScript
   |
   | HTTP / JSON
   v
FastAPI
   |
   +---- mT5
   |
   +---- Qwen
   |
   +---- Faithfulness Evaluator
   |
   +---- Existing Experiment Results
```

Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide React.

Backend: Python, FastAPI, PyTorch, Transformers, PEFT, Pandas.

---

## 3. Suggested Repository Structure

```text
LoRA_Experiment/
|
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   │
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       │
│       ├── api/
│       │   └── comparison.ts
│       │
│       ├── components/
│       │   ├── Header.tsx
│       │   ├── MetricCard.tsx
│       │   ├── ModelPanel.tsx
│       │   ├── ComparisonChart.tsx
│       │   └── ExampleSelector.tsx
│       │
│       └── data/
│           └── examples.ts
│
├── backend/
│   ├── api.py
│   ├── inference.py
│   ├── metrics.py
│   ├── model_manager.py
│   └── results.py
│
├── lora_experiment_matrix.py
├── qwen_experiment_matrix.py
│
├── experiment_results/
├── experiment_results_qwen/
│
├── docs/
│   ├── PROJECT_GUIDE.md
│   └── WEBSITE_GUIDE.md
│
└── requirements.txt
```

---

## 4. Website Pages

### Dataset (Arabic / Malay) — primary offline path

Curated paired examples from logged experiment JSONL (no GPU, works on static Vercel hosting):

- Source files: `experiment_results/checkpoint_full_qlora_{arabic,malay}.jsonl` (mT5 `B_ce_lora` / QLoRA) and `experiment_results_qwen/checkpoint_full_qwen_qlora_{arabic,malay}.jsonl` (Qwen `C_composite_lora` / QLoRA), seed 42.
- Static asset: `frontend/public/data/dataset-examples.json` (12 Arabic + 12 Malay).
- UI: language filter → example picker → context / question / **reference** → side-by-side mT5 vs Qwen answers + faithfulness / EM / F1.
- **Best model rule:** higher faithfulness first, then Exact Match, then Token F1; equal on all three → **tie**. Implemented in `frontend/src/lib/scoring.ts`.

### Compare

English interactive demos (Simple Fact, Distractor, Hallucination Trap, Knowledge Conflict). Optional FastAPI `/api/compare`; without a backend the UI falls back to the built-in Knowledge Conflict sample. Kept for local lab use; Dataset path does not need the API.

### Theme

Dark and light modes via `data-theme` on `<html>`, toggle in the nav, preference stored in `localStorage` (`rag-lab-theme`), default from `prefers-color-scheme`.

### Results / About

Still planned for later phases (full CSV dashboards, research write-up).

---

## 5–7. Demonstration Cases

Include English demos for Simple Fact (Marie Curie / 1911), Distractor (Venus), Hallucination Trap (PhD not in context), and Knowledge Conflict (capital of France = Lyon).

---

## 8. Metrics

- **Faithfulness**: NLI entailment of answer given context (same pipeline as experiments when possible).
- **Exact Match**: normalized prediction vs reference identity.
- **Token F1**: token overlap with reference.
- **Latency**: `time.perf_counter()` per model independently.

---

## 9. Scientific Comparison Rules

Keep identical context, question, reference, examples, max generation length, metrics, splits, and seeds where applicable. Do not unequalize prompts across models.

---

## 10. Training vs Website Inference

Training stays offline. The website only loads saved checkpoints and runs inference + evaluation. Never train on Compare.

---

## 11. Model Loading Strategy

Load one selected mT5 config + one selected Qwen config at a time. Cache if memory allows; otherwise unload before switching.

---

## 12. Development Order

1. React interface + static demonstration data
2. FastAPI + frozen mT5/Qwen
3. Exact Match, Token F1, Latency
4. NLI faithfulness
5. Results dashboard from existing CSV/JSONL
6. LoRA-family trained checkpoints
7. Production deployment

---

## 13. API Contract

POST `/api/compare` request:

```json
{
  "context": "...",
  "question": "...",
  "reference": "...",
  "mt5_configuration": "frozen",
  "qwen_configuration": "frozen"
}
```

Response:

```json
{
  "mt5": {
    "answer": "Lyon",
    "faithfulness": 0.97,
    "exact_match": 1.0,
    "f1": 1.0,
    "latency_ms": 713
  },
  "qwen": {
    "answer": "Paris",
    "faithfulness": 0.31,
    "exact_match": 0.0,
    "f1": 0.0,
    "latency_ms": 482
  }
}
```

Also expose `GET /api/health`.

---

## 14. Local Development

```bash
# backend
uvicorn backend.api:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Swagger: http://localhost:8000/docs

---

## 15. Recommended UI

Dark research-dashboard design: Compare / Examples / Results / About; context/question/reference inputs; configuration selects; side-by-side mT5 vs Qwen panels; metric chart.

---

## 16. Production Deployment (Vercel)

Deploy the **frontend** as a static SPA (Root Directory = `frontend`).

```bash
cd frontend
npm install
npm run build
# Vercel: import the GitHub repo, set Root Directory to frontend,
# Framework Preset = Vite, Build = npm run build, Output = dist
```

- `frontend/vercel.json` rewrites unknown paths to `index.html` for SPA routing.
- Dataset comparison loads `/data/dataset-examples.json` from `public/` — **no FastAPI required** on Vercel.
- Live Compare still needs a separate GPU/API host when you wire real models; until then the Compare tab uses the mock/fallback only.

Do not run model inference in the frontend-only host.

---

## 17. Final Goal

An interactive RAG faithfulness laboratory where visitors supply evidence, ask a question, run both models, inspect answers, compare faithfulness vs correctness, explore demos, review experiment results, and compare adapter configs.
