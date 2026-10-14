# Website guide

The dashboard lives in `web/` and is deployed at https://lo-ra-experiment.vercel.app.

## Data flow

```text
results/  ──►  scripts/build_web_data.py  ──►  web/public/data/results.json  ──►  React pages
```

`results.json` is tracked in git so that Vercel and GitHub Pages can build without Python. After any change to `results/`, run `make web-data` and commit the new JSON.

## Pages (`web/src/pages/`)

| Tab | File | Content |
|---|---|---|
| Overview | `Overview.tsx` | headline numbers, condition bars, study status |
| Corrected study | `Corrected.tsx` | per-seed strip plot, condition means, pre-specified contrasts, bootstraps, checkpointing |
| T5Gemma 2 & other backbones | `Backbones.tsx` | T5Gemma 2, Qwen3 and first-study mT5 (original protocol, flagged as not comparable) |
| Validation | `Validation.tsx` | independent scorer, human judgements, answer length, run diagnostics |
| Worked examples | `Worked.tsx` | five items with both answers, both scorers and annotator ratings |
| First-study logs | `Examples.tsx` | seed-42 prediction browser (mT5 vs Qwen3) |

## Develop

```bash
cd web
npm ci
npm run dev      # http://localhost:5173
npm run build    # type-check + production build into web/dist
```

## Deploy

* **Vercel:** the repository-root `vercel.json` installs and builds from `web/` and serves `web/dist` with an SPA rewrite.
* **GitHub Pages:** `.github/workflows/deploy-pages.yml` builds `web/` on pushes that touch `web/**`.
* Vite uses `base: "./"` so both hosts work from the same build.

## Optional backend

`web/api/api.py` is an optional FastAPI service for live generation. The dashboard does not need it.
