# Website guide

The dashboard lives in `web/` and is deployed at https://lo-ra-experiment.vercel.app.

## What the site shows

The top-level **Current results** tab presents the five-seed Qwen3 and T5Gemma 2 study. Its sections are:

| Section | What readers can do |
|---|---|
| Results | Choose Arabic or Malay; compare both models and read the verdict. |
| Compare | View standard and reward training side by side, with error bars, exact values, and expandable intervals. |
| Reward checks | Choose a model and language; inspect controls, answer length, matched-length results, and late reward activity. |
| Answers | Browse five questions per language with Previous/Next, the reference and both models’ standard/reward answers. |

The separate **Earlier study** tab retains the previous mT5 dashboard and labels it as a legacy baseline. The footer on Current results states the limits: models differ in size, reward contrasts are exploratory, human evaluation covers the earlier mT5 study, and the manuscript is under review.

## Data flow

```text
results/hero_v2/ ──► scripts/build_hero_web_data.py ──► web/public/data/hero.json ──► Current results
results/          ──► scripts/build_web_data.py      ──► web/public/data/results.json ──► Earlier study
```

Both JSON files are tracked so Vercel can build without running the analysis. After updating hero-v2 results, run `python3 scripts/build_hero_web_data.py` and commit the refreshed JSON with the source changes.

## Develop and build

```bash
cd web
npm ci
npm run dev      # local development at http://localhost:5173
npm run build    # type-check and build into web/dist
```

## Deploy

The repository-root `vercel.json` builds `web/` and serves `web/dist` with an SPA rewrite. Pushing the site changes and generated data to `main` triggers the Vercel deployment. Vite uses a relative base path so the same build works on the configured static hosts.

`web/api/api.py` is an optional FastAPI service for live generation; the dashboard reads the saved results and does not need it.
