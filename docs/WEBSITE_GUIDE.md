# Website guide

The website lives in `web/` and is deployed at https://lo-ra-experiment.vercel.app.

## Visitor experience

The landing page (`web/src/pages/Home.tsx`) is a guided answer explorer. One Arabic/Malay control updates every section:

1. A featured question shows unchanged saved outputs for all three models, reference-match labels, observed-output explanations, and highlighted evidence. Featured examples include an optional human-written English guide.
2. A short explanation introduces closed-passage QA, standard versus NLI-reward training, and the difference between exact match, token F1 and entailment.
3. The training comparison uses existing matched B/C records for the same question, model, adapter and seed 42. Ten exported questions are available (five per language), with QLoRA and DoRA. Missing pairs stay explicitly unavailable. Original and corrected protocols are labelled separately.
4. The example browser exposes the full saved sample, five examples per page. Search includes questions, references, outputs and passages. Filters cover every model symmetrically. All-model reference-match filters require all three outputs. “Different answers” compares output wording, not semantic correctness. Downloads contain the original records selected by the filters.
5. Overall results show the full evaluation with bars on a common 0–100% scale. Research details retain the complete stored precision and explain the separate corrected mT5 study and protocol limitations.

Explanations describe observable text and scores. They do not assert internal model reasoning. Word overlap is never labelled “partly correct.” An output occurring in the passage is not automatically a correct answer to the question. Highlighted spans show literal matches, not a claim of semantic entailment. The Gingrich/Romney example has a specific reviewed explanation; other examples use conservative descriptions of matches, output length and passage overlap. No generative explanation service runs on the site.

## Data flow and integrity

- `results/` → `scripts/build_web_data.py` → `web/public/data/results.json`.
- `scripts/build_compare_examples.py` → `results/compare/examples.json` and `web/public/data/compare-examples.json`.
- Existing `compare.outputs` records contain the matched training pairs; `scripts/build_compare_outputs.py` documents their lineage.
- Raw answers, reference answers and metric values are displayed from the exports without rewriting them. Six-digit or percentage display formatting does not replace the original stored values, which remain available in source details.
- Per-answer run IDs and source-log paths are absent from the current example export. The UI states this limitation instead of inventing provenance. QLoRA/seed-42 metadata follows the export scripts.
- The 100-question export and the full evaluation are different populations, sometimes with different objectives. The UI labels these scopes next to the corresponding results.
- Existing analysis-page modules remain available in source; the main application renders the answer explorer.

After changes to `results/`, run `make web-data` and commit the generated JSON. Refresh example exports from real saved logs using the documented script; do not manually fill missing outputs.

## Develop and verify

```bash
cd web
npm ci
npm test          # behavioral regressions and integrity checks on real saved examples
npm run lint     # zero lint warnings
npm run build    # type-check and production build
npm run dev
```

Tests cover third-model filter handling, missing outputs, exact-match versus overlap labels, explanation wording, passage preservation during highlighting, English-guide selection, and real training-pair availability. CI runs these checks on pull requests.

## Deployment

The repository-root `vercel.json` builds `web/` and serves `web/dist`. Vercel can create a preview from a pull-request branch; merging to the configured production branch updates the production site. The GitHub Pages workflow also runs on qualifying pushes to main. Vite uses relative asset paths for compatibility with both hosts.

The optional FastAPI backend is not required: all comparisons on the landing page use saved experimental outputs.
