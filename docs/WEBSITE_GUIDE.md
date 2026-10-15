# Website guide

The website lives in `web/` and is deployed at https://lo-ra-experiment.vercel.app.

## Visitor experience

The main application is a simple three-tab interface in `web/src/pages/Home.tsx`:

- **Start here** introduces the task and offers a direct button into all 100 saved examples or a guided example. Label explanations, training background and full evaluation results stay collapsed until requested.
- **100 examples** shows one question at a time and all three model outputs together. All 100 source records are present in the question picker by default. Users can jump directly to any question, use Previous/Next at either end of the card, search, filter by language (50 Arabic and 50 Malay), or filter by model/reference-match outcomes. Each answer's explanation and scores open under “Why this result?”. Evidence, English guides for featured examples, raw records and downloads are optional expandable sections. Outputs are never truncated or regenerated.
- **Training comparison** retains the existing ten saved matched B/C questions with model and adapter selectors. These are separate from the 100-question example export; missing pairs remain explicitly unavailable. Model protocols and seed 42 remain labelled.

Tabs support arrow keys, Home/End and direct `#examples` / `#training` links. Switching tabs preserves the selected example, filters and training controls. The sticky tab bar keeps navigation accessible while reading long answers. Panel and question changes use short slide/fade transitions, the question-position indicator animates between selections, and controls have gentle hover feedback. All motion switches off for `prefers-reduced-motion`.

The language selector is shared between the examples and training tabs. The “all” setting exposes all 100 examples and both languages in the saved training question picker. Filters compare recorded scores or output wording and never invent semantic correctness. Literal reference and answer highlights preserve the entire source text. Explanations describe visible output behaviour rather than hidden model reasoning.

The main page deliberately avoids a long stack of questions and visible research controls. Detailed results, protocol limitations, stored precision and source links remain accessible through expandable sections.

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
