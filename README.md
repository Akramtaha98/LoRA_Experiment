<div align="center">

# NLI-Faithfulness Reward × LoRA Adapters

**Does an NLI entailment reward make small multilingual QA models more faithful? A corrected, pre-specified study on Arabic and Malay.**

[Live results dashboard](https://lo-ra-experiment.vercel.app) · [Findings](#findings) · [Repository layout](#repository-layout) · [Reproduce](#reproduce) · [Cite](#citation)

</div>

A leave-one-out REINFORCE objective with an NLI entailment reward is added to cross-entropy training of LoRA, QLoRA and DoRA adapters on **mT5-base**, evaluated on **Arabic (XQuAD)** and **Malay (Belebele `zsm_Latn`)** closed-passage question answering. The headline study is the **corrected, pre-specified v2 run** (228 run records, 8 seeds per cell). An earlier study that reported the reward *harming* LoRA and DoRA is superseded: that effect was an evaluation artefact of gradient checkpointing.

## Findings

| | Result (corrected study, mT5-base, 8 seeds) |
|---|---|
| NLI score, Arabic | **+0.601** (95% CI +0.561 to +0.641): about 0.15 → 0.72–0.77 |
| NLI score, Malay | **+0.008** (95% CI −0.001 to +0.016): no reliable change |
| Controls | Zero-weight and shuffled-reward runs stay within 0.01 of cross-entropy training (equivalence shown), so the gain is specific to the reward |
| Independent scorer | `joeddav/xlm-roberta-large-xnli`, not used in training, reproduces the Arabic gain (pooled +0.257) |
| Other measures | Arabic exact match falls 0.075 → 0.004; answers grow from 2.9 to 16.5 words; 79.6% are more than 3× the gold length |
| Length | Holding answer-length bins fixed removes about three quarters of the estimated gain (descriptive, not causal) |
| Human check | Two blinded annotators, 60 items per language, κ = 0.38 (support): one rates reward-trained Arabic answers as less supported, the other sees no difference; correctness unchanged |
| Checkpointing | Neutral in eval mode. A checkpointed T5 left in train mode decodes from the last token only, which produced the first study's spurious −0.16 |

The NLI score is also the training reward, so the pattern is consistent with over-optimisation of the score. It is not evidence that the answers are less faithful, and the human evidence is limited (two annotators, low agreement).

### Model comparison (which model wins)

With the same adapters (DoRA, QLoRA) and seeds (42, 123, 2026), standard fine-tuning (config B), exact match, Arabic and Malay combined: **Qwen3-0.6B-Base 0.480**, T5Gemma 2 (270M) 0.368, mT5-base 0.166 (token F1 0.592 / 0.510 / 0.256). Qwen3 leads the runner-up by +0.111 EM (95% CI +0.105 to +0.118). The NLI score is not used to rank models because it is the training reward. The models differ in size, architecture and pre-training, and Qwen3 and T5Gemma 2 use the original protocol, so this ranks the fine-tuned systems, not the architectures. Side-by-side answers for five shared questions per language are in `results/compare/outputs.json` (`python scripts/build_compare_outputs.py --runs_v2 runs_v2`).

### Other backbones (original protocol, not comparable)

T5Gemma 2 (`google/t5gemma-2-270m-270m`) and Qwen3-0.6B-Base were run with the first-study code path (3 seeds, before the training and evaluation fixes). Neither shows a detectable reward effect on the NLI score (T5Gemma 2 pooled +0.0005, 95% CI −0.0095 to +0.0105). No corrected-protocol run exists for them, so nothing here says whether the reward would help them once the fixes are applied. They are included in the dashboard and in `results/first_study/`.

## Repository layout

```text
src/lora_nli/            library code
  matrix.py              training + evaluation (LORA_V2=1 selects the corrected path)
  analyze.py             pre-specified seed-paired analysis
  independent_eval.py    independent NLI scorer
  human_eval.py          human-validation package builder / scorer
  qwen_matrix.py         Qwen3 variant (original protocol)
  t5gemma2_matrix.py     T5Gemma 2 variant (original protocol)
scripts/                 run_v2.py (job scheduler), estimator_checks.py, ckpt_repro.py,
                         summarize_answer_length.py, build_web_data.py, legacy/ (first-study runners)
tests/                   CPU unit tests for the training path, analysis, scorers
results/
  corrected_v2/          228 run records, contrasts, bootstraps, length analysis, examples
  independent_scorer/    independent-scorer summary
  human_validation/      human-validation labels and scripts
  first_study/           mT5, Qwen3 and T5Gemma 2 original-protocol records
analysis_first_study/    scripts and tables of the first study (incl. T5Gemma 2 reanalysis)
docs/                    runbooks, original-protocol README, project and website guides
web/                     React + TypeScript dashboard (Vite, Tailwind, Recharts) and optional FastAPI backend
```

Per-example predictions for the 228 runs are in `runs_v2.tar.gz`, which is not tracked by git (attach it to a GitHub release).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pinned.txt && pip install -e .
make test                                  # CPU tests; each prints ALL_PASS / *_TEST_PASS
```

**Rebuild the dashboard data from saved results (CPU, seconds).**

```bash
python scripts/build_web_data.py           # writes web/public/data/results.json
cd web && npm ci && npm run dev
```

**Re-run the corrected study (GPU).** Restartable; finished jobs are skipped. See `docs/runbooks/RUNBOOK_v2.md` for the full protocol and the pre-specified analysis.

```bash
python scripts/run_v2.py --stage core     --slots-per-gpu 2 --nli-gpu   # 48 jobs, B + C
python scripts/run_v2.py --stage controls --slots-per-gpu 2 --nli-gpu   # 96 jobs, λ₁=0 and shuffled reward
python scripts/run_v2.py --stage ckptfact --slots-per-gpu 2 --nli-gpu   # 18 jobs, checkpointing factorial
python -m lora_nli.analyze runs_v2 --out results/corrected_v2 --margin 0.01
python -m lora_nli.independent_eval runs_v2 --out results/independent_scorer
```

Pre-specified design: seeds 42, 123, 2026, 7, 11, 13, 17, 19; seed-paired t intervals (7 df); Holm correction over the three adapters; equivalence margin ±0.01 (90% CI); item- and passage-cluster bootstraps.

## Web dashboard

`web/` is a static React results explorer published on Vercel. Four sections — Overview, Experiments, Compare answers, and Leaderboard — expose the saved original A–D matrix and corrected mT5 conditions, with protocol labels, measured scores, source-linked examples, and explicit missing-data states. The 100 shared questions remain available in Compare answers. Study evidence includes controls, independent scoring, human validation, uncertainty and length analysis.

`python scripts/build_web_data.py` rebuilds the analysis summary; `python scripts/build_experiment_catalog.py` rebuilds the experiment index and lazily loaded answer files. The latter follows the checked-in per-seed analysis tables and full-evaluation A/D logs, preserving all actually saved outputs. Corrected per-cell predictions require the untracked prediction archive and are labelled unavailable in the website. No missing combinations or answers are fabricated. See [the website guide](docs/WEBSITE_GUIDE.md) for provenance and checks.

## Data and licence

Datasets: XQuAD (Arabic) and Belebele `zsm_Latn` (Malay), used under their own licences. Reward model `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`; independent scorer `joeddav/xlm-roberta-large-xnli`; base model `google/mt5-base`. Code is released under the licence in `LICENSE`.

## Citation

The manuscript is under review. Until it appears, please cite the repository:

```bibtex
@misc{zeyad2026nlireward,
  author = {Taha Zeyad, Akram},
  title  = {{NLI}-Faithfulness Reward {\texttimes} {LoRA} Adapters: a corrected study on Arabic and Malay question answering},
  year   = {2026},
  howpublished = {\url{https://github.com/Akramtaha98/LoRA_Experiment}}
}
```
