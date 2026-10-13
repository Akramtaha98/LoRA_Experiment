# RUNBOOK (v2): corrected training path, corrected evaluation, pre-specified analysis

This is the only runbook shipped with the code. `RUNBOOK_legacy_pgfix.md` documents the first (provisional) study.
Everything below was unit-tested on CPU (`tests_v2.py`, `tests_analysis.py`, `tests_human_eval.py`, `tests_indep.py`);
none of the v2 GPU runs has been executed yet.

## 0. What changed and why (LORA_V2=1)
| # | Defect found in the first study | v2 behaviour | CPU test |
|---|---|---|---|
| 1 | Training scored the decoded *prompt* (question + Malay options + passage) as the NLI premise; evaluation scored the passage | the raw passage travels through the dataset and collator and is the premise in both | T1 (negative control: the legacy path scores options) |
| 2 | Held-out generation ran without `model.eval()` (dropout on; with checkpointing enabled, `generate()` decodes from the last token only) | `model.eval()` + mode assertions before evaluation | T2, `ckpt_generate_mt5_repro.py`, `ckpt_repro.py` |
| 3 | Policy sampling and re-scoring drew independent dropout masks; estimator verified only in eval mode on a tiny T5 | policy branch (sampling and re-scoring) runs with dropout off, train mode restored for CE; tested on a PEFT-wrapped model with dropout > 0, bf16, checkpointing | T3 (train-mode mismatch 17-22 nats, eval-mode 1e-5) |
| 4 | generate() used +-1e4, the scorer +-1e9 as non-finite fill | one shared sanitizer, counters in every run record | T4 |
| 5 | AdaLoRA never called `update_and_allocate` (it behaved as fixed rank 12) | callback after each optimizer step; rank trajectory logged | T5 (432 -> 288 non-zero singular values; constant without the callback) |
| 6 | Checkpointing flag not recorded; eval examples had no stable IDs; 4 held-out items shared a passage with training; truncation unlogged | `grad_ckpt`, `src_id`, `passage_id`, `passage_in_train`, prompt-truncation flags, per-example predictions (`preds/*.jsonl`); headline metrics exclude overlapping-passage items (`*_all` keeps them) | T6 |
| 7 | checkpointing applied to Config C only, non-QLoRA only | a pure memory switch for B and C alike, off by default (`--grad_ckpt on` to enable) | T7 |

## 1. Environment (stage 0)
```bash
python3 -m venv ~/venv_v2 && source ~/venv_v2/bin/activate
pip install -r requirements_pinned.txt
pip freeze > env_freeze.txt                 # authoritative environment record; ship it with the data
python tests_v2.py && python tests_analysis.py && python estimator_checks_fixed.py   # each must end in ALL_PASS / *_TEST_PASS
python ckpt_generate_mt5_repro.py           # on the GPU node: confirms the generate()/checkpointing defect on YOUR stack (expected: train/ckpt == last-token-only)
```

## 2. Runs (restartable; finished jobs are skipped; add --dry-run to list)
Pre-specified: 8 seeds per cell = 42, 123, 2026, 7, 11, 13, 17, 19. Do not add or drop seeds after seeing results.
```bash
python run_v2.py --stage core       --slots-per-gpu 2 --nli-gpu   # 48 jobs: dora/lora/qlora x 2 langs x 8 seeds, B + C
python run_v2.py --stage controls   --slots-per-gpu 2 --nli-gpu   # 96 jobs: lambda1=0 and shuffled reward, same cells
python run_v2.py --stage ckptfact   --slots-per-gpu 2 --nli-gpu   # 18 jobs: B and C with checkpointing ON (CE x checkpointing factorial)
python run_v2.py --stage policymode --slots-per-gpu 2 --nli-gpu   # 18 jobs: legacy policy branch (dropout on) for C
python run_v2.py --stage adaptive   --slots-per-gpu 2 --nli-gpu   # 12 jobs: AdaLoRA (rank reallocation active) and VeRA
python run_v2.py --stage baselines  --slots-per-gpu 2 --nli-gpu   # 8 jobs: frozen mT5 and full fine-tuning (competence check)
```
Every run also re-scores the same adapter in the legacy evaluation state (train mode) and stores it as `trainmode_eval_f_faith`;
this isolates the evaluation artifact on the *same* weights (`LORA_EVAL_ALSO_TRAINMODE=0` disables it).

## 3. Analysis (pre-specified; 95% CIs primary)
```bash
python analyze_v2.py runs_v2 --out analysis_v2 --margin 0.01
```
* Primary estimand: pooled C - B in F_faith over the six adapter x language cells, seed-paired, t interval with n_seeds - 1 df.
* Secondary: per adapter (Holm over the three), per language, EM, token F1, Malay option accuracy, extractive-copy rate; C - lambda1=0, C - shuffled; B/C checkpointing contrasts; item-level cluster bootstrap.
* Equivalence: reported only when the 90% CI lies inside +-margin. **Fix the margin in writing before looking at data**; 0.01 is the proposal (the first study's pooled effect was 0.018).
* Everything else is exploratory and must be labelled so.

## 4. Independent validation
```bash
python independent_eval.py runs_v2 --out indep_eval --device cuda            # second NLI model (different architecture/data)
python human_eval.py make runs_v2 --out human_eval --n 100 --adapter lora    # blinded sheets for two annotators + key.csv
python human_eval.py score human_eval                                        # kappa, per-condition support, C-B with item-bootstrap CI
```
Annotators must not see `key.csv`. Obtain institutional ethics guidance (or an exemption statement) before recruiting annotators.
Both NLI scorers lack Malay in their NLI training data; the human check is the language-independent evidence.

## 5. Release
Ship `runs_v2/**/checkpoint_*.jsonl`, `runs_v2/**/preds/*.jsonl` (complete per-example predictions with `src_id`), `env_freeze.txt`, the analysis outputs, and (separately, size permitting) adapter weights.
