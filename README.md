# LoRA Faithfulness Study: Arabic and Malay Closed-Passage QA

Code, results and analysis scripts for a controlled study of a training-time NLI faithfulness penalty combined with
LoRA variants (QLoRA, AdaLoRA, DoRA, VeRA, plus a post hoc plain LoRA baseline) on two backbones (mT5-base and
Qwen3-0.6B-Base), evaluated on Arabic (XQuAD) and Malay (Belebele `zsm_Latn`) closed-passage question answering.

## Findings (see the paper for the full analysis)
- **DoRA collapse (robust).** On mT5 the composite objective lowers DoRA's NLI faithfulness score by 0.154. The effect holds
  in every seed and under pooled, Welch and seed-paired analyses, and a blinded human validation of 350 generations confirms it.
- **Plain LoRA collapses too** (post hoc, mT5 only, -0.146), so the collapse is not specific to DoRA's decomposition.
- **QLoRA gain (small, not robust).** +0.023, same sign in all seeds and both languages. Significant under the pooled-variance
  analysis (p = 2e-7) but not under a Welch test (p = 0.012; threshold 0.00625) or a seed-paired test (p = 0.070). The human
  sample can neither confirm nor exclude it. QLoRA also differs from the other variants in quantization and in its
  gradient-checkpointing policy, which are confounded with the variant.
- **Qwen3:** no variant-level effect detected under this protocol (three seeds per cell; absence of detection is not evidence of absence).
- The NLI score is also the training reward and correlates only weakly with human judgment (Spearman 0.30 Arabic, 0.16 Malay).
- mT5 Arabic is in a near-floor regime (EM <= 0.022).

## Design
- Configs: A frozen baseline, B CE-only LoRA, C composite LoRA (CE + lambda_1 (1 - F_faith), lambda_1 = 0.3, optimized with a
  corrected self-critical policy-gradient (SCST) estimator, K = 4 samples), D full fine-tuning (single seed, different optimizer).
- Training set: 470 examples per language. Held-out evaluation: 720 Arabic and 430 Malay examples.
- Configs B and C: 3 seeds (42, 123, 2026). Configs A and D: seed 42 only.

## Repository contents
| Path | What it is |
|---|---|
| `lora_experiment_matrix.py` | mT5-base experiment matrix |
| `qwen_experiment_matrix.py` | Qwen3-0.6B-Base experiment matrix |
| `run_plain_lora.py` | Plain LoRA baseline (mT5) |
| `analysis/per_seed_results.csv` | 108 per-seed runs (F_faith, EM, F1) behind Tables 1 and 3 |
| `analysis/seed42_rerun_lambda0.3.csv` | Later seed-42 lambda = 0.3 reruns (used by the lambda sweep, not by the 3-seed tables) |
| `analysis/verify_table.py` | Checks that the per-seed file reproduces the published cell means and SDs (32/32) |
| `analysis/factorial_reanalysis.py` | Planned contrasts and interaction tests from the cell means/SDs |
| `analysis/robust_reanalysis.py` | Welch, seed-paired and Holm sensitivity analyses |
| `human_validation/` | 350 adjudicated human labels, analysis script and README |
| `RUNBOOK.md` | Running the experiments on a rented GPU |

## Reproduce the statistics
```bash
pip install numpy scipy
python analysis/verify_table.py        # expect: TABLE_REPRODUCED cells 32 mismatch 0
python analysis/factorial_reanalysis.py
python analysis/robust_reanalysis.py analysis/per_seed_results.csv
python human_validation/analyze_human_validation.py
```

## Notes
- XQuAD (Apache-2.0) and Belebele (CC-BY-SA-4.0) are third-party datasets and are not redistributed.
- Results from earlier 30-example evaluations are superseded and should not be used.
- Open (not run): plain LoRA Config C without gradient checkpointing, a lambda = 0 control through the composite path,
  a shuffled-reward control, full-output independent rescoring, and a tuned Arabic CE baseline.
