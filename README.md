# LoRA Faithfulness Study: Arabic and Malay Closed-Passage QA

Code, results and analysis scripts for a controlled study of a training-time NLI faithfulness penalty combined with
LoRA variants (QLoRA, AdaLoRA, DoRA, VeRA) on two backbones (mT5-base and Qwen3-0.6B-Base), evaluated on
Arabic (XQuAD) and Malay (Belebele `zsm_Latn`) closed-passage question answering.

**Main finding.** On mT5-base the identical composite objective raises the NLI faithfulness score of QLoRA and collapses
that of DoRA; on Qwen3-0.6B-Base no variant-level effect is significant after correction, and backbone x objective
interaction tests show the two backbones differ. Faithfulness is measured by an NLI proxy that is also the training
reward; human validation is in progress.

## Design
- Configs: A frozen baseline, B CE-only LoRA, C composite LoRA (CE + lambda_1 (1 - F_faith), lambda_1 = 0.3, optimized with a
  corrected self-critical policy-gradient (SCST) estimator, K = 4 samples), D full fine-tuning (single seed, different optimizer).
- Training set: 470 examples per language. Held-out evaluation: 720 Arabic and 430 Malay examples.
- Configs B and C: 3 seeds (42, 123, 2026). Configs A and D: seed 42.

## Repository contents
| Path | What it is |
|---|---|
| `lora_experiment_matrix.py` | mT5-base experiment matrix |
| `qwen_experiment_matrix.py` | Qwen3-0.6B-Base experiment matrix |
| `analysis/factorial_reanalysis.py` | Reproduces the paper's planned contrasts and interaction tests from the cell means/SDs |
| `data/paper_tables_current.csv` | All values in the result tables (means and SDs) |
| `RUNBOOK.md` | Running the experiments on a rented GPU |

## Reproduce the statistics
```bash
pip install numpy scipy
python analysis/factorial_reanalysis.py
```

## Notes
- XQuAD (Apache-2.0) and Belebele (CC-BY-SA-4.0) are third-party datasets and are not redistributed.
- Results from earlier 30-example evaluations are superseded and should not be used.
