# NLI faithfulness reward for LoRA fine-tuning: a checkpointing confound and a small reward-attributable gain

Arabic (XQuAD) and Malay (Belebele `zsm_Latn`) closed-passage QA on mT5-base. This release accompanies the corrected
version of the manuscript. **It supersedes the earlier claim that the reward harms DoRA and LoRA but helps QLoRA.**

## What the corrected study shows

| Finding | Evidence |
|---|---|
| The earlier "harm" is a gradient-checkpointing artifact | With checkpointing on (original policy: Config C only, non-quantized adapters only), Config C is lower than Config B by about 0.16 for LoRA and DoRA; the same collapse occurs with the reward weight at 0 and with a shuffled reward, and for QLoRA when checkpointing is forced on |
| With checkpointing off the reward gives a small gain | C(off) - B = +0.018 (90% CI +0.005, +0.031), positive in 16 of 18 variant-language-seed cells; the shuffled-reward and lambda1=0 controls are within 0.002 of B |
| Strength of evidence | 3 seeds; no single-adapter test survives Holm correction; no exact-match improvement; the NLI score is also the reward |
| Determinism | Four repeats each of DoRA/Malay and QLoRA/Malay (seed 42) are identical to the reported decimals, with and without deterministic kernels |

The mechanism of the checkpointing failure is not identified.

## Reproduce

**From saved results (CPU, minutes).** `code/analysis/pgfix_corrected/all.csv` (also `data/pgfix_all_runs.csv`) has one record per run
(188 runs), including GPU, driver and library versions.

```
cd code/analysis/pgfix_corrected
python tables.py && python stats.py && python mk_tables.py && python mk_figs.py
```

This regenerates the manuscript tables (`tab_*.tex`), Figure 1 and `stats.json` (every contrast). `merge.py DIR...` rebuilds `all.csv`
from raw `runs_pgfix/` outputs after a re-run.

**From training (GPU, about 62 GPU-hours).** See `code/RUNBOOK_H200.md`. `code/run_pgfix.py` runs the stages `core`, `extra`, `controls`,
`ckptoff`, `attrib` and `repeat` with the corrected estimator (`LORA_PG_FIX=1`). Results were produced on an NVIDIA H200 NVL
(driver 595.71.05, CUDA 13.0, PyTorch 2.12.1, Transformers 5.5.0, PEFT 0.21.2). Other stacks should show the same pattern, not
necessarily the same decimals.

**Estimator test (CPU).** `code/analysis/estimator_checks_fixed.py` checks that the aligned scoring equals the generation log-probability,
that banned tokens are never sampled, that gradients are nonzero and that a constant reward gives zero gradient.

## Original-protocol material (superseded)

With `LORA_PG_FIX` unset, `code/lora_experiment_matrix.py` runs the original estimator. The Qwen3-0.6B-Base and T5Gemma 2 scripts, the older per-seed CSVs,
the lambda sweep and the human-validation labels come from the original protocol (original estimator and checkpointing policy) and appear in
the manuscript's appendices only as such. They are not comparable with the corrected mT5 results.

## Data

XQuAD (Apache-2.0) and Belebele (CC-BY-SA-4.0) are not redistributed; `data/split_*.json` give the exact example indices.
