<div align="center">

# LoRA-Based Mitigation of RAG Faithfulness Failures

**Parameter-efficient fine-tuning for context-faithful generation, evaluated on Arabic and Malay QA**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/paper-under%20review-yellow)
![Model](https://img.shields.io/badge/backbone-mT5--base-lightgrey)

</div>

---

Training code and experiment logs for a closed-passage evaluation of four parameter-efficient fine-tuning (PEFT) methods, **QLoRA**, **AdaLoRA**, **DoRA**, and **VeRA**, as a mitigation for faithfulness failures in retrieval-augmented generation (RAG). Tested on Arabic (XQuAD) and Malay (Belebele) question answering with `mt5-base`.

This repo backs a paper submitted to IEEE/ACM TASLP. It documents a real implementation bug found during the GPU rerun (a non-differentiable training objective), the fix, and the corrected results, including honest reporting of what did and didn't hold up under a second, independent verification pass. For the full story, see [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Overview

Four training configurations are compared across four LoRA-family adapters and two low-resource languages.

| Config | Objective |
|---|---|
| **A** | Frozen `mt5-base`, no fine-tuning (baseline) |
| **B** | LoRA fine-tuning, standard cross-entropy loss |
| **C** | LoRA fine-tuning, composite loss (cross-entropy + a faithfulness reward), optimized via self-critical sequence training (SCST; Rennie et al., 2017) |
| **D** | Full fine-tuning (all 580M params), cross-entropy loss |

Faithfulness is scored with **F_faith**: the entailment probability between a generated answer and its source context, judged by a frozen multilingual NLI classifier (`mDeBERTa-v3-base-mnli-xnli`) that's separate from the model being trained. Exact match (EM) and token F1 track answer *correctness*, so faithfulness and correctness can be told apart.

## Key finding

Composite-loss training (Config C) produces a small, direction-mixed effect on faithfulness: mean ΔF_faith = +0.0074 across 8 conditions (5 positive, 3 negative), not statistically significant by a binomial sign test (p ≈ 0.73). Alongside that, there's a consistent **faithfulness-for-correctness trade-off** on Malay, where the composite objective's only nonzero-EM condition is the CE-only baseline, not the composite-trained one.

An independent re-scoring with a second, architecturally distinct NLI model (XLM-RoBERTa, trained on SNLI+MNLI+ANLI+XNLI) finds the two classifiers' rankings significantly correlated overall (Spearman ρ = 0.762, p = 0.028), with agreement strong for Malay and weak for Arabic. That weakness lines up with Arabic's own near-flat, non-significant spread.

No inflated claims. This project reports what the data actually shows, including the bug, the fix, and the parts of the result that didn't replicate cleanly.

## Repository structure

```
.
├── lora_experiment_matrix.py    Main experiment script: trains + evaluates all conditions
├── experiment_results/          Per-condition checkpoint logs (JSONL) + results CSV
├── analysis/
│   ├── generate_figures.py      Regenerates the paper's result figures from logged data
│   └── xlmr_rescore.py          Independent NLI re-scoring (verification pass)
├── docs/
│   └── PROJECT_GUIDE.md         Full narrative: motivation, bug, fix, GPU rerun, verification
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU (24GB+ recommended) for real training runs. CPU is fine for the smoke test.

## Usage

```bash
# Fast sanity check: validates the pipeline without real training
python3 lora_experiment_matrix.py --smoke_test

# Full run: all conditions, both languages
python3 lora_experiment_matrix.py --full

# Full run, sharded to one condition (for parallel multi-GPU runs)
python3 lora_experiment_matrix.py --full --variant qlora --lang arabic
```

Every completed run auto-commits its result to `experiment_results/`, so a killed or disconnected process can resume without re-paying for finished conditions.

## Results

Mean F_faith by condition (held-out slice, 30 examples/condition):

| Condition | Arabic (XQuAD) | Malay (Belebele) |
|---|---|---|
| A: Frozen | 0.4532 | 0.2719 |
| D: Full FT | 0.3768 | 0.1721 |
| B: QLoRA (CE) | 0.4361 | 0.2676 |
| **C: QLoRA (composite)** | **0.4462** | **0.2925** |
| B: AdaLoRA (CE) | 0.4343 | 0.1439 |
| C: AdaLoRA (composite) | 0.4371 | 0.1372 |
| B: DoRA (CE) | **0.4716** | 0.1931 |
| C: DoRA (composite) | 0.4440 | 0.2468 |
| B: VeRA (CE) | 0.4522 | 0.0598 |
| C: VeRA (composite) | 0.4408 | 0.0734 |

QLoRA leads both languages under the composite objective (Config C). DoRA leads Arabic under the CE-only objective (Config B). Full analysis, statistical tests, and the independent verification pass are in the paper.

## Reproducibility

- Seed fixed at 42 for all runs (`set_all_seeds()`), applied before every condition so Config B and C differ only in training objective.
- 12 epochs, batch size 4, `K=4` SCST samples per training example.
- Every reported number traces to a checkpoint file in `experiment_results/`. See `docs/PROJECT_GUIDE.md` for the full bug and fix history behind the current numbers.

## Citation

```bibtex
@article{taha2026lora,
  author  = {Taha Zeyad, Akram and Qadri Zakaria, Lailatul},
  title   = {LoRA-Based Mitigation of Retrieval-Augmented Generation Failure Modes: A Closed-Passage Evaluation on Arabic and Malay},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  note    = {Under review}
}
```

**Authors:** Akram Taha Zeyad and Lailatul Qadri Zakaria, Faculty of Information Science and Technology (FTSM), Universiti Kebangsaan Malaysia.

## License

MIT. See [`LICENSE`](LICENSE).
