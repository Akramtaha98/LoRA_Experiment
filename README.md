<div align="center">

# NLI-Faithfulness Reward × LoRA

**Does an NLI-entailment training reward make small multilingual QA models more faithful? A controlled study on Arabic and Malay, with a null result.**

[Findings](#findings) · [Study design](#study-design) · [Repository layout](#repository-layout) · [Reproduce](#reproduce) · [Disclosures](#disclosures-and-limits) · [Legacy mT5 study](#legacy-baseline-mt5-base)

</div>

A leave-one-out REINFORCE objective with an NLI-entailment reward (K = 4 samples, λ = 0.3) is added to cross-entropy training of **LoRA** adapters (r = 8, α = 16) on **Qwen3-0.6B-Base** and **T5Gemma 2 270M-270M**. Models are evaluated on **Arabic (XQuAD)** and **Malay (Belebele `zsm_Latn`)** closed-passage question answering. The reward model is `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`; an independent scorer, `joeddav/xlm-roberta-large-xnli`, is never used in training.

## Findings

| | Result (84 runs, 5 seeds per trained cell) |
|---|---|
| Fine-tuning | Raises F1 and exact match sharply in every cell (e.g. Qwen3 Arabic F1 0.211 → 0.592, EM 0.075 → 0.423). |
| Reward effect on NLI score | **At most 0.012** in any cell. Nine of twelve reward-vs-control contrasts include zero. |
| The three that do not | All in Qwen3 on Arabic (C−B +0.011, C−C_l0 +0.012, C−C_shuf +0.008). The independent scorer gives +0.002, and at matched answer length the difference is +0.003 (95% CI −0.004 to +0.009). |
| Controls | Zero-weight (`C_l0`) and shuffled-reward (`C_shuf`) runs match cross-entropy training within about ±0.006. |
| Length | Most of the frozen → fine-tuned NLI drop is a change in answer length, not in faithfulness. |
| Why the reward is inert (indirect evidence) | Cross-entropy memorises the 470 training examples, so the K samples are identical, the leave-one-out advantage is zero, and the policy-gradient term is below 1e-3 in 80–100 % of logged steps in the second half of training. |
| Legacy mT5-base | Does not memorise; the reward raises the Arabic NLI score by +0.60 while exact match falls and answers grow sixfold, consistent with over-optimisation of the score. |

An NLI reward can therefore do nothing or do harm, and the NLI score alone cannot tell which. Controls, length-matched analysis and reward-independent metrics are needed.

### Cell means (NLI score `faith`, mean over seeds)

| Model / language | A frozen | B (CE) | C (reward) | C_l0 | C_shuf |
|---|---|---|---|---|---|
| Qwen3 / Arabic | 0.506 | 0.144 | 0.156 | 0.144 | 0.147 |
| Qwen3 / Malay | 0.292 | 0.293 | 0.302 | 0.295 | 0.299 |
| T5Gemma 2 / Arabic | 0.186 | 0.132 | 0.131 | 0.130 | 0.136 |
| T5Gemma 2 / Malay | 0.092 | 0.295 | 0.301 | 0.291 | 0.293 |

Full tables (F1, EM, second scorer, answer length, contrasts with bootstrap intervals) are in `results/hero_v2/analysis/summary.md`.

## Study design

| Item | Setting |
|---|---|
| Models | Qwen3-0.6B-Base, T5Gemma 2 270M-270M (primary); mT5-base (legacy baseline) |
| Data | XQuAD Arabic, Belebele Malay; 470 fine-tuning examples; held-out evaluation sets |
| Conditions | `A_frozen` (single deterministic run), `B` (cross-entropy), `C` (CE + NLI reward), `C_l0` (λ = 0), `C_shuf` (rewards rotated across the batch) |
| Seeds | 5 per trained cell → 2 models × 2 languages × (4 × 5 + 1) = **84 runs** |
| Training | 12 epochs, batch size 4 (1,416 steps), lr 2e-4, LoRA r = 8, α = 16 |
| Scorers | Training reward: mDeBERTa-v3 MNLI/XNLI. Evaluation: same scorer (`faith`) and an independent XLM-R-large XNLI scorer (`faith_2nd`) |

## Repository layout

```text
src/lora_nli/            library code
  hero_v2.py             training + evaluation for the primary study (Qwen3, T5Gemma 2)
  matrix.py              legacy mT5 pipeline (LORA_V2=1 selects the corrected path)
  analyze.py, independent_eval.py, human_eval.py   legacy-study analysis and scorers
scripts/
  run_hero_v2.sh         launcher for the 84 runs (resumable, one process per GPU)
  analyze_hero_v2.py     cell means, contrasts, bootstrap intervals
  lenmatch.py            matched-answer-length analysis
  gen_paper_assets.py    regenerates every table and figure of the paper from saved results
H200/                    self-contained test + train folder used on the GPU server
results/
  hero_v2/               primary study: per-run summaries and training logs, predictions archive,
                         analysis/, length_matched/, logs/
  corrected_v2/          legacy mT5 run records, contrasts, bootstraps
  human_validation/      legacy human-validation labels and scripts
  first_study/           first (pre-correction) study
docs/runbooks/           RUNBOOK_HERO_V2.md and legacy runbooks
tests/                   CPU unit tests
web/                     dashboard (legacy mT5 results)
```

Per-example predictions for the 84 primary runs are in `results/hero_v2/predictions.tgz`. Legacy mT5 predictions are in `runs_v2.tar.gz` (not tracked by git; attach it to a release).

## Reproduce

**From saved results (CPU, minutes):**

```bash
pip install numpy pandas scipy matplotlib
cd results/hero_v2 && tar xzf predictions.tgz && cd ../..
python scripts/analyze_hero_v2.py results/hero_v2                       # cell means, contrasts
python scripts/lenmatch.py results/hero_v2 results/hero_v2/length_matched
python scripts/gen_paper_assets.py results/hero_v2 results/hero_v2/length_matched out/   # tables + figures
```

**From training (GPU):**

```bash
cd H200
GPU=0 bash run_test.sh          # 4-stage pre-flight: access, smoke test, 2-epoch learning check, timing
GPUS="0 1 2 3" bash run_train.sh   # launches the 84 runs; completed runs are skipped on restart
python analyze.py results_train
```

T5Gemma 2 is a gated model: accept its license on Hugging Face and run `huggingface-cli login` first. Tested stack: PyTorch 2.12.1, Transformers 5.5.0, PEFT 0.21.2 on NVIDIA H200 GPUs. On other hardware or library versions expect the same qualitative pattern, not the same decimals: the same seed run twice on the same stack already differs by up to 0.012 in the NLI score.

## Disclosures and limits

- 12 Qwen3 runs were killed by CUDA out-of-memory errors and restarted; as a side effect, 4 `C_l0` cells were trained twice and the stored result is the later finisher. Duplicate runs differ by 0.003–0.012 in `faith` and 0.007–0.021 in F1.
- The memorisation explanation is indirect (logs are written every 50 steps and reward variance is not logged). A direct test needs new runs.
- The 12 contrasts and the length analysis are exploratory and not adjusted for multiplicity.
- Human evaluation covers the legacy mT5 generations only (two annotators, κ = 0.38 for support).
- The dashboard under `web/` shows the legacy mT5 results and has not been updated for the primary study.

## Legacy baseline: mT5-base

The earlier corrected, pre-specified study (mT5-base, LoRA / QLoRA / DoRA, 228 run records, 8 seeds per cell) found an Arabic NLI gain of +0.601 (95 % CI +0.561 to +0.641) and no reliable Malay change (+0.008), with exact match falling and answers growing from 2.9 to 16.5 words. Holding answer-length bins fixed removes about three quarters of the estimated Arabic gain. The first study, which reported the reward *harming* LoRA and DoRA, is superseded: that effect was an evaluation artefact (a gradient-checkpointed T5 left in train mode decodes from the last token only). Records and scripts are in `results/corrected_v2/` and `results/first_study/`.

## Data and licences

XQuAD (Apache 2.0) and Belebele (CC-BY-SA 4.0) are public and are not redistributed; exact split indices are released. Model weights and the two NLI models are used from their public releases. Code is released under the licence in `LICENSE`.

## Citation

A manuscript is under review; a citation entry will be added after the review.
