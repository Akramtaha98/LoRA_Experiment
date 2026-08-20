# Project Guide: What This Is and What Happened

This is the working log behind the paper *"LoRA-Based Mitigation of Retrieval-Augmented Generation Failure Modes: A Closed-Passage Evaluation on Arabic and Malay."* It's written to be read start to finish — it explains the motivation, the method, a real bug that was found partway through the GPU rerun, how it was fixed, and what the corrected results actually show.

## 1. Motivation

Retrieval-augmented generation (RAG) systems fail in characteristic ways: a model can generate fluent, plausible-sounding text that isn't actually supported by the retrieved context. This project asks a narrow, testable question: **can parameter-efficient fine-tuning (LoRA and its variants) be used to directly optimize for faithfulness to context, and does that come at a cost to answer correctness?**

The setting is deliberately low-resource: Arabic (XQuAD) and Malay (Belebele), two languages with far less NLP tooling and training data than English, using `mt5-base` (580M params) as the backbone — small enough to fine-tune on a single consumer GPU, which matters for who can actually reproduce or extend this work.

## 2. Method, in brief

Four training configurations, crossed with four LoRA-family adapters (QLoRA, AdaLoRA, DoRA, VeRA) and two languages — 20 conditions total, 8 of which (the composite-loss ones) required a full rerun after a bug was found (Section 4).

- **Config A** — frozen baseline, no fine-tuning.
- **Config B** — LoRA fine-tuning, standard cross-entropy loss on the gold answer.
- **Config C** — LoRA fine-tuning, a *composite* loss: cross-entropy plus a faithfulness-reward term, meant to directly push the model toward context-grounded generations.
- **Config D** — full fine-tuning (all 580M params), cross-entropy loss.

Faithfulness is measured by **F_faith**: the entailment probability between a generated answer and its source context, scored by a frozen multilingual NLI classifier (`mDeBERTa-v3-base-mnli-xnli`) that is never trained and is architecturally separate from `mt5-base`. Exact match and token F1 are tracked alongside it, because F_faith measures *faithfulness*, not *correctness* — a model can copy plausible-looking context verbatim and score well on F_faith while being factually wrong, so both numbers are needed to tell the two apart.

## 3. The composite-loss objective (Config C)

Turning F_faith into a training signal is not trivial: it's computed by running a separate frozen classifier on *generated text*, and `argmax`/sampling-based text generation is not differentiable. The intended design uses **self-critical sequence training** (SCST; Rennie et al., 2017, "Self-Critical Sequence Training for Image Captioning"):

1. Sample `K=4` generations per training example from the current policy (no gradient — sampling is non-differentiable by construction).
2. Score each sampled generation with the frozen F_faith classifier (no gradient — the reward model is never updated).
3. Build a leave-one-out advantage per sample: `advantage_k = reward_k - mean(reward_{j != k})`.
4. Recompute the log-probability of each *sampled* sequence via a **separate, teacher-forced forward pass** through the trainable model — this is the step that actually carries gradient.
5. Weight the policy-gradient loss term by the (detached) advantage, and add it to the standard cross-entropy loss.

## 4. The bug, found on the GPU rerun

The first implementation of this objective computed the reward correctly but **never routed it back through a differentiable path** — the composite loss term was, in effect, disconnected from the trainable LoRA parameters. Config C was training, converging, and producing numbers, but the "composite" part of the composite loss was a no-op: Config C was quietly training identically to Config B.

This was caught during the GPU rerun of the full 20-condition matrix, before the paper's Config C numbers were finalized. It's disclosed here (and in the paper's Limitations section) rather than fixed silently, because a training bug that produces plausible-looking numbers without an error is exactly the kind of failure a reader can't catch from the results table alone.

**The fix**: rewrote the training step to implement the leave-one-out SCST objective in step 3–5 above, so the faithfulness reward has a genuine, differentiable path into the LoRA adapter weights. All 8 affected conditions (4 LoRA variants × 2 languages) were retrained from scratch under the corrected objective; Config A, B, and D were unaffected (they never touched the composite loss) and were not retrained.

## 5. Retraining infrastructure

The 8-condition rerun was parallelized across 8 single-GPU RunPod instances (RTX 4090s) instead of run sequentially, to bring wall-clock time down from an estimated 13–14 hours to roughly 2–3 hours. Each pod ran one `(variant, language)` shard:

```bash
python3 lora_experiment_matrix.py --full --variant <qlora|adalora|dora|vera> --lang <arabic|malay>
```

Each pod wrote to its own checkpoint file (`checkpoint_full_<variant>_<lang>.jsonl`) to avoid write conflicts when 8 processes push to the same GitHub repo simultaneously, then merged after the fact with:

```bash
cat experiment_results/checkpoint_full_*.jsonl > experiment_results/checkpoint_full.jsonl
```

Two further bugs surfaced during this rerun, both fixed and both documented in code comments in `lora_experiment_matrix.py`:

- **CUDA OOM on non-QLoRA variants.** SCST's `K=4` sampling inflates the effective training batch through the frozen backbone to `4 × 4 = 16` sequences. For DoRA, AdaLoRA, and VeRA — which don't run a 4-bit quantized backbone the way QLoRA does — this pushed a 24GB GPU out of memory at step 4 of 1416. Fixed by enabling gradient checkpointing for Config C on all non-QLoRA variants.
- **Silent empty-generation bug (QLoRA/Malay).** One run produced an empty string for every single evaluation example — a `mean_f_faith` of exactly 0.0 that looked like a legitimate "worst case" result but wasn't. The root cause was NaN/Inf logits under 4-bit quantization collapsing greedy decoding to an immediate end-of-sequence token. A NaN-safety guard (logit sanitization + renormalization) had already been added to the training-time generation call for a related crash; it turned out the *evaluation-time* generation call needed the same guard and didn't have it. Fixed by applying the same guard to both call sites.

## 6. Results, honestly reported

With the corrected objective, the composite-loss effect on faithfulness is real but small: mean $\Delta F_{\text{faith}} = +0.0074$ across the 8 conditions, with 5 positive and 3 negative deltas. A binomial sign test on that 5/8 split gives $p \approx 0.73$ — **not statistically distinguishable from chance** at this sample size. The paper reports this directly rather than treating "5 out of 8 positive" as evidence of a real effect.

A consistent secondary pattern held up better: on Malay, the composite objective's only nonzero exact-match condition is the *CE-only* baseline (Config B), not the composite-trained one (Config C) — a faithfulness-for-correctness trade-off that shows up independently of the (non-significant) mean effect above.

## 7. Independent verification

Because F_faith is scored by the same NLI classifier used during training, a natural question is whether any composite-loss "gain" reflects genuine faithfulness improvement or the model learning to exploit that specific classifier. To check, generations from all 8 corrected Config C conditions were independently re-scored with a second, architecturally distinct multilingual NLI model (`symanto/xlm-roberta-base-snli-mnli-anli-xnli`, XLM-RoBERTa trained on SNLI+MNLI+ANLI+XNLI — a different architecture and training mixture from the mDeBERTa model used at training time).

This check is a spot-check, not a full re-scoring: only the first 5 of each condition's 30 held-out generations are retained in the experiment logs, so the comparison is between a 5-example XLM-R mean and a 30-example mDeBERTa mean, not two directly comparable point estimates. Across the 8 conditions, the two classifiers' rankings correlate significantly (Spearman $\rho = 0.762$, $p = 0.028$). Splitting by language: Malay shows perfect rank agreement ($\rho = 1.0$), while Arabic shows weak agreement ($\rho = 0.6$, not significant) — which tracks with Arabic's own composite-loss deltas already sitting in a near-flat, non-significant band. The independent check corroborates the paper's central caveat rather than contradicting it: the Malay result looks more robust under a second classifier than the Arabic one does.

## 8. What's still open

- **Multi-seed replication.** Every condition here is a single run. The single most important next step is re-running the 8 composite-loss conditions under ≥3 seeds each to determine whether the modest mean effect and the Arabic variant-ranking pattern are real or single-seed noise.
- **Full-scale independent re-scoring.** The XLM-R check above covers 5 of 30 examples per condition; extending it to the full 30, ideally with human-annotated labels, would tighten the estimate considerably.
- **External baseline.** No comparison yet against an established RAG-faithfulness baseline outside this LoRA-variant matrix.

## 9. Reproducing this

```bash
git clone https://github.com/Akramtaha98/lora_experiment.git
cd lora_experiment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Sanity check (no real training, validates the pipeline)
python3 lora_experiment_matrix.py --smoke_test

# Full run
python3 lora_experiment_matrix.py --full
```

Seed is fixed at 42 (`set_all_seeds()` in `lora_experiment_matrix.py`) and applied before every condition, so Config B and Config C differ only in training objective, not in incidental LoRA initialization or batch order.
