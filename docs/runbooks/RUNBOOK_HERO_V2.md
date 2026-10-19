# RUNBOOK — Hero rerun (Qwen3-0.6B-Base + T5Gemma 2 270M), 5 seeds

mT5 is the finished *Legacy Baseline* and is NOT rerun.  Seeds: 42 123 2026 7 99.
Every run saves per-example predictions (`predictions.jsonl`), `summary.json`, `train_log.jsonl`.

## 0. Setup (once)
    cd ~/LoRA_Experiment && git pull   # or copy the 3 new files
    huggingface-cli login              # T5Gemma 2 is a gated repo (needs your token)

## 1. Smoke test (≈5 min, 1 GPU) — MUST PASS before the full run
    export PYTHONPATH=src
    HERO_MAX_PROMPT=160 python -u src/lora_nli/hero_v2.py --smoke --models qwen3 t5gemma2 \
        --langs arabic malay --conds A_frozen B C C_shuf --seeds 42 --ckpt_control --out /tmp/hero_smoke
Check: no traceback; `B` and `B_ckpt` give identical `faith/f1` (checkpointing = memory switch);
`grep -c . /tmp/hero_smoke/*/*/B/seed42/predictions.jsonl` > 0.
(The cloud smoke test validated Qwen3 end-to-end and the seq2seq path with a stand-in T5;
the real T5Gemma 2 weights are gated there, so this step is the first real T5Gemma 2 check.
If `decoder_input_ids` is rejected the code falls back to `labels=` automatically.)

## 2. Time profile (≈40 min) — gives the real ETA
    python -u src/lora_nli/hero_v2.py --models qwen3 --langs arabic --conds C --seeds 42 --out /tmp/hero_prof
Read `train_seconds` in `/tmp/hero_prof/qwen3/arabic/C/seed42/summary.json`.
Estimate (to be replaced by the measured number): C-type run ≈ 25–45 min, B ≈ 5–10 min,
eval ≈ 2–4 min.  Per (backbone, language): 3 C-type × 5 seeds + 5 B + A ≈ 9–12 h; four such jobs
in parallel on 4 GPUs ≈ 10–14 h wall-clock (longer if the GPUs are shared).

## 3. Full run
    bash scripts/run_hero_v2.sh                 # one job per GPU, nohup, resumable
    EXTRA="--ckpt_control" bash scripts/run_hero_v2.sh   # also runs the B_ckpt control
Monitor: `grep DONE results/hero_v2/logs/*.log`.  Re-running the same command resumes.
Order is seed-major, so after the first seed every condition already has a result.

## 4. Analysis
    python scripts/analyze_hero_v2.py results/hero_v2
→ `results/hero_v2/analysis/{summary.md,contrasts.json,per_seed.csv}`

## 5. Send back
Zip `results/hero_v2` (predictions are small) and attach it; the paper, website and anonymous
package are then rebuilt around Qwen3 + T5Gemma 2 with mT5 as the Legacy Baseline.

## What changed vs. the legacy runners (why earlier T5Gemma 2 / Qwen numbers are not reusable)
NLI premise = raw passage (not decoded prompt); sampling = re-scoring distribution (T=1, full softmax,
eval mode, cache on); reward text = log-prob tokens; held-out decode always in eval mode with
asserts and gradient checkpointing off; controls C_l0 / C_shuf; per-seed per-example outputs.
