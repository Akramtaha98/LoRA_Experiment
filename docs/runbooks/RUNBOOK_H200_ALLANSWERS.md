# Runbook: save every answer of all three models (for the website)

Goal: Qwen3-0.6B-Base and T5Gemma 2 (270M) currently log only 5 standard-training answers per language. This run re-trains them
(QLoRA, seed 42, Arabic and Malay, config B and C) with `--log_all_examples`, so the website can show every model on every question.
mT5 already has full answers in `runs_v2/`.

## 1. On the H200 machine
```bash
# one-time environment
python3 -m venv ~/venv_ans && source ~/venv_ans/bin/activate
pip install torch transformers peft datasets accelerate bitsandbytes sentencepiece tabulate pandas scipy
huggingface-cli login            # needed if google/t5gemma-2-270m-270m is gated for your account

git clone https://github.com/Akramtaha98/LoRA_Experiment.git && cd LoRA_Experiment

# run everything (use tmux so the session can drop)
tmux new -s ans
bash scripts/run_h200_allanswers.sh
# detach: Ctrl-b d      re-attach: tmux attach -t ans
```
Expected wall time: about 2 to 3 hours on one H200 (the 4 jobs run side by side; slowest is Qwen3 Config C Arabic).
The run is restartable: re-running the same command skips finished runs.

## 2. Check the output
```bash
cd ~/h200_allanswers/results/first_study
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("*/*.jsonl")):
    for l in open(f):
        r = json.loads(l); n = len(json.loads(r["samples"])) if r["samples"].startswith("[") else 0
        print(f"{f:58s} {r['config']:18s} {r['status']:5s} n_answers={n} EM={r['mean_em']:.3f}")
PY
```
Every line must show `PASS` and `n_answers` = 720 (Arabic) or 430 (Malay).

## 3. Bring the files back and rebuild the website
```bash
# on the machine that has the repo (the Mac), from the repo root:
mkdir -p results/first_study_allanswers
scp -r <user>@<h200-host>:~/h200_allanswers/results/first_study/* results/first_study_allanswers/

python scripts/build_compare_examples.py --runs_v2 runs_v2 \
    --h200_dir results/first_study_allanswers --config B_ce_lora
cd web && npm ci && npm run build        # or just push; Vercel builds it
git add -A && git commit -m "All-answers run: 3 models x 100 questions" && git push
```
`runs_v2` is the corrected-study folder (not in git); copy it from wherever you ran the corrected study.

## Notes
- This re-runs the original-protocol code (same as the first study), so Qwen3 and T5Gemma 2 numbers may differ slightly from the
  earlier seed-42 runs (GPU nondeterminism). The website's ranking table keeps using the 3-seed results.
- Use `--config C_composite_lora` in the last build step to show the NLI-reward setup instead.
