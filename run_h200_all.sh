#!/usr/bin/env bash
# One-shot T5Gemma 2 run on a single H200: work in a fresh folder, keep only a results tarball, then delete the folder.
set -e
WORK=/workspace/t5gemma2_run_$(date +%Y%m%d_%H%M)      # fresh working folder
OUT=/workspace/t5gemma2_results_$(date +%Y%m%d_%H%M).tgz  # results kept OUTSIDE the folder
REPO=https://github.com/Akramtaha98/LoRA_Experiment.git
mkdir -p "$WORK" && cd "$WORK"
git clone "$REPO" code && cd code
pip install -q -U transformers peft bitsandbytes accelerate datasets sentencepiece tabulate scipy
python3 t5gemma2_smoke.py                                  # stops here (set -e) if anything is broken
python3 run_t5gemma2.py --stage all --parallel 6           # 18 jobs (B+C each), 6 at a time
python3 run_t5gemma2.py --summary_only | tee summary.txt
tar czf "$OUT" experiment_results_t5gemma2 t5gemma2_logs summary.txt
tar tzf "$OUT" > /dev/null && echo "Results verified: $OUT"
cd / && rm -rf "$WORK"                                     # delete the folder only after the tarball verified
echo "DONE. Download $OUT, then delete it if you want."
