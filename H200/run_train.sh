#!/usr/bin/env bash
# Full 5-seed study: 4 jobs (qwen3/t5gemma2 x arabic/malay), one per GPU, nohup, resumable.
#   bash run_train.sh                     # refuses to start unless run_test.sh passed
#   GPUS="0 1 2 3" bash run_train.sh      # choose GPUs
#   EXTRA="--ckpt_control" bash run_train.sh
#   FORCE=1 bash run_train.sh             # skip the test gate (not recommended)
set -euo pipefail
cd "$(dirname "$0")"
[ -f results_test/PASSED ] || [ "${FORCE:-0}" = "1" ] || { echo "Run 'bash run_test.sh' first (no results_test/PASSED)."; exit 1; }
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-results_train}; GPUS=(${GPUS:-0 1 2 3}); mkdir -p "$OUT/logs"
JOBS=("qwen3 arabic" "qwen3 malay" "t5gemma2 arabic" "t5gemma2 malay")
for i in "${!JOBS[@]}"; do
  read -r M L <<<"${JOBS[$i]}"; G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "launch $M/$L on GPU $G"
  CUDA_VISIBLE_DEVICES=$G nohup python -u hero_v2.py --models "$M" --langs "$L" --out "$OUT" ${EXTRA:-} \
      > "$OUT/logs/${M}_${L}.log" 2>&1 &
  echo $! > "$OUT/logs/${M}_${L}.pid"
done
echo "progress : grep -h DONE $OUT/logs/*.log | tail"
echo "analysis : python analyze.py $OUT      (any time; re-running this script resumes)"
