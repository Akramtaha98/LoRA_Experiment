#!/usr/bin/env bash
# Launch the corrected 5-seed hero study: 4 jobs (2 backbones x 2 languages), one per GPU.
# Resumable: finished (cond, seed) runs are skipped. Usage:
#   bash scripts/run_hero_v2.sh            # full study
#   GPUS="0 1 2 3" OUT=results/hero_v2 bash scripts/run_hero_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-results/hero_v2}
GPUS=(${GPUS:-0 1 2 3})
mkdir -p "$OUT/logs"
JOBS=("qwen3 arabic" "qwen3 malay" "t5gemma2 arabic" "t5gemma2 malay")
for i in "${!JOBS[@]}"; do
  read -r M L <<<"${JOBS[$i]}"
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "launch $M/$L on GPU $G"
  CUDA_VISIBLE_DEVICES=$G nohup python -u src/lora_nli/hero_v2.py --models "$M" --langs "$L" \
      --out "$OUT" ${EXTRA:-} > "$OUT/logs/${M}_${L}.log" 2>&1 &
  echo $! > "$OUT/logs/${M}_${L}.pid"
done
echo "logs: $OUT/logs/*.log   progress: grep DONE $OUT/logs/*.log"
