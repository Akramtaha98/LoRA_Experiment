#!/usr/bin/env bash
# Full study, maximum parallelism: one process per (backbone, language, seed) = 20 processes,
# spread round-robin over the GPUs (5 per GPU on 4 GPUs). Resumable; refuses to start unless run_test.sh passed.
#   bash run_train.sh                       GPUS="0 1 2 3" SEEDS="42 123 2026 7 99" EXTRA="--ckpt_control"
set -euo pipefail
cd "$(dirname "$0")"
[ -f results_test/PASSED ] || [ "${FORCE:-0}" = "1" ] || { echo "Run 'bash run_test.sh' first."; exit 1; }
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-results_train}; GPUS=(${GPUS:-0 1 2 3}); SEEDS=(${SEEDS:-42 123 2026 7 99}); mkdir -p "$OUT/logs"
JOBS=("qwen3 arabic" "qwen3 malay" "t5gemma2 arabic" "t5gemma2 malay")
TOTAL=$(( ${#JOBS[@]} * ${#SEEDS[@]} ))
T=$(( $(nproc) / TOTAL )); [ "$T" -lt 1 ] && T=1; [ "$T" -gt 8 ] && T=8
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
n=0
for S in "${SEEDS[@]}"; do
  for J in "${JOBS[@]}"; do
    read -r M L <<<"$J"; G=${GPUS[$((n % ${#GPUS[@]}))]}; n=$((n+1))
    CONDS="B C C_l0 C_shuf"; [ "$S" = "${SEEDS[0]}" ] && CONDS="A_frozen $CONDS"   # frozen baseline once
    CUDA_VISIBLE_DEVICES=$G nohup python -u hero_v2.py --models "$M" --langs "$L" --seeds "$S" \
        --conds $CONDS --out "$OUT" ${EXTRA:-} > "$OUT/logs/${M}_${L}_seed${S}.log" 2>&1 &
    echo $! > "$OUT/logs/${M}_${L}_seed${S}.pid"
    echo "launched $M/$L seed $S on GPU $G (pid $!)"
  done
done
echo "$TOTAL processes running. Expected total DONE lines: $(( ${#JOBS[@]} * (4*${#SEEDS[@]} + 1) ))"
echo "progress: grep -h DONE $OUT/logs/*.log | wc -l"
