#!/usr/bin/env bash
# Re-run Qwen3-0.6B-Base and T5Gemma 2 (270M) for QLoRA, seed 42, Arabic + Malay, config B (standard fine-tuning) and
# config C (NLI reward), logging EVERY held-out answer (--log_all_examples). 4 jobs run in parallel on one H200.
#
#   bash run_h200_allanswers.sh            # from the repo root, anywhere with the 2 matrix files at src/lora_nli/
#
# Output (fresh folder, so the old checkpoints are never skipped or overwritten):
#   ~/h200_allanswers/results/first_study/qwen3/checkpoint_full_qwen_qlora_{arabic,malay}.jsonl
#   ~/h200_allanswers/results/first_study/t5gemma2/checkpoint_full_qlora_{arabic,malay}.jsonl
set -euo pipefail
REPO="${REPO:-$PWD}"
RUN="$HOME/h200_allanswers"
mkdir -p "$RUN/logs" && cd "$RUN"            # NOT a git repo, so nothing is pushed
cp "$REPO/src/lora_nli/qwen_matrix.py" "$REPO/src/lora_nli/t5gemma2_matrix.py" .
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false

for lang in arabic malay; do
  nohup python qwen_matrix.py       --full --bc_only --variant qlora --lang "$lang" --seed 42 --log_all_examples \
        > "logs/qwen3_${lang}.log"    2>&1 &
  nohup python t5gemma2_matrix.py   --full --bc_only --variant qlora --lang "$lang" --seed 42 --log_all_examples --no_git \
        > "logs/t5gemma2_${lang}.log" 2>&1 &
done
echo "4 jobs started. Watch:  tail -f $RUN/logs/*.log   |  nvidia-smi"
wait
echo "ALL DONE"
