#!/usr/bin/env bash
# One-command pre-flight test for the hero_v2 study.  ~20-35 min on ONE H200.
#   bash run_test.sh                # uses GPU 0
#   GPU=2 bash run_test.sh          # pick another GPU
# Writes everything to ./results_test and creates ./results_test/PASSED only if all stages pass.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0}
T=results_test; rm -rf "$T"; mkdir -p "$T/logs"
PY=${PY:-python}
stage() { echo; echo "=================== $1 ==================="; }
die()   { echo "!!! STAGE FAILED: $1  (see $T/logs/)"; exit 1; }

stage "0/4 environment + model access (gated T5Gemma 2, Qwen3, NLI scorers)"
$PY - <<'PY' 2>&1 | tee $T/logs/0_env.log || die "environment"
import torch, transformers, peft, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_properties(0); print("GPU:", p.name, f"{p.total_memory/2**30:.0f} GB")
from huggingface_hub import hf_hub_download
for repo in ("google/t5gemma-2-270m-270m", "Qwen/Qwen3-0.6B-Base",
             "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "joeddav/xlm-roberta-large-xnli"):
    hf_hub_download(repo, "config.json"); print("access OK:", repo)
datasets.load_dataset("google/xquad", "xquad.ar", split="validation[:2]")
datasets.load_dataset("facebook/belebele", "zsm_Latn", split="test[:2]")
print("datasets OK")
PY

stage "1/4 smoke: 2 models x 2 languages x A/B/C/C_l0/C_shuf/B_ckpt (tiny slices, ~8 min)"
$PY hero_v2.py --smoke --models qwen3 t5gemma2 --langs arabic malay \
    --conds A_frozen B C C_l0 C_shuf --seeds 42 --ckpt_control --out $T/smoke 2>&1 | tee $T/logs/1_smoke.log \
    | grep -E "RUN|DONE|Traceback|Error" ; [ ${PIPESTATUS[0]} -eq 0 ] || die "smoke run crashed"
$PY check_results.py smoke $T/smoke 2>&1 | tee $T/logs/1_smoke_check.log || die "smoke checks"

stage "2/4 learning sanity: real train set, 2 epochs, Arabic, 128 held-out items (~6 min)"
for M in qwen3 t5gemma2; do
  $PY hero_v2.py --models $M --langs arabic --conds A_frozen B --seeds 42 --epochs 2 --eval_limit 128 \
      --no_second --out $T/learn 2>&1 | tee -a $T/logs/2_learn.log | grep -E "RUN|DONE|Traceback|Error"
  [ ${PIPESTATUS[0]} -eq 0 ] || die "learn run $M"
done
$PY check_results.py learn $T/learn 2>&1 | tee $T/logs/2_learn_check.log || die "learning check"

stage "3/4 timing at REAL settings (30 steps of B and C per model/language) -> ETA (~12 min)"
for M in qwen3 t5gemma2; do for L in arabic malay; do
  $PY hero_v2.py --models $M --langs $L --conds B C --seeds 42 --max_steps 30 --eval_limit 64 \
      --no_second --out $T/timing 2>&1 | tee -a $T/logs/3_timing.log | grep -E "DONE|Traceback|Error"
  [ ${PIPESTATUS[0]} -eq 0 ] || die "timing run $M/$L"
done; done
$PY check_results.py timing $T/timing 2>&1 | tee $T/logs/3_timing_check.log || die "timing"

stage "4/4 analysis script on the test outputs"
$PY analyze.py $T/smoke > $T/logs/4_analyze.log 2>&1 && echo "analysis OK" || die "analysis script"

date > $T/PASSED
echo; echo "############ ALL STAGES PASSED -> safe to run: bash run_train.sh ############"
