# H200 folder — hero study (Qwen3-0.6B-Base + T5Gemma 2 270M, 5 seeds)

Self-contained. Everything runs from this folder. mT5 is the finished Legacy Baseline (not rerun).

## 0. One-time
    cd ~/LoRA_Experiment/H200
    source <your uv env>/bin/activate        # torch, transformers, peft, datasets installed
    huggingface-cli login                    # T5Gemma 2 is gated (accept the license on its HF page first)

## 1. TEST everything first (~25-35 min, ONE GPU) — the full command
    cd ~/LoRA_Experiment/H200 && GPU=0 nohup bash run_test.sh > test.log 2>&1 &
    tail -f test.log          # Ctrl-C only stops the tail, not the test

Stages: (0) GPU + gated-model + dataset access, (1) smoke of every condition for both models and both
languages incl. the gradient-checkpointing control, (2) real 2-epoch training must beat the frozen model,
(3) speed at real settings -> prints the estimated hours for the full run, (4) analysis script.
Success ends with:  `############ ALL STAGES PASSED -> safe to run: bash run_train.sh ############`
and the file `results_test/PASSED`.  Any `FAIL` line = do not start training; send me `test.log`.

## 2. TRAIN (only after PASSED) — 4 jobs, one per GPU, resumable
    GPUS="0 1 2 3" bash run_train.sh
    grep -h DONE results_train/logs/*.log | tail      # progress
    python analyze.py results_train                   # any time

## 3. Send back
    tar czf results_train.tgz results_train results_test/logs && (upload results_train.tgz)
