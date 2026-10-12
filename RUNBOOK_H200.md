# H200 rerun runbook (corrected REINFORCE estimator)

All commands run on the GPU node (`warisan-gpu-01`, 4x H200). Everything new is env-gated; the default path reproduces the legacy results.

## 0. Get the code
```bash
cd ~ && rm -rf LoRA_Experiment && git clone https://github.com/Akramtaha98/LoRA_Experiment.git && cd LoRA_Experiment
git log --oneline | head -3          # must show the "PG-fix rerun" commit
nvidia-smi --query-gpu=name,driver_version --format=csv ; nproc
```
## 1. Environment
```bash
python3 -m venv ~/venv_pgfix && source ~/venv_pgfix/bin/activate
pip install -q torch transformers peft datasets accelerate bitsandbytes sentencepiece tabulate pandas scipy
python - <<'PY'
import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.device_count())
PY
pip freeze > env_pgfix_freeze.txt
```
## 2. Gate: unit tests (stop if either fails)
```bash
python analysis/estimator_checks_fixed.py        # must end with ALL_PASS
```
## 3. Gate: one-job GPU smoke test (~10-20 min)
```bash
LORA_PG_FIX=1 LORA_OUTPUT_DIR=./smoke_gpu CUDA_VISIBLE_DEVICES=0 python lora_experiment_matrix.py --smoke_test --no_git
```
## 4. Stages (each is restartable; finished jobs are skipped)
Use `tmux` so the SSH session can drop. Add `--nli-gpu` to move the NLI scorer to the GPU (faster; applies identically to B and C). Start with `--slots-per-gpu 2`; raise to 3 if `nproc` >= 64 and GPU memory < 60%.
```bash
tmux new -s pgfix
source ~/venv_pgfix/bin/activate
python run_pgfix.py --stage core --slots-per-gpu 2 --nli-gpu --dry-run     # inspect: 18 jobs
python run_pgfix.py --stage core --slots-per-gpu 2 --nli-gpu 2>&1 | tee core.log
```
Monitor from a second tmux pane: `nvidia-smi -l 30` and `tail -n 3 runs_pgfix/core/*/run.log`.
Then, in order of priority:
```bash
python run_pgfix.py --stage repeat   --slots-per-gpu 2 --nli-gpu 2>&1 | tee repeat.log     # 8 jobs, determinism / same-seed
python run_pgfix.py --stage controls --slots-per-gpu 2 --nli-gpu 2>&1 | tee controls.log   # 24 jobs, ckpt/lambda0/shuffled
python run_pgfix.py --stage extra    --slots-per-gpu 2 --nli-gpu 2>&1 | tee extra.log      # 12 jobs, AdaLoRA/VeRA (optional)
```
Failed jobs print `FAILED`; just rerun the same command (it resumes).
## 5. Collect and pack
```bash
python collect_pgfix.py                                  # runs_pgfix/summary.csv + C-B table
tar czf pgfix_results.tgz runs_pgfix/summary.csv runs_pgfix/*/*/checkpoint_*.jsonl runs_pgfix/*/*/run.log env_pgfix_freeze.txt core.log repeat.log controls.log extra.log 2>/dev/null
ls -lh pgfix_results.tgz ; sha256sum pgfix_results.tgz
```
## 6. Bring results back
From the Mac: `scp <user>@warisan-gpu-01:~/LoRA_Experiment/pgfix_results.tgz ~/LoRA_Experiment/` then give it to Claude (attach or say the folder) to rewrite the results section.
## 7. Cleanup
```bash
tmux kill-server ; deactivate ; rm -rf ~/venv_pgfix ~/LoRA_Experiment/smoke_gpu
```
Revoke the old GitHub token and stop the GPU instance.

## What each stage answers
| Stage | Reviewer issue |
|---|---|
| core | Corrected-estimator B vs C (DoRA, plain LoRA, QLoRA; both languages; 3 seeds) |
| repeat | Same-seed discrepancy: 2 nondeterministic + 2 deterministic repeats per cell, hardware logged per run |
| controls | Checkpointing on/off, lambda1=0 through composite path, shuffled-reward control |
| extra | AdaLoRA / VeRA under the corrected estimator |

Per-run records now include pg_fix, reward_mode, deterministic, gpu, driver, torch/CUDA/transformers/peft versions.
