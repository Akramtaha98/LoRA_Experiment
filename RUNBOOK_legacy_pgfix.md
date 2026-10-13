# Runbook: corrected-estimator study (188 runs, 118 jobs)

Hardware used: one node with 4x NVIDIA H200 NVL, driver 595.71.05, CUDA 13.0. Software: PyTorch 2.12.1, Transformers 5.5.0, PEFT 0.21.2
(plus `bitsandbytes`, `sentencepiece`, `pandas`, `scipy`). Each run record stores GPU, driver and library versions.

## 1. Environment and unit test
```bash
python3 -m venv ~/venv_pgfix && source ~/venv_pgfix/bin/activate
pip install torch transformers peft datasets accelerate bitsandbytes sentencepiece tabulate pandas scipy
pip freeze > env_pgfix_freeze.txt
python analysis/estimator_checks_fixed.py        # must end with ALL_PASS
```

## 2. Stages (restartable; finished jobs are skipped)
Use `tmux` so the session can drop. `--nli-gpu` runs the NLI scorer on the GPU; `--slots-per-gpu 2` runs two jobs per GPU.
```bash
python run_pgfix.py --stage core     --slots-per-gpu 2 --nli-gpu   # 18 jobs: LoRA/QLoRA/DoRA x 2 languages x 3 seeds, B and C, original checkpointing policy
python run_pgfix.py --stage extra    --slots-per-gpu 2 --nli-gpu   # 12 jobs: AdaLoRA/VeRA
python run_pgfix.py --stage controls --slots-per-gpu 2 --nli-gpu   # 24 jobs, seed 42: checkpointing on/off, lambda1=0, shuffled reward
python run_pgfix.py --stage repeat   --slots-per-gpu 2 --nli-gpu   # 8 jobs: determinism repeats
python run_pgfix.py --stage ckptoff  --slots-per-gpu 2 --nli-gpu   # 20 jobs: five adapters, seeds 123/2026, checkpointing OFF
python run_pgfix.py --stage attrib   --slots-per-gpu 2 --nli-gpu   # 36 jobs: lambda1=0 and shuffled reward, checkpointing OFF, 3 seeds
```
Add `--dry-run` to list jobs. Failed jobs print `FAILED`; rerun the same command to resume.

## 3. Collect and analyse
```bash
python collect_pgfix.py                                      # runs_pgfix/summary.csv
cd analysis/pgfix_corrected && python merge.py ../..          # rebuild all.csv from raw runs
python tables.py && python stats.py && python mk_tables.py && python mk_figs.py
```
Total compute: about 62 GPU-hours (Config B runs 3-8 minutes, Config C runs 20-45 minutes).
