#!/usr/bin/env python3
"""
Launcher for the plain-LoRA baseline (reviewer-requested), run on ONE GPU.

What it runs
------------
Configs B (CE loss) and C (composite/SCST loss) with plain, unmodified LoRA
(`--variant lora`), for both languages and three seeds (42, 123, 2026):

    2 configs x 2 languages x 3 seeds = 12 training runs

Each launched process handles one (language, seed) pair and runs Config B then
Config C back to back (`--bc_only`), so there are 6 processes. They are run
`--parallel` at a time on the same GPU. Protocol is identical to every other
run in the paper (12 epochs, AdamW 2e-4, r=8, alpha=16, dropout 0.05, same
target modules, same fixed 470-example split) because it reuses the same
script, `lora_experiment_matrix.py`.

Usage (run from the folder that contains lora_experiment_matrix.py)
-------------------------------------------------------------------
    python3 run_plain_lora.py                 # 3 processes at a time (default)
    python3 run_plain_lora.py --parallel 2    # if you hit out-of-memory errors
    python3 run_plain_lora.py --parallel 6    # everything at once (needs ~24GB+)
    python3 run_plain_lora.py --dry_run       # print the commands, run nothing
    python3 run_plain_lora.py --summary_only  # just print results gathered so far

It is safe to re-run after an interruption: the training script resumes from
its per-shard checkpoint files and skips runs that already PASSED.

Outputs
-------
    experiment_results/checkpoint_full_lora_<lang>[_seed<N>].jsonl   (results)
    plain_lora_logs/<lang>_seed<N>.log                               (stdout)
    experiment_results/plain_lora_summary.csv                        (summary)
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = "lora_experiment_matrix.py"
LANGS = ["arabic", "malay"]
SEEDS = [42, 123, 2026]
LOG_DIR = Path("plain_lora_logs")
RESULTS_DIR = Path("experiment_results")


def build_jobs():
    jobs = []
    for seed in SEEDS:
        for lang in LANGS:
            cmd = [sys.executable, SCRIPT, "--full", "--bc_only",
                   "--variant", "lora", "--lang", lang,
                   "--seed", str(seed), "--no_git"]
            jobs.append({"name": f"{lang}_seed{seed}", "cmd": cmd})
    return jobs


def print_summary():
    rows = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "checkpoint_full_lora_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "config": r.get("config"), "language": r.get("language"),
                    "seed": r.get("seed"), "status": r.get("status"),
                    "F_faith": r.get("mean_f_faith"), "EM": r.get("mean_em"),
                    "F1": r.get("mean_f1"), "n_eval": r.get("n_eval"),
                    "runtime_min": round((r.get("runtime_sec") or 0) / 60, 1),
                    "error": r.get("error", ""),
                })
    if not rows:
        print("No plain-LoRA results found yet.")
        return
    rows.sort(key=lambda r: (r["language"], r["config"], r["seed"]))
    hdr = ["config", "language", "seed", "status", "F_faith", "EM", "F1",
           "n_eval", "runtime_min"]
    print("\n" + " | ".join(f"{h:>16}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{str(r[h]):>16}" for h in hdr))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "plain_lora_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(r["status"] == "PASS" for r in rows)
    print(f"\n{n_pass}/12 runs PASSED. Summary written to {out}")
    if n_pass < 12:
        print("Some runs are missing or FAILED. Re-run this launcher to "
              "resume; check plain_lora_logs/*.log for errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3,
                    help="Processes to run at once on this GPU (default 3).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        print_summary()
        return
    if not Path(SCRIPT).exists():
        sys.exit(f"ERROR: {SCRIPT} not found in {os.getcwd()}. "
                 f"Run this launcher from the folder that contains it.")

    jobs = build_jobs()
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.dry_run:
        for j in jobs:
            print(" ".join(j["cmd"]))
        return

    print(f"Launching {len(jobs)} processes, {args.parallel} at a time "
          f"(each runs Config B then Config C).")
    pending, running = list(jobs), []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            log = open(LOG_DIR / f"{job['name']}.log", "a", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT)
            running.append((job, proc, log))
            print(f"[{(time.time()-t0)/60:6.1f} min] started  {job['name']}")
            time.sleep(20)  # stagger starts so model downloads/loads don't collide
        for item in running[:]:
            job, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "ok" if proc.returncode == 0 else f"EXIT {proc.returncode}"
                print(f"[{(time.time()-t0)/60:6.1f} min] finished {job['name']} ({status})")
        time.sleep(10)

    print(f"\nAll processes finished in {(time.time()-t0)/60:.1f} min.")
    print_summary()


if __name__ == "__main__":
    main()
