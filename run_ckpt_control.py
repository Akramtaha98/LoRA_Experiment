#!/usr/bin/env python3
"""
Gradient-checkpointing control (reviewer-requested). Config C only; Config B already exists for every cell.

  lora_off : plain LoRA, Config C, gradient checkpointing OFF   (QLoRA's training path; needs a 48GB+ GPU)
  qlora_on : QLoRA,      Config C, gradient checkpointing ON    (the other variants' training path; fits 24GB)

Usage (from the folder containing lora_experiment_matrix.py):
    python3 run_ckpt_control.py --which qlora_on --parallel 3     # 24GB+ GPU
    python3 run_ckpt_control.py --which lora_off --parallel 2     # 48GB GPU (80GB: --parallel 3)
    python3 run_ckpt_control.py --which lora_off --seeds 42 --dry_run
    python3 run_ckpt_control.py --summary_only
Safe to re-run: finished runs are skipped. Results: experiment_results/checkpoint_full_<variant>_<lang>[_seedN]_ckpt<on|off>.jsonl
"""
import argparse, glob, json, subprocess, sys, time
from pathlib import Path
SCRIPT = "lora_experiment_matrix.py"
CFG = {"lora_off": ("lora", "off"), "qlora_on": ("qlora", "on")}
LOG = Path("ckpt_control_logs")

def jobs(which, seeds):
    out = []
    for w in which:
        var, mode = CFG[w]
        for seed in seeds:
            for lang in ("arabic", "malay"):
                out.append({"name": f"{var}_ckpt{mode}_{lang}_seed{seed}",
                            "cmd": [sys.executable, SCRIPT, "--full", "--composite_only", "--variant", var,
                                    "--lang", lang, "--seed", str(seed), "--grad_ckpt", mode, "--no_git"]})
    return out

def summary():
    print(f"{'file':70s} status  F_faith    EM     F1  minutes")
    for p in sorted(glob.glob("experiment_results/checkpoint_full_*_ckpt*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                print(f"{Path(p).name:70s} {r.get('status')}  {r.get('mean_f_faith')}  {r.get('mean_em')}  {r.get('mean_f1')}  {round((r.get('runtime_sec') or 0)/60,1)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["lora_off", "qlora_on", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    if a.summary_only: return summary()
    js = jobs(["lora_off", "qlora_on"] if a.which == "both" else [a.which], a.seeds)
    if a.dry_run:
        for j in js: print(" ".join(j["cmd"]))
        return
    LOG.mkdir(exist_ok=True); pend, run, t0 = list(js), [], time.time()
    while pend or run:
        while pend and len(run) < a.parallel:
            j = pend.pop(0); f = open(LOG / f"{j['name']}.log", "a")
            run.append((j, subprocess.Popen(j["cmd"], stdout=f, stderr=subprocess.STDOUT), f))
            print(f"[{(time.time()-t0)/60:6.1f} min] started {j['name']}", flush=True); time.sleep(20)
        for it in run[:]:
            if it[1].poll() is not None:
                it[2].close(); run.remove(it)
                print(f"[{(time.time()-t0)/60:6.1f} min] finished {it[0]['name']} (exit {it[1].returncode})", flush=True)
        time.sleep(15)
    summary()

if __name__ == "__main__":
    main()
