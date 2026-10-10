#!/usr/bin/env python3
"""T5Gemma 2 runner. Each job = Config B (CE) + Config C (composite) for one variant/lang/seed (same seed => matched pair).
  python3 run_t5gemma2.py --stage pilot --parallel 3     # seed 42, lora/dora/qlora x ar/ms = 6 jobs (competence gate)
  python3 run_t5gemma2.py --stage rest  --parallel 3     # seeds 123, 2026 = 12 jobs
  python3 run_t5gemma2.py --summary_only
Safe to re-run (finished runs are skipped by the script's checkpoint logic)."""
import argparse, glob, json, subprocess, sys, time
from pathlib import Path
SCRIPT = "t5gemma2_experiment_matrix.py"; LOG = Path("t5gemma2_logs")
def jobs(seeds, variants):
    return [{"name": f"{v}_{l}_seed{s}", "cmd": [sys.executable, SCRIPT, "--full", "--bc_only", "--variant", v,
             "--lang", l, "--seed", str(s), "--no_git"]} for s in seeds for v in variants for l in ("arabic", "malay")]
def summary():
    print(f"{'file':62s} {'config':18s} status  F_faith    EM     F1")
    for p in sorted(glob.glob("experiment_results_t5gemma2/checkpoint_full_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                print(f"{Path(p).name:62s} {str(r.get('config')):18s} {r.get('status')}  {r.get('mean_f_faith')}  {r.get('mean_em')}  {r.get('mean_f1')}")
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["pilot", "rest", "all"], default="pilot")
ap.add_argument("--variants", nargs="+", default=["lora", "dora", "qlora"])
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--dry_run", action="store_true"); ap.add_argument("--summary_only", action="store_true")
a = ap.parse_args()
if a.summary_only: summary(); sys.exit()
js = jobs({"pilot":[42],"rest":[123,2026],"all":[42,123,2026]}[a.stage], a.variants)
if a.dry_run:
    for j in js: print(" ".join(j["cmd"]))
    sys.exit()
def any_fail():
    for p in glob.glob("experiment_results_t5gemma2/checkpoint_full_*.jsonl"):
        for l in open(p, encoding="utf-8"):
            if l.strip() and json.loads(l).get("status") == "FAIL": return True
    return False
LOG.mkdir(exist_ok=True); pend, run, t0, aborted = list(js), [], time.time(), False
while pend or run:
    while pend and len(run) < a.parallel and not aborted:
        j = pend.pop(0); f = open(LOG / f"{j['name']}.log", "a")
        run.append((j, subprocess.Popen(j["cmd"], stdout=f, stderr=subprocess.STDOUT), f)); print(f"[{(time.time()-t0)/60:6.1f} min] started {j['name']}", flush=True); time.sleep(20)
    for it in run[:]:
        if it[1].poll() is not None:
            it[2].close(); run.remove(it); print(f"[{(time.time()-t0)/60:6.1f} min] finished {it[0]['name']} rc={it[1].returncode}", flush=True)
            if any_fail() and not aborted:
                aborted = True; pend.clear(); print("!! A run FAILED - no new jobs will start. Check t5gemma2_logs/", flush=True)
    time.sleep(15)
summary()
sys.exit(1 if any_fail() else 0)
