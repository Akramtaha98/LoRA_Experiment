#!/usr/bin/env python3
"""LORA_V2 rerun launcher (corrected training path + corrected evaluation). One process = one variant x language x seed.

PRE-SPECIFIED ANALYSIS PLAN (fixed before any v2 data are seen; see RUNBOOK_V2.md, section "Analysis plan"):
  seeds: 8 per cell = [42, 123, 2026, 7, 11, 13, 17, 19]; no seed is added or dropped after seeing results.
Stages:
  core        dora, lora, qlora x {arabic, malay} x 8 seeds, Config B + C, checkpointing off                (48 jobs)
  controls    same cells, Config C only: lambda1 = 0, and shuffled reward                                  (96 jobs)
  ckptfact    dora, lora, qlora x 2 langs x seeds {42,123,2026}, B + C with checkpointing ON               (18 jobs)
  policymode  Config C with the legacy policy branch (dropout ON, LORA_POLICY_MODE=train), same 3 seeds    (18 jobs)
  adaptive    adalora (rank reallocation active) and vera x 2 langs x 3 seeds, B + C                       (12 jobs)
  baselines   Config A (frozen) and D (full fine-tuning), both languages (D: 3 seeds)                       (8 jobs)
Usage: python scripts/run_v2.py --stage core [--gpus 0,1,2,3] [--slots-per-gpu 2] [--seeds 42,123,...] [--dry-run]
"""
import argparse, itertools, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs_v2"
SEEDS = [42, 123, 2026]
LANGS = ["arabic", "malay"]

def job(stage, name, variant, lang, seed, extra=(), env=None):
    return dict(stage=stage, name=name, variant=variant, lang=lang, seed=seed, extra=list(extra), env=env or {})

SEEDS8 = [42, 123, 2026, 7, 11, 13, 17, 19]
SEEDS3 = [42, 123, 2026]
STAGES = ["core", "controls", "ckptfact", "policymode", "adaptive", "baselines"]

def build(stage, seeds):
    J = []
    V3 = ["dora", "lora", "qlora"]
    if stage in ("core", "all"):
        for v, l, s in itertools.product(V3, LANGS, seeds):
            J.append(job("core", f"{v}_{l}_s{s}", v, l, s, ["--bc_only"]))
    if stage in ("controls", "all"):
        for v, l, s in itertools.product(V3, LANGS, seeds):
            J.append(job("controls", f"lambda0_{v}_{l}_s{s}", v, l, s, ["--composite_only", "--lambda1", "0.0"]))
            J.append(job("controls", f"shuffled_{v}_{l}_s{s}", v, l, s, ["--composite_only"], {"LORA_REWARD_MODE": "shuffled"}))
    if stage in ("ckptfact", "all"):
        for v, l, s in itertools.product(V3, LANGS, SEEDS3):
            J.append(job("ckptfact", f"ckpton_{v}_{l}_s{s}", v, l, s, ["--bc_only", "--grad_ckpt", "on"]))
    if stage in ("policymode", "all"):
        for v, l, s in itertools.product(V3, LANGS, SEEDS3):
            J.append(job("policymode", f"policytrain_{v}_{l}_s{s}", v, l, s, ["--composite_only"], {"LORA_POLICY_MODE": "train"}))
    if stage in ("adaptive", "all"):
        for v, l, s in itertools.product(["adalora", "vera"], LANGS, SEEDS3):
            J.append(job("adaptive", f"{v}_{l}_s{s}", v, l, s, ["--bc_only"]))
    if stage in ("baselines", "all"):
        for l in LANGS:
            J.append(job("baselines", f"A_{l}", None, l, 42, ["--ad_only"]))
            for s in SEEDS3:
                J.append(job("baselines", f"D_{l}_s{s}", None, l, s, ["--ad_only"]))
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--nli-gpu", action="store_true", help="run the NLI scorer on the GPU (faster; same for all arms)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS8)), help="override seeds for core/controls (default: the 8 pre-specified)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gpus = a.gpus.split(",")
    nslots = len(gpus) * a.slots_per_gpu
    threads = max(1, (os.cpu_count() or 8) // nslots)
    jobs = build(a.stage, [int(x) for x in a.seeds.split(",")])
    todo = [j for j in jobs if not (OUT / j["stage"] / j["name"] / "DONE").exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {nslots} slots, {threads} CPU threads each")
    if a.dry_run:
        for j in todo: print(" ", j["stage"], j["name"], j["extra"], j["env"])
        return
    running = {}   # slot -> (Popen, job)
    slots = [(g, k) for g in gpus for k in range(a.slots_per_gpu)]
    q = list(todo)
    while q or running:
        for slot in slots:
            if slot in running:
                p, j = running[slot]
                if p.poll() is not None:
                    d = OUT / j["stage"] / j["name"]
                    ok = p.returncode == 0 and any(d.glob("checkpoint_*.jsonl"))
                    if ok: (d / "DONE").write_text(time.strftime("%F %T"))
                    print(f"[{time.strftime('%T')}] {'DONE ' if ok else 'FAILED'} {j['stage']}/{j['name']} (gpu {slot[0]})", flush=True)
                    del running[slot]
            if slot not in running and q:
                j = q.pop(0)
                d = OUT / j["stage"] / j["name"]; d.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], LORA_V2="1", LORA_EVAL_ALSO_TRAINMODE="1",
                           LORA_OUTPUT_DIR=str(d), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                           TOKENIZERS_PARALLELISM="false")
                if a.nli_gpu: env["LORA_NLI_DEVICE"] = "cuda"
                env.update(j["env"])
                cmd = [sys.executable, str(ROOT / "src" / "lora_nli" / "matrix.py"), "--full", "--no_git",
                       *(["--variant", j["variant"]] if j["variant"] else []), "--lang", j["lang"],
                       "--seed", str(j["seed"]), "--log_all_examples"] + j["extra"]
                log = open(d / "run.log", "a")
                running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), j)
                print(f"[{time.strftime('%T')}] START {j['stage']}/{j['name']} on gpu {slot[0]}", flush=True)
        time.sleep(20)
    print("stage finished")

if __name__ == "__main__":
    main()
