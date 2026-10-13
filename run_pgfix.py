#!/usr/bin/env python3
"""Corrected-estimator rerun launcher for a multi-GPU node (e.g. 4x H200).

Every job is one `lora_experiment_matrix.py` process (one variant x language x seed; Config B and C are trained
back-to-back in the SAME process/environment). Jobs are packed onto GPUs (--slots-per-gpu processes per GPU),
each in its own output directory under runs_pgfix/<stage>/<job>/, and finished jobs are skipped on restart.

Stages:  core | extra | controls | repeat | ckptoff | attrib | all
  core      dora, lora, qlora x {arabic, malay} x seeds {42,123,2026}, --bc_only            (18 jobs)
  extra     adalora, vera    x {arabic, malay} x seeds {42,123,2026}, --bc_only            (12 jobs)
  controls  seed 42, Config C only: gradient-checkpointing on/off (B+C), lambda1=0 via composite path,
            shuffled-reward control; dora/lora/qlora x both languages                      (24 jobs)
  repeat    dora, qlora x malay, seed 42, B+C, 2 nondeterministic + 2 deterministic repeats (8 jobs)
  ckptoff   all five adapters x both languages x seeds {123,2026}, B+C with Config C gradient checkpointing OFF (20 jobs)
  attrib    dora, lora, qlora x both languages x 3 seeds, Config C with checkpointing OFF: lambda1=0 and shuffled reward (36 jobs)
Usage:  python run_pgfix.py --stage core [--gpus 0,1,2,3] [--slots-per-gpu 2] [--dry-run]
"""
import argparse, itertools, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs_pgfix"
SEEDS = [42, 123, 2026]
LANGS = ["arabic", "malay"]

def job(stage, name, variant, lang, seed, extra=(), env=None):
    return dict(stage=stage, name=name, variant=variant, lang=lang, seed=seed, extra=list(extra), env=env or {})

def build(stage):
    J = []
    if stage in ("core", "all"):
        for v, l, s in itertools.product(["dora", "lora", "qlora"], LANGS, SEEDS):
            J.append(job("core", f"{v}_{l}_s{s}", v, l, s, ["--bc_only"]))
    if stage in ("extra", "all"):
        for v, l, s in itertools.product(["adalora", "vera"], LANGS, SEEDS):
            J.append(job("extra", f"{v}_{l}_s{s}", v, l, s, ["--bc_only"]))
    if stage in ("controls", "all"):
        for v, l in itertools.product(["dora", "lora", "qlora"], LANGS):
            for ck in ("on", "off"):
                J.append(job("controls", f"ckpt{ck}_{v}_{l}_s42", v, l, 42, ["--bc_only", "--grad_ckpt", ck]))
            J.append(job("controls", f"lambda0_{v}_{l}_s42", v, l, 42, ["--composite_only", "--lambda1", "0.0"]))
            J.append(job("controls", f"shuffled_{v}_{l}_s42", v, l, 42, ["--composite_only"],
                         {"LORA_REWARD_MODE": "shuffled"}))
    if stage in ("ckptoff", "all"):   # B+C with gradient checkpointing OFF for Config C, seeds 123/2026 (42 is in controls)
        for v, l, sd in itertools.product(["dora", "lora", "qlora", "adalora", "vera"], LANGS, [123, 2026]):
            J.append(job("ckptoff", f"ckptoff_{v}_{l}_s{sd}", v, l, sd, ["--bc_only", "--grad_ckpt", "off"]))
    if stage in ("attrib", "all"):   # reward attribution with checkpointing OFF: lambda1=0 and shuffled reward, all seeds
        for v, l, sd in itertools.product(["dora", "lora", "qlora"], LANGS, SEEDS):
            J.append(job("attrib", f"lambda0_{v}_{l}_s{sd}", v, l, sd,
                         ["--composite_only", "--grad_ckpt", "off", "--lambda1", "0.0"]))
            J.append(job("attrib", f"shuffled_{v}_{l}_s{sd}", v, l, sd,
                         ["--composite_only", "--grad_ckpt", "off"], {"LORA_REWARD_MODE": "shuffled"}))
    if stage in ("repeat", "all"):
        for v in ("dora", "qlora"):
            for tag, det in (("nd1", "0"), ("nd2", "0"), ("det1", "1"), ("det2", "1")):
                J.append(job("repeat", f"{tag}_{v}_malay_s42", v, "malay", 42, ["--bc_only"],
                             {"LORA_DETERMINISTIC": det}))
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["core", "extra", "controls", "repeat", "ckptoff", "attrib", "all"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--nli-gpu", action="store_true", help="run the NLI scorer on the GPU (faster; same for all arms)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gpus = a.gpus.split(",")
    nslots = len(gpus) * a.slots_per_gpu
    threads = max(1, (os.cpu_count() or 8) // nslots)
    jobs = build(a.stage)
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
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], LORA_PG_FIX="1",
                           LORA_OUTPUT_DIR=str(d), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                           TOKENIZERS_PARALLELISM="false")
                if a.nli_gpu: env["LORA_NLI_DEVICE"] = "cuda"
                env.update(j["env"])
                cmd = [sys.executable, str(ROOT / "lora_experiment_matrix.py"), "--full", "--no_git",
                       "--variant", j["variant"], "--lang", j["lang"], "--seed", str(j["seed"]),
                       "--log_all_examples"] + j["extra"]
                log = open(d / "run.log", "a")
                running[slot] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), j)
                print(f"[{time.strftime('%T')}] START {j['stage']}/{j['name']} on gpu {slot[0]}", flush=True)
        time.sleep(20)
    print("stage finished")

if __name__ == "__main__":
    main()
