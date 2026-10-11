#!/usr/bin/env python3
"""Empty-output audit for the mT5 seed-42 runs that logged every held-out generation.

Reports, per (variant, language, config): number of held-out examples, the share of empty
generations, F_faith over all examples, F_faith over non-empty generations only, and EM.
Only runs whose checkpoint stores all held-out examples (720 Arabic / 430 Malay) can be audited;
the seed-123/2026 and other variants store 5 sample generations only and are skipped.
Usage: python analysis/empty_output_audit.py  (writes analysis/empty_output_audit.csv)
"""
import csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(ROOT, "experiment_results", "checkpoint_full_*_*.jsonl"))):
    if "seed" in f or "lambda" in f:
        continue
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            s = json.loads(d["samples"]) if isinstance(d["samples"], str) else d["samples"]
        except Exception:
            continue
        if d.get("status") != "PASS" or len(s) not in (720, 430) or "f_faith" not in s[0]:
            continue
        key = (d["lora_variant"], d["language"], d["config"], d["seed"])
        seen.add(key)
        gen = [str(x.get("generated", "")).strip() for x in s]
        ff = [float(x["f_faith"]) for x in s]
        em = [float(x["em"]) for x in s]
        ne = [i for i, g in enumerate(gen) if g]
        rows.append({
            "variant": d["lora_variant"], "language": d["language"], "config": d["config"],
            "seed": d["seed"], "n": len(s), "empty": len(s) - len(ne),
            "empty_rate": round((len(s) - len(ne)) / len(s), 4),
            "f_faith_all": round(sum(ff) / len(ff), 4),
            "f_faith_nonempty": round(sum(ff[i] for i in ne) / max(1, len(ne)), 4),
            "em": round(sum(em) / len(em), 4),
        })
# keep the last record per condition (later lines supersede earlier ones)
last = {}
for r in rows:
    last[(r["variant"], r["language"], r["config"], r["seed"])] = r
out = sorted(last.values(), key=lambda r: (r["variant"], r["language"], r["config"]))
with open(os.path.join(HERE, "empty_output_audit.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
for r in out:
    print(r["variant"], r["language"], r["config"][0], "n", r["n"], "empty", r["empty"], r["empty_rate"],
          "ff_all", r["f_faith_all"], "ff_nonempty", r["f_faith_nonempty"], "em", r["em"])
