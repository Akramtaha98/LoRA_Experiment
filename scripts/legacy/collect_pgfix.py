#!/usr/bin/env python3
"""Merge runs_pgfix/**/checkpoint_*.jsonl into runs_pgfix/summary.csv (drops the bulky per-example 'samples' column)
and print a Config C - B table per variant x language (mean over seeds)."""
import json, glob, sys
import pandas as pd
rows = []
for f in glob.glob("runs_pgfix/*/*/checkpoint_*.jsonl"):
    stage, job = f.split("/")[1:3]
    for line in open(f, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); r.pop("samples", None); r["stage"], r["job"] = stage, job; rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("runs_pgfix/summary.csv", index=False)
print(f"{len(df)} rows; status counts:\n{df.status.value_counts()}")
core = df[(df.stage == "core") & (df.status == "PASS")]
if len(core):
    piv = core.pivot_table(index=["lora_variant", "language", "seed"], columns="config", values="mean_f_faith")
    piv["C_minus_B"] = piv["C_composite_lora"] - piv["B_ce_lora"]
    print(piv.groupby(level=[0, 1])["C_minus_B"].agg(["mean", "std", "count"]).round(4))
