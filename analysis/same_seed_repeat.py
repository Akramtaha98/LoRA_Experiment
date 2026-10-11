#!/usr/bin/env python3
"""Same-seed repeat check: the eight seed-42 mT5 cells (QLoRA/DoRA x Arabic/Malay x Config B/C) were trained twice
with the same variant, language, config, seed and lambda_1 = 0.3: once for the main 3-seed tables (per_seed_results.csv)
and once later as a repeat that also logged every held-out generation (seed42_rerun_lambda0.3.csv, also the lambda-sweep
lambda_1 = 0.3 point). This script quantifies how far the two runs differ.
Config B has no sampling step, so a difference there cannot come from the training-time sampling stream.
Usage: python analysis/same_seed_repeat.py   (writes analysis/same_seed_repeat.csv)
"""
import csv, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
main = {(r["variant"], r["language"], r["config"]): r for r in csv.DictReader(open(os.path.join(HERE, "per_seed_results.csv")))
        if r["backbone"] == "mt5" and r["seed"] == "42"}
allseeds = {}
for r in csv.DictReader(open(os.path.join(HERE, "per_seed_results.csv"))):
    if r["backbone"] == "mt5":
        allseeds.setdefault((r["variant"], r["language"], r["config"]), []).append(float(r["f_faith"]))
def sd(x):
    m = sum(x) / len(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
rows = []
for r in csv.DictReader(open(os.path.join(HERE, "seed42_rerun_lambda0.3.csv"))):
    k = (r["variant"], r["language"], r["config"]); m = main[k]
    rows.append(dict(variant=r["variant"], language=r["language"], config=r["config"][0],
                     ff_main=float(m["f_faith"]), ff_repeat=float(r["f_faith"]), diff=float(r["f_faith"]) - float(m["f_faith"]),
                     em_main=float(m["em"]), em_repeat=float(r["em"]), seed_sd=sd(allseeds[k]),
                     ratio=abs(float(r["f_faith"]) - float(m["f_faith"])) / sd(allseeds[k])))
with open(os.path.join(HERE, "same_seed_repeat.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows:
    print(f"{x['variant']:6s} {x['language']:7s} {x['config']}  main {x['ff_main']:.4f} repeat {x['ff_repeat']:.4f} diff {x['diff']:+.4f}  EM {x['em_main']:.4f}->{x['em_repeat']:.4f}  3-seed SD {x['seed_sd']:.4f}  |diff|/SD {x['ratio']:.1f}")
ad = [abs(x["diff"]) for x in rows]; adB = [abs(x["diff"]) for x in rows if x["config"] == "B"]; adC = [abs(x["diff"]) for x in rows if x["config"] == "C"]
print(f"mean |diff| all {sum(ad)/8:.4f}; Config B (no sampling) {sum(adB)/4:.4f}; Config C {sum(adC)/4:.4f}; max {max(ad):.4f}")
# C minus B (language-averaged) for QLoRA, main vs repeat
for v in ("qlora", "dora"):
    for tag in ("ff_main", "ff_repeat"):
        d = sum(x[tag] for x in rows if x["variant"] == v and x["config"] == "C") / 2 - sum(x[tag] for x in rows if x["variant"] == v and x["config"] == "B") / 2
        print(f"{v} language-averaged C-B at seed 42 ({tag}): {d:+.4f}")
