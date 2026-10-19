"""Aggregate hero_v2 results over seeds.  Usage: python scripts/analyze_hero_v2.py results/hero_v2
Outputs <root>/analysis/{per_seed.csv, summary.md, contrasts.json}.
Contrasts (paired by example, averaged over seeds, bootstrap 95% CI over examples; plus
seed-level sign counts): C-B, C-C_l0, C-C_shuf on faith, faith_2nd, f1, em, option_correct, n_words."""
import json, sys, csv, random
from pathlib import Path
from collections import defaultdict
import numpy as np

root = Path(sys.argv[1]); out = root / "analysis"; out.mkdir(exist_ok=True)
METR = ["faith", "faith_2nd", "f1", "em", "option_correct", "n_words"]
runs = defaultdict(lambda: defaultdict(dict))        # (model,lang) -> cond -> seed -> {ex_id: row}
for f in root.glob("*/*/*/seed*/predictions.jsonl"):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    if not rows: continue
    r0 = rows[0]
    runs[(r0["model"], r0["lang"])][r0["cond"]][r0["seed"]] = {r["ex_id"]: r for r in rows}

def val(r, m):
    v = r.get(m); return None if v is None else float(v)

def seed_mean(d, m):
    xs = [val(r, m) for r in d.values() if val(r, m) is not None]
    return float(np.mean(xs)) if xs else None

with open(out / "per_seed.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "lang", "cond", "seed", "n"] + METR)
    for (mk, lg), cs in sorted(runs.items()):
        for c, ss in sorted(cs.items()):
            for s, d in sorted(ss.items()):
                w.writerow([mk, lg, c, s, len(d)] + [seed_mean(d, m) for m in METR])

def per_example(cs, c, m):
    """example -> mean over seeds of metric m for condition c"""
    acc = defaultdict(list)
    for d in cs.get(c, {}).values():
        for i, r in d.items():
            v = val(r, m)
            if v is not None: acc[i].append(v)
    return {i: float(np.mean(v)) for i, v in acc.items()}

def boot(diffs, n=4000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(diffs)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

summary, contr = [], {}
for (mk, lg), cs in sorted(runs.items()):
    summary.append(f"\n### {mk} / {lg}\n\n| cond | seeds | " + " | ".join(METR) + " |\n|---|---|" + "---|" * len(METR))
    for c in ["A_frozen", "B", "B_ckpt", "C", "C_l0", "C_shuf"]:
        if c not in cs: continue
        cells = []
        for m in METR:
            xs = [x for x in (seed_mean(d, m) for d in cs[c].values()) if x is not None]
            cells.append("-" if not xs else f"{np.mean(xs):.3f}±{np.std(xs, ddof=1) if len(xs) > 1 else 0:.3f}")
        summary.append(f"| {c} | {len(cs[c])} | " + " | ".join(cells) + " |")
    for a, b in [("C", "B"), ("C", "C_l0"), ("C", "C_shuf")]:
        if a not in cs or b not in cs: continue
        for m in METR:
            pa, pb = per_example(cs, a, m), per_example(cs, b, m)
            ids = sorted(set(pa) & set(pb))
            if not ids: continue
            diffs = [pa[i] - pb[i] for i in ids]
            lo, hi = boot(diffs)
            common = sorted(set(cs[a]) & set(cs[b]))
            sd = [seed_mean(cs[a][s], m) - seed_mean(cs[b][s], m) for s in common]
            contr[f"{mk}/{lg}/{a}-{b}/{m}"] = dict(mean_diff=float(np.mean(diffs)), ci95=[lo, hi], n_examples=len(ids),
                                                   seeds_positive=int(sum(x > 0 for x in sd)), n_seeds=len(sd))
(out / "contrasts.json").write_text(json.dumps(contr, indent=2))
summary.append("\n### Key contrasts (mean diff [95% bootstrap CI over examples], seeds with C>other)\n")
for k, v in contr.items():
    if k.endswith(("/faith", "/f1")):
        summary.append(f"- {k}: {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
                       f"{v['seeds_positive']}/{v['n_seeds']} seeds")
(out / "summary.md").write_text("\n".join(summary))
print("\n".join(summary)); print(f"\nwritten to {out}")
