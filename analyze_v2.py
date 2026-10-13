#!/usr/bin/env python3
"""Pre-specified analysis of the LORA_V2 reruns (95% CIs are primary; p-values are secondary).

  python analyze_v2.py RUNS_DIR [--out analysis_v2] [--margin 0.01] [--n-boot 5000]

Inputs : RUNS_DIR/**/checkpoint_*.jsonl (one record per run) and the per-example prediction files they point to.
Outputs: conditions.csv (all runs, labelled), cells.csv (cell means), contrasts.csv, equivalence in contrasts.csv,
         item_bootstrap.json (cluster bootstrap over source items), secondary.csv, summary.md.

Estimand (primary): Delta = mean over the 6 adapter x language cells {dora,lora,qlora} x {arabic,malay} of
    F_faith(Config C, lambda1=0.3, real reward) - F_faith(Config B),  paired by seed.
The seed-level contrast (average over the 6 cells within each seed) is the unit; the 95% CI is a t interval with
n_seeds-1 df. Per-adapter and per-language contrasts are secondary; per-adapter p-values are Holm-adjusted.
Equivalence: a contrast is called 'equivalent' only if its 90% CI lies inside +-margin; the margin must be fixed
in the analysis plan BEFORE the data are seen (default 0.01).
Exploratory (labelled as such in the paper): everything not listed above.
"""
import argparse, glob, json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats

def load(runs):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "**", "checkpoint_*.jsonl"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            r = json.loads(line); r["_dir"] = os.path.dirname(f); rows.append(r)
    return pd.DataFrame(rows)

def label(r):
    if r.get("status") != "PASS": return "FAIL"
    c, lam, rm = r["config"], r.get("lambda1", 0.3), r.get("reward_mode", "real")
    ck = bool(r.get("grad_ckpt", False)); pm = r.get("policy_mode", "eval")
    if c == "A_frozen": return "A"
    if c == "D_full_ft": return "D"
    if c == "B_ce_lora": return "B_ckpt" if ck else "B"
    if c == "C_composite_lora":
        if pm == "train": return "C_policytrain"
        if rm == "shuffled": return "C_shuf"
        if lam == 0: return "C_l0"
        return "C_ckpt" if ck else "C"
    return "other"

def tci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (float("nan"),) * 4
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else float("nan")
    return m, m - h, m + h, p

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1, run)
    return adj

def contrast(df, a, b, metric, cells=None, by="seed"):
    """seed-level paired contrast a-b averaged over the cells present for both; returns Series indexed by seed."""
    d = df[df.label.isin([a, b])].pivot_table(index=["lora_variant", "language", "seed"], columns="label", values=metric)
    if a not in d or b not in d: return pd.Series(dtype=float)
    d = d.dropna(subset=[a, b])
    if cells is not None: d = d[d.index.droplevel("seed").isin(cells)]
    diff = (d[a] - d[b]).rename("d").reset_index()
    return diff.groupby("seed").d.mean()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--out", default="analysis_v2")
    ap.add_argument("--margin", type=float, default=0.01); ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = load(a.runs); df["label"] = df.apply(label, axis=1)
    n_fail = int((df.label == "FAIL").sum()); df = df[df.label != "FAIL"].copy()
    df.drop(columns=["samples"], errors="ignore").to_csv(f"{a.out}/conditions.csv", index=False)
    V3, L2 = ["dora", "lora", "qlora"], ["arabic", "malay"]
    core = [(v, l) for v in V3 for l in L2]
    metrics = ["mean_f_faith", "mean_em", "mean_f1"] + [m for m in ("option_acc", "in_context_rate") if m in df]
    cells = df.groupby(["label", "lora_variant", "language"])[metrics + ["seed"]].agg(["mean", "std"]).round(4)
    cells.to_csv(f"{a.out}/cells.csv")
    pairs = [("C", "B"), ("C", "C_l0"), ("C", "C_shuf"), ("C_l0", "B"), ("C_shuf", "B"),
             ("B_ckpt", "B"), ("C_ckpt", "C"), ("C_policytrain", "C")]
    out = []
    for (x, y) in pairs:
        for metric in metrics:
            s = contrast(df, x, y, metric, cells=core)
            if len(s) < 2: continue
            m, lo, hi, p = tci(s); m90, lo90, hi90, _ = tci(s, 0.90)
            out.append(dict(contrast=f"{x}-{y}", scope="pooled(6 cells)", metric=metric, n_seeds=len(s), mean=m, ci95_lo=lo,
                            ci95_hi=hi, p=p, ci90_lo=lo90, ci90_hi=hi90,
                            equivalent_within_margin=bool(lo90 > -a.margin and hi90 < a.margin),
                            cells_positive=f"{int((s > 0).sum())}/{len(s)} seeds"))
            if metric == "mean_f_faith" and (x, y) == ("C", "B"):
                ps = []
                for v in V3:
                    sv = contrast(df, x, y, metric, cells=[(v, l) for l in L2])
                    if len(sv) < 2: continue
                    mv, lov, hiv, pv = tci(sv); ps.append(pv)
                    out.append(dict(contrast="C-B", scope=f"adapter={v}", metric=metric, n_seeds=len(sv), mean=mv, ci95_lo=lov, ci95_hi=hiv, p=pv))
                for l in L2:
                    sl = contrast(df, x, y, metric, cells=[(v, l) for v in V3])
                    if len(sl) < 2: continue
                    ml, lol, hil, pl = tci(sl)
                    out.append(dict(contrast="C-B", scope=f"language={l}", metric=metric, n_seeds=len(sl), mean=ml, ci95_lo=lol, ci95_hi=hil, p=pl))
                ad = [o for o in out if o["scope"].startswith("adapter=") and o["contrast"] == "C-B" and o["metric"] == metric]
                for o, ph in zip(ad, holm([o["p"] for o in ad])): o["p_holm"] = ph
    pd.DataFrame(out).to_csv(f"{a.out}/contrasts.csv", index=False)

    # item-level cluster bootstrap for the primary contrast (resample source items within each cell)
    boot = {}
    def preds(r):
        f = r.get("preds_file")
        if not isinstance(f, str): return None
        p = os.path.join(r["_dir"], f)
        if not os.path.exists(p): return None
        d = pd.read_json(p, lines=True)
        return d[~d.passage_in_train].set_index("src_id").f_faith
    if "preds_file" in df:
        per_cell = {}
        for (v, l) in core:
            xs, ys = [], []
            for lab, acc in (("C", xs), ("B", ys)):
                sub = df[(df.label == lab) & (df.lora_variant == v) & (df.language == l)]
                ser = [preds(r) for _, r in sub.iterrows()]; ser = [x for x in ser if x is not None]
                if ser: acc.append(pd.concat(ser, axis=1).mean(axis=1))
            if xs and ys:
                per_cell[(v, l)] = (xs[0] - ys[0]).dropna().values
        if per_cell:
            rng = np.random.default_rng(0); bs = []
            for _ in range(a.n_boot):
                bs.append(np.mean([rng.choice(d, len(d)).mean() for d in per_cell.values()]))
            pt = float(np.mean([d.mean() for d in per_cell.values()]))
            boot = dict(contrast="C-B", metric="mean_f_faith", point=pt, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        unit="source item (seeds averaged within item)", n_cells=len(per_cell))
    json.dump(boot, open(f"{a.out}/item_bootstrap.json", "w"), indent=1)

    # diagnostics: sanitizer counts, truncation, evaluation-state diagnostic, AdaLoRA ranks
    diag = {}
    if "sanitize_stats" in df:
        ss = pd.DataFrame(df.sanitize_stats.dropna().tolist()); diag["sanitize_total"] = ss.sum().to_dict()
        diag["runs_with_nonfinite_logits"] = int(((ss.get("gen_nan", 0) + ss.get("gen_posinf", 0) + ss.get("score_nan", 0) + ss.get("score_posinf", 0)) > 0).sum())
    for k in ("frac_eval_prompt_truncated", "frac_train_prompt_truncated", "n_passage_overlap_excluded"):
        if k in df: diag[k] = {"mean": float(df[k].astype(float).mean()), "max": float(df[k].astype(float).max())}
    if "trainmode_eval_f_faith" in df:
        d2 = df.dropna(subset=["trainmode_eval_f_faith"]); g = d2.groupby("label").apply(lambda t: (t.trainmode_eval_f_faith - t.mean_f_faith).mean())
        diag["legacy_eval_state_minus_headline_by_label"] = g.round(4).to_dict()
    if "adalora_rank_log" in df:
        ar = df[df.lora_variant == "adalora"].adalora_rank_log.dropna().tolist()
        diag["adalora_rank_first_last"] = [[x[0], x[-1]] for x in ar if x]
    diag["n_failed_runs"] = n_fail; diag["n_runs"] = int(len(df))
    json.dump(diag, open(f"{a.out}/diagnostics.json", "w"), indent=1, default=str)

    c = pd.DataFrame(out); prim = c[(c.contrast == "C-B") & (c.scope == "pooled(6 cells)") & (c.metric == "mean_f_faith")]
    with open(f"{a.out}/summary.md", "w") as fh:
        fh.write(f"runs={len(df)} failed={n_fail} margin=+-{a.margin}\n\n")
        fh.write(c.round(4).to_markdown(index=False) + "\n\nPRIMARY:\n" + prim.round(4).to_markdown(index=False) + "\n")
        fh.write("\nITEM BOOTSTRAP:\n" + json.dumps(boot) + "\n\nDIAGNOSTICS:\n" + json.dumps(diag, default=str, indent=1) + "\n")
    print(open(f"{a.out}/summary.md").read()[:6000])

if __name__ == "__main__":
    main()
