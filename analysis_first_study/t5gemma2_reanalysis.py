#!/usr/bin/env python3
"""Third-backbone (T5Gemma 2) analysis, mirroring analysis/factorial_reanalysis.py.

Inputs : analysis/per_seed_results.csv   (mT5 and Qwen3, 3 seeds per cell)
         experiment_results_t5gemma2/checkpoint_full_*.jsonl  (T5Gemma 2 raw runs)
Outputs: analysis/t5gemma2_per_seed.csv, analysis/t5gemma2_reanalysis_results.json and a printed summary.

Model (cell means, one pooled within-cell variance per backbone; seeds are replicates):
  Delta_v = mean over {Arabic, Malay} of F_faith(C) - F_faith(B), per LoRA variant v.
  SE(Delta_v) = pooled_SD / sqrt(3)  (4 cell means, each with variance s^2/3, weights 1/2).
T5Gemma 2 effects use its own pooled SD (3 variants x 2 languages x 2 objectives = 12 cells, 24 df).
Cross-backbone contrasts use Welch-Satterthwaite t on the difference of two Deltas.
These analyses were added after the primary analysis of the manuscript was fixed (post hoc extension).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "experiment_results_t5gemma2")
rows = []
for f in sorted(glob.glob(os.path.join(RAW, "checkpoint_full_*.jsonl"))):
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            rows.append(dict(backbone="t5gemma2", variant=r["lora_variant"], language=r["language"],
                             config=r["config"], seed=r["seed"], f_faith=r["mean_f_faith"],
                             em=r["mean_em"], f1=r["mean_f1"], n_eval=r["n_eval"]))
t = pd.DataFrame(rows).drop_duplicates(["variant", "language", "config", "seed"])
assert len(t) == 36, f"expected 36 runs, found {len(t)}"
t.to_csv(os.path.join(HERE, "t5gemma2_per_seed.csv"), index=False)
old = pd.read_csv(os.path.join(HERE, "per_seed_results.csv"))
N = 3

def effect(df, variants):
    g = df.groupby(["variant", "language", "config"]).f_faith.agg(["mean", "std"]).reset_index()
    s2 = g["std"].pow(2).mean(); dfp = len(g) * (N - 1)
    out = {}
    for v in variants:
        gv = g[g.variant == v]
        m = lambda l, c: float(gv[(gv.language == l) & (gv.config == c)]["mean"].iloc[0])
        sd = lambda l, c: float(gv[(gv.language == l) & (gv.config == c)]["std"].iloc[0])
        d = 0.5 * sum(m(l, "C_composite_lora") - m(l, "B_ce_lora") for l in ("arabic", "malay"))
        se = np.sqrt(s2 / N); tc = stats.t.ppf(0.975, dfp)
        terms = [0.25 * sd(l, c) ** 2 / N for l in ("arabic", "malay") for c in ("B_ce_lora", "C_composite_lora")]
        vw = sum(terms); dfw = vw ** 2 / sum(x ** 2 / (N - 1) for x in terms)
        pv = df[df.variant == v].pivot_table(index=["seed", "language"], columns="config", values="f_faith")
        ps = (pv["C_composite_lora"] - pv["B_ce_lora"]).groupby("seed").mean()
        out[v] = dict(delta=d, se=se, lo=d - tc * se, hi=d + tc * se, t=d / se, df=dfp,
                      p=2 * stats.t.sf(abs(d / se), dfp), p_welch=2 * stats.t.sf(abs(d / np.sqrt(vw)), dfw),
                      df_welch=dfw, p_paired=stats.ttest_1samp(ps.values, 0).pvalue,
                      seeds_same_sign=int((np.sign(ps.values) == np.sign(d)).sum()), per_seed=[float(x) for x in ps.values])
    return out, float(np.sqrt(s2)), dfp

def contrast(a, dfa, b, dfb):
    va, vb = a["se"] ** 2, b["se"] ** 2
    d = a["delta"] - b["delta"]; se = np.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / dfa + vb ** 2 / dfb); tc = stats.t.ppf(0.975, dfw)
    return dict(diff=d, lo=d - tc * se, hi=d + tc * se, t=d / se, df=dfw, p=2 * stats.t.sf(abs(d / se), dfw))

tg, sd_t, df_t = effect(t, ["qlora", "dora", "lora"])
m4, sd_m4, df_m4 = effect(old[(old.backbone == "mt5") & (old.variant != "lora")], ["qlora", "adalora", "dora", "vera"])
m5, sd_m5, df_m5 = effect(old[old.backbone == "mt5"], ["lora"])        # post hoc plain LoRA, pooled over all 5 mT5 variants
q3, sd_q3, df_q3 = effect(old[old.backbone == "qwen3"], ["qlora", "adalora", "dora", "vera"])

res = dict(t5gemma2=tg, t5gemma2_pooled_sd=sd_t, t5gemma2_df=df_t, mt5_planned=m4, mt5_lora_posthoc=m5["lora"],
           mt5_pooled_sd_planned=sd_m4, mt5_pooled_sd_5var=sd_m5, qwen3=q3, contrasts={})
for v in ["qlora", "dora"]:
    res["contrasts"][f"mt5_vs_t5g_{v}"] = contrast(m4[v], df_m4, tg[v], df_t)
    res["contrasts"][f"qwen3_vs_t5g_{v}"] = contrast(q3[v], df_q3, tg[v], df_t)
res["contrasts"]["mt5_vs_t5g_lora_posthoc"] = contrast(m5["lora"], df_m5, tg["lora"], df_t)
d = np.array([tg[v]["delta"] for v in ["qlora", "dora", "lora"]]); w = np.array([1 / tg[v]["se"] ** 2 for v in ["qlora", "dora", "lora"]])
chi = float((w * (d - (w * d).sum() / w.sum()) ** 2).sum()); res["t5g_heterogeneity"] = dict(chi2=chi, df=2, p=float(stats.chi2.sf(chi, 2)))
res["t5g_ci_halfwidth"] = float(stats.t.ppf(0.975, df_t) * sd_t / np.sqrt(N))
json.dump(res, open(os.path.join(HERE, "t5gemma2_reanalysis_results.json"), "w"), indent=1, default=float)
print(f"T5Gemma 2 pooled SD {sd_t:.4f} ({df_t} df); 95% CI half-width {res['t5g_ci_halfwidth']:.4f}")
for v, o in tg.items():
    print(f"{v:6s} Delta {o['delta']:+.4f} [{o['lo']:+.4f},{o['hi']:+.4f}] p={o['p']:.3f} welch p={o['p_welch']:.3f} paired p={o['p_paired']:.3f} seeds {o['seeds_same_sign']}/3")
for k, c in res["contrasts"].items():
    print(f"{k:28s} {c['diff']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] t={c['t']:+.2f} df={c['df']:.1f} p={c['p']:.2g}")
print("heterogeneity across T5Gemma 2 variants: chi2(2)=%.2f p=%.3f" % (chi, res["t5g_heterogeneity"]["p"]))
