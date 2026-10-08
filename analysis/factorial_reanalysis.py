#!/usr/bin/env python3
"""Single primary analysis for the paper (Section 4.1).

Model (fixed-effects, cell-means form):
    F_faith ~ Backbone x Variant x Language x Objective   (n = 3 seeds per cell)
with one pooled within-cell variance PER BACKBONE (mT5, Qwen3).
Seed variance was estimated at ~0 for mT5 (mixed model, Sec. 4.1); seeds are
treated as replicates within a cell.

Input : cell means and SDs of the 3-seed replications (Tables 1 and 3 of the
        manuscript; per-seed values are in the released run logs).
Output: printed tables + factorial_reanalysis_results.json

Planned contrasts (defined before looking at p-values, single family):
  Family 1 (8 tests): objective effect Delta = F(C) - F(B), averaged over the
                      two languages, for each variant x backbone.
                      Bonferroni alpha = 0.05 / 8.
  Family 2 (4 tests): backbone x objective interaction per variant,
                      Delta(mT5) - Delta(Qwen3). Bonferroni alpha = 0.05 / 4.
  Omnibus            : 3-df test of Backbone x Variant x Objective (does the
                      variant-specific objective effect differ by backbone?).
Language-specific contrasts are reported as secondary/descriptive only.
"""
import json, numpy as np
from scipy import stats

N = 3  # seeds per cell
V = ["QLoRA", "AdaLoRA", "DoRA", "VeRA"]
# (mean, sd) ; order: [backbone][language][variant][config]
D = {
 "mT5": {
  "Arabic": {"QLoRA": {"B": (0.1450, 0.0076), "C": (0.1657, 0.0121)},
             "AdaLoRA": {"B": (0.0686, 0.0033), "C": (0.0601, 0.0003)},
             "DoRA": {"B": (0.1395, 0.0024), "C": (0.0473, 0.0007)},
             "VeRA": {"B": (0.0669, 0.0051), "C": (0.0664, 0.0021)}},
  "Malay":  {"QLoRA": {"B": (0.2806, 0.0152), "C": (0.3065, 0.0046)},
             "AdaLoRA": {"B": (0.0363, 0.0040), "C": (0.0256, 0.0035)},
             "DoRA": {"B": (0.2817, 0.0055), "C": (0.0658, 0.0063)},
             "VeRA": {"B": (0.0360, 0.0017), "C": (0.0266, 0.0013)}}},
 "Qwen3": {
  "Arabic": {"QLoRA": {"B": (0.1474, 0.0018), "C": (0.1443, 0.0103)},
             "AdaLoRA": {"B": (0.1521, 0.0029), "C": (0.1488, 0.0071)},
             "DoRA": {"B": (0.1377, 0.0064), "C": (0.1395, 0.0091)},
             "VeRA": {"B": (0.1465, 0.0009), "C": (0.1477, 0.0017)}},
  "Malay":  {"QLoRA": {"B": (0.3060, 0.0100), "C": (0.2926, 0.0050)},
             "AdaLoRA": {"B": (0.3083, 0.0114), "C": (0.3185, 0.0019)},
             "DoRA": {"B": (0.2965, 0.0135), "C": (0.3109, 0.0165)},
             "VeRA": {"B": (0.3203, 0.0011), "C": (0.3138, 0.0044)}}},
}
LANGS = ["Arabic", "Malay"]

def pooled_var(bb):
    sds = [D[bb][l][v][c][1] for l in LANGS for v in V for c in "BC"]
    return float(np.mean(np.square(sds))), 16 * (N - 1)   # equal n -> mean of variances

out = {"pooled": {}, "family1": {}, "family2": {}, "secondary": {}}
pv = {}
for bb in D:
    s2, df = pooled_var(bb)
    pv[bb] = (s2, df)
    out["pooled"][bb] = {"sd": s2 ** .5, "df": df}
    print(f"{bb}: pooled residual SD = {s2**.5:.4f}, df = {df}")

def delta(bb, v, langs=LANGS):
    return float(np.mean([D[bb][l][v]["C"][0] - D[bb][l][v]["B"][0] for l in langs]))

def var_delta(bb, langs=LANGS):
    # Var of mean over languages of (C_l - B_l): 2 cells per language, each var s2/N
    s2, _ = pv[bb]
    return (len(langs) * 2 * s2 / N) / len(langs) ** 2

print("\nFamily 1: objective effect (C-B), language-averaged, 8 tests, Bonferroni 0.05/8 = 0.00625")
for bb in D:
    s2, df = pv[bb]
    for v in V:
        d = delta(bb, v); se = var_delta(bb) ** .5; t = d / se
        p = 2 * stats.t.sf(abs(t), df); ci = stats.t.ppf(.975, df) * se
        sd_pool = s2 ** .5
        out["family1"][f"{bb}/{v}"] = dict(delta=d, ci=[d - ci, d + ci], t=t, df=df, p=p,
                                             cohen_d=d / sd_pool, bonf_sig=bool(p < .05 / 8))
        print(f"{bb:6s} {v:8s} d={d:+.4f} CI[{d-ci:+.4f},{d+ci:+.4f}] t({df})={t:+.2f} p={p:.2e} "
              f"d/SD={d/sd_pool:+.1f} Bonf={'Y' if p<.05/8 else 'n'}")

print("\nFamily 2: Backbone x Objective interaction per variant, Bonferroni 0.05/4 = 0.0125")
c, Sig = [], []
for v in V:
    cv = delta("mT5", v) - delta("Qwen3", v)
    var = var_delta("mT5") + var_delta("Qwen3")
    se = var ** .5; t = cv / se
    # Welch-Satterthwaite df
    vm, vq = var_delta("mT5"), var_delta("Qwen3")
    df = var ** 2 / (vm ** 2 / pv["mT5"][1] + vq ** 2 / pv["Qwen3"][1])
    p = 2 * stats.t.sf(abs(t), df); ci = stats.t.ppf(.975, df) * se
    out["family2"][v] = dict(interaction=cv, ci=[cv - ci, cv + ci], t=t, df=df, p=p, bonf_sig=bool(p < .05 / 4))
    print(f"{v:8s} diff={cv:+.4f} CI[{cv-ci:+.4f},{cv+ci:+.4f}] t({df:.0f})={t:+.2f} p={p:.2e} Bonf={'Y' if p<.05/4 else 'n'}")
    c.append(cv); Sig.append(var)

c = np.array(c); Sig = np.diag(Sig)
H = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float)   # 3-df: variant heterogeneity of the interaction
hc = H @ c; W = float(hc @ np.linalg.inv(H @ Sig @ H.T) @ hc)
p3 = float(stats.chi2.sf(W, 3))
W4 = float(c @ np.linalg.inv(Sig) @ c); p4 = float(stats.chi2.sf(W4, 4))
out["omnibus"] = {"backbone_x_variant_x_objective_chi2_df3": W, "p3": p3,
                  "backbone_x_objective_within_variants_chi2_df4": W4, "p4": p4}
print(f"\nOmnibus Backbone x Variant x Objective  : chi2(3) = {W:.1f}, p = {p3:.2e}")
print(f"Omnibus Backbone x Objective (4 variants): chi2(4) = {W4:.1f}, p = {p4:.2e}")

print("\nSecondary (descriptive, unadjusted): per-language objective effect")
for bb in D:
    s2, df = pv[bb]
    for l in LANGS:
        for v in V:
            d = delta(bb, v, [l]); se = (2 * s2 / N) ** .5; t = d / se
            p = 2 * stats.t.sf(abs(t), df)
            out["secondary"][f"{bb}/{l}/{v}"] = dict(delta=d, t=t, p=p)
            print(f"{bb:6s} {l:7s} {v:8s} d={d:+.4f} t={t:+.2f} p={p:.3g}")

# worst-case bound for removing the 4 overlapping Arabic held-out examples (4/720)
m = 4
shift = 0.5 * (2 * m / (720 - m))   # language-averaged, per-example difference in [-1,1]
out["overlap_bound"] = {"max_shift_language_avg_delta": shift}
print(f"\nWorst-case shift of a language-averaged Delta if 4/720 Arabic items are dropped: {shift:.4f}")
json.dump(out, open("factorial_reanalysis_results.json", "w"), indent=1)
