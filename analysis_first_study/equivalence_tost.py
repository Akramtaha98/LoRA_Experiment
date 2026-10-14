#!/usr/bin/env python3
"""Equivalence (TOST) check for the 'no detected effect' cells (Qwen3, T5Gemma 2).

Reads analysis/t5gemma2_reanalysis_results.json (language-averaged Delta, SE and df per cell).
For each cell it reports
  * the smallest symmetric margin m* that the two one-sided 5% tests can establish:
        m* = |Delta| + t_{0.95, df} * SE   (the larger bound of the 90% CI);
  * whether equivalence within a pre-chosen margin of +/-0.03 F_faith holds
    (TOST p = max of the two one-sided p-values).
The margin 0.03 is a judgement, not derived from data: it is about one fifth of the mT5 DoRA collapse
(-0.154) and of the order of the mT5 QLoRA gain (+0.023). It was chosen after the data were seen,
so equivalence statements are post hoc and descriptive. Equivalence is established only for the stated margin.
"""
import json, os
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "t5gemma2_reanalysis_results.json")))
MARGIN = 0.03
out = []
for bb, cells in (("qwen3", R["qwen3"]), ("t5gemma2", R["t5gemma2"])):
    for v, o in cells.items():
        d, se, df = o["delta"], o["se"], o["df"]
        p_lo = stats.t.sf((d + MARGIN) / se, df)      # H0: Delta <= -margin
        p_hi = stats.t.cdf((d - MARGIN) / se, df)     # H0: Delta >= +margin
        mstar = abs(d) + stats.t.ppf(0.95, df) * se
        out.append(dict(backbone=bb, variant=v, delta=d, se=se, df=df, tost_p=max(p_lo, p_hi), m_star=mstar))
        print(f"{bb:9s} {v:8s} Delta {d:+.4f} SE {se:.4f} df {df}  TOST p (+/-{MARGIN}) = {max(p_lo,p_hi):.4f}  smallest margin m* = {mstar:.4f}")
json.dump(dict(margin=MARGIN, cells=out), open(os.path.join(HERE, "equivalence_tost_results.json"), "w"), indent=1)
