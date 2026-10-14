#!/usr/bin/env python3
"""Build web/public/data/results.json (the single data file behind the results dashboard) from results/.

Inputs (all tracked in the repository):
  results/corrected_v2/{conditions.csv,contrasts.csv,joint_bootstrap.json,length_matched.json,diagnostics.json,
                        human_sens.json,annot_vs_option.json,answer_length.json,examples.json}
  results/human_validation/human_eval_results.json  (optional)
  results/independent_scorer/indep_results.json
  analysis_first_study/{per_seed_results.csv,t5gemma2_per_seed.csv,t5gemma2_reanalysis_results.json}

Usage: python scripts/build_web_data.py [--out web/public/data/results.json]
Everything on the site is read from this file; no number is typed into the front end.
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results" / "corrected_v2"
FS = ROOT / "analysis_first_study"
ADAPTERS = ["lora", "dora", "qlora"]
LANGS = ["arabic", "malay"]

def jload(p, default=None):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else default

def clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

def label(r):
    ck = bool(r.grad_ckpt)
    if r.config.startswith("B"): return "B_ckpt" if ck else "B"
    if r.reward_mode == "shuffled": return "C_shuf"
    if float(r.lambda1) == 0.0: return "C_l0"
    return "C_ckpt" if ck else "C"

def t_ci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return dict(mean=m, lo=None, hi=None, n=n)
    se = x.std(ddof=1) / math.sqrt(n); h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return dict(mean=m, lo=m - h, hi=m + h, n=n, p=float(stats.ttest_1samp(x, 0).pvalue) if x.std(ddof=1) > 0 else None)

def corrected():
    c = pd.read_csv(R2 / "conditions.csv")
    c = c[c.status == "PASS"].copy(); c["label"] = c.apply(label, axis=1)
    metrics = ["mean_f_faith", "mean_em", "mean_f1"]
    cells = []
    for (lab, ad, lg), g in c.groupby(["label", "lora_variant", "language"]):
        d = dict(label=lab, adapter=ad, language=lg, n_seeds=int(len(g)))
        for m in metrics: d[m] = float(g[m].mean()); d[m + "_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else None
        if "option_acc" in g and g.option_acc.notna().any(): d["option_acc"] = float(g.option_acc.mean())
        d["trainmode_eval_f_faith"] = float(g.trainmode_eval_f_faith.mean()) if "trainmode_eval_f_faith" in g else None
        cells.append(d)
    # per-seed paired values C vs B (core stage, checkpointing off)
    key = ["lora_variant", "language", "seed"]
    piv = c[c.label.isin(["B", "C", "C_l0", "C_shuf"])].pivot_table(index=key, columns="label", values=metrics)
    seeds = []
    for idx, row in piv.iterrows():
        seeds.append(dict(adapter=idx[0], language=idx[1], seed=int(idx[2]),
                          **{f"{m}|{lab}": row.get((m, lab)) for m in metrics for lab in ["B", "C", "C_l0", "C_shuf"]}))
    # headline numbers recomputed from run records (seed-paired t, df = seeds-1)
    def delta(lang=None, ad=None, a="C", b="B", metric="mean_f_faith"):
        g = c[c.label.isin([a, b])]
        if lang: g = g[g.language == lang]
        if ad: g = g[g.lora_variant == ad]
        p = g.pivot_table(index=["lora_variant", "language", "seed"], columns="label", values=metric)
        per_seed = (p[a] - p[b]).dropna().groupby(level="seed").mean()   # pooled over cells within seed
        return t_ci(per_seed.values)
    heads = {"pooled": delta(), "arabic": delta("arabic"), "malay": delta("malay"),
             "em_pooled": delta(metric="mean_em"), "em_arabic": delta("arabic", metric="mean_em"),
             "em_malay": delta("malay", metric="mean_em"), "f1_pooled": delta(metric="mean_f1"),
             "c_vs_l0": delta(a="C", b="C_l0"), "c_vs_shuf": delta(a="C", b="C_shuf"),
             "l0_vs_b": delta(a="C_l0", b="B"), "shuf_vs_b": delta(a="C_shuf", b="B"),
             "ckpt_B": delta(a="B_ckpt", b="B"), "ckpt_C": delta(a="C_ckpt", b="C")}
    by_adapter = {ad: delta(ad=ad) for ad in ADAPTERS}
    contrasts = pd.read_csv(R2 / "contrasts.csv").replace({np.nan: None}).to_dict("records")
    return dict(
        n_runs=int(len(c)), seeds=sorted(int(s) for s in c.seed.unique()),
        headline=heads, by_adapter=by_adapter, cells=cells, per_seed=seeds, contrasts=contrasts,
        joint_bootstrap=jload(R2 / "joint_bootstrap.json"), length_matched=jload(R2 / "length_matched.json"),
        answer_length=jload(R2 / "answer_length.json"), diagnostics=jload(R2 / "diagnostics.json"),
        examples=jload(R2 / "examples.json", []))

def human():
    hs = jload(R2 / "human_sens.json", {}); av = jload(R2 / "annot_vs_option.json", {})
    he = jload(ROOT / "results" / "human_validation" / "human_eval_results.json", {}) \
        or jload(R2 / "human_eval_results.json", {})
    return dict(sensitivity=hs, annotator_vs_option=av, summary=he)

def first_study():
    ps = pd.read_csv(FS / "per_seed_results.csv")
    tg = pd.read_csv(FS / "t5gemma2_per_seed.csv"); tg["backbone"] = "t5gemma2"
    keep = ["backbone", "variant", "language", "config", "seed", "f_faith", "em", "f1", "n_eval"]
    d = pd.concat([ps[keep], tg[keep]], ignore_index=True)
    rows = []
    for (bb, ad, lg), g in d.groupby(["backbone", "variant", "language"]):
        p = g.pivot_table(index="seed", columns="config", values=["f_faith", "em", "f1"])
        if ("f_faith", "C_composite_lora") not in p or ("f_faith", "B_ce_lora") not in p: continue
        dl = (p[("f_faith", "C_composite_lora")] - p[("f_faith", "B_ce_lora")])
        rows.append(dict(backbone=bb, adapter=ad, language=lg, seeds=[int(s) for s in p.index],
                         B=float(p[("f_faith", "B_ce_lora")].mean()), C=float(p[("f_faith", "C_composite_lora")].mean()),
                         delta=float(dl.mean()), delta_per_seed=[float(x) for x in dl.values],
                         em_B=float(p[("em", "B_ce_lora")].mean()), em_C=float(p[("em", "C_composite_lora")].mean()),
                         f1_B=float(p[("f1", "B_ce_lora")].mean()), f1_C=float(p[("f1", "C_composite_lora")].mean())))
    summary = {}
    for bb in d.backbone.unique():
        sub = d[d.backbone == bb]
        if bb in ("mt5", "qwen3"): sub = sub[sub.variant.isin(["lora", "dora", "qlora", "adalora"]) if bb == "mt5" else sub.variant.isin(["dora", "qlora", "adalora"])]
        p = sub.pivot_table(index=["variant", "language", "seed"], columns="config", values="f_faith").dropna()
        per_seed = (p["C_composite_lora"] - p["B_ce_lora"]).groupby(level="seed").mean()
        summary[bb] = dict(pooled=t_ci(per_seed.values), n_runs=int(len(sub)))
    return dict(rows=rows, summary=summary, reanalysis=jload(FS / "t5gemma2_reanalysis_results.json", {}))


MODELS = {"mt5": "mT5-base", "qwen3": "Qwen3-0.6B-Base", "t5gemma2": "T5Gemma 2 (270M)"}
CMP_ADAPTERS, CMP_SEEDS = ["dora", "qlora"], [42, 123, 2026]

def compare():
    """Like-for-like model comparison: same two adapters (DoRA, QLoRA), same three seeds, same held-out items.
    mT5 numbers come from the corrected study; Qwen3 and T5Gemma 2 from the original protocol (flagged in the UI)."""
    c = pd.read_csv(R2 / "conditions.csv"); c = c[(c.status == "PASS") & (~c.grad_ckpt.astype(bool))].copy()
    c["label"] = c.apply(label, axis=1); c = c[c.label.isin(["B", "C"])]
    mt = pd.DataFrame(dict(model="mt5", adapter=c.lora_variant, language=c.language, config=c.label, seed=c.seed,
                           f_faith=c.mean_f_faith, em=c.mean_em, f1=c.mean_f1))
    ps = pd.read_csv(FS / "per_seed_results.csv"); tg = pd.read_csv(FS / "t5gemma2_per_seed.csv")
    other = pd.concat([ps[ps.backbone == "qwen3"], tg], ignore_index=True)
    other = pd.DataFrame(dict(model=other.backbone, adapter=other.variant, language=other.language,
                              config=other.config.map({"B_ce_lora": "B", "C_composite_lora": "C"}), seed=other.seed,
                              f_faith=other.f_faith, em=other.em, f1=other.f1))
    d = pd.concat([mt, other], ignore_index=True)
    d = d[d.adapter.isin(CMP_ADAPTERS) & d.seed.isin(CMP_SEEDS)]
    units = ["adapter", "seed"]
    res = {}
    for cfg in ["B", "C"]:
        for lang in LANGS + ["all"]:
            sub = d[(d.config == cfg) & ((d.language == lang) if lang != "all" else True)]
            for metric in ["em", "f1", "f_faith"]:
                # one value per (model, adapter, seed[, language]); 'all' averages the two languages within a unit
                g = sub.groupby(["model"] + units, as_index=False)[metric].mean()
                wide = g.pivot(index=units, columns="model", values=metric)
                rows = [dict(model=m, name=MODELS[m], mean=float(wide[m].mean()), sd=float(wide[m].std(ddof=1)), n=int(wide[m].notna().sum()),
                             per_unit=[float(x) for x in wide[m].values]) for m in wide.columns]
                rows.sort(key=lambda r: -r["mean"])
                top, runner = rows[0], rows[1]
                diff = (wide[top["model"]] - wide[runner["model"]]).values
                t = t_ci(diff)
                res[f"{cfg}|{lang}|{metric}"] = dict(ranking=rows, winner=top["model"], runner_up=runner["model"],
                    margin=t["mean"], lo=t["lo"], hi=t["hi"], p=t.get("p"), clear=bool(t["lo"] is not None and t["lo"] > 0))
    return dict(models=MODELS, adapters=CMP_ADAPTERS, seeds=CMP_SEEDS, results=res,
                protocol_note="mT5 from the corrected study; Qwen3 and T5Gemma 2 from the original protocol (3 seeds, DoRA and QLoRA).",
                outputs=jload(ROOT / "results" / "compare" / "outputs.json", []))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "web" / "public" / "data" / "results.json"))
    a = ap.parse_args()
    out = dict(
        meta=dict(title="NLI-faithfulness reward x LoRA adapters", base_model="google/mt5-base",
                  reward_model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", independent_scorer="joeddav/xlm-roberta-large-xnli",
                  languages=LANGS, adapters=ADAPTERS, margin=0.01),
        corrected=corrected(), independent=jload(ROOT / "results" / "independent_scorer" / "indep_results.json", {}),
        human=human(), first_study=first_study(), compare=compare())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(clean(out), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    h = out["corrected"]["headline"]
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1024:.0f} KB); runs={out['corrected']['n_runs']}  "
          f"pooled C-B={h['pooled']['mean']:.4f} [{h['pooled']['lo']:.4f},{h['pooled']['hi']:.4f}]  "
          f"arabic={h['arabic']['mean']:.3f} malay={h['malay']['mean']:.4f}")

if __name__ == "__main__":
    main()
