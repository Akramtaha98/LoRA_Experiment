#!/usr/bin/env python3
"""Blinded human validation of F_faith (two subcommands).

  python -m lora_nli.human_eval make  RUNS_DIR --out human_eval --n 100 --adapter lora --seed 42
      -> sheet_annotator1.csv / sheet_annotator2.csv (same items, different random order, no condition shown) and key.csv.
         Each row: language, passage, question, [options], candidate answer, empty columns `supported` (0/1/2) and `correct` (0/1).
         supported: 0 = not supported by the passage, 1 = partly supported, 2 = fully supported.
         correct:   1 = the answer is a correct answer to the question given the passage (Malay: selects the correct option).
         Identical answers produced by different conditions for the same item are shown once.
  python -m lora_nli.human_eval score human_eval  [--n-boot 5000]
      -> weighted Cohen's kappa (supported, quadratic), per-condition mean human support, paired C-B and C-C_l0 differences
         with item-bootstrap 95% CIs, Spearman correlation between the NLI reward and human support.
Passages are fetched from the public datasets (google/xquad xquad.ar; facebook/belebele zsm_Latn) by `ds_index`.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

def load_runs(runs):
    rows = []
    for f in glob.glob(os.path.join(runs, "**", "checkpoint_*.jsonl"), recursive=True):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); r["_dir"] = os.path.dirname(f); rows.append(r)
    return rows

def cond(r):
    if r["config"] == "B_ce_lora": return "B"
    if r.get("reward_mode") == "shuffled": return "C_shuf"
    return "C_l0" if r.get("lambda1") == 0 else "C"

def get_passage_fn(language):
    from datasets import load_dataset
    ds = load_dataset("google/xquad", "xquad.ar", split="validation") if language == "arabic" else load_dataset("facebook/belebele", "zsm_Latn", split="test")
    def f(i):
        e = ds[int(i)]
        if language == "arabic": return e["context"], None
        return e["flores_passage"], [e[f"mc_answer{k}"] for k in (1, 2, 3, 4)]
    return f

def make(a, passage_fn=None):
    rng = np.random.default_rng(a.seed_rng); os.makedirs(a.out, exist_ok=True)
    rows = [r for r in load_runs(a.runs) if r.get("status") == "PASS" and r["lora_variant"] == a.adapter and r["seed"] == a.seed
            and r["config"] in ("B_ce_lora", "C_composite_lora") and r.get("preds_file")]
    items = []
    for lang in ("arabic", "malay"):
        recs = []
        for r in rows:
            if r["language"] != lang: continue
            d = pd.read_json(os.path.join(r["_dir"], r["preds_file"]), lines=True); d = d[~d.passage_in_train]; d["cond"] = cond(r); recs.append(d)
        if not recs: continue
        allp = pd.concat(recs); ids = sorted(set.intersection(*[set(x.src_id) for x in recs]))
        ids = list(rng.choice(ids, size=min(a.n, len(ids)), replace=False))
        pf = passage_fn(lang) if passage_fn else get_passage_fn(lang)
        for sid in ids:
            sub = allp[allp.src_id == sid]; seen = {}
            for _, r in sub.iterrows():
                k = " ".join(str(r.generated).split()).lower()
                seen.setdefault(k, []).append(r.cond)
                if len(seen[k]) == 1:
                    psg, opts = pf(r.ds_index)
                    items.append(dict(language=lang, src_id=sid, passage=psg, question=r.get("question", ""),
                                      options=" | ".join(opts) if opts else "", candidate=r.generated, _key=k, f_faith=r.f_faith))
            for it in items:
                if it["src_id"] == sid: it["conds"] = ",".join(seen.get(it["_key"], []))
    df = pd.DataFrame(items); df["item_uid"] = [f"i{n:04d}" for n in rng.permutation(len(df))]
    df[["item_uid", "language", "src_id", "conds", "f_faith"]].to_csv(f"{a.out}/key.csv", index=False)
    for k in (1, 2):
        s = df.sample(frac=1, random_state=k + a.seed_rng)[["item_uid", "language", "passage", "question", "options", "candidate"]].copy()
        s["supported"] = ""; s["correct"] = ""; s["notes"] = ""; s.to_csv(f"{a.out}/sheet_annotator{k}.csv", index=False)
    print(f"{len(df)} unique judgments -> {a.out}/sheet_annotator[12].csv (annotators must not see key.csv)")

def boot_diff(d, a_, b_, n):
    rng = np.random.default_rng(0); pivot = d.pivot_table(index="src_id", columns="cond", values="h")
    pivot = pivot.dropna(subset=[a_, b_]) if a_ in pivot and b_ in pivot else pd.DataFrame()
    if pivot.empty: return None
    x = (pivot[a_] - pivot[b_]).values
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return dict(mean=float(x.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_items=int(len(x)))

def score(a):
    key = pd.read_csv(f"{a.dir}/key.csv"); s1 = pd.read_csv(f"{a.dir}/sheet_annotator1.csv"); s2 = pd.read_csv(f"{a.dir}/sheet_annotator2.csv")
    m = key.merge(s1[["item_uid", "supported", "correct"]], on="item_uid").merge(s2[["item_uid", "supported", "correct"]], on="item_uid", suffixes=("_1", "_2"))
    m = m.dropna(subset=["supported_1", "supported_2"]); res = {}
    from sklearn.metrics import cohen_kappa_score
    res["kappa_supported_quadratic"] = float(cohen_kappa_score(m.supported_1.astype(int), m.supported_2.astype(int), weights="quadratic"))
    res["kappa_correct"] = float(cohen_kappa_score(m.correct_1.astype(int), m.correct_2.astype(int))) if m.correct_1.notna().all() else None
    m["h"] = (m.supported_1 + m.supported_2) / 2
    res["spearman_nli_vs_human_support"] = {l: float(stats.spearmanr(g.f_faith, g.h)[0]) for l, g in m.groupby("language")}
    long = m.assign(cond=m.conds.str.split(",")).explode("cond")
    res["mean_human_support_by_condition"] = long.groupby(["language", "cond"]).h.mean().round(3).unstack().to_dict()
    for lang in ("arabic", "malay", "all"):
        d = long if lang == "all" else long[long.language == lang]
        res[f"C-B[{lang}]"] = boot_diff(d, "C", "B", a.n_boot); res[f"C-C_l0[{lang}]"] = boot_diff(d, "C", "C_l0", a.n_boot)
    json.dump(res, open(f"{a.dir}/human_eval_results.json", "w"), indent=1); print(json.dumps(res, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("runs"); m.add_argument("--out", default="human_eval"); m.add_argument("--n", type=int, default=100)
    m.add_argument("--adapter", default="lora"); m.add_argument("--seed", type=int, default=42); m.add_argument("--seed-rng", type=int, default=0, dest="seed_rng")
    s = sp.add_parser("score"); s.add_argument("dir"); s.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(); make(a) if a.cmd == "make" else score(a)
