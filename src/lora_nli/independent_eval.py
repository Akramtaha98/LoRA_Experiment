#!/usr/bin/env python3
"""Independent faithfulness evaluator: re-score every held-out prediction with a DIFFERENT NLI model than the reward.

  python -m lora_nli.independent_eval RUNS_DIR --out results/independent_scorer [--model joeddav/xlm-roberta-large-xnli] [--device cuda]

The training reward is MoritzLaurer/mDeBERTa-v3-base-mnli-xnli. This script re-scores B / C / C_l0 / C_shuf predictions
(deduplicated by passage + answer) with another architecture trained on different data, then reports
  * the pooled C - B contrast under the independent scorer (seed-level t 95% CI, as in analyze_v2.py);
  * the correlation between reward-model and independent scores.
Caveat (state it in the paper): Arabic is an XNLI language; Malay is not covered by either model's NLI training, so Malay
rests on cross-lingual transfer for both. Use human_eval.py for a language-independent check.
"""
import argparse, glob, json, os, math
import numpy as np, pandas as pd
from scipy import stats

def entail_scores(model_name, pairs, device, bs=16):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_name); m = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    lab = {str(v).lower(): int(k) for k, v in m.config.id2label.items()}; ent = lab["entailment"]
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(): pr = torch.softmax(m(**enc).logits, -1)[:, ent]
        out += pr.cpu().tolist()
    return out

def main(argv=None, scorer=entail_scores, passage_fn=None):
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--out", default="results/independent_scorer")
    ap.add_argument("--model", default="joeddav/xlm-roberta-large-xnli"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    try:
        from lora_nli import human_eval as H
    except ImportError:
        import human_eval as H
    recs = []
    for f in glob.glob(os.path.join(a.runs, "**", "checkpoint_*.jsonl"), recursive=True):
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("status") != "PASS" or not r.get("preds_file") or r["config"] not in ("B_ce_lora", "C_composite_lora"): continue
            if r.get("policy_mode") == "train" or r.get("grad_ckpt"): continue
            d = pd.read_json(os.path.join(os.path.dirname(f), r["preds_file"]), lines=True); d = d[~d.passage_in_train]
            d["cond"] = H.cond(r); d["variant"] = r["lora_variant"]; d["language"] = r["language"]; d["seed"] = r["seed"]; recs.append(d)
    df = pd.concat(recs, ignore_index=True)
    pf = {}
    def passage(lang, i):
        if lang not in pf: pf[lang] = passage_fn(lang) if passage_fn else H.get_passage_fn(lang)
        return pf[lang](i)[0]
    df["key"] = df.language + "|" + df.src_id + "|" + df.generated.astype(str).str.strip().str.lower()
    uniq = df.drop_duplicates("key")[["key", "language", "ds_index", "generated"]]
    pairs = [(passage(r.language, r.ds_index), str(r.generated)) for r in uniq.itertuples()]
    uniq = uniq.assign(f_indep=scorer(a.model, pairs, a.device)); df = df.merge(uniq[["key", "f_indep"]], on="key")
    df.drop(columns=["key"]).to_csv(f"{a.out}/indep_scores.csv", index=False)
    cell = df.groupby(["cond", "variant", "language", "seed"])[["f_faith", "f_indep"]].mean().reset_index()
    res = {"model": a.model, "spearman_reward_vs_indep": float(stats.spearmanr(df.f_faith, df.f_indep)[0])}
    for col in ("f_faith", "f_indep"):
        p = cell.pivot_table(index=["variant", "language", "seed"], columns="cond", values=col)
        for x, y in (("C", "B"), ("C", "C_l0"), ("C", "C_shuf")):
            if x in p and y in p:
                s = (p[x] - p[y]).dropna().groupby("seed").mean(); n = len(s)
                if n > 1:
                    se = s.std(ddof=1) / math.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
                    res[f"{col}:{x}-{y}"] = dict(mean=float(s.mean()), ci95=[float(s.mean() - h), float(s.mean() + h)], n_seeds=int(n))
    json.dump(res, open(f"{a.out}/indep_results.json", "w"), indent=1); print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
