#!/usr/bin/env python3
"""Summarise answer length / extractive-copy rate per condition from per-example predictions.

Usage: python scripts/summarize_answer_length.py runs_v2 --out results/corrected_v2/answer_length.json
Reads runs_v2/{core,controls}/*/preds/*.jsonl (the checkpointing-factorial stage is excluded) and reports, per
language x condition (B, C, C_l0, C_shuf): mean answer words, mean gold words, share of answers occurring verbatim
in the passage (`in_context`), and share of answers more than 3x the gold length. Means are over all
(seed, adapter, example) triples of the headline evaluation set (passages also seen in training are excluded); counts are reported.
"""
import argparse, glob, json, os, re

def words(s):
    """Word count after replacing punctuation by spaces (the tokenisation used for the paper's mechanism table)."""
    return len(re.sub(r"[^\w\s]", " ", str(s)).split())

from collections import defaultdict

def label(name):
    m = re.match(r"(B_ce_lora|C_composite_lora)_(\w+?)_(arabic|malay)_seed(\d+)_l([\d.]+)_(real|shuffled)_ckpt(\d)", name)
    if not m: return None
    cfg, _, lang, _, lam, rm, ck = m.groups()
    if ck != "0": return None
    if cfg.startswith("B"): return lang, "B"
    if rm == "shuffled": return lang, "C_shuf"
    return lang, ("C" if float(lam) > 0 else "C_l0")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--out", default="results/corrected_v2/answer_length.json")
    a = ap.parse_args()
    acc = defaultdict(lambda: dict(n=0, words=0, gold=0, ctx=0, long3=0))
    for stage in ("core", "controls"):
        for f in glob.glob(os.path.join(a.runs, stage, "*", "preds", "*.jsonl")):
            k = label(os.path.basename(f))
            if k is None: continue
            for line in open(f, encoding="utf-8"):
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("passage_in_train"): continue   # headline set: overlapping-passage items excluded
                w = words(r["generated"]); g = max(1, words(r["gold"]))
                s = acc[k]; s["n"] += 1; s["words"] += w; s["gold"] += g; s["ctx"] += bool(r.get("in_context")); s["long3"] += w > 3 * g
    out = {f"{lang}|{lab}": dict(n=s["n"], mean_words=s["words"] / s["n"], mean_gold_words=s["gold"] / s["n"],
                                 in_context_rate=s["ctx"] / s["n"], share_gt_3x_gold=s["long3"] / s["n"])
           for (lang, lab), s in sorted(acc.items())}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in out.items(): print(k, {x: round(y, 3) for x, y in v.items()})

if __name__ == "__main__":
    main()
