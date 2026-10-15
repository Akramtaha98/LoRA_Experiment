#!/usr/bin/env python3
"""Build the 100-question example set (first 50 held-out Arabic + first 50 Malay) for the website.

Only runs whose answers were saved in full can be shown for every question: Qwen3 (NLI-reward config C) and
mT5 (corrected study, config C). T5Gemma 2 stored its first five answers only, so it appears on those five.
All answers are QLoRA, seed 42. Questions are the first 50 per language in held-out order (not cherry-picked).

After the H200 full re-run (docs/runbooks/RUNBOOK_H200_ALLANSWERS.md) pass --h200_dir to use the new logs, which hold
every answer of all three models for the same training setup (default config B = standard fine-tuning, as in the ranking):
  python scripts/build_compare_examples.py --runs_v2 runs_v2 --h200_dir ~/h200_allanswers/results/first_study --config B_ce_lora

Usage: python scripts/build_compare_examples.py --runs_v2 runs_v2
Writes results/compare/examples.json and web/public/data/compare-examples.json
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "results" / "first_study"
N = 50

def recs(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def smp(r): s = r["samples"]; return json.loads(s) if isinstance(s, str) else s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs_v2", required=True); ap.add_argument("--h200_dir", default=None); ap.add_argument("--config", default="B_ce_lora"); a = ap.parse_args()
    out = []
    cfg = a.config if a.h200_dir else "C_composite_lora"
    base = Path(a.h200_dir) if a.h200_dir else None
    def full(path):
        """Per-example records of the seed-42 run of `cfg` that logged every held-out answer."""
        for r in recs(path):
            if r["seed"] == 42 and r["config"] == cfg and len(smp(r)) > 100: return smp(r)
        raise SystemExit(f"no full log for {cfg} in {path}")
    for lang in ["arabic", "malay"]:
        qwen = full((base or FS) / "qwen3" / f"checkpoint_full_qwen_qlora_{lang}.jsonl")
        if base:
            t5full = {s["question"]: s for s in full(base / "t5gemma2" / f"checkpoint_full_qlora_{lang}.jsonl")}
        else:
            t5 = {s["gold"]: s for r in recs(FS / "t5gemma2" / f"checkpoint_full_qlora_{lang}.jsonl")
                  if r["seed"] == 42 and r["config"] == cfg for s in smp(r)}
        f = Path(a.runs_v2) / "core" / f"qlora_{lang}_s42" / "preds" / f"{cfg}_qlora_{lang}_seed42_l0.3_real_ckpt0.jsonl"
        mt5 = {p["question"]: p for p in map(json.loads, open(f, encoding="utf-8"))}
        for s in qwen[:N]:
            m = mt5[s["question"]]
            ans = {"qwen3": dict(a=s["generated"], em=int(s["em"]), f1=round(s["f1"], 3)),
                   "mt5": dict(a=m["generated"], em=int(m["em"]), f1=round(m["f1"], 3))}
            t = t5full.get(s["question"]) if base else t5.get(s["gold"])
            if t: ans["t5gemma2"] = dict(a=t["generated"], em=int(t["em"]), f1=round(t["f1"], 3))
            out.append(dict(id=f"{lang}-{s['example_index']}", config=cfg, language=lang, question=s["question"], context=s["context"], gold=s["gold"], answers=ans))
    for p in [ROOT / "results/compare/examples.json", ROOT / "web/public/data/compare-examples.json"]:
        p.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    for lang in ["arabic", "malay"]:
        it = [o for o in out if o["language"] == lang]
        print(lang, len(it), {m: sum(o["answers"][m]["em"] for o in it if m in o["answers"]) for m in ["qwen3", "mt5", "t5gemma2"]},
              sum("t5gemma2" in o["answers"] for o in it))
if __name__ == "__main__": main()
