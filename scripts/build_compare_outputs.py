#!/usr/bin/env python3
"""Collect the answers of mT5-base, Qwen3-0.6B-Base and T5Gemma 2 to the same held-out questions.

Why only five questions per language: the T5Gemma 2 run records store per-example answers for the first five
held-out items only (the other two models log all of them). Those five items are identical across models, so they are
the common set for a side-by-side comparison.

Sources
  Qwen3 C      results/first_study/qwen3/checkpoint_full_qwen_<adapter>_<lang>.jsonl   (seed 42, original protocol)
  Qwen3 B      results/first_study/qwen3/checkpoint_full_qwen_<lang>.jsonl             (first five answers per run)
  T5Gemma 2    results/first_study/t5gemma2/checkpoint_full_<adapter>_<lang>.jsonl      (seed 42, original protocol)
  mT5 (corrected study, optional)  <runs_v2>/core/<adapter>_<lang>_s42/preds/*.jsonl   (matched on question text)

Usage: python scripts/build_compare_outputs.py [--runs_v2 runs_v2] [--out results/compare/outputs.json]
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "results" / "first_study"
LANGS, ADAPTERS, CFGS = ["arabic", "malay"], ["qlora", "dora"], ["B_ce_lora", "C_composite_lora"]
N_ITEMS = 5

def samples(rec):
    s = rec["samples"]
    return json.loads(s) if isinstance(s, str) else s

def read_records(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_v2", default=None); ap.add_argument("--out", default=str(ROOT / "results" / "compare" / "outputs.json"))
    a = ap.parse_args()
    out = []
    for lang in LANGS:
        qwen = {}
        for ad in ADAPTERS:
            for r in read_records(FS / "qwen3" / f"checkpoint_full_qwen_{ad}_{lang}.jsonl"):
                if r["seed"] == 42 and r["config"] == CFGS[1] and r.get("samples"): qwen[(ad, CFGS[1])] = samples(r)
            for r in read_records(FS / "qwen3" / f"checkpoint_full_qwen_{lang}.jsonl"):
                if (r["seed"] == 42 and r["config"] == CFGS[0] and r["lora_variant"] == ad and r.get("samples")
                        and (ad, CFGS[0]) not in qwen):
                    qwen[(ad, CFGS[0])] = samples(r)
        base = qwen[("qlora", CFGS[1])][:N_ITEMS]
        items = [dict(id=f"{lang}-{s['example_index']}", language=lang, question=s["question"], context=s["context"],
                      gold=s["gold"], answers={}) for s in base]
        def put(model, ad, cfg, i, gen, em, f1):
            items[i]["answers"].setdefault(model, {}).setdefault(ad, {})[cfg] = dict(generated=gen, em=float(em), f1=float(f1))
        for ad in ADAPTERS:
            for cfg in CFGS:
                for i, s in enumerate(qwen[(ad, cfg)][:N_ITEMS]):
                    assert s["gold"] == items[i]["gold"], "Qwen item mismatch"
                    put("qwen3", ad, cfg, i, s["generated"], s["em"], s["f1"])
            for r in read_records(FS / "t5gemma2" / f"checkpoint_full_{ad}_{lang}.jsonl"):
                if r["seed"] != 42: continue
                for i, s in enumerate(samples(r)[:N_ITEMS]):
                    assert s["gold"] == items[i]["gold"], f"T5Gemma2 item mismatch {lang} {ad} {i}"
                    put("t5gemma2", ad, r["config"], i, s["generated"], s["em"], s["f1"])
            if a.runs_v2:
                for cfg in CFGS:
                    f = Path(a.runs_v2) / "core" / f"{ad}_{lang}_s42" / "preds" / f"{cfg}_{ad}_{lang}_seed42_l0.3_real_ckpt0.jsonl"
                    by_q = {p["question"]: p for p in map(json.loads, open(f, encoding="utf-8"))}
                    for i, it in enumerate(items):
                        p = by_q.get(it["question"])
                        if p: put("mt5", ad, cfg, i, p["generated"], p["em"], p["f1"])
        out += items
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: {len(out)} items; models per item: {sorted({m for it in out for m in it['answers']})}")

if __name__ == "__main__":
    main()
