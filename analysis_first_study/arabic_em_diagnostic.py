"""
Arabic exact-match diagnostic (docs/EXPERIMENTAL_DEBT_ROADMAP.md-adjacent
item: "Arabic EM diagnostic" from the JKSUCI critique). Lowest-priority of
the four remaining items because it is pure analysis of logs you will
already have from the mT5 --log_all_examples reruns done for the human
validation study -- no separate GPU run needed if you run the Arabic
QLoRA/DoRA Config B/C conditions with --log_all_examples (see RUNBOOK.md
/ the GPUHub command block).

INPUT: checkpoint JSONL file(s) for Arabic conditions run with
--log_all_examples (same format sample_for_human_annotation.py reads).

WHAT THIS COMPUTES, per condition (config x variant):
  1. Generated-answer length distribution (word count): a generator stuck
     producing near-empty or wildly long outputs on Arabic specifically
     (vs. Malay) is a generation-quality symptom, not a scoring artifact.
  2. Script consistency: what fraction of generated answers contain at
     least one Arabic-script character (U+0600-U+06FF block) vs. being
     entirely Latin/CJK/other -- a cheap, deterministic proxy for the kind
     of pretraining-artifact leakage already documented qualitatively in
     Section sec:results-qual (e.g. the frozen baseline's Chinese-script
     fragment on Malay). A low Arabic-script fraction on nominally-Arabic
     generations would point at a tokenizer/generation-config issue rather
     than "the model just isn't very good yet."
  3. A random 50-example CSV for manual reading (gold, generated, context,
     EM, F1) -- the "manual audit of 50 Arabic outputs" the critique asked
     for. This script does not itself judge fluency/correctness; that is
     the manual step you do by reading the CSV.

USAGE:
    python arabic_em_diagnostic.py \
        --checkpoints ckpt_B_ce_lora_qlora_arabic_logall.jsonl ckpt_C_composite_lora_qlora_arabic_logall.jsonl ... \
        --out_summary arabic_diagnostic_summary.csv \
        --out_sample arabic_diagnostic_manual_audit_sample.csv \
        --sample_size 50 --seed 20260913
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

ARABIC_RE = re.compile(r"[؀-ۿ]")


def load_records(paths):
    records = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def flatten(records):
    rows = []
    skipped = 0
    for rec in records:
        if rec.get("language") != "arabic":
            continue
        samples_raw = rec.get("samples", "")
        if not samples_raw:
            continue
        try:
            samples = json.loads(samples_raw)
        except json.JSONDecodeError:
            continue
        for s in samples:
            if "context" not in s:
                skipped += 1
                continue
            rows.append({
                "config": rec.get("config"), "variant": rec.get("lora_variant"),
                "seed": rec.get("seed"), "example_index": s.get("example_index"),
                "context": s["context"], "question": s.get("question", ""),
                "gold": s["gold"], "generated": s["generated"],
                "em": s.get("em", ""), "f1": s.get("f1", ""),
                "f_faith": s.get("f_faith", ""),
            })
    if skipped:
        print(f"NOTE: skipped {skipped} records without a 'context' field "
              f"(run without --log_all_examples).", file=sys.stderr)
    return rows


def summarize(rows):
    by_cond = {}
    for r in rows:
        key = (r["config"], r["variant"])
        by_cond.setdefault(key, []).append(r)

    summary = []
    for (config, variant), group in sorted(by_cond.items()):
        lengths = [len(r["generated"].split()) for r in group]
        empty = sum(1 for r in group if not r["generated"].strip())
        has_arabic = sum(1 for r in group if ARABIC_RE.search(r["generated"]))
        n = len(group)
        summary.append({
            "config": config, "variant": variant, "n": n,
            "mean_gen_len_words": round(statistics.mean(lengths), 2) if lengths else 0,
            "median_gen_len_words": statistics.median(lengths) if lengths else 0,
            "pct_empty_generation": round(100 * empty / n, 1) if n else 0,
            "pct_contains_arabic_script": round(100 * has_arabic / n, 1) if n else 0,
            "mean_em": round(statistics.mean([float(r["em"]) for r in group if r["em"] != ""]), 4) if group else 0,
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out_summary", default="arabic_diagnostic_summary.csv")
    ap.add_argument("--out_sample", default="arabic_diagnostic_manual_audit_sample.csv")
    ap.add_argument("--sample_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()

    records = load_records(args.checkpoints)
    rows = flatten(records)
    if not rows:
        print("No eligible Arabic per-example rows found -- rerun the "
              "Arabic B/C conditions with --log_all_examples first.",
              file=sys.stderr)
        sys.exit(1)

    summary = summarize(rows)
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote per-condition summary ({len(summary)} conditions) to "
          f"{args.out_summary}")
    for s in summary:
        print(f"  {s['config']:20s} {s['variant']:8s} n={s['n']:4d} "
              f"mean_len={s['mean_gen_len_words']:5.1f}w "
              f"empty%={s['pct_empty_generation']:5.1f} "
              f"arabic_script%={s['pct_contains_arabic_script']:5.1f} "
              f"EM={s['mean_em']:.4f}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sample = rows[:min(args.sample_size, len(rows))]
    fieldnames = ["config", "variant", "seed", "example_index", "context",
                  "question", "gold", "generated", "em", "f1", "f_faith",
                  "manual_notes"]
    with open(args.out_sample, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample:
            row["manual_notes"] = ""
            writer.writerow(row)
    print(f"Wrote {len(sample)}-example manual-audit sample to "
          f"{args.out_sample} -- read this by hand for the qualitative "
          f"part of the diagnostic (fluency, script mixing, truncation, "
          f"off-topic generation).")


if __name__ == "__main__":
    main()
