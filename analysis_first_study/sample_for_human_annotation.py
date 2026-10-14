"""
Sample a stratified set of examples for the F_faith human-annotation
validation study (docs/EXPERIMENTAL_DEBT_ROADMAP.md, item 7).

INPUT: one or more checkpoint JSONL files produced by
lora_experiment_matrix.py / qwen_experiment_matrix.py run with the
--log_all_examples flag (added for this purpose -- the default run only
logs the first 5 examples per condition with no context field, which is
not enough to build a human-annotation set from).

Each line of an input JSONL is one condition's summary record; the
per-example generations live in its "samples" field as a JSON-encoded
list of dicts with keys: example_index, context, question, gold,
generated, em, f1, f_faith (only present when --log_all_examples was
used; the default run's "samples" field has gold/generated/em/f1 only,
for the first 5 examples, and is skipped here).

USAGE:
    python sample_for_human_annotation.py \
        --checkpoints ckpt_B_ce_lora_qlora_arabic.jsonl ckpt_C_composite_lora_qlora_arabic.jsonl ... \
        --n_per_language 175 \
        --seed 20260913 \
        --out human_annotation_set.csv

Produces a CSV (one row per sampled example) with the columns the
annotation spreadsheet expects (see build_annotation_template.py in this
same folder for the empty, annotator-facing spreadsheet). This script
only does deterministic sampling and formatting -- it never invents,
edits, or scores a generation; every row's context/generated/gold text
is copied verbatim from the input JSONL.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path


# Stratification target: the paper's two headline variants, both training
# configs (CE-only and composite-loss), both languages. This directly
# validates F_faith on the conditions the paper's two headline claims
# (QLoRA's gain, DoRA's collapse) depend on, not F_faith in the abstract.
TARGET_VARIANTS = {"qlora", "dora"}
TARGET_CONFIGS = {"B_ce_lora", "C_composite_lora"}

# The first 5 examples of every condition are already in the paper's
# exploratory n=15-per-cell re-scoring diagnostic (Section
# sec:results-robustness). Excluding them keeps this human-annotation set
# an independent check, not the same handful of examples scaled up.
EXCLUDE_EXAMPLE_INDICES = {0, 1, 2, 3, 4}


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
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def extract_candidates(records, backbone_label):
    """Flatten every eligible per-example generation into a flat list of
    annotation-ready dicts."""
    candidates = []
    skipped_no_context = 0
    for rec in records:
        config = rec.get("config")
        variant = rec.get("lora_variant")
        language = rec.get("language")
        seed = rec.get("seed")
        if config not in TARGET_CONFIGS or variant not in TARGET_VARIANTS:
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
                # This condition was logged without --log_all_examples
                # (old-format, first-5-only, no context). Can't use it.
                skipped_no_context += 1
                continue
            idx = s.get("example_index")
            if idx in EXCLUDE_EXAMPLE_INDICES:
                continue
            candidates.append({
                "backbone": backbone_label,
                "config": config,
                "variant": variant,
                "language": language,
                "seed": seed,
                "example_index": idx,
                "context": s["context"],
                "question": s.get("question", ""),
                "gold_answer": s["gold"],
                "generated_answer": s["generated"],
                "em": s.get("em", ""),
                "f1": s.get("f1", ""),
                "f_faith_model_score": s.get("f_faith", ""),
            })
    if skipped_no_context:
        print(f"NOTE: skipped {skipped_no_context} per-example records that "
              f"had no 'context' field -- these came from a run without "
              f"--log_all_examples and cannot be used for annotation.",
              file=sys.stderr)
    return candidates


def stratified_sample(candidates, n_per_language, seed):
    rng = random.Random(seed)
    by_language = {}
    for c in candidates:
        by_language.setdefault(c["language"], []).append(c)

    sampled = []
    for language, pool in by_language.items():
        rng.shuffle(pool)
        take = min(n_per_language, len(pool))
        if take < n_per_language:
            print(f"WARNING: only {len(pool)} eligible candidates for "
                  f"'{language}', requested {n_per_language}. Taking all "
                  f"{take}.", file=sys.stderr)
        sampled.extend(pool[:take])
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                     help="One or more checkpoint JSONL files produced "
                          "with --log_all_examples.")
    ap.add_argument("--backbone", default="unspecified",
                     help="Label to record in the 'backbone' column "
                          "(e.g. 'mt5-base' or 'qwen3-0.6b-base'). Run this "
                          "script separately per backbone if annotating "
                          "both.")
    ap.add_argument("--n_per_language", type=int, default=175,
                     help="Target sample size per language (default 175, "
                          "the midpoint of the paper's stated 150-200 "
                          "range).")
    ap.add_argument("--seed", type=int, default=20260913,
                     help="RNG seed for reproducible sampling.")
    ap.add_argument("--out", default="human_annotation_sample.csv")
    args = ap.parse_args()

    records = load_records(args.checkpoints)
    if not records:
        print("No records loaded -- check --checkpoints paths.", file=sys.stderr)
        sys.exit(1)

    candidates = extract_candidates(records, args.backbone)
    if not candidates:
        print("No eligible per-example generations found. Did you run the "
              "affected conditions with --log_all_examples?", file=sys.stderr)
        sys.exit(1)

    sampled = stratified_sample(candidates, args.n_per_language, args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(sampled)  # de-block by condition so annotators can't guess

    fieldnames = ["annotation_id", "backbone", "language", "config",
                  "variant", "seed", "example_index", "context", "question",
                  "gold_answer", "generated_answer", "em", "f1",
                  "f_faith_model_score",
                  # blank columns for the annotators to fill in:
                  "annotator_1_judgment", "annotator_1_notes",
                  "annotator_2_judgment", "annotator_2_notes",
                  "adjudicated_judgment"]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(sampled):
            row["annotation_id"] = f"A{i+1:04d}"
            for k in ("annotator_1_judgment", "annotator_1_notes",
                      "annotator_2_judgment", "annotator_2_notes",
                      "adjudicated_judgment"):
                row[k] = ""
            writer.writerow(row)

    print(f"Wrote {len(sampled)} examples to {args.out} "
          f"({args.n_per_language} requested per language).")


if __name__ == "__main__":
    main()
