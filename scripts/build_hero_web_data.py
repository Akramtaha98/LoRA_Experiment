#!/usr/bin/env python3
"""Build the public hero-study dashboard data from the saved experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hero_v2"
MODELS = ("qwen3", "t5gemma2")
LANGUAGES = ("arabic", "malay")
CONDITIONS = ("B", "C", "C_l0", "C_shuf")
METRICS = ("faith", "faith_2nd", "f1", "em", "n_words")


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def summarize(values):
    present = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(present) if present else None,
        "sd": statistics.stdev(present) if len(present) > 1 else (0.0 if present else None),
        "n": len(present),
    }


def build_cells():
    rows = read_csv(RESULTS / "analysis" / "per_seed.csv")
    indexed = defaultdict(list)
    for row in rows:
        key = (row["model"], row["lang"], row["cond"])
        indexed[key].append(row)

    cells = {}
    for model in MODELS:
        cells[model] = {}
        for language in LANGUAGES:
            cells[model][language] = {}
            for condition in ("A_frozen", *CONDITIONS):
                selected = indexed.get((model, language, condition), [])
                if not selected:
                    continue
                cells[model][language][condition] = {
                    "seeds": sorted(int(row["seed"]) for row in selected),
                    "metrics": {
                        metric: summarize([number(row.get(metric)) for row in selected])
                        for metric in METRICS
                    },
                }
                cells[model][language][condition]["n_eval"] = sorted(
                    {int(row["n"]) for row in selected}
                )
    return cells


def build_contrasts():
    raw = load_json(RESULTS / "analysis" / "contrasts.json")
    output = defaultdict(lambda: defaultdict(dict))
    for key, record in raw.items():
        model, language, contrast, metric = key.split("/")
        output[model][language].setdefault(contrast, {})[metric] = {
            "mean": record["mean_diff"],
            "lo": record["ci95"][0],
            "hi": record["ci95"][1],
            "n_examples": record["n_examples"],
            "seeds_positive": record["seeds_positive"],
            "n_seeds": record["n_seeds"],
        }
    return output


def build_length_data():
    bins = read_csv(RESULTS / "length_matched" / "len_bins.csv")
    contrasts = read_csv(RESULTS / "length_matched" / "len_contrasts.csv")
    return {
        "bins": [
            {**row, "n": int(row["n"]), "faith": number(row["faith"])}
            for row in bins
        ],
        "contrasts": [
            {
                **row,
                "n": int(row["n"]),
                "difference": number(row["d"]),
                "lo": number(row["lo"]),
                "hi": number(row["hi"]),
            }
            for row in contrasts
        ],
    }


def build_training_diagnostics():
    grouped = defaultdict(list)
    for model in MODELS:
        for language in LANGUAGES:
            for path in sorted((RESULTS / model / language / "C").glob("seed*/train_log.jsonl")):
                with path.open(encoding="utf-8") as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                if not rows:
                    continue
                final_step = max(int(row["step"]) for row in rows)
                for row in rows:
                    quarter = min(3, (int(row["step"]) - 1) * 4 // final_step)
                    grouped[(model, language, quarter)].append(row)

    quarters = defaultdict(dict)
    for (model, language, quarter), rows in sorted(grouped.items()):
        pg = [abs(float(row["pg"])) for row in rows]
        quarters[model][language] = quarters[model].get(language, [])
        quarters[model][language].append(
            {
                "quarter": quarter + 1,
                "inactive_share": sum(value < 1e-3 for value in pg) / len(pg),
                "mean_pg_abs": statistics.fmean(pg),
                "mean_reward": statistics.fmean(float(row["reward"]) for row in rows),
                "mean_ce": statistics.fmean(float(row["ce"]) for row in rows),
                "logged_steps": len(rows),
            }
        )
    return quarters


def prediction_map(archive, model, language, condition):
    member = f"{model}/{language}/{condition}/seed42/predictions.jsonl"
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Missing saved predictions: {member}")
    with handle:
        return {
            int(row["ex_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def saved_answer(row):
    return {
        "answer": row.get("pred", ""),
        "faith": number(row.get("faith")),
        "faith_2nd": number(row.get("faith_2nd")),
        "f1": number(row.get("f1")),
        "em": number(row.get("em")),
        "n_words": number(row.get("n_words")),
    }


def build_examples():
    output = {}
    with tarfile.open(RESULTS / "predictions.tgz", "r:gz") as archive:
        for language in LANGUAGES:
            records = {
                (model, condition): prediction_map(archive, model, language, condition)
                for model in MODELS
                for condition in ("B", "C")
            }
            shared_ids = set.intersection(*(set(value) for value in records.values()))
            candidates = []
            for ex_id in sorted(shared_ids):
                rows = [records[(model, condition)][ex_id] for model in MODELS for condition in ("B", "C")]
                if any(row.get("passage_in_train") for row in rows):
                    continue
                # First 50 shared held-out items, including unchanged answers.
                candidates.append(ex_id)
            if len(candidates) < 50:
                raise ValueError(f"Only {len(candidates)} shared held-out examples for {language}")

            examples = []
            for ex_id in candidates[:50]:
                qwen = records[("qwen3", "B")][ex_id]
                t5gemma = records[("t5gemma2", "B")][ex_id]
                for model in MODELS:
                    for condition in ("B", "C"):
                        row = records[(model, condition)][ex_id]
                        if (row.get("question"), row.get("gold")) != (qwen.get("question"), qwen.get("gold")):
                            raise ValueError(f"Question/gold mismatch for {language} example {ex_id}")
                examples.append(
                    {
                        "id": ex_id,
                        "question": qwen["question"],
                        "context": qwen.get("passage", ""),
                        "gold": qwen.get("gold", ""),
                        "gold_option": qwen.get("gold_option"),
                        "options": qwen.get("options"),
                        "answers": {
                            model: {
                                condition: saved_answer(records[(model, condition)][ex_id])
                                for condition in ("B", "C")
                            }
                            for model in MODELS
                        },
                    }
                )
            output[language] = examples
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "web" / "public" / "data" / "hero.json"))
    args = parser.parse_args()
    cells = build_cells()
    seeds = sorted({seed for model in cells.values() for language in model.values() for condition in language.values() for seed in condition["seeds"]})
    payload = {
        "meta": {
            "title": "Qwen3 and T5Gemma 2 multilingual QA results",
            "models": list(MODELS),
            "languages": list(LANGUAGES),
            "conditions": list(CONDITIONS),
            "seeds": seeds,
            "source": "results/hero_v2/analysis/per_seed.csv and saved hero_v2 run outputs",
            "protocol": "New Qwen3/T5Gemma 2 run set; five seeds; results are exploratory.",
        },
        "cells": cells,
        "contrasts": build_contrasts(),
        "length": build_length_data(),
        "trainingDiagnostics": build_training_diagnostics(),
        "examples": build_examples(),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    example_count = sum(len(items) for items in payload["examples"].values())
    print(f"HERO_WEB_DATA_OK cells=4 examples={example_count} source_rows={sum(len(v) for m in payload['cells'].values() for v in m.values())}")


if __name__ == "__main__":
    main()
