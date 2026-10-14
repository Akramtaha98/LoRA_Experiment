"""
Reconstructs the source context/premise text for the first 5 held-out
examples per language (indices 0-4 of eval_indices = range(30) in
lora_experiment_matrix.py). The experiment checkpoint logs only retain
{gold, generated, em, f1} for the 5 logged samples per condition -- not
the context passage -- so this script reloads the same datasets, with
the same deterministic slicing (no shuffle), to recover it.

Run this before xlmr_rescore.py.

Output: nli_rescore_contexts.json
"""
from datasets import load_dataset
import json

ar = load_dataset("google/xquad", "xquad.ar", split="validation")
ar = ar.select(range(min(500, len(ar))))

ms = load_dataset("facebook/belebele", "zsm_Latn", split="test")
ms = ms.select(range(min(500, len(ms))))

data = {"arabic": [], "malay": []}
for i in range(5):
    ex = ar[i]
    gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
    data["arabic"].append({"idx": i, "context": ex["context"], "question": ex["question"], "gold": gold})

for i in range(5):
    ex = ms[i]
    correct = ex["correct_answer_num"]
    gold = ex[f"mc_answer{correct}"]
    data["malay"].append({"idx": i, "context": ex["flores_passage"], "question": ex["question"], "gold": gold})

with open("nli_rescore_contexts.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("Wrote nli_rescore_contexts.json")
