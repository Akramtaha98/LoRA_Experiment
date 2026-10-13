import json, os, subprocess, sys, tempfile, types, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_eval as H
root = tempfile.mkdtemp(); rng = np.random.default_rng(0)
for cfg, lam, rm in (("B_ce_lora", 0.3, "real"), ("C_composite_lora", 0.3, "real"), ("C_composite_lora", 0.0, "real"), ("C_composite_lora", 0.3, "shuffled")):
    for lang in ("arabic", "malay"):
        d = os.path.join(root, f"{cfg}{lam}{rm}{lang}"); os.makedirs(d + "/preds")
        f = "preds/p.jsonl"
        with open(os.path.join(d, f), "w") as fh:
            for i in range(60):
                ans = "same answer" if (i % 3 == 0) else f"ans {cfg} {i}"
                fh.write(json.dumps(dict(src_id=f"s{i}", ds_index=i, passage_in_train=False, generated=ans, question="q", f_faith=float(rng.random()))) + "\n")
        open(d + "/checkpoint_x.jsonl", "w").write(json.dumps(dict(status="PASS", config=cfg, lora_variant="lora", language=lang, seed=42, lambda1=lam, reward_mode=rm, preds_file=f)) + "\n")
out = os.path.join(root, "he")
a = types.SimpleNamespace(runs=root, out=out, n=20, adapter="lora", seed=42, seed_rng=0)
H.make(a, passage_fn=lambda lang: (lambda i: (f"passage {i}", None if lang == "arabic" else ["a", "b", "c", "d"])))
key = pd.read_csv(out + "/key.csv")
for k in (1, 2):
    s = pd.read_csv(f"{out}/sheet_annotator{k}.csv"); s["supported"] = rng.integers(0, 3, len(s)); s["correct"] = rng.integers(0, 2, len(s)); s.to_csv(f"{out}/sheet_annotator{k}.csv", index=False)
H.score(types.SimpleNamespace(dir=out, n_boot=200))
assert "conds" in key and key.conds.str.contains(",").any(), "identical answers should be merged across conditions"
print("HUMAN_EVAL_TEST_PASS")
