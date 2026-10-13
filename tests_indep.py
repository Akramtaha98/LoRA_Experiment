import json, os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import independent_eval as I
root = tempfile.mkdtemp(); rng = np.random.default_rng(0)
for cfg, lam, rm, eff in (("B_ce_lora", 0.3, "real", 0), ("C_composite_lora", 0.3, "real", .05), ("C_composite_lora", 0.0, "real", 0)):
    for seed in (1, 2, 3, 4):
        d = os.path.join(root, f"{cfg}{lam}{seed}"); os.makedirs(d + "/preds")
        with open(d + "/preds/p.jsonl", "w") as fh:
            for i in range(30): fh.write(json.dumps(dict(src_id=f"s{i}", ds_index=i, passage_in_train=False, generated=f"{cfg}{lam}{seed}-{i}", f_faith=float(rng.random()))) + "\n")
        open(d + "/checkpoint_x.jsonl", "w").write(json.dumps(dict(status="PASS", config=cfg, lora_variant="lora", language="malay", seed=seed, lambda1=lam, reward_mode=rm, preds_file="preds/p.jsonl", eff=eff)) + "\n")
fake = lambda model, pairs, dev: [float(np.clip(0.4 + (0.05 if "composite" in h and "0.3" in h else 0) + rng.normal(0, .02), 0, 1)) for _, h in pairs]
I.main([root, "--out", os.path.join(root, "o")], scorer=fake, passage_fn=lambda lang: (lambda i: (f"p{i}", None)))
res = json.load(open(os.path.join(root, "o", "indep_results.json")))
assert res["f_indep:C-B"]["ci95"][0] > 0 and res["f_indep:C-C_l0"]["ci95"][0] > 0
print("INDEP_EVAL_TEST_PASS")
