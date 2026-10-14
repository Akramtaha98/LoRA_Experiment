#!/usr/bin/env python3
"""Synthetic end-to-end test of analyze_v2.py: plants a known C-B effect, checks the CI covers it and controls ~0."""
import json, os, subprocess, sys, tempfile, numpy as np
rng = np.random.default_rng(0); root = tempfile.mkdtemp(); TRUE = 0.02
def write(d, rec, items):
    os.makedirs(os.path.join(d, "preds"), exist_ok=True)
    f = f"preds/{rec['config']}_{rec['lora_variant']}_{rec['language']}_{rec['seed']}_{rec['reward_mode']}_{rec['lambda1']}.jsonl"
    with open(os.path.join(d, f), "w") as fh:
        for i, v in enumerate(items): fh.write(json.dumps(dict(src_id=f"s{i}", f_faith=float(v), passage_in_train=False)) + "\n")
    rec["preds_file"] = f; open(os.path.join(d, "checkpoint_x.jsonl"), "a").write(json.dumps(rec) + "\n")
base = {(v, l): 0.25 + 0.05 * i for i, (v, l) in enumerate([(v, l) for v in ("dora", "lora", "qlora") for l in ("arabic", "malay")])}
item_eff = {k: rng.normal(0, 0.1, 100) for k in base}
for (v, l), b in base.items():
    for s in (1, 2, 3, 4, 5, 6, 7, 8):
        d = os.path.join(root, f"{v}_{l}_{s}"); os.makedirs(d)
        sn = rng.normal(0, 0.01)
        for cfg, lam, rm, eff in (("B_ce_lora", 0.3, "real", 0), ("C_composite_lora", 0.3, "real", TRUE), ("C_composite_lora", 0.0, "real", 0), ("C_composite_lora", 0.3, "shuffled", 0)):
            items = np.clip(b + item_eff[(v, l)] + eff + sn + rng.normal(0, 0.02, 100), 0, 1)
            rec = dict(status="PASS", config=cfg, lora_variant=v, language=l, seed=s, lambda1=lam, reward_mode=rm, grad_ckpt=False,
                       policy_mode="eval", mean_f_faith=float(items.mean()), mean_em=0.1, mean_f1=0.2)
            write(d, rec, items)
out = os.path.join(root, "out")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "lora_nli", "analyze.py"), root, "--out", out, "--n-boot", "300"], check=True, capture_output=True)
import pandas as pd
c = pd.read_csv(f"{out}/contrasts.csv"); p = c[(c.contrast == "C-B") & (c.scope == "pooled(6 cells)") & (c.metric == "mean_f_faith")].iloc[0]
l0 = c[(c.contrast == "C_l0-B") & (c.metric == "mean_f_faith")].iloc[0]
bt = json.load(open(f"{out}/item_bootstrap.json"))
ok = p.ci95_lo < TRUE < p.ci95_hi and l0.ci95_lo < 0 < l0.ci95_hi and bt["ci95"][0] < TRUE < bt["ci95"][1] and "p_holm" in c.columns
print(f"pooled C-B = {p['mean']:.4f} [{p.ci95_lo:.4f},{p.ci95_hi:.4f}]  null control {l0['mean']:.4f} [{l0.ci95_lo:.4f},{l0.ci95_hi:.4f}]  bootstrap {bt['ci95']}")
print("ANALYSIS_TEST_PASS" if ok else "ANALYSIS_TEST_FAIL"); sys.exit(0 if ok else 1)
