"""Automatic pass/fail checks for the hero_v2 test run.
usage: python check_results.py smoke  results_test/smoke
       python check_results.py learn  results_test/learn
       python check_results.py timing results_test/timing
Exit code 1 on any FAIL.  WARN does not fail."""
import json, math, sys
from pathlib import Path

mode, root = sys.argv[1], Path(sys.argv[2])
fails, warns = [], []
def ok(c, msg, warn=False):
    print(("PASS  " if c else ("WARN  " if warn else "FAIL  ")) + msg)
    if not c: (warns if warn else fails).append(msg)

def runs():
    for f in sorted(root.glob("*/*/*/seed*/summary.json")):
        yield f.parent, json.load(open(f))
def preds(d):
    return [json.loads(l) for l in open(d / "predictions.jsonl", encoding="utf-8")]

if mode == "smoke":
    expected = {(m, l, c) for m in ("qwen3", "t5gemma2") for l in ("arabic", "malay")
                for c in ("A_frozen", "B", "C", "C_l0", "C_shuf", "B_ckpt")}
    got = {(s["model"], s["lang"], s["cond"]) for _, s in runs()}
    ok(expected <= got, f"all {len(expected)} model/lang/condition runs finished (missing: {sorted(expected - got)})")
    byk = {}
    for d, s in runs():
        rows = preds(d)
        k = (s["model"], s["lang"], s["cond"]); byk[k] = rows
        ok(len(rows) > 0 and all(r["ex_id"] is not None for r in rows), f"{k}: predictions saved with ids ({len(rows)})")
        vals = [s.get(x) for x in ("faith", "f1", "em")]
        ok(all(v is not None and math.isfinite(v) for v in vals), f"{k}: metrics finite {vals}")
        ok(s["empty_rate"] < 1.0, f"{k}: not all answers empty (empty_rate={s['empty_rate']:.2f})", warn=True)
        if s["cond"] in ("C", "C_l0", "C_shuf"):
            tl = [json.loads(l) for l in open(d / "train_log.jsonl")]
            ok(bool(tl) and all(t["reward"] is not None and math.isfinite(t["reward"]) and math.isfinite(t["ce"]) for t in tl),
               f"{k}: train log has finite CE and NLI reward")
    for m in ("qwen3", "t5gemma2"):
        for l in ("arabic", "malay"):
            a, b = byk.get((m, l, "B")), byk.get((m, l, "B_ckpt"))
            if a and b:
                ce = {c: [json.loads(x)["ce"] for x in open(root / m / l / c / "seed42" / "train_log.jsonl")] for c in ("B", "B_ckpt")}
                d1 = abs(ce["B"][0] - ce["B_ckpt"][0])
                d2 = abs(ce["B"][1] - ce["B_ckpt"][1]) / max(1e-6, abs(ce["B"][1]))
                ok(d1 < 1e-3 and d2 < 0.15, f"{m}/{l}: checkpointing ON/OFF give the same loss (step1 diff {d1:.4f}, step2 rel diff {d2:.3f})")
                agree = sum(x["pred"] == y["pred"] for x, y in zip(a, b))
                print(f"      info: {agree}/{len(a)} identical decoded answers after 3 noisy steps (bf16 noise can flip a few)")
elif mode == "learn":
    for m in ("qwen3", "t5gemma2"):
        res = {s["cond"]: s for _, s in runs() if s["model"] == m and s["lang"] == "arabic"}
        if "A_frozen" in res and "B" in res:
            a, b = res["A_frozen"], res["B"]
            print(f"      {m}: frozen F1={a['f1']:.3f} faith={a['faith']:.3f} | B F1={b['f1']:.3f} faith={b['faith']:.3f} "
                  f"words {a['mean_words']:.1f}->{b['mean_words']:.1f}")
            ok(b["f1"] > a["f1"], f"{m}: LoRA training improves F1 over frozen (learning actually works)")
            ok(b["empty_rate"] < 0.2, f"{m}: trained model rarely outputs empty answers ({b['empty_rate']:.2f})")
        else:
            ok(False, f"{m}: learn runs missing")
elif mode == "timing":
    total_h = 0.0
    tab = {}
    for _, s in runs():
        if s["n_steps"]:
            tab[(s["model"], s["lang"], s["cond"])] = s
    for m in ("qwen3", "t5gemma2"):
        for l in ("arabic", "malay"):
            c, b = tab.get((m, l, "C")), tab.get((m, l, "B"))
            if not (c and b):
                ok(False, f"{m}/{l}: timing runs missing"); continue
            spc, spb = c["train_seconds"] / c["n_steps"], b["train_seconds"] / b["n_steps"]
            full_steps = 1416 if l == "arabic" else 1416   # 470 train x 12 epochs / bs 4 (same split both languages)
            ev_per_ex = (c["total_seconds"] - c["train_seconds"]) / max(1, c["n_eval"])
            n_eval = 720 if l == "arabic" else 430
            ev = ev_per_ex * n_eval
            hrs = (15 * (spc * full_steps + ev) + 5 * (spb * full_steps + ev)) / 3600
            print(f"      {m}/{l}: {spb:.2f}s/step (B)  {spc:.2f}s/step (C)  eval~{ev/60:.1f} min  -> full 5-seed job ~ {hrs:.1f} h")
            total_h = max(total_h, hrs)
    print(f"\n      ESTIMATED WALL-CLOCK with 4 free GPUs (slowest job): ~{total_h:.1f} h ; sequential on 1 GPU would be several times longer")
if fails:
    print(f"\n{len(fails)} FAIL(s) -> do NOT start the long run"); sys.exit(1)
print(f"\nALL CHECKS PASSED ({len(warns)} warning(s))")
