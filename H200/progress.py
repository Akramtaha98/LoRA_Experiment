"""Live progress + ETA + running results for the hero_v2 training.   usage: python progress.py [results_train]
Run it live:  watch -n 30 python progress.py"""
import glob, json, os, re, sys, time
from collections import defaultdict
root = sys.argv[1] if len(sys.argv) > 1 else "results_train"
STEPS, EVAL_S = 1416, 90
MODELS, LANGS = ("qwen3", "t5gemma2"), ("arabic", "malay")
SEEDS = [42, 123, 2026, 7, 99]
CONDS = ["B", "C", "C_l0", "C_shuf"]
done = {}
for f in glob.glob(f"{root}/*/*/*/seed*/summary.json"):
    d = json.load(open(f)); done[(d["model"], d["lang"], d["cond"], d["seed"])] = d
total = len(MODELS) * len(LANGS) * (len(SEEDS) * len(CONDS) + 1)
cur, speeds = {}, []
for m in MODELS:
    for l in LANGS:
        for s in SEEDS:
            lf = f"{root}/logs/{m}_{l}_seed{s}.log"
            if not os.path.exists(lf): continue
            last = None
            for line in open(lf, errors="ignore"):
                mm = re.search(r"\[(\w+) s(\d+)\] step (\d+)/(\d+) .* (\d+)s$", line.strip())
                if mm: last = (mm.group(1), int(mm.group(3)), int(mm.group(4)), int(mm.group(5)))
            if last: cur[(m, l, s)] = last
            if last and last[0] != "B" and last[1] >= 30: speeds.append(last[3] / last[1])
x = sorted(speeds)[len(speeds) // 2] if speeds else 1.5
b = x / 6
rem = {}
for m in MODELS:
    for l in LANGS:
        for s in SEEDS:
            t = 0.0
            for c in (["A_frozen"] if s == SEEDS[0] else []) + CONDS:
                if (m, l, c, s) in done: continue
                full = (0 if c == "A_frozen" else STEPS * (b if c == "B" else x)) + EVAL_S
                c0 = cur.get((m, l, s))
                if c0 and c0[0] == c: full -= c0[1] * (b if c == "B" else x)
                t += max(full, 0)
            rem[(m, l, s)] = t
print(f"{time.strftime('%H:%M:%S')}   runs finished: {len(done)}/{total} ({100*len(done)/total:.0f}%)   "
      f"C speed: {x:.2f} s/step   ETA: ~{max(rem.values())/3600:.1f} h  (mean process {sum(rem.values())/len(rem)/3600:.1f} h)\n")
print("process                     now running                 progress")
for (m, l, s), (c, st, tot, t) in sorted(cur.items()):
    fin = sum(1 for k in done if k[:2] == (m, l) and k[3] == s)
    print(f"{m:9s} {l:7s} seed {s:<5d} {c:8s} step {st:4d}/{tot}  ({100*st/tot:3.0f}%)   runs done {fin}/{5 if s==SEEDS[0] else 4}   left ~{rem[(m,l,s)]/3600:.1f} h")
print("\nLIVE RESULTS (mean over finished seeds; faith = NLI entailment, F1/EM vs gold)")
print(f"{'model':9s} {'lang':7s} {'cond':8s} {'seeds':>5s} {'faith':>7s} {'F1':>7s} {'EM':>7s} {'words':>6s}")
order = ["A_frozen"] + CONDS
g = defaultdict(list)
for (m, l, c, s), d in done.items(): g[(m, l, c)].append(d)
for (m, l, c), ds in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], order.index(kv[0][2]))):
    mean = lambda k: sum(d[k] for d in ds) / len(ds)
    print(f"{m:9s} {l:7s} {c:8s} {len(ds):5d} {mean('faith'):7.3f} {mean('f1'):7.3f} {mean('em'):7.3f} {mean('mean_words'):6.1f}")
