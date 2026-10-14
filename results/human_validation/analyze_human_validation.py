#!/usr/bin/env python3
"""Reproduce the human-validation tables from human_validation_labels.csv.
Usage: python analyze_human_validation.py human_validation_labels.csv
q1: 2 supported, 1 partly, 0 unsupported, X empty/unreadable (scored 0 in ordinal summaries)."""
import csv, math, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
ORD = {"2": 2, "1": 1, "0": 0, "X": 0}

def kappa(a, b, labels, quad=False):
    ix = {l: i for i, l in enumerate(labels)}; k = len(labels); m = np.zeros((k, k))
    for x, y in zip(a, b): m[ix[x], ix[y]] += 1
    e = np.outer(m.sum(1), m.sum(0)) / m.sum()
    W = np.array([[0 if i == j else (((i - j) / (k - 1)) ** 2 if quad else 1) for j in range(k)] for i in range(k)])
    return 1 - (W * m).sum() / (W * e).sum()

def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

for lang in ("arabic", "malay"):
    R = [r for r in rows if r["language"] == lang]
    a, b = [r["q1_rater1"] for r in R], [r["q1_rater2"] for r in R]
    print(f"{lang}: n={len(R)} raw={np.mean([x == y for x, y in zip(a, b)]):.3f} "
          f"kappa4={kappa(a, b, ['2','1','0','X']):.3f} "
          f"wkappa={kappa([str(ORD[x]) for x in a], [str(ORD[x]) for x in b], ['0','1','2'], True):.3f} "
          f"binary={kappa([str(int(x=='2')) for x in a], [str(int(x=='2')) for x in b], ['0','1']):.3f}")
    rho, p = spearmanr([ORD[r["q1_final"]] for r in R], [float(r["f_faith_nli_score"]) for r in R])
    print(f"  Spearman(NLI, human) = {rho:.3f} (p={p:.3g})")
    for var in ("qlora", "dora"):
        g = lambda c, col: [r[col] == "2" for r in R if r["lora_variant"] == var and r["config"] == c]
        for col in ("q1_final", "q1_rater1", "q1_rater2"):
            B, C = g("B_ce_lora", col), g("C_composite_lora", col)
            pb, pc = np.mean(B), np.mean(C); se = math.sqrt(pb*(1-pb)/len(B) + pc*(1-pc)/len(C)); d = pc - pb
            print(f"  {var:5s} {col:10s} B={pb:.3f}(n={len(B)}) C={pc:.3f}(n={len(C)}) diff={d:+.3f} [{d-1.96*se:+.3f},{d+1.96*se:+.3f}]")
