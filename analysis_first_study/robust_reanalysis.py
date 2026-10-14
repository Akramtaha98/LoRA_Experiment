#!/usr/bin/env python3
"""Robustness reanalysis of the Family-1 contrasts from RAW per-seed results (per_seed_results.csv).
Three estimators of Delta = F(C)-F(B), averaged over Arabic and Malay, for every backbone x variant:
  pooled : original analysis (one pooled residual SD per backbone, 32 df)  [reproduced from raw data]
  welch  : unequal-variance, each of the 4 cells keeps its own variance, Welch-Satterthwaite df
  paired : per-seed language-averaged difference, one-sample t over the 3 seeds (df=2)
Also: Holm correction over the 8 Family-1 tests per estimator, and the aggregate mean effect over 4 variants."""
import csv, math, sys, json
import numpy as np
from scipy import stats
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else 'per_seed_results.csv')))
V = ['qlora','adalora','dora','vera']; L = ['arabic','malay']; S = ['42','123','2026']
def get(bb,v,l,c):
    d = {r['seed']: float(r['f_faith']) for r in rows if r['backbone']==bb and r['variant']==v and r['language']==l and r['config']==c}
    assert sorted(d)==sorted(S), (bb,v,l,c,d.keys()); return np.array([d[s] for s in S])
res = {}
for bb in ('mt5','qwen3'):
    cell_var = {}
    for v in V:
        for l in L:
            for c in ('B_ce_lora','C_composite_lora'): cell_var[(v,l,c)] = get(bb,v,l,c).var(ddof=1)
    s2 = np.mean(list(cell_var.values())); dfp = 32
    for v in V:
        B = {l:get(bb,v,l,'B_ce_lora') for l in L}; C = {l:get(bb,v,l,'C_composite_lora') for l in L}
        delta = np.mean([C[l].mean()-B[l].mean() for l in L])
        # pooled
        se_p = math.sqrt(s2*(4/4)/3 * 1.0) * math.sqrt(4*(0.5**2)/1) / math.sqrt(1) if False else math.sqrt(sum(0.25*s2/3 for _ in range(4)))
        t_p = delta/se_p; p_p = 2*stats.t.sf(abs(t_p), dfp)
        # welch
        vs = [cell_var[(v,l,c)]/3*0.25 for l in L for c in ('B_ce_lora','C_composite_lora')]
        se_w = math.sqrt(sum(vs)); dfw = se_w**4/sum(x**2/2 for x in vs)
        t_w = delta/se_w; p_w = 2*stats.t.sf(abs(t_w), dfw)
        # paired over seeds
        d_seed = np.mean([C[l]-B[l] for l in L], axis=0)
        t_s, p_s = stats.ttest_1samp(d_seed, 0.0)
        res[(bb,v)] = dict(delta=delta, p_pooled=p_p, t_pooled=t_p, se_pooled=se_p, p_welch=p_w, df_welch=dfw, t_welch=t_w, se_welch=se_w, p_paired=float(p_s), t_paired=float(t_s), seeds_pos=int((d_seed>0).sum()), d_seed=d_seed.tolist())
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, idx in enumerate(o): run = max(run, (m-i)*ps[idx]); adj[idx] = min(1, run)
    return adj
keys = list(res)
for est in ('pooled','welch','paired'):
    a = holm([res[k]['p_'+est] for k in keys])
    for k, x in zip(keys, a): res[k]['holm_'+est] = float(x)
print(f"{'bb':5s}{'variant':8s}{'delta':>9s} | pooled p  holm | welch  t   df    p    holm | paired t   p   holm | seeds(+)")
for k in keys:
    r = res[k]
    print(f"{k[0]:5s}{k[1]:8s}{r['delta']:+9.4f} | {r['p_pooled']:.2e} {r['holm_pooled']:.2e} | {r['t_welch']:+7.2f} {r['df_welch']:5.1f} {r['p_welch']:.4f} {r['holm_welch']:.4f} | {r['t_paired']:+8.2f} {r['p_paired']:.4f} {r['holm_paired']:.4f} | {r['seeds_pos']}/3")
# aggregate (mean over 4 variants x 2 languages) per backbone
agg = {}
for bb in ('mt5','qwen3'):
    per_seed = np.mean([res[(bb,v)]['d_seed'] for v in V], axis=0)
    t, p = stats.ttest_1samp(per_seed, 0.0)
    var_delta = [res[(bb,v)]['delta'] for v in V]
    t2, p2 = stats.ttest_1samp(var_delta, 0.0)
    agg[bb] = dict(mean=float(np.mean(var_delta)), p_seedpaired=float(p), t_seedpaired=float(t), p_across_variants=float(p2), t_across_variants=float(t2), per_seed=per_seed.tolist())
    print(f"{bb} aggregate mean effect over 4 variants = {np.mean(var_delta):+.4f}; seed-paired t(2)={t:+.2f} p={p:.3f}; across-variant t(3)={t2:+.2f} p={p2:.3f}")
# QLoRA unpaired Welch per-language and paired per-language
for l in L:
    for v in ('qlora','dora'):
        B = get('mt5',v,l,'B_ce_lora'); C = get('mt5',v,l,'C_composite_lora')
        t,p = stats.ttest_ind(C,B,equal_var=False); tp,pp = stats.ttest_rel(C,B)
        print(f"mt5 {v} {l}: delta={C.mean()-B.mean():+.4f} welch p={p:.4f} paired p={pp:.4f} seeds+={int((C>B).sum())}/3")
# plain LoRA language-averaged paired
for bb,v in (('mt5','lora'),):
    B = {l:get(bb,v,l,'B_ce_lora') for l in L}; C = {l:get(bb,v,l,'C_composite_lora') for l in L}
    d = np.mean([C[l]-B[l] for l in L],axis=0); t,p = stats.ttest_1samp(d,0.0); print('plain LoRA',d.mean(),t,p)
json.dump({'family1':{f'{k[0]}/{k[1]}':v for k,v in res.items()},'aggregate':agg}, open('robust_reanalysis_results.json','w'), indent=1)
