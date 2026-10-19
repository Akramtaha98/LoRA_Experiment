"""Matched-answer-length analysis of the NLI faithfulness score (hero_v2 results).
Usage: python lenmatch.py <results_train_dir> <out_dir>
For each (model, lang):
  (1) mean faith by answer-length bin and condition;
  (2) matched pairs: answers of two conditions to the SAME item and seed that fall in the SAME bin
      (difference = first - second; item-cluster bootstrap 95% CI; reported if >= MINP pairs);
  (3) length-adjusted contrast: OLS of faith on condition dummies + length-bin fixed effects (item-cluster bootstrap),
      compared with the unadjusted contrast.
"""
import json, sys, glob, os, numpy as np, pandas as pd
R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BINS = [(1,1),(2,3),(4,6),(7,10),(11,10**9)]; BL = ["1","2-3","4-6","7-10","11+"]
def b(n):
    for i,(lo,hi) in enumerate(BINS):
        if lo<=n<=hi: return i
    return 0 if n<1 else len(BINS)-1
rows=[]
for f in glob.glob(f"{R}/*/*/*/seed*/predictions.jsonl"):
    for l in open(f):
        r=json.loads(l); rows.append((r["model"],r["lang"],r["cond"],r["seed"],r["ex_id"],r["n_words"],r["faith"],r["f1"]))
D=pd.DataFrame(rows,columns="model lang cond seed ex w faith f1".split()); D["bin"]=D.w.map(b)
D=D[~D.ex.isin([])]  # no exclusions applied here: ex_ids are unique per (lang, item)
rng=np.random.default_rng(0); NB=2000; MINP=30
def boot(items, fn):
    """cluster bootstrap over item ids; fn(index array of items) -> stat"""
    n=len(items); vals=[fn(rng.integers(0,n,n)) for _ in range(NB)]
    return np.percentile(vals,[2.5,97.5])
res=[]; binrows=[]
for (m,l),G in D.groupby(["model","lang"]):
    # (1) bin means
    for c,H in G.groupby("cond"):
        for i,h in H.groupby("bin"):
            binrows.append(dict(model=m,lang=l,cond=c,bin=BL[i],n=len(h),faith=h.faith.mean()))
    # (2) matched pairs & (3) adjusted contrast, vs reference conds
    for a,ref in [("C","B"),("C","C_l0"),("C","C_shuf"),("B","A_frozen")]:
        A=G[G.cond==a].set_index(["seed","ex"]) if a!="B" or ref!="A_frozen" else G[G.cond=="B"].set_index(["seed","ex"])
        Rf=G[G.cond==ref]
        if ref=="A_frozen":   # frozen has one seed: pair every trained seed with it
            Rf=Rf.drop(columns="seed").set_index("ex"); P=A.reset_index().join(Rf,on="ex",lsuffix="_a",rsuffix="_r")
        else:
            P=A.join(Rf.set_index(["seed","ex"]),lsuffix="_a",rsuffix="_r",how="inner").reset_index()
        P["d"]=P.faith_a-P.faith_r; P["same"]=P.bin_a==P.bin_r
        raw=P.groupby("ex").d.mean()
        lo,hi=boot(raw.index,lambda ix:raw.values[ix].mean())
        res.append(dict(model=m,lang=l,a=a,ref=ref,kind="unadjusted",bin="all",n=len(P),d=raw.mean(),lo=lo,hi=hi))
        # matched, per bin
        for i in range(len(BINS)):
            S=P[P.same&(P.bin_a==i)]
            if len(S)<MINP: res.append(dict(model=m,lang=l,a=a,ref=ref,kind="matched",bin=BL[i],n=len(S),d=np.nan,lo=np.nan,hi=np.nan)); continue
            s=S.groupby("ex").d.mean(); lo,hi=boot(s.index,lambda ix:s.values[ix].mean())
            res.append(dict(model=m,lang=l,a=a,ref=ref,kind="matched",bin=BL[i],n=len(S),d=s.mean(),lo=lo,hi=hi))
        # matched pooled over bins: weight pairs equally
        S=P[P.same]
        if len(S)>=MINP:
            s=S.groupby("ex").d.mean(); lo,hi=boot(s.index,lambda ix:s.values[ix].mean())
            res.append(dict(model=m,lang=l,a=a,ref=ref,kind="matched",bin="pooled",n=len(S),d=s.mean(),lo=lo,hi=hi))
        # length-adjusted: pair-level d regressed on difference in bin dummies (bin FE for each side) -> intercept = adjusted effect
        # equivalently OLS faith ~ cond + bin FE on stacked long data (items clustered)
        L=pd.concat([P[["ex","faith_a","bin_a"]].rename(columns={"faith_a":"y","bin_a":"bin"}).assign(t=1),
                     P[["ex","faith_r","bin_r"]].rename(columns={"faith_r":"y","bin_r":"bin"}).assign(t=0)])
        X=pd.get_dummies(L.bin.astype(int),dtype=float); X.insert(0,"t",L.t.values); Xv=X.values; y=L.y.values
        beta=np.linalg.lstsq(Xv,y,rcond=None)[0][0]
        exs=L.ex.values; uex=np.unique(exs); idx={e:np.where(exs==e)[0] for e in uex}
        bs=[]
        for _ in range(500):
            ii=np.concatenate([idx[e] for e in rng.choice(uex,len(uex))]); bs.append(np.linalg.lstsq(Xv[ii],y[ii],rcond=None)[0][0])
        lo,hi=np.percentile(bs,[2.5,97.5])
        res.append(dict(model=m,lang=l,a=a,ref=ref,kind="adjusted",bin="FE",n=len(L)//2,d=beta,lo=lo,hi=hi))
        # share of pairs in same bin
        res.append(dict(model=m,lang=l,a=a,ref=ref,kind="share_same_bin",bin="-",n=len(P),d=P.same.mean(),lo=np.nan,hi=np.nan))
pd.DataFrame(binrows).to_csv(f"{OUT}/len_bins.csv",index=False); pd.DataFrame(res).to_csv(f"{OUT}/len_contrasts.csv",index=False)
print("done")
