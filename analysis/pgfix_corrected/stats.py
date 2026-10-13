import pandas as pd, numpy as np, json
from scipy import stats
w=pd.read_csv("per_seed_wide.csv")
out={}
def comp(a,b,name,variants=("dora","lora","qlora"),lang=None):
    d=w[w.lora_variant.isin(variants)].copy()
    if lang: d=d[d.language==lang]
    d=d.dropna(subset=[a,b]); d["diff"]=d[a]-d[b]
    res={}
    for v,g in d.groupby("lora_variant"):
        # language-avg per seed
        s=g.groupby("seed")["diff"].mean()
        t,p=stats.ttest_1samp(s,0) if len(s)>1 else (np.nan,np.nan)
        res[v]=dict(delta=round(s.mean(),4),sd=round(s.std(ddof=1),4),n=len(s),t=round(t,3),p=round(p,4),pos_cells=int((g["diff"]>0).sum()),cells=len(g),
                    arab=round(g[g.language=="arabic"]["diff"].mean(),4),malay=round(g[g.language=="malay"]["diff"].mean(),4))
    # pooled over adapters: per seed mean of cells
    s=d.groupby("seed")["diff"].mean()
    t,p=stats.ttest_1samp(s,0)
    se=s.std(ddof=1)/np.sqrt(len(s)); tc=stats.t.ppf(.95,len(s)-1)
    res["pooled"]=dict(delta=round(s.mean(),4),n=len(s),t=round(t,3),p=round(p,4),ci90=[round(s.mean()-tc*se,4),round(s.mean()+tc*se,4)],
        pos_cells=int((d["diff"]>0).sum()),cells=len(d),sign_p=round(stats.binomtest(int((d["diff"]>0).sum()),len(d),.5).pvalue,4),cellsd=round(d["diff"].std(ddof=1),4))
    out[name]=res
    return res
comp("C_ckptoff","B","ckptoff_vs_B")
comp("C_default","B","default_vs_B")
comp("C_ckptoff","C_shuf_ckptoff","real_vs_shuffled")
comp("C_ckptoff","C_l0_ckptoff","real_vs_lambda0")
comp("C_shuf_ckptoff","B","shuffled_vs_B")
comp("C_l0_ckptoff","B","lambda0_vs_B")
comp("C_ckptoff","C_ckpton","off_vs_on")
comp("C_l0_ckpton","B","l0_ckpton_vs_B")
comp("C_shuf_ckpton","B","shuf_ckpton_vs_B")
# adalora/vera
comp("C_ckptoff","B","ckptoff_vs_B_adalora_vera",variants=("adalora","vera"))
print(json.dumps(out,indent=1))
json.dump(out,open("stats.json","w"),indent=1)
