import pandas as pd, numpy as np, re
from scipy import stats
df=pd.read_csv("all.csv")
df["prefix"]=df.job.str.replace(r"_(dora|lora|qlora|adalora|vera)_(arabic|malay)_s\d+$","",regex=True)
def cond(r):
    s,p,c=r.stage,r.prefix,r.config
    if c.startswith("B"): return "B"
    if s in("core","extra"): return "C_default"
    if s=="ckptoff" or p=="ckptoff": return "C_ckptoff"
    if p=="ckpton": return "C_ckpton"
    if s=="controls" and p=="shuffled": return "C_shuf_ckpton"
    if s=="controls" and p=="lambda0": return "C_l0_ckpton"
    if s=="attrib" and p=="shuffled": return "C_shuf_ckptoff"
    if s=="attrib" and p=="lambda0": return "C_l0_ckptoff"
    return "other"
df["cond"]=df.apply(cond,axis=1)
df=df[df.stage!="repeat"]
# B: use core/extra B as the canonical; ckptoff-stage B are duplicates of different run -> keep as B_rerun
df.loc[(df.cond=="B")&(df.stage.isin(["ckptoff","controls"])),"cond"]="B_rerun"
df.to_csv("conditions.csv",index=False)
piv=df.pivot_table(index=["lora_variant","language","seed"],columns="cond",values="mean_f_faith")
piv.to_csv("per_seed_wide.csv")
m=df.groupby(["lora_variant","language","cond"]).mean_f_faith.mean().unstack().round(4)
print(m.to_string())
print(df.groupby(["lora_variant","language","cond"]).mean_em.mean().unstack().round(3).to_string())
print(df.groupby("cond").size())
