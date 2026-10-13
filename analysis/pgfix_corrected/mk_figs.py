import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9.5,"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})
w=pd.read_csv("per_seed_wide.csv")
V=[("qlora","QLoRA"),("lora","LoRA"),("dora","DoRA")]
fig,axs=plt.subplots(1,2,figsize=(9,3.4),sharey=False)
# panel a: default vs B and ckptoff vs B, per adapter (language-avg per seed)
ax=axs[0]; x=np.arange(3); wd=0.36
for j,(c,lab,col) in enumerate([("C_default","Original setting (ckpt on)","#B5483A"),("C_ckptoff","Checkpointing off","#2F6F8F")]):
    for i,(v,_) in enumerate(V):
        g=w[w.lora_variant==v].copy(); g["d"]=g[c]-g["B"]; s=g.groupby("seed").d.mean()
        ax.bar(x[i]+(j-.5)*wd,s.mean(),wd,color=col,label=lab if i==0 else None,alpha=.9)
        ax.scatter([x[i]+(j-.5)*wd]*len(s),s.values,color="k",s=8,zorder=3)
ax.axhline(0,color="#444",lw=.8); ax.set_xticks(x); ax.set_xticklabels([n for _,n in V]); ax.set_ylabel(r"$\Delta F_{\rm faith}$ (C $-$ B)"); ax.set_title("(a) Checkpointing confound",fontsize=10); ax.set_ylim(-0.21,0.07); ax.legend(frameon=False,fontsize=8,loc="lower left",bbox_to_anchor=(-0.02,0.0))
# panel b: attribution
ax=axs[1]; wd=0.26
for j,(c,lab,col) in enumerate([("C_l0_ckptoff",r"$\lambda_1=0$","#9AA3AD"),("C_shuf_ckptoff","Shuffled reward","#C9A227"),("C_ckptoff","Real reward","#2F6F8F")]):
    for i,(v,_) in enumerate(V):
        g=w[w.lora_variant==v].copy(); g["d"]=g[c]-g["B"]; s=g.groupby("seed").d.mean()
        ax.bar(x[i]+(j-1)*wd,s.mean(),wd,color=col,label=lab if i==0 else None,alpha=.9)
        ax.scatter([x[i]+(j-1)*wd]*len(s),s.values,color="k",s=8,zorder=3)
ax.axhline(0,color="#444",lw=.8); ax.set_xticks(x); ax.set_xticklabels([n for _,n in V]); ax.set_title("(b) Reward attribution (ckpt off)",fontsize=10); ax.set_ylim(-0.04,0.07); ax.legend(frameon=False,fontsize=8,loc="upper center",ncol=3,columnspacing=0.8,handlelength=1)
ax.set_ylabel(r"$\Delta F_{\rm faith}$ vs. B")
plt.tight_layout(); plt.savefig("fig_ckpt_attrib.pdf"); print("fig ok")
