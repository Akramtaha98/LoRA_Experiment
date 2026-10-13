import pandas as pd, numpy as np, json
from scipy import stats
df=pd.read_csv("conditions.csv"); S=json.load(open("stats.json"))
V=[("qlora","QLoRA"),("lora","LoRA"),("dora","DoRA"),("adalora","AdaLoRA"),("vera","VeRA")]
L=[("arabic","Arabic"),("malay","Malay")]
def ms(v,l,c,col="mean_f_faith",dec=4):
    x=df[(df.lora_variant==v)&(df.language==l)&(df.cond==c)][col]
    if len(x)==0: return "--"
    if len(x)==1: return f"{x.iloc[0]:.{dec}f}"
    return f"{x.mean():.{dec}f} $\\pm$ {x.std(ddof=1):.{dec}f}"
o=[]
# Table 1: main
o.append(r"""\begin{table}[t]
\centering
\caption{mT5-base, corrected pipeline (Section~\ref{sec:method-train}), seeds 42, 123 and 2026; mean~$\pm$~SD over seeds. B: CE-only LoRA. C(off): composite objective, gradient checkpointing off. C(default): composite objective under the checkpointing setting of the original protocol (on for every non-quantized variant, off for QLoRA, so QLoRA's C(default) is the same run as C(off)). Arabic: 720 held-out examples; Malay: 430. For AdaLoRA and VeRA, C(off) has seeds 123 and 2026 only ($n=2$).}
\label{tab:main}
\footnotesize
\begin{tabular}{@{}llccccc@{}}
\toprule
& & \multicolumn{3}{c}{$F_{\text{faith}}$} & \multicolumn{2}{c}{EM} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
\textbf{Lang} & \textbf{Variant} & B & C(off) & C(default) & B & C(off) \\
\midrule""")
for l,ln in L:
    for v,vn in V:
        cd = "same as C(off)" if v=="qlora" else ms(v,l,"C_default")
        o.append(f"{ln} & {vn} & {ms(v,l,'B')} & {ms(v,l,'C_ckptoff')} & {cd} & {ms(v,l,'B','mean_em',3)} & {ms(v,l,'C_ckptoff','mean_em',3)} \\\\")
    if l=="arabic": o.append(r"\midrule")
o.append(r"\botrule"+"\n"+r"\end{tabular}"+"\n"+r"\end{table}")
open("tab_main.tex","w").write("\n".join(o))
# Table 2: attribution
o=[r"""\begin{table}[t]
\centering
\caption{Reward-attribution controls (gradient checkpointing off; three seeds; mean~$\pm$~SD of $F_{\text{faith}}$). $\lambda_1{=}0$: the sampling-and-scoring path runs but the policy-gradient term has zero weight. Shuffled: the NLI rewards are permuted across the batch before advantages are computed, which preserves the reward distribution but removes its dependence on the sampled answer. B is repeated from Table~\ref{tab:main}.}
\label{tab:attrib}
\footnotesize
\begin{tabular}{@{}lllcccc@{}}
\toprule
\textbf{Lang} & \textbf{Variant} & & B & $\lambda_1{=}0$ & Shuffled & Real reward \\
\midrule"""]
for l,ln in L:
    for v,vn in V[:3]:
        o.append(f"{ln} & {vn} & & {ms(v,l,'B')} & {ms(v,l,'C_l0_ckptoff')} & {ms(v,l,'C_shuf_ckptoff')} & {ms(v,l,'C_ckptoff')} \\\\")
    if l=="arabic": o.append(r"\midrule")
o.append(r"\botrule"+"\n"+r"\end{tabular}"+"\n"+r"\end{table}")
open("tab_attrib.tex","w").write("\n".join(o))
# Table 3: contrasts
def holm(ps):
    idx=np.argsort(ps); m=len(ps); out=[None]*m; run=0
    for r,i in enumerate(idx):
        run=max(run,(m-r)*ps[i]); out[i]=min(1,run)
    return out
rows=[("ckptoff_vs_B","C(off) $-$ B"),("real_vs_lambda0","C(off) $-$ ($\\lambda_1{=}0$)"),("real_vs_shuffled","C(off) $-$ shuffled"),("default_vs_B","C(default) $-$ B")]
o=[r"""\begin{table}[t]
\centering
\caption{Language-averaged contrasts in $F_{\text{faith}}$, seed-paired ($n=3$ seeds, 2 degrees of freedom). $p$: two-sided paired $t$; $p_{\text{Holm}}$: Holm correction over the three variants within a contrast. Cells: variant-by-language-by-seed differences that are positive (descriptive; the 18 cells are not independent). ``All'' pools the three variants (per-seed mean over six cells; 90\% CI shown); it treats variants as exchangeable and is descriptive.}
\label{tab:contrasts}
\footnotesize
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Contrast} & \textbf{Variant} & $\Delta$ & $p$ & $p_{\text{Holm}}$ & Cells $>0$ & 90\% CI \\
\midrule"""]
for k,nm in rows:
    R=S[k]; ps=[R[v]["p"] for v in ("qlora","lora","dora")]; h=holm(ps)
    for i,(v,vn) in enumerate([("qlora","QLoRA"),("lora","LoRA"),("dora","DoRA")]):
        r=R[v]; o.append(f"{nm if i==0 else ''} & {vn} & ${r['delta']:+.4f}$ & {r['p']:.3f} & {h[i]:.3f} & {r['pos_cells']}/{r['cells']} & \\\\")
    r=R["pooled"]; o.append(f" & All & ${r['delta']:+.4f}$ & {r['p']:.3f} & -- & {r['pos_cells']}/{r['cells']} & [${r['ci90'][0]:+.4f}$, ${r['ci90'][1]:+.4f}$] \\\\")
    o.append(r"\midrule")
o[-1]=r"\botrule"
o.append(r"\end{tabular}"+"\n"+r"\end{table}")
open("tab_contrasts.tex","w").write("\n".join(o))
# Table 4: ckpt seed 42
o=[r"""\begin{table}[t]
\centering
\caption{Gradient-checkpointing audit, seed 42 (one run per cell). $F_{\text{faith}}$ of Config~C with checkpointing on and off, and of the two controls with checkpointing on, against Config~B. The two control columns follow the original protocol's default (checkpointing on, except QLoRA, which was never checkpointed). With checkpointing on, Config~C collapses whether the reward is real, shuffled or absent ($\lambda_1{=}0$), and QLoRA collapses too when checkpointing is forced on for it; with checkpointing off the same code does not collapse.}
\label{tab:ckpt}
\footnotesize
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Lang} & \textbf{Variant} & B & C, ckpt on & C, ckpt off & $\lambda_1{=}0$, on & Shuffled, on \\
\midrule"""]
d=df[df.seed==42]
def one(v,l,c,s42=True):
    x=df[(df.lora_variant==v)&(df.language==l)&(df.cond==c)&(df.seed==42)].mean_f_faith
    return f"{x.iloc[0]:.4f}" if len(x) else "--"
for l,ln in L:
    for v,vn in V[:3]:
        # controls ckpt on/off rows: ckpton cond; off = 'C_ckptoff' seed42
        o.append(f"{ln} & {vn} & {one(v,l,'B')} & {one(v,l,'C_ckpton')} & {one(v,l,'C_ckptoff')} & {one(v,l,'C_l0_ckpton')} & {one(v,l,'C_shuf_ckpton')} \\\\")
    if l=="arabic": o.append(r"\midrule")
o.append(r"\botrule"+"\n"+r"\end{tabular}"+"\n"+r"\end{table}")
open("tab_ckpt.tex","w").write("\n".join(o))
print("ok")
