"""Generate LaTeX tables and figures for the manuscript from the hero_v2 results.
usage: python gen_paper_assets.py <results_train> <len_dir> <out_src_dir>"""
import sys, json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, LEN, OUT = sys.argv[1:4]
import os; os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
ps = pd.read_csv(f"{R}/analysis/per_seed.csv")
con = json.load(open(f"{R}/analysis/contrasts.json"))
lc = pd.read_csv(f"{LEN}/len_contrasts.csv"); lb = pd.read_csv(f"{LEN}/len_bins.csv")
M = {"qwen3":"Qwen3-0.6B-Base","t5gemma2":"T5Gemma~2"}; MS = {"qwen3":"Qwen3","t5gemma2":"T5G2"}; L = {"arabic":"Arabic","malay":"Malay"}
CO = ["A_frozen","B","C","C_l0","C_shuf"]; CN = {"A_frozen":"A (frozen)","B":"B (CE)","C":"C (CE+NLI)","C_l0":"C, $\\lambda{=}0$","C_shuf":"C, shuffled"}
def f3(x): return f"{x:.3f}"
def fm(x):
    t=f"{x:+.3f}".replace("+0.","+.").replace("-0.","-.")
    if t in ("+.000","-.000"): t=".000"
    return t.replace("-","$-$")
def fi(lo,hi): return f"[{fm(lo)},{fm(hi)}]"
def sgn(x): return f"{x:+.3f}"
# ---------- main table
rows=[]
for m in M:
  for l in L:
    rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{M[m]}, {L[l]}}}}}\\\\")
    for c in CO:
      d=ps[(ps.model==m)&(ps.lang==l)&(ps.cond==c)]
      def ms(col):
        v=d[col].dropna(); 
        if len(v)==0: return "--"
        return f"{v.mean():.3f}" if len(v)==1 else f"{v.mean():.3f}\\,{{\\scriptsize$\\pm${v.std(ddof=1):.3f}}}"
      rows.append(f"\\quad {CN[c]} & {len(d)} & {ms('faith')} & {ms('faith_2nd')} & {ms('f1')} & {ms('em')} & {ms('option_correct')} & {d.n_words.mean():.1f}\\\\")
    rows.append("\\addlinespace")
tab=("\\begin{table}[t]\n\\centering\n\\caption{Primary study: mean over seeds (and standard deviation over seeds where more than one) of held-out scores. "
 "Faith: NLI entailment score under the reward model; Faith$_2$: independent NLI model; Opt.: option accuracy (Malay only; the answer selects the gold option). "
 "Condition A has one seed because it involves no training. Held-out sets: 720 Arabic and 430 Malay items.}\n\\label{tab:h-main}\n\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n"
 "\\begin{tabular}{@{}lrcccccr@{}}\n\\toprule\n\\textbf{Condition} & $n_{\\text{seeds}}$ & \\textbf{Faith} & \\textbf{Faith$_2$} & \\textbf{F1} & \\textbf{EM} & \\textbf{Opt.} & \\textbf{Words}\\\\\n\\midrule\n"
 + "\n".join(rows) + "\n\\botrule\n\\end{tabular}\n\\end{table}\n")
open(f"{OUT}/tab_h_main.tex","w").write(tab)
# ---------- contrasts table
from scipy import stats as _st
def seedci(m,l,a,b):
  x=ps[(ps.model==m)&(ps.lang==l)&(ps.cond==a)].set_index("seed").faith; y=ps[(ps.model==m)&(ps.lang==l)&(ps.cond==b)].set_index("seed").faith
  d=(x-y).dropna(); t=_st.t.ppf(.975,len(d)-1); se=d.std(ddof=1)/np.sqrt(len(d)); lo,hi=d.mean()-t*se,d.mean()+t*se
  return fi(lo,hi)+("$^{\\dagger}$" if (lo>0 or hi<0) else "")
def cell(k):
  v=con.get(k)
  if not v: return "--"
  lo,hi=v["ci95"]; star="$^{*}$" if (lo>0 or hi<0) else ""
  return f"{fm(v['mean_diff'])}{star} {fi(lo,hi)}"
def ctab(cols, caption, label, fname):
  rows=[]
  for m in M:
    for l in L:
      for (a,b,lab) in [("C","B","C$-$B"),("C","C_l0","C$-$C$_{\\lambda=0}$"),("C","C_shuf","C$-$C$_{\\text{shuf}}$")]:
        pre=f"{m}/{l}/{a}-{b}"
        cells=[]
        for c in cols:
          cells.append(seedci(m,l,a,b) if c=="seeds" else cell(pre+"/"+c))
        rows.append(f"{MS[m]}-{L[l][:2]} & {lab} & "+" & ".join(cells)+"\\\\")
      rows.append("\\addlinespace")
  hd={"faith":"Faith","seeds":"Faith, seeds","faith_2nd":"Faith$_2$","f1":"F1","em":"EM"}
  tab=("\\begin{table}[t]\n\\centering\n\\caption{"+caption+"}\n\\label{"+label+"}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
   "\\begin{tabular}{@{}ll"+"l"*len(cols)+"@{}}\n\\toprule\n\\textbf{Cell} & \\textbf{Contrast} & "+" & ".join("\\textbf{"+hd[c]+"}" for c in cols)+"\\\\\n\\midrule\n"
   +"\n".join(rows)+"\\botrule\n\\end{tabular}\n\\end{table}\n")
  open(f"{OUT}/{fname}","w").write(tab)
ctab(["faith","seeds","faith_2nd"],
 "Paired contrasts of the NLI score (Ar: Arabic, Ma: Malay, T5G2: T5Gemma~2; mean difference). Faith: reward-model score, 95\\% bootstrap interval over held-out items (seed-averaged item scores); $^{*}$: interval excludes zero. Faith, seeds: same contrast, 95\\% $t$ interval over the five seeds (4 degrees of freedom); $^{\\dagger}$: excludes zero. Faith$_2$: independent scorer, item bootstrap. None is adjusted for the 12 contrasts per outcome. C$-$B tests the composite objective as a whole; C against the two controls tests whether the reward information matters.",
 "tab:h-contrasts","tab_h_contrasts.tex")
ctab(["f1","em"],
 "Paired contrasts of F1 and exact match (Ar: Arabic, Ma: Malay, T5G2: T5Gemma~2; mean difference with 95\\% bootstrap interval over held-out items; $^{*}$: interval excludes zero; no multiplicity adjustment).",
 "tab:h-contrasts-acc","tab_h_contrasts_acc.tex")
# ---------- length-adjusted table
def g(m,l,a,ref,kind,bin_):
  r=lc[(lc.model==m)&(lc.lang==l)&(lc.a==a)&(lc.ref==ref)&(lc.kind==kind)&(lc.bin==bin_)]
  return r.iloc[0] if len(r) else None
rows=[]
for m in M:
  for l in L:
    for (a,ref,lab) in [("C","B","C$-$B"),("C","C_l0","C$-$C$_{\\lambda=0}$"),("C","C_shuf","C$-$C$_{\\text{shuf}}$"),("B","A_frozen","B$-$A")]:
      u=g(m,l,a,ref,"unadjusted","all"); ad=g(m,l,a,ref,"adjusted","FE"); mp=g(m,l,a,ref,"matched","pooled"); sh=g(m,l,a,ref,"share_same_bin","-")
      f=lambda r: "n/a" if r is None or pd.isna(r.d) else f"{fm(r.d)}\\,{fi(r.lo,r.hi)}"
      rows.append(f"{MS[m]}-{L[l][:2]} & {lab} & {f(u)} & {f(ad)} & {f(mp)} & {100*sh.d:.0f}\\%\\\\")
    rows.append("\\addlinespace")
tab=("\\begin{table}[t]\n\\centering\n\\caption{Faithfulness contrasts with answer length controlled (Ar: Arabic, Ma: Malay, T5G2: T5Gemma~2). Unadjusted: mean paired difference. "
 "Adjusted: coefficient of the condition indicator in a regression of the NLI score on the condition and answer-length-bin fixed effects "
 "(bins 1, 2--3, 4--6, 7--10, 11+ words; the same items and seeds as the unadjusted contrast). Matched: mean difference over pairs of answers "
 "to the same item and seed that fall in the same length bin. Same bin: share of pairs that fall in the same bin. All intervals are 95\\% bootstrap intervals over held-out items. "
 "For B$-$A the frozen model has one seed, and its answers are rarely in the same bin as trained answers.}\n\\label{tab:h-len}\n\\scriptsize\n\\setlength{\\tabcolsep}{1pt}\n"
 "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n\\textbf{Cell} & \\textbf{Contrast} & \\textbf{Unadjusted} & \\textbf{Length-adjusted} & \\textbf{Matched} & \\textbf{Same bin}\\\\\n\\midrule\n"
 + "\n".join(rows) + "\\botrule\n\\end{tabular}\n\\end{table}\n")
open(f"{OUT}/tab_h_len.tex","w").write(tab)
# ---------- length bin table (appendix)
BL=["1","2-3","4-6","7-10","11+"]; rows=[]
for m in M:
  for l in L:
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{M[m]}, {L[l]}}}}}\\\\")
    for c in CO:
      cells=[]
      for b in BL:
        r=lb[(lb.model==m)&(lb.lang==l)&(lb.cond==c)&(lb.bin==b)]
        cells.append("--" if len(r)==0 else f"{r.faith.iloc[0]:.2f}\\,({int(r.n.iloc[0])})")
      rows.append(f"\\quad {CN[c]} & "+" & ".join(cells)+"\\\\")
    rows.append("\\addlinespace")
tab=("\\begin{table}[t]\n\\centering\n\\caption{Mean NLI score (number of answers) by answer length in words, all seeds pooled.}\n\\label{tab:h-lenbins}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
 "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n\\textbf{Condition} & \\textbf{1} & \\textbf{2--3} & \\textbf{4--6} & \\textbf{7--10} & \\textbf{11+}\\\\\n\\midrule\n"
 + "\n".join(rows) + "\\botrule\n\\end{tabular}\n\\end{table}\n")
open(f"{OUT}/tab_h_lenbins.tex","w").write(tab)
# ---------- training diagnostics (condition C)
agg={}
for f in glob.glob(f"{R}/*/*/C/seed*/train_log.jsonl"):
  m,l=f.split("/")[-5:-3]
  for r in map(json.loads,open(f)):
    q=min(3,(r["step"]-1)*4//1416); agg.setdefault((m,l,q),[]).append((r["ce"],abs(r["pg"]),r["reward"],abs(r["pg"])<1e-3))
Bce={}
for f in glob.glob(f"{R}/*/*/B/seed*/train_log.jsonl"):
  m,l=f.split("/")[-5:-3]
  for r in map(json.loads,open(f)):
    if r["step"]>1062: Bce.setdefault((m,l),[]).append(r["ce"])
rows=[]
for m in M:
  for l in L:
    for q in range(4):
      a=np.array(agg[(m,l,q)],float).mean(0)
      rows.append(f"{M[m] if (l=='arabic' and q==0) else ''} & {L[l] if q==0 else ''} & {q+1} & {a[0]:.3f} & {a[1]:.3f} & {a[2]:.3f} & {100*a[3]:.0f}\\%\\\\")
    rows.append("\\addlinespace")
tab=("\\begin{table}[t]\n\\centering\n\\caption{Training diagnostics of Config~C (logged every 50 steps; all seeds; quarters of the 1,416 steps). CE: token-level cross-entropy of the minibatch; $|$PG$|$: absolute policy-gradient term; "
 "Reward: mean NLI reward of the sampled answers; Inactive: share of logged steps with $|$PG$|<10^{-3}$, i.e. all $K{=}4$ samples receive (almost) the same reward or the same answer.}\n\\label{tab:h-diag}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
 "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n\\textbf{Model} & \\textbf{Lang.} & \\textbf{Qtr} & \\textbf{CE} & $|$\\textbf{PG}$|$ & \\textbf{Reward} & \\textbf{Inactive}\\\\\n\\midrule\n"
 + "\n".join(rows) + "\\botrule\n\\end{tabular}\n\\end{table}\n")
open(f"{OUT}/tab_h_diag.tex","w").write(tab)
json.dump({f"{k[0]}/{k[1]}":float(np.mean(v)) for k,v in Bce.items()}, open(f"{LEN}/B_final_quarter_ce.json","w"))
# ---------- figures
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6),constrained_layout=True)
cols={"A_frozen":"#9a9a9a","B":"#4c78a8","C":"#e45756","C_l0":"#72b7b2","C_shuf":"#f2a93b"}
cells=[(m,l) for m in M for l in L]; xlab=[f"{MS[m].replace('~','')}\n{L[l]}" for m,l in cells]
for a_,(col,tt) in zip(ax,[("faith","NLI score (reward model)"),("f1","F1"),("faith_2nd","NLI score (independent)")]):
  for i,(m,l) in enumerate(cells):
    for j,c in enumerate(CO):
      d=ps[(ps.model==m)&(ps.lang==l)&(ps.cond==c)][col]; x=i+(j-2)*0.15
      a_.scatter([x]*len(d),d,s=7,color=cols[c],alpha=.55,lw=0); a_.hlines(d.mean(),x-.065,x+.065,color=cols[c],lw=1.6)
  a_.set_xticks(range(4)); a_.set_xticklabels(xlab,fontsize=5.5); a_.set_title(tt,fontsize=8); a_.tick_params(labelsize=7); a_.set_ylim(0,None)
h=[plt.Line2D([],[],color=cols[c],lw=2,label=CN[c].replace("$\\lambda{=}0$","λ=0")) for c in CO]
fig.legend(handles=h,loc="lower center",ncol=5,fontsize=6.5,frameon=False,bbox_to_anchor=(.5,-.09))
fig.savefig(f"{OUT}/figures/fig_h_summary.pdf",bbox_inches="tight"); plt.close(fig)
fig,ax=plt.subplots(1,4,figsize=(7.4,2.3),constrained_layout=True,sharey=True)
for a_,(m,l) in zip(ax,cells):
  for c in ["A_frozen","B","C"]:
    xs=[];ys=[];ns=[]
    for i,b in enumerate(BL):
      r=lb[(lb.model==m)&(lb.lang==l)&(lb.cond==c)&(lb.bin==b)]
      if len(r) and r.n.iloc[0]>=30: xs.append(i);ys.append(r.faith.iloc[0])
    a_.plot(xs,ys,marker="o",ms=3,lw=1.2,color=cols[c],label=CN[c])
  a_.set_xticks(range(5)); a_.set_xticklabels(BL,fontsize=7); a_.set_title(f"{M[m].replace('~',' ')}, {L[l]}",fontsize=7); a_.tick_params(labelsize=7); a_.set_xlabel("answer length (words)",fontsize=7)
ax[0].set_ylabel("mean NLI score",fontsize=7); ax[0].legend(fontsize=6,frameon=False)
fig.savefig(f"{OUT}/figures/fig_h_lenbins.pdf",bbox_inches="tight"); plt.close(fig)
print("ok")
