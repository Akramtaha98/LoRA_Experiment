import csv,re,os,numpy as np,sys
HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,'per_seed_results.csv'))))
src=open(os.path.join(HERE,'factorial_reanalysis.py')).read()
D=eval(re.search(r'^D = (\{.*?^\})\nLANGS',src,re.S|re.M).group(1))
bbm={'mT5':'mt5','Qwen3':'qwen3'};bad=0;n=0
for bb,d in D.items():
  for lang,dv in d.items():
    for var,dc in dv.items():
      for c,(m,s) in dc.items():
        cfg='B_ce_lora' if c=='B' else 'C_composite_lora'
        x=np.array([float(r['f_faith']) for r in rows if r['backbone']==bbm[bb] and r['variant']==var.lower() and r['language']==lang.lower() and r['config']==cfg])
        n+=1
        if not(len(x)==3 and abs(x.mean()-m)<6e-5 and abs(x.std(ddof=1)-s)<1.2e-4): bad+=1; print('MISMATCH',bb,lang,var,c,round(x.mean(),4),round(x.std(ddof=1),4),m,s)
print('TABLE_REPRODUCED cells',n,'mismatch',bad)
sys.exit(1 if bad else 0)
