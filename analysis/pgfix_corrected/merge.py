import json,glob,pandas as pd
rows=[]
import sys
for d in (sys.argv[1:] or ["."]):   # directories that contain runs_pgfix/<stage>/<job>/checkpoint_*.jsonl
    for f in glob.glob(f"{d}/runs_pgfix/*/*/checkpoint_*.jsonl"):
        stage,job=f.split("/")[-3:-1]
        for l in open(f):
            if l.strip():
                r=json.loads(l);r.pop("samples",None);r["stage"],r["job"]=stage,job;rows.append(r)
df=pd.DataFrame(rows).drop_duplicates(["stage","job","config","lora_variant","language","seed"] if False else None)
df.to_csv("all.csv",index=False)
print(len(df));print(df.status.value_counts());print(df.groupby("stage").size())
print(df.columns.tolist())
print(df[["config","lora_variant","lambda1","reward_mode"]].drop_duplicates().to_string())
