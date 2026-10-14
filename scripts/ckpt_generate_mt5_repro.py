import torch,transformers,peft
from transformers import MT5ForConditionalGeneration,AutoTokenizer
from peft import LoraConfig,get_peft_model,TaskType
print("transformers",transformers.__version__,"torch",torch.__version__,"peft",peft.__version__,flush=True)
tok=AutoTokenizer.from_pretrained("google/mt5-base")
m=MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
m=get_peft_model(m,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=8,lora_alpha=16,lora_dropout=0.0,target_modules=["q","k","v","o","wi_0","wi_1","wo"]))
m.enable_input_require_grads()
for mod in m.modules():
    if isinstance(mod,torch.nn.Dropout): mod.p=0.0
    if hasattr(mod,"dropout") and isinstance(mod.dropout,float): mod.dropout=0.0
    if hasattr(mod,"dropout_rate") and isinstance(mod.dropout_rate,float): mod.dropout_rate=0.0
# make the adapter non-trivial so the decoder is not the bare pretrained one
torch.manual_seed(1)
for n,p in m.named_parameters():
    if "lora_B" in n: p.data.normal_(0,0.02)
ban=[tok.encode(f"<extra_id_{i}>",add_special_tokens=False)[0] for i in range(100)]
texts=["question: Where is the capital of France?  context: Paris is the capital and largest city of France, located on the Seine.",
       "question: Who wrote Hamlet?  context: Hamlet is a tragedy written by William Shakespeare around 1600."]
enc=tok(texts,return_tensors="pt",padding=True)
def gen(mode,ckpt):
    if ckpt: m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else: m.base_model.model.gradient_checkpointing_disable()
    getattr(m,mode)()
    with torch.no_grad(): return m.generate(**enc,max_new_tokens=12,do_sample=False,renormalize_logits=True,suppress_tokens=ban)
def manual(last_only,steps=12):
    m.eval();seq=torch.zeros(2,1,dtype=torch.long)
    with torch.no_grad():
        for _ in range(steps):
            dec=seq[:,-1:] if last_only else seq
            lg=m(**enc,decoder_input_ids=dec).logits[:,-1].float();lg[:,ban]=-1e9
            seq=torch.cat([seq,lg.argmax(-1,keepdim=True)],1)
    return seq
def cut(r):
    r=r[1:].tolist();o=[]
    for t in r:
        o.append(t)
        if t==1: break
    return o
def eqn(a,b):
    return all(cut(a[i])==cut(b[i]) for i in range(a.shape[0]))
ref=gen("eval",False);tr=gen("train",False);tc=gen("train",True)
full,lo=manual(False),manual(True)
print("dropout=0 everywhere (isolates checkpointing):")
print(" train/no-ckpt == eval            :",eqn(tr,ref))
print(" train/ckpt    == eval            :",eqn(tc,ref))
print(" eval          == full-prefix ref :",eqn(ref,full))
print(" train/ckpt    == last-token-only :",eqn(tc,lo))
print(" decoded eval      :",repr(tok.decode(ref[0],skip_special_tokens=True)[:60]))
print(" decoded train+ckpt:",repr(tok.decode(tc[0],skip_special_tokens=True)[:60]))
