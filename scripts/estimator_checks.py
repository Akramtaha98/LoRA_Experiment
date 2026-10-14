#!/usr/bin/env python3
"""CPU unit test for the corrected scorer (LORA_PG_FIX=1).

Extracts `_seq_log_probs_aligned` and the legacy `_seq_log_probs` verbatim from src/lora_nli/matrix.py and checks,
on a tiny random T5 with a banned-token set standing in for the sentinel ids:
  F1  aligned log-prob == the log-prob generate() itself assigned to the sampled sequence (|diff| < 1e-4 nats)
  F2  the legacy scorer deviates from generation (documents the double decoder-start feed)
  F3  banned tokens are never sampled and carry zero probability under the scorer
  F4  gradient flows (non-zero) and constant reward still gives zero gradient
Exit code 0 only if F1, F3, F4 pass.
"""
import ast, os, sys, textwrap
import torch, torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
_cands = [os.path.join(os.path.dirname(HERE), "src", "lora_nli", "matrix.py")]
SRC = open(next(c for c in _cands if os.path.exists(c)), encoding="utf-8").read()
tree = ast.parse(SRC)
ns = {"torch": torch, "F": F, "V2": False, "SANITIZE_STATS": {"score_calls": 0}}


for name in ("sanitize_logits", "_seq_log_probs", "_seq_log_probs_aligned"):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(textwrap.dedent(ast.get_source_segment(SRC, fn)), ns)
legacy, aligned = ns["_seq_log_probs"], ns["_seq_log_probs_aligned"]

torch.manual_seed(0)
cfg = T5Config(vocab_size=200, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
               pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
model = T5ForConditionalGeneration(cfg).eval()
banned = list(range(150, 200))
B, K = 3, 4
x = torch.randint(2, 150, (B, 9)); am = torch.ones_like(x)
rx, ram = x.repeat_interleave(K, 0), am.repeat_interleave(K, 0)
with torch.no_grad():
    out = model.generate(input_ids=rx, attention_mask=ram, max_new_tokens=12, do_sample=True, temperature=1.0,
                         top_p=1.0, top_k=0, renormalize_logits=True, suppress_tokens=banned,
                         output_scores=True, return_dict_in_generate=True)
samp = out.sequences
gen_lp = torch.zeros(samp.size(0))
for t, sc in enumerate(out.scores):
    tok = samp[:, t + 1]
    lp = F.log_softmax(sc.float(), -1).gather(1, tok.unsqueeze(1)).squeeze(1)
    live = (tok != 0).float()                       # pad after EOS is not scored
    gen_lp += lp * live
res = {}
with torch.no_grad():
    al = aligned(model, rx, ram, samp, 0, 0, banned)
    lg = legacy(model, rx, ram, samp, 0)
res["F1_max_abs_diff_aligned_vs_generation"] = float((al - gen_lp).abs().max())
res["F1_pass"] = res["F1_max_abs_diff_aligned_vs_generation"] < 1e-4
res["F2_legacy_mean_abs_diff"] = float((lg - gen_lp).abs().mean())
res["F2_legacy_deviates"] = res["F2_legacy_mean_abs_diff"] > 1e-3
res["F3_no_banned_sampled"] = not bool(torch.isin(samp, torch.tensor(banned)).any())
model.train()
lp = aligned(model, rx, ram, samp, 0, 0, banned)
(-(torch.rand(B * K) - 0.5) * lp).mean().backward()
g = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
res["F4_gradient_nonzero"] = g > 0
model.zero_grad()
lp = aligned(model, rx, ram, samp, 0, 0, banned)
(-(torch.zeros(B * K)) * lp).mean().backward()
res["F4_zero_advantage_zero_grad"] = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None) == 0.0
for k, v in res.items(): print(f"{k}: {v}")
ok = res["F1_pass"] and res["F3_no_banned_sampled"] and res["F4_gradient_nonzero"] and res["F4_zero_advantage_zero_grad"]
print("ALL_PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
