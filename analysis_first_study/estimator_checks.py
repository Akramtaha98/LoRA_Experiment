#!/usr/bin/env python3
"""CPU checks of the corrected composite-loss (REINFORCE with leave-one-out baseline) implementation.

The function `_seq_log_probs` is extracted *verbatim* from lora_experiment_matrix.py (via `ast`), and the
advantage computation mirrors the three lines of the training step. A tiny randomly initialised T5 is used,
so no download or GPU is needed. Checks:
  C1  constant reward  -> all leave-one-out advantages are exactly 0 and the policy-gradient term has zero gradient
  C2  leave-one-out advantages sum to zero within each group of K samples
  C3  masking/EOS: the sequence log-probability counts every non-pad token including EOS and nothing after it
  C4  consistency with generation: the teacher-forced log-probability of a sampled sequence equals the log-probability
      that generation conditioned on (decoder input = start token + previous tokens). REPORTED, may differ.
  C5  gradient direction: a small descent step on -(A * logp) raises logp for A>0 and lowers it for A<0
Usage: python analysis/estimator_checks.py
"""
import ast, os, sys, textwrap
import torch, torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(os.path.dirname(HERE), "lora_experiment_matrix.py"), encoding="utf-8").read()
fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "_seq_log_probs")
ns = {"torch": torch, "F": F}
exec(textwrap.dedent(ast.get_source_segment(SRC, fn)), ns)
seq_log_probs = ns["_seq_log_probs"]

def loo_advantages(rewards_t):                       # mirrors the training step (K > 1 branch)
    K = rewards_t.shape[1]
    baseline = (rewards_t.sum(dim=1, keepdim=True) - rewards_t) / (K - 1)
    return (rewards_t - baseline).view(-1).detach()

torch.manual_seed(0)
cfg = T5Config(vocab_size=200, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
               pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
model = T5ForConditionalGeneration(cfg).eval()      # eval(): no dropout, as in the unit test of Appendix A
B, K = 3, 4
x = torch.randint(2, 200, (B, 9)); am = torch.ones_like(x)
rx, ram = x.repeat_interleave(K, 0), am.repeat_interleave(K, 0)
with torch.no_grad():
    samp = model.generate(input_ids=rx, attention_mask=ram, max_new_tokens=12, do_sample=True, top_p=0.95, temperature=1.0)
res = {}

# C1 / C2
const = torch.full((B, K), 0.37)
res["C1_constant_reward_zero_advantage"] = bool(torch.all(loo_advantages(const) == 0))
lp = seq_log_probs(model, rx, ram, samp, 0)
(-(loo_advantages(const) * lp).mean()).backward()
gmax = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
res["C1_constant_reward_zero_gradient"] = gmax == 0.0
model.zero_grad()
r = torch.rand(B, K)
res["C2_advantages_sum_to_zero"] = bool(torch.allclose(loo_advantages(r).view(B, K).sum(1), torch.zeros(B), atol=1e-6))

# C3 masking / EOS
mask = (samp != 0)
with torch.no_grad():
    lp_all = seq_log_probs(model, rx, ram, samp, 0)
    logits = model(input_ids=rx, attention_mask=ram, labels=samp).logits
    tok = F.log_softmax(logits, -1).gather(2, samp.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    manual = (tok * mask.float()).sum(1)
res["C3_masked_sum_matches_manual"] = bool(torch.allclose(lp_all, manual, atol=1e-5))
has_eos = (samp == 1).any(1)
res["C3_eos_counted_in_mask"] = bool(all(mask[i][samp[i] == 1].all() for i in range(len(samp)) if has_eos[i]))
res["C3_nothing_after_eos_counted"] = True
for i in range(len(samp)):
    if has_eos[i]:
        pos = (samp[i] == 1).nonzero()[0].item()
        res["C3_nothing_after_eos_counted"] &= bool((samp[i, pos + 1:] == 0).all())

# C4 consistency with generation conditioning
with torch.no_grad():
    dec_in = samp[:, :-1]                              # [start, y1, ..., y_{n-1}] -- what generation conditioned on
    tgt = samp[:, 1:]                                  # [y1, ..., y_n]
    lg = model(input_ids=rx, attention_mask=ram, decoder_input_ids=dec_in).logits
    lpg = F.log_softmax(lg, -1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    gen_lp = (lpg * (tgt != 0).float()).sum(1)
diff = (lp_all - gen_lp).abs()
res["C4_max_abs_diff_teacher_forced_vs_generation_conditioning"] = float(diff.max())
res["C4_mean_abs_diff"] = float(diff.mean())
res["C4_mean_abs_logp"] = float(gen_lp.abs().mean())

# C5 gradient direction (full-parameter toy step; the adapter case is the same chain rule)
import copy
for sign, name in ((+1.0, "C5_positive_advantage_raises_logp"), (-1.0, "C5_negative_advantage_lowers_logp")):
    m = copy.deepcopy(model)
    l0 = seq_log_probs(m, rx[:1], ram[:1], samp[:1], 0)
    (-(sign * l0).mean()).backward()
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None: p -= 1e-2 * p.grad
        l1 = seq_log_probs(m, rx[:1], ram[:1], samp[:1], 0)
    res[name] = bool((l1 - l0).item() * sign > 0)

for k, v in res.items():
    print(f"{k:62s} {v}")
ok = all(v for k, v in res.items() if isinstance(v, bool))
print("ESTIMATOR_CHECKS_BOOLEAN_ALL_PASS" if ok else "ESTIMATOR_CHECKS_SOME_FAIL")
