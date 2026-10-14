#!/usr/bin/env python3
"""Minimal CPU reproducer for the checkpointing question (tiny random T5 + PEFT-LoRA, no data, no GPU).

For each (dropout, mode) it runs the SAME composite step twice, gradient checkpointing OFF vs ON, with identical weights
and RNG, and reports: (a) max |grad_on - grad_off| over LoRA params, (b) whether the K sampled sequences are identical.
Expected for a correct implementation: identical (differences ~1e-6). A non-zero result localises a defect; zero on this
CPU toy does NOT clear the GPU/bf16/mT5 path (that needs the factorial in RUNBOOK_V2.md).
"""
import copy, os, sys, torch, torch.nn.functional as F
os.environ.setdefault("LORA_V2", "1"); os.environ.setdefault("LORA_OUTPUT_DIR", "/tmp/ckpt_repro_out")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
from lora_nli import matrix as M
from transformers import T5Config, T5ForConditionalGeneration

def build(dropout):
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=420, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, pad_token_id=0, eos_token_id=1,
                   decoder_start_token_id=0, dropout_rate=dropout, feed_forward_proj="gated-gelu")
    m = M.get_peft_model(T5ForConditionalGeneration(cfg), M.LoraConfig(task_type=M.TaskType.SEQ_2_SEQ_LM, r=4, lora_alpha=8,
                         lora_dropout=dropout, target_modules=["q", "v", "wi_0", "wi_1", "wo"]))
    m.enable_input_require_grads()
    return m

def step(base, ckpt, mode, policy_mode):
    m = copy.deepcopy(base)
    if ckpt: m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    torch.manual_seed(11)
    B, K = 2, 4
    x = torch.randint(3, 280, (B, 9)); am = torch.ones_like(x); lab = torch.randint(3, 280, (B, 4))
    ce = m(input_ids=x, attention_mask=am, labels=lab).loss
    if policy_mode == "eval": m.eval()
    rx, ram = x.repeat_interleave(K, 0), am.repeat_interleave(K, 0)
    torch.manual_seed(5)
    with torch.no_grad():
        s = m.generate(input_ids=rx, attention_mask=ram, max_new_tokens=8, do_sample=True, top_k=0, top_p=1.0,
                       renormalize_logits=True, suppress_tokens=list(range(300, 400)))
    lp = M._seq_log_probs_aligned(m, rx, ram, s, 0, 0, list(range(300, 400)))
    adv = torch.tensor([0.5, -0.5, 0.2, -0.2] * B)
    loss = ce + 0.3 * (-(adv * lp).mean())
    if policy_mode == "eval": m.train()
    loss.backward()
    g = torch.cat([p.grad.flatten() for n, p in m.named_parameters() if p.grad is not None])
    return g, s, float(loss)

# --- (1) generate() under train mode + gradient checkpointing: which decoder prefix does it condition on? -------------------
import transformers
print("transformers", transformers.__version__, "torch", torch.__version__)
def _gen(base, mode, ckpt, seed=5):
    m = copy.deepcopy(base)
    if ckpt: m.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    getattr(m, mode)(); torch.manual_seed(seed)
    x = torch.randint(3, 280, (6, 9), generator=torch.Generator().manual_seed(2)); am = torch.ones_like(x)
    with torch.no_grad():
        s = m.generate(input_ids=x, attention_mask=am, max_new_tokens=8, do_sample=True, top_k=0, top_p=1.0,
                       renormalize_logits=True, suppress_tokens=list(range(300, 400)))
    return m, x, am, s
def _manual(m, x, am, last_only, steps=8, seed=5):
    m.eval(); torch.manual_seed(seed); seq = torch.zeros(x.size(0), 1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(steps):
            dec = seq[:, -1:] if last_only else seq
            lg = m(input_ids=x, attention_mask=am, decoder_input_ids=dec).logits[:, -1].float(); lg[:, 300:400] = -float("inf")
            nx = torch.multinomial(torch.softmax(lg, -1), 1)
            fin = (seq[:, -1] == 1) | ((seq[:, -1] == 0) & (seq.size(1) > 1))
            seq = torch.cat([seq, torch.where(fin[:, None], torch.zeros_like(nx), nx)], 1)
    return seq
def _eq(a, b):
    n = min(a.shape[1], b.shape[1]); return bool((a[:, :n] == b[:, :n]).all())
base0 = build(0.0)
_, _, _, s_eval = _gen(base0, "eval", False)
_, _, _, s_tr = _gen(base0, "train", False)
m_tc, x, am, s_tc = _gen(base0, "train", True)
print("\n[generate path, dropout=0]  (identical = sampled sequences equal)")
print("  train/no-ckpt  vs eval            :", _eq(s_tr, s_eval))
print("  train/ckpt     vs eval            :", _eq(s_tc, s_eval))
print("  train/ckpt     vs last-token-only :", _eq(s_tc, _manual(m_tc, x, am, True)))
print("  eval           vs full-prefix     :", _eq(s_eval, _manual(m_tc, x, am, False)))
print("  => when the answers above are (True, False, True, True): generate() under train-mode checkpointing\n"
      "     decodes from the last token only (the KV cache is discarded but the input is still sliced).\n")
rows = []
for dropout in (0.0, 0.3):
    base = build(dropout)
    for policy_mode in ("train", "eval"):
        g0, s0, l0 = step(base, False, "train", policy_mode)
        g1, s1, l1 = step(base, True, "train", policy_mode)
        rows.append((dropout, policy_mode, float((g0 - g1).abs().max()), bool(s0.shape == s1.shape and (s0 == s1).all()), abs(l0 - l1)))
print("dropout  policy_branch  max|dgrad(on-off)|  samples_identical  |dloss|")
for r in rows: print(f"{r[0]:<8} {r[1]:<14} {r[2]:<19.3g} {str(r[3]):<18} {r[4]:.3g}")
