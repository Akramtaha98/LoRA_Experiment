#!/usr/bin/env python3
"""Smoke test for T5Gemma 2 (run first). Checks: load, text-only LoRA targets, 4 adapters, forward w/ labels, generate."""
import sys, torch
import t5gemma2_experiment_matrix as M
from transformers import AutoTokenizer
print("model:", M.BASE_MODEL)
tok = AutoTokenizer.from_pretrained(M.BASE_MODEL)
print("tokenizer:", type(tok).__name__, "| bos", tok.bos_token_id, "eos", tok.eos_token_id, "pad", tok.pad_token_id)
prompt = M.build_prompt("ما هي عاصمة ماليزيا؟", "كوالالمبور هي عاصمة ماليزيا وأكبر مدنها.", None)
enc = tok(prompt, return_tensors="pt").to(M.DEVICE)
lab = tok(text_target="كوالالمبور", add_special_tokens=False, return_tensors="pt")["input_ids"]
lab = torch.cat([lab, torch.tensor([[tok.eos_token_id]])], 1).to(M.DEVICE)
ok = True
for v in ["lora", "dora", "qlora", "adalora"]:
    try:
        m = M.load_model_for_config("C_composite_lora", v, total_steps=30)
        names = M.T5G_TARGETS
        if v == "lora": print("example targets:", names[:3], "...", names[-2:], "| total", len(names))
        m.train()
        out = m(**enc, labels=lab)
        out.loss.backward()
        m.eval()
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=12, do_sample=False)
        print(f"[OK] {v}: loss={out.loss.item():.3f} gen_first_ids={g[0,:4].tolist()} text={tok.decode(g[0], skip_special_tokens=True)!r}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        ok = False; print(f"[FAIL] {v}: {type(e).__name__}: {e}")
sys.exit(0 if ok else 1)
