#!/usr/bin/env python3
"""Generate t5gemma2_experiment_matrix.py from lora_experiment_matrix.py by exact, asserted replacements."""
import re
src = open("lora_experiment_matrix.py", encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:70])
    src = src.replace(old, new)

rep("    MT5ForConditionalGeneration,\n", "    AutoModelForSeq2SeqLM,\n")
rep('BASE_MODEL    = "google/mt5-base"',
    'import os as _os\nBASE_MODEL    = _os.environ.get("T5G_MODEL", "google/t5gemma-2-270m-270m")')
rep('OUTPUT_DIR = Path("./experiment_results")', 'OUTPUT_DIR = Path("./experiment_results_t5gemma2")')
# no <extra_id_N> sentinels in Gemma vocab
rep("def get_sentinel_token_ids(tokenizer) -> list:",
    "def get_sentinel_token_ids(tokenizer):\n    return None  # T5Gemma 2 has no T5 sentinel tokens\n\n\ndef _unused_get_sentinel_token_ids(tokenizer) -> list:")
# target modules: discovered text-only Linear layers
rep('MT5_FFN_MODULES = ["wi_0", "wi_1", "wo"]',
    'T5G_TARGETS = []\n_T5G_NAMES = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")\n'
    'def discover_targets(model):\n'
    '    out = []\n'
    '    for n, m in model.named_modules():\n'
    '        if n.split(".")[-1] in _T5G_NAMES and isinstance(m, torch.nn.Linear):\n'
    '            if any(b in n.lower() for b in ("vision","siglip","image","mm_proj","multi_modal")):\n'
    '                continue\n'
    '            out.append(n)\n'
    '    assert out, "no text Linear targets found - run t5gemma2_smoke.py --inspect"\n'
    '    return out\n')
rep('attn_and_ffn = ["q", "k", "v", "o"] + MT5_FFN_MODULES', 'attn_and_ffn = list(T5G_TARGETS)')
rep("""    model = MT5ForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=quant_config,
    )""", """    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config,
    )
    global T5G_TARGETS
    T5G_TARGETS = discover_targets(model)
    print(f"  [T5Gemma2] {len(T5G_TARGETS)} text-only LoRA target modules")""")
# labels: no BOS in targets, append EOS
rep("""labels = tokenizer(text_target=(gold or ""), truncation=True,
                                    max_length=64)
                model_inputs["labels"] = labels["input_ids"]""",
    """labels = tokenizer(text_target=(gold or ""), truncation=True,
                                    max_length=63, add_special_tokens=False)
                model_inputs["labels"] = labels["input_ids"] + [tokenizer.eos_token_id]""")
# SCST: drop the leading decoder-start token from sampled ids (it is pad for mT5, BOS for Gemma)
rep("        # ---- Step 3: leave-one-out baseline / advantage (detached) ----",
    "        sampled_ids = sampled_ids[:, 1:]  # strip decoder-start token\n\n        # ---- Step 3: leave-one-out baseline / advantage (detached) ----")
rep("data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)", "data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)  # T5Gemma2 shifts labels itself")
rep("        _nli_model.eval()\n    return _nli_tokenizer, _nli_model", "        _nli_model.to(DEVICE).eval()  # NLI scorer on GPU (was CPU: GPU sat idle)\n    return _nli_tokenizer, _nli_model")
rep("""    enc = tok(context, answer, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits""", """    enc = tok(context, answer, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits""")
open("t5gemma2_experiment_matrix.py", "w", encoding="utf-8").write(src)
print("wrote t5gemma2_experiment_matrix.py")
