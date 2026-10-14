#!/usr/bin/env python3
"""CPU test suite for the LORA_V2 corrections (tiny random T5, no network, no GPU).

  T1  premise      the NLI premise scored in TRAINING is the raw passage (what evaluation scores), never the prompt
                   (question + Malay options). Synthetic wrong-option marker; negative control = legacy path.
  T2  modes        policy branch (sampling + re-scoring) runs in eval mode, CE pass in train mode, held-out generation
                   in eval mode, train mode restored. Negative control = legacy path (generation in train mode).
  T3  estimator    |log-prob generate() assigned - log-prob the scorer returns| on the REAL wrapped model
                   (PEFT-LoRA, lora_dropout>0, model dropout>0) x {eval, train} x {ckpt off, on} x {fp32, bf16}.
  T4  sanitizer    one shared fill value for generation and scoring; counters count NaN / +inf.
  T5  adalora      update_and_allocate is called: total rank falls from init_r toward target_r; without the callback it
                   never changes.
  T6  records      run record carries the checkpointing flag, stable source IDs, passage-overlap flags, truncation and
                   option-accuracy fields; headline means exclude overlapping-passage items.
Usage: python tests/test_v2_training_path.py        (exit 0 only if every assertion passes)
"""
import json, os, sys, tempfile, types
os.environ["LORA_V2"] = "1"
os.environ["LORA_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="v2tests_")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
import torch, torch.nn.functional as F
from datasets import Dataset
from transformers import T5Config, T5ForConditionalGeneration, BatchEncoding
from lora_nli import matrix as M

RES, FAILS = {}, []
def check(name, cond, info=""):
    RES[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info != "" else ""))
    if not cond: FAILS.append(name)

# ----------------------------------------------------------------------------- stub tokenizer / tiny model
class StubTok:
    pad_token_id, eos_token_id, unk_token_id = 0, 1, 2
    def __init__(self): self.v = {}; self.inv = {}
    def _id(self, w):
        if w.startswith("<extra_id_"): return 300 + int(w[10:-1])
        if w not in self.v:
            i = 3 + len(self.v); assert i < 290; self.v[w] = i; self.inv[i] = w
        return self.v[w]
    def encode(self, t, add_special_tokens=True): return [self._id(w) for w in t.split()]
    def __call__(self, text=None, text_target=None, truncation=False, max_length=512, return_tensors=None, **kw):
        ids = self.encode(text if text is not None else text_target) + [1]
        if truncation: ids = ids[:max_length]
        if text_target is not None: return BatchEncoding({"input_ids": ids})
        d = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_tensors == "pt": d = {k: torch.tensor([v]) for k, v in d.items()}
        return BatchEncoding(d)
    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return " ".join(self.inv.get(i, "") for i in ids if i > 2 and i in self.inv)
    def batch_decode(self, seqs, skip_special_tokens=True): return [self.decode(s) for s in seqs]
    def pad(self, feats, **kw): raise NotImplementedError

class StubCollator:
    def __init__(self, *a, **k): pass
    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats); T = max(len(f["labels"]) for f in feats)
        ids = torch.tensor([f["input_ids"] + [0] * (L - len(f["input_ids"])) for f in feats])
        am = torch.tensor([[1] * len(f["input_ids"]) + [0] * (L - len(f["input_ids"])) for f in feats])
        lab = torch.tensor([f["labels"] + [-100] * (T - len(f["labels"])) for f in feats])
        return BatchEncoding({"input_ids": ids, "attention_mask": am, "labels": lab})

def tiny_cfg(dropout=0.3):
    return T5Config(vocab_size=420, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, pad_token_id=0,
                    eos_token_id=1, decoder_start_token_id=0, dropout_rate=dropout, feed_forward_proj="gated-gelu")

def tiny_loader(variant_override=None):
    def load(config_name, lora_variant=None, total_steps=30):
        torch.manual_seed(0)
        m = T5ForConditionalGeneration(tiny_cfg())
        if config_name == "A_frozen":
            m.eval(); return m
        m = M.get_peft_model(M.build_peft_config(lora_variant, total_steps=total_steps), ) if False else \
            M.get_peft_model(m, M.build_peft_config(lora_variant, total_steps=total_steps))
        m.enable_input_require_grads()
        return m
    return load

PASSAGES = [f"passage{i} alpha{i} beta{i} gamma{i} delta{i}" for i in range(10)]
WRONG = "WRONGOPTIONXYZ"
def make_ds(n=14):
    rows = []
    for i in range(n):
        k = i % 6 if i < 8 else (i - 2 if i < 12 else 0)       # eval items 12,13 reuse passage0 (in train) -> overlap
        p = PASSAGES[k]
        rows.append({"flores_passage": p, "question": f"q{i} what", "mc_answer1": f"alpha{k}",
                     "mc_answer2": WRONG, "mc_answer3": f"beta{k}", "mc_answer4": f"gamma{k}",
                     "correct_answer_num": "1", "link": f"https://x/{i // 3}", "question_number": i % 3 + 1})
    return Dataset.from_list(rows)

class Probe:
    """Wraps f_faith and records (answer, premise)."""
    def __init__(self): self.calls = []
    def __call__(self, answer, ctx):
        self.calls.append((answer, ctx)); return round(0.1 + 0.8 * (len(set(answer.split()) & set(ctx.split())) / 5), 4)

def setup_e2e(v2=True, ckpt_override=None, probe=None, modes=None):
    tok = StubTok()
    M.V2 = v2; M.PG_FIX = True; M.POLICY_MODE = "eval" if v2 else "train"
    M.TRAIN_SIZE, M.TRAIN_EPOCHS, M.TRAIN_BATCH_SIZE, M.PG_NUM_SAMPLES = 8, 2, 2, 2
    M.DEVICE = "cpu"
    M.GRAD_CKPT_OVERRIDE = ckpt_override
    M.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tok)
    M.DataCollatorForSeq2Seq = StubCollator
    M.load_qa_data = lambda language, smoke_test=False: make_ds()
    M.get_sentinel_token_ids = lambda t: list(range(300, 400))
    M._SENTINEL_TOKEN_IDS = None
    M.f_faith = probe or Probe()
    base_load = tiny_loader()
    def load(*a, **k):
        m = base_load(*a, **k)
        if modes is not None:
            orig_gen = m.generate
            def gen(*ga, **gk):
                modes.append(("generate_sample" if gk.get("do_sample") else "generate_greedy", m.training)); return orig_gen(*ga, **gk)
            m.generate = gen
            def pre(mod, args, kwargs):
                if kwargs.get("labels") is not None: modes.append(("ce_forward", m.training))
                elif kwargs.get("decoder_input_ids") is not None: modes.append(("rescore_forward", m.training))
            m.register_forward_pre_hook(pre, with_kwargs=True)
        return m
    M.load_model_for_config = load
    return tok

# ----------------------------------------------------------------------------- T1 + T2 + T6 (end-to-end, V2)
probe, modes = Probe(), []
setup_e2e(True, "off", probe, modes)
r = M.run_single_experiment("C_composite_lora", "lora", "malay", seed=1)
check("T0_run_passes", r["status"] == "PASS", r.get("error", ""))
prem = {c for _, c in probe.calls if c in PASSAGES or True}
train_calls = probe.calls[: -r["n_eval_all"]] if r["n_eval_all"] else probe.calls
check("T1_training_premise_is_raw_passage", all(c in PASSAGES for _, c in train_calls) and len(train_calls) > 0, len(train_calls))
check("T1_no_prompt_or_wrong_option_in_premise",
      all(("options:" not in c and "question:" not in c and WRONG not in c) for _, c in train_calls))
check("T1_premise_matches_eval_premise", all(c in PASSAGES for _, c in probe.calls))
# negative control: legacy path scores the decoded prompt
probe_l = Probe(); setup_e2e(False, "off", probe_l)
rl = M.run_single_experiment("C_composite_lora", "lora", "malay", seed=1)
leg_train = probe_l.calls[: len(probe_l.calls) - rl["n_eval"]]
check("T1_negative_control_legacy_scores_prompt",
      any(("options:" in c and WRONG in c) for _, c in leg_train), "legacy premise contains question+options")

pol = [m for m in modes if m[0] in ("generate_sample", "rescore_forward")]
check("T2_policy_branch_in_eval_mode", len(pol) > 0 and all(not t for _, t in pol), len(pol))
check("T2_ce_pass_in_train_mode", all(t for k, t in modes if k == "ce_forward") and any(k == "ce_forward" for k, _ in modes))
_gr = [t for k, t in modes if k == "generate_greedy"]
check("T2_heldout_generation_in_eval_mode", len(_gr) >= r["n_eval_all"] and all(not t for t in _gr[: r["n_eval_all"]]),
      "headline eval calls only; later calls are the labelled train-mode diagnostic")
check("T2_diagnostic_is_train_mode", all(t for t in _gr[r["n_eval_all"]:]) and len(_gr) > r["n_eval_all"])
check("T2_record_flag", r.get("eval_mode_asserted") is True and r["policy_mode"] == "eval")
modes_l = []; setup_e2e(False, "off", None, modes_l)
M.run_single_experiment("C_composite_lora", "lora", "malay", seed=1)
check("T2_negative_control_legacy_eval_in_train_mode",
      any(t for k, t in modes_l if k == "generate_greedy"), "legacy: held-out generation ran with dropout ON")

# T6 records (reuse r)
check("T6_ckpt_flag_recorded_off", r["grad_ckpt"] is False and r["grad_ckpt_override"] == "off")
setup_e2e(True, "on"); r_on = M.run_single_experiment("C_composite_lora", "lora", "malay", seed=1)
check("T6_ckpt_flag_recorded_on", r_on["status"] == "PASS" and r_on["grad_ckpt"] is True, r_on.get("error", ""))
pf = os.path.join(os.environ["LORA_OUTPUT_DIR"], r["preds_file"])
recs = [json.loads(l) for l in open(pf, encoding="utf-8")]
check("T6_preds_have_stable_ids", all(x["src_id"].startswith("belebele.zsm:https://x/") and "passage_id" in x for x in recs), recs[0]["src_id"])
check("T6_overlap_flagged_and_excluded", r["n_passage_overlap_excluded"] >= 1 and r["n_eval"] == r["n_eval_all"] - r["n_passage_overlap_excluded"],
      f'excl={r["n_passage_overlap_excluded"]}')
check("T6_option_and_truncation_fields", r["option_acc"] is not None and r["frac_eval_prompt_truncated"] is not None and "in_context_rate" in r)

check("T7_v2_ckpt_on_equals_off_headline", abs(r["mean_f_faith"] - r_on["mean_f_faith"]) < 0.05 and abs(r["mean_em"] - r_on["mean_em"]) < 0.26,
      f'off={r["mean_f_faith"]} on={r_on["mean_f_faith"]}')
print("   (legacy-state eval of the same adapters) off:", r.get("trainmode_eval_f_faith"), " on:", r_on.get("trainmode_eval_f_faith"))
check("T7_trainmode_diagnostic_recorded", "trainmode_eval_f_faith" in r and "trainmode_eval_f_faith" in r_on)

# ----------------------------------------------------------------------------- T3 estimator under production conditions
def estimator_diff(mode, ckpt, dtype, peft):
    torch.manual_seed(1)
    cfg = tiny_cfg(0.3)
    m = T5ForConditionalGeneration(cfg)
    if peft:
        m = M.get_peft_model(m, M.LoraConfig(task_type=M.TaskType.SEQ_2_SEQ_LM, r=4, lora_alpha=8, lora_dropout=0.3,
                                             target_modules=["q", "v", "wi_0", "wi_1", "wo"]))
        m.enable_input_require_grads()
    if ckpt:
        (m.base_model.model if peft else m).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if dtype == torch.bfloat16: m = m.to(dtype)
    getattr(m, mode)()
    banned = list(range(300, 400)); B, K = 3, 4
    x = torch.randint(3, 280, (B, 9)); am = torch.ones_like(x)
    rx, ram = x.repeat_interleave(K, 0), am.repeat_interleave(K, 0)
    torch.manual_seed(5)
    with torch.no_grad():
        out = m.generate(input_ids=rx, attention_mask=ram, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         renormalize_logits=True, suppress_tokens=banned,
                         logits_processor=M.LogitsProcessorList([M._NaNSafeLogitsProcessor()]),
                         output_scores=True, return_dict_in_generate=True)
    samp = out.sequences
    gen_lp = torch.zeros(samp.size(0))
    for t, sc in enumerate(out.scores):
        tok = samp[:, t + 1]
        lp = F.log_softmax(sc.float(), -1).gather(1, tok.unsqueeze(1)).squeeze(1)
        gen_lp += lp * (tok != 0).float()
    with torch.no_grad() if mode == "eval" else torch.enable_grad():
        sc_lp = M._seq_log_probs_aligned(m, rx, ram, samp, 0, 0, banned).detach()
    return float((sc_lp - gen_lp).abs().max())

rows = []
for peft in (False, True):
    for dtype in (torch.float32, torch.bfloat16):
        for ckpt in (False, True):
            for mode in ("eval", "train"):
                try: d = estimator_diff(mode, ckpt, dtype, peft)
                except Exception as e: d = float("nan"); print("   err", mode, ckpt, dtype, peft, type(e).__name__, str(e)[:80])
                rows.append((peft, str(dtype).split(".")[-1], ckpt, mode, d))
print("\n  wrapped_peft  dtype     ckpt   mode   max|logp_generate - logp_scorer| (nats)")
for r_ in rows: print(f"  {str(r_[0]):12s}  {r_[1]:8s}  {str(r_[2]):5s}  {r_[3]:5s}  {r_[4]:.3g}")
ev32 = [d for p, dt, c, m_, d in rows if m_ == "eval" and dt == "float32"]
evbf = [d for p, dt, c, m_, d in rows if m_ == "eval" and dt == "bfloat16"]
tr32 = [d for p, dt, c, m_, d in rows if m_ == "train" and dt == "float32"]
check("T3_eval_mode_estimator_matches_fp32", len(ev32) == 4 and max(ev32) < 1e-4, f"max={max(ev32):.2g}")
check("T3_eval_mode_estimator_matches_bf16", len(evbf) == 4 and max(evbf) < 0.1, f"max={max(evbf):.2g}")
check("T3_train_mode_dropout_breaks_match (why the policy branch is eval)", min(tr32) > 1e-2, f"min={min(tr32):.2g}")
RES["T3_table"] = rows

# ----------------------------------------------------------------------------- T4 sanitizer
for k in M.SANITIZE_STATS: M.SANITIZE_STATS[k] = 0
x = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0]])
a, fa = M.sanitize_logits(x.clone(), "gen", 1e4); b, fb = M.sanitize_logits(x.clone(), "score", 1e9)
check("T4_same_fill_gen_and_score", fa == fb == 1e4 and torch.equal(a, b))
check("T4_counters", int(M.SANITIZE_STATS["gen_nan"]) == 1 and int(M.SANITIZE_STATS["score_posinf"]) == 1)

# ----------------------------------------------------------------------------- T5 AdaLoRA
setup_e2e(True, "off"); M.TRAIN_EPOCHS = 12
ra = M.run_single_experiment("B_ce_lora", "adalora", "malay", seed=1)
log = ra.get("adalora_rank_log") or []
check("T5_adalora_ran", ra["status"] == "PASS" and len(log) > 1, ra.get("error", ""))
if log: check("T5_adalora_rank_changes", log[-1][1] < log[0][1], f"{log[0]} -> {log[-1]}")
orig = M.AdaLoraAllocateCallback.on_optimizer_step
M.AdaLoraAllocateCallback.on_optimizer_step = lambda self, args, state, control, model=None, **kw: self.rank_log.append([int(state.global_step), self.total_rank(model)])
setup_e2e(True, "off"); M.TRAIN_EPOCHS = 12
rb = M.run_single_experiment("B_ce_lora", "adalora", "malay", seed=1)
lb = rb.get("adalora_rank_log") or []
check("T5_negative_control_without_allocate_rank_constant", len(lb) > 1 and lb[-1][1] == lb[0][1], f"{lb[:1]}..{lb[-1:]}")
M.AdaLoraAllocateCallback.on_optimizer_step = orig

print("\nALL_PASS" if not FAILS else f"\nFAILED: {FAILS}")
json.dump({k: v for k, v in RES.items() if k != "T3_table"}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_v2_results.json"), "w"), indent=1)
sys.exit(1 if FAILS else 0)
