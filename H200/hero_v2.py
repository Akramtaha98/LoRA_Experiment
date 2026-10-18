"""
hero_v2.py -- corrected multi-seed runner for the NLI-faithfulness-reward LoRA study.

Hero backbones : Qwen3-0.6B-Base (causal)  and  T5Gemma 2 270M-270M (seq2seq).
Legacy baseline: mT5-base (already finished; NOT rerun here).

Conditions (per backbone x language x seed)
  A_frozen : no adaptation, greedy decode (deterministic; run once, seed dir "seed42")
  B        : LoRA, cross-entropy only
  C        : LoRA, CE + lambda * self-critical policy gradient, NLI-entailment reward
  C_l0     : identical code path to C (sampling, scoring, re-scoring) but lambda = 0
  C_shuf   : like C, but each prompt receives ANOTHER prompt's reward vector
  B_ckpt   : (optional, --ckpt_control) B trained WITH gradient checkpointing, decoded
             correctly (checkpointing off, eval mode) -- shows checkpointing is a pure
             memory switch once the evaluation state is right.

Corrections relative to the legacy Qwen3 / T5Gemma 2 runners
  1. NLI premise = the RAW passage (legacy: the decoded, possibly truncated prompt).
  2. Sampling and re-scoring use the same distribution: temperature 1, full softmax
     (no top-p / top-k), eval mode (dropout off), KV cache on.  One code path
     (`token_logps`) is used for CE, for re-scoring and for the decode-time checks.
  3. Reward text and log-prob cover exactly the same tokens (generation stops at
     EOS or at the first token that contains a newline; both are included in the
     log-prob, the answer text is cut at the newline).
  4. Held-out decoding is ALWAYS in eval mode with the cache on, gradient
     checkpointing off, and this is asserted (the gradient-checkpointing artifact:
     a model left in train mode at decode time corrupts generations).
  5. Per-example predictions with stable ids are written for every run and seed.
  6. Shared NaN/Inf logit sanitizer for sampling and greedy decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

# --------------------------------------------------------------------------- config
MODELS = {
    "qwen3": dict(hf="Qwen/Qwen3-0.6B-Base", kind="causal"),
    "t5gemma2": dict(hf=os.environ.get("T5G_MODEL", "google/t5gemma-2-270m-270m"), kind="seq2seq"),
}
NLI_REWARD = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_SECOND = "joeddav/xlm-roberta-large-xnli"
SEEDS = [42, 123, 2026, 7, 99]
CONDS = ["A_frozen", "B", "C", "C_l0", "C_shuf"]
TRAIN_SIZE = 470
MAX_PROMPT = int(os.environ.get("HERO_MAX_PROMPT", 512))
MAX_ANSWER = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" or os.environ.get("HERO_BF16")) else torch.float32
LORA_NAMES = os.environ.get("HERO_LORA_NAMES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------------------------------------------------------------- data
def load_qa(lang: str):
    if lang == "arabic":
        return load_dataset("google/xquad", "xquad.ar", split="validation")
    if lang == "malay":
        return load_dataset("facebook/belebele", "zsm_Latn", split="test")
    raise ValueError(lang)


def parse_example(ex: dict, lang: str):
    """-> (passage, question, gold, options|None, gold_option_idx|None)"""
    if lang == "arabic":
        a = ex.get("answers", {})
        gold = a["text"][0] if isinstance(a, dict) and a.get("text") else ""
        return ex["context"], ex["question"], gold, None, None
    n = int(ex.get("correct_answer_num", 1))
    opts = [ex.get(f"mc_answer{i}", "") for i in range(1, 5)]
    return ex["flores_passage"], ex["question"], opts[n - 1], opts, n - 1


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", (s or "").strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(p: str, g: str) -> int:
    return int(_norm(p) == _norm(g))


def token_f1(p: str, g: str) -> float:
    pt, gt = _norm(p).split(), _norm(g).split()
    if not pt or not gt:
        return float(pt == gt)
    gc = {}
    for t in gt:
        gc[t] = gc.get(t, 0) + 1
    ov, pc = 0, {}
    for t in pt:
        pc[t] = pc.get(t, 0) + 1
    for t, c in pc.items():
        ov += min(c, gc.get(t, 0))
    if ov == 0:
        return 0.0
    pr, rc = ov / len(pt), ov / len(gt)
    return 2 * pr * rc / (pr + rc)


# --------------------------------------------------------------------------- NLI scorers
class NLIScorer:
    def __init__(self, name: str):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
        lm = {str(v).lower(): int(k) for k, v in self.m.config.id2label.items()}
        self.ent = lm["entailment"]

    @torch.no_grad()
    def __call__(self, premises, hyps, bs: int = 32):
        out = [0.0] * len(hyps)
        idx = [i for i, h in enumerate(hyps) if h.strip() and premises[i].strip()]
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            enc = self.tok([premises[i] for i in b], [hyps[i] for i in b], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(DEVICE)
            p = torch.softmax(self.m(**enc).logits.float(), -1)[:, self.ent].cpu().tolist()
            for i, v in zip(b, p):
                out[i] = round(v, 4)
        return out


# --------------------------------------------------------------------------- sanitizer
class Sanitize(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)


def set_eval(model):
    model.eval()
    assert not any(m.training for m in model.modules()), "a submodule is still in train mode"


def set_train(model):
    model.train()


# --------------------------------------------------------------------------- backbone
class Backbone:
    def __init__(self, key: str, lang: str):
        self.key, self.lang = key, lang
        self.hf, self.kind = MODELS[key]["hf"], MODELS[key]["kind"]
        self.tok = AutoTokenizer.from_pretrained(self.hf)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos = self.tok.eos_token_id
        self._stop = None

    # ---- model ----
    def load_model(self, adapter: bool, seed: int):
        cls = AutoModelForCausalLM if self.kind == "causal" else AutoModelForSeq2SeqLM
        model = cls.from_pretrained(self.hf, dtype=DTYPE).to(DEVICE)
        model.config.use_cache = True
        if not adapter:
            for p in model.parameters():
                p.requires_grad = False
            return model
        targets = [n for n, m in model.named_modules()
                   if n.split(".")[-1] in LORA_NAMES and isinstance(m, torch.nn.Linear)
                   and not any(b in n.lower() for b in ("vision", "siglip", "image", "mm_proj", "multi_modal"))]
        assert targets, "no LoRA targets found"
        set_seed(seed)  # adapter init depends on the seed
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM if self.kind == "causal" else TaskType.SEQ_2_SEQ_LM,
                         r=8, lora_alpha=16, lora_dropout=0.05, target_modules=targets)
        return get_peft_model(model, cfg)

    # ---- stop tokens: EOS and every token whose text contains a newline ----
    @property
    def stop_ids(self):
        if self._stop is None:
            ids = {self.eos}
            for i in range(len(self.tok)):
                try:
                    if "\n" in self.tok.decode([i]):
                        ids.add(i)
                except Exception:
                    pass
            self._stop = sorted(ids)
        return self._stop

    # ---- prompts ----
    def prompt_ids(self, passage, question, options):
        if self.kind == "causal":
            head, tail = f"Question: {question}\nContext: ", "\n"
            if options:
                tail += "Options: " + " ".join(f"({i+1}) {o}" for i, o in enumerate(options)) + "\n"
            tail += "Answer:"
        else:
            head, tail = f"question: {question}  context: ", ""
            if options:
                tail = "  options: " + " ".join(f"({i+1}) {o}" for i, o in enumerate(options))
        h = self.tok(head, add_special_tokens=True).input_ids
        t = self.tok(tail, add_special_tokens=False).input_ids if tail else []
        p = self.tok(passage, add_special_tokens=False).input_ids
        budget = max(16, MAX_PROMPT - len(h) - len(t))
        return h + p[:budget] + t     # the PASSAGE is truncated, never the options/question

    def target_ids(self, gold):
        pre = " " if self.kind == "causal" else ""
        ids = self.tok(pre + gold, add_special_tokens=False).input_ids[:MAX_ANSWER - 1]
        return ids + [self.eos]

    def _pad(self, seqs, left):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            if left:
                ids[i, L - len(s):] = torch.tensor(s); mask[i, L - len(s):] = 1
            else:
                ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        return ids.to(DEVICE), mask.to(DEVICE)

    def batch_prompts(self, exs):
        return self._pad([e["p_ids"] for e in exs], left=True)

    def batch_targets(self, exs):
        return self._pad([e["t_ids"] for e in exs], left=False)

    # ---- the single differentiable path ----
    def token_logps(self, model, p_ids, p_mask, t_ids):
        """(B,T) log p(t_j | prompt, t_<j).  Same code for CE and for policy re-scoring."""
        T = t_ids.size(1)
        if self.kind == "causal":
            ids = torch.cat([p_ids, t_ids], 1)
            am = torch.cat([p_mask, torch.ones_like(t_ids)], 1)
            pos = (am.cumsum(1) - 1).clamp(min=0)
            out = model(input_ids=ids, attention_mask=am, position_ids=pos,
                        use_cache=False, logits_to_keep=T + 1)
            logits = out.logits[:, :T]          # position P-1 .. P+T-2 predicts t_0 .. t_{T-1}
        else:
            start = self._dec_start(model)
            dec_in = torch.cat([torch.full_like(t_ids[:, :1], start), t_ids[:, :-1]], 1)
            try:
                out = model(input_ids=p_ids, attention_mask=p_mask, decoder_input_ids=dec_in, use_cache=False)
            except (TypeError, ValueError):   # fallback: let the model shift the labels itself
                out = model(input_ids=p_ids, attention_mask=p_mask, labels=t_ids, use_cache=False)
            logits = out.logits
        logits = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        lp = torch.log_softmax(logits, -1)
        return lp.gather(2, t_ids.unsqueeze(-1)).squeeze(-1)

    def _dec_start(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for cfg in (base.generation_config, base.config, getattr(base.config, "decoder", None)):
            v = getattr(cfg, "decoder_start_token_id", None)
            if v is not None:
                return v
        return self.pad

    # ---- generation (eval mode is the caller's responsibility and is asserted) ----
    @torch.no_grad()
    def generate(self, model, p_ids, p_mask, K: int = 1, sample: bool = False):
        assert not any(m.training for m in model.modules()), "generate() called in train mode"
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        assert not getattr(base, "is_gradient_checkpointing", False), "checkpointing must be off at decode"
        if K > 1:
            p_ids, p_mask = p_ids.repeat_interleave(K, 0), p_mask.repeat_interleave(K, 0)
        kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        out = model.generate(input_ids=p_ids, attention_mask=p_mask, max_new_tokens=MAX_ANSWER,
                             eos_token_id=self.stop_ids, pad_token_id=self.pad, use_cache=True,
                             logits_processor=LogitsProcessorList([Sanitize()]),
                             num_beams=1, **kw)
        cont = out[:, p_ids.size(1):] if self.kind == "causal" else out[:, 1:]
        stop = torch.tensor(self.stop_ids, device=cont.device)
        hit = torch.isin(cont, stop)
        first = torch.where(hit.any(1), hit.float().argmax(1), torch.full((cont.size(0),), cont.size(1) - 1,
                                                                          device=cont.device))
        mask = (torch.arange(cont.size(1), device=cont.device)[None] <= first[:, None]).long()
        cont = torch.where(mask.bool(), cont, torch.full_like(cont, self.pad))
        return cont, mask

    def texts(self, cont, mask):
        res = []
        for ids, m in zip(cont.tolist(), mask.tolist()):
            seq = [t for t, k in zip(ids, m) if k]
            res.append(self.tok.decode(seq, skip_special_tokens=True).split("\n")[0].strip())
        return res


# --------------------------------------------------------------------------- evaluation
def evaluate(bb: Backbone, model, eval_exs, scorers, bs: int, tag: dict, out_path: Path):
    set_eval(model)
    rows = []
    for s in range(0, len(eval_exs), bs):
        b = eval_exs[s:s + bs]
        p_ids, p_mask = bb.batch_prompts(b)
        cont, mask = bb.generate(model, p_ids, p_mask)
        for e, pred in zip(b, bb.texts(cont, mask)):
            rows.append(dict(e["meta"], pred=pred))
    prem = [r["passage"] for r in rows]
    hyp = [r["pred"] for r in rows]
    f_main = scorers["reward"](prem, hyp)
    f_2nd = scorers["second"](prem, hyp) if scorers.get("second") else [None] * len(rows)
    for r, a, b2 in zip(rows, f_main, f_2nd):
        r["faith"], r["faith_2nd"] = a, b2
        r["em"] = exact_match(r["pred"], r["gold"])
        r["f1"] = round(token_f1(r["pred"], r["gold"]), 4)
        r["n_words"] = len(r["pred"].split())
        if r["options"]:
            sc = [token_f1(r["pred"], o) for o in r["options"]]
            r["option_correct"] = int(int(np.argmax(sc)) == r["gold_option"] and max(sc) > 0)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**tag, **r}, ensure_ascii=False) + "\n")
    m = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(
        r.get(k) is not None for r in rows) else None
    return dict(n_eval=len(rows), faith=m("faith"), faith_2nd=m("faith_2nd"), em=m("em"), f1=m("f1"),
                option_acc=m("option_correct"), mean_words=m("n_words"),
                empty_rate=float(np.mean([not r["pred"].strip() for r in rows])))


# --------------------------------------------------------------------------- training
def train_run(bb: Backbone, cond: str, seed: int, train_exs, args, scorers):
    use_pg = cond in ("C", "C_l0", "C_shuf")
    lam = 0.0 if cond == "C_l0" else args.lam
    ckpt = cond == "B_ckpt"
    model = bb.load_model(adapter=True, seed=seed)
    base = model.get_base_model()
    if ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (len(train_exs) + args.bs - 1) // args.bs
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    log, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        order = torch.randperm(len(train_exs), generator=gen).tolist()
        for s in range(0, len(order), args.bs):
            if step >= total:
                break
            b = [train_exs[i] for i in order[s:s + args.bs]]
            p_ids, p_mask = bb.batch_prompts(b)
            t_ids, t_mask = bb.batch_targets(b)
            set_train(model)
            lp = bb.token_logps(model, p_ids, p_mask, t_ids)
            ce = -(lp * t_mask).sum() / t_mask.sum()
            loss, pg_val, rew_mean = ce, 0.0, None
            if use_pg:
                set_eval(model)                      # policy branch: dropout off
                B, K = len(b), args.k
                with torch.no_grad():
                    cont, cmask = bb.generate(model, p_ids, p_mask, K=K, sample=True)
                    texts = bb.texts(cont, cmask)
                    prem = [b[i // K]["meta"]["passage"] for i in range(B * K)]
                    r = torch.tensor(scorers["reward"](prem, texts), device=DEVICE).view(B, K)
                    rew_mean = r.mean().item()
                    if cond == "C_shuf" and B > 1:   # each prompt gets another prompt's rewards
                        r = torch.roll(r, shifts=rng.randint(1, B - 1), dims=0)
                    adv = r - (r.sum(1, keepdim=True) - r) / (K - 1)     # leave-one-out
                lp2 = bb.token_logps(model, p_ids.repeat_interleave(K, 0), p_mask.repeat_interleave(K, 0), cont)
                seq_lp = (lp2 * cmask).sum(1).view(B, K)
                pg = -(adv.detach() * seq_lp).mean()
                pg_val = pg.item()
                loss = ce + lam * pg
                set_train(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % args.log_every == 0 or step == total:
                log.append(dict(step=step, ce=round(ce.item(), 4), pg=round(pg_val, 4),
                                reward=None if rew_mean is None else round(rew_mean, 4),
                                sec=round(time.time() - t0, 1)))
                print(f"    [{cond} s{seed}] step {step}/{total} ce={ce.item():.3f} "
                      f"rew={rew_mean} {time.time()-t0:.0f}s", flush=True)
    if ckpt:   # the fix: checkpointing is a memory switch -- turn it OFF before decoding
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
    return model, log, time.time() - t0


# --------------------------------------------------------------------------- main
def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3", "t5gemma2"])
    ap.add_argument("--langs", nargs="+", default=["arabic", "malay"])
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default="results/hero_v2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_control", action="store_true", help="also run B_ckpt")
    ap.add_argument("--no_second", action="store_true", help="skip the second NLI scorer")
    ap.add_argument("--max_steps", type=int, default=0, help="cap training steps (profiling/tests)")
    ap.add_argument("--eval_limit", type=int, default=0, help="evaluate only the first N held-out items (tests)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.bs, args.k, args.train_size, args.eval_bs, args.log_every = 1, 2, 2, 6, 4, 1
        args.no_second = True
    conds = list(args.conds) + (["B_ckpt"] if args.ckpt_control and "B_ckpt" not in args.conds else [])
    scorers = dict(reward=NLIScorer(NLI_REWARD),
                   second=None if args.no_second else NLIScorer(NLI_SECOND))
    for mk in args.models:
        for lang in args.langs:
            ds = load_qa(lang)
            if args.smoke:
                ds = ds.select(range(12))
            exs = []
            for i in range(len(ds)):
                passage, q, gold, opts, gi = parse_example(ds[i], lang)
                exs.append(dict(i=i, passage=passage, q=q, gold=gold, opts=opts, gi=gi))
            tr_raw, ev_raw = exs[:args.train_size], exs[args.train_size:]
            if args.smoke:
                ev_raw = ev_raw[:6]
            train_ctx = {e["passage"] for e in tr_raw}
            bb = Backbone(mk, lang)

            def prep(e):
                return dict(p_ids=bb.prompt_ids(e["passage"], e["q"], e["opts"]),
                            t_ids=bb.target_ids(e["gold"]),
                            meta=dict(ex_id=e["i"], passage=e["passage"], question=e["q"], gold=e["gold"],
                                      options=e["opts"], gold_option=e["gi"],
                                      passage_in_train=e["passage"] in train_ctx))
            if args.eval_limit:
                ev_raw = ev_raw[:args.eval_limit]
            train_exs, eval_exs = [prep(e) for e in tr_raw], [prep(e) for e in ev_raw]
            print(f"\n=== {mk} / {lang}: train={len(train_exs)} eval={len(eval_exs)} "
                  f"overlap_eval={sum(e['meta']['passage_in_train'] for e in eval_exs)}", flush=True)
            # seed-major order: after the first seed every condition already has a result
            for seed, cond in [(sd, c) for sd in args.seeds for c in conds if not (c == "A_frozen" and sd != args.seeds[0])]:
                if True:
                    rd = Path(args.out) / mk / lang / cond / f"seed{seed}"
                    if (rd / "summary.json").exists():
                        print(f"  skip {mk}/{lang}/{cond}/seed{seed} (done)"); continue
                    rd.mkdir(parents=True, exist_ok=True)
                    print(f"  RUN {mk}/{lang}/{cond}/seed{seed}", flush=True)
                    t0 = time.time(); set_seed(seed)
                    if cond == "A_frozen":
                        model, log, ttrain = bb.load_model(adapter=False, seed=seed), [], 0.0
                    else:
                        model, log, ttrain = train_run(bb, cond, seed, train_exs, args, scorers)
                    set_seed(seed)
                    tag = dict(model=mk, lang=lang, cond=cond, seed=seed)
                    res = evaluate(bb, model, eval_exs, scorers, args.eval_bs, tag, rd / "predictions.jsonl")
                    with open(rd / "train_log.jsonl", "w") as f:
                        for r in log:
                            f.write(json.dumps(r) + "\n")
                    summ = dict(**tag, **res, train_seconds=round(ttrain, 1), n_steps=(log[-1]['step'] if log else 0), total_seconds=round(time.time() - t0, 1),
                                config=dict(vars(args), hf_model=bb.hf, nli_reward=NLI_REWARD, nli_second=NLI_SECOND,
                                            dtype=str(DTYPE), torch=torch.__version__, git=git_rev(),
                                            gradient_checkpointing=(cond == "B_ckpt")))
                    (rd / "summary.json.tmp").write_text(json.dumps(summ, indent=2))
                    os.replace(rd / "summary.json.tmp", rd / "summary.json")   # written last = resume marker
                    print(f"  DONE {mk}/{lang}/{cond}/seed{seed}: faith={res['faith']:.3f} f1={res['f1']:.3f} "
                          f"em={res['em']:.3f} words={res['mean_words']:.1f}", flush=True)
                    del model
                    import gc; gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
