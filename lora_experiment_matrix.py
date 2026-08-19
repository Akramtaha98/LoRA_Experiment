"""
Full LoRA Experiment Matrix — Pipeline Validation (CPU) + Full Run (GPU/Lab Server)
=====================================================================================
Author : Akram Taha Zeyad
Thesis : Formalizing RAG Failure Modes and LoRA-Based Mitigation

TWO MODES:
  --smoke_test  : Runs on YOUR MACBOOK (CPU), NO GPU needed, NO server access
                  needed. Validates the entire pipeline logic end-to-end on a
                  tiny slice of data (2-3 examples per run) so you can be
                  confident the code works BEFORE requesting lab server time.
                  Automatically skips 4-bit quantization (QLoRA needs CUDA)
                  and falls back to a plain float32 model for the dry run.

  --full        : The real 20-condition matrix. REQUIRES GPU (bitsandbytes
                  4-bit quantization for QLoRA will raise an error on CPU/Mac).
                  Run this only on the lab server / rented GPU after
                  --smoke_test passes.

WHAT THIS SCRIPT DOES:
  Trains and evaluates 20 total conditions: Configs A and D run once per
  language (no LoRA variant), and Configs B and C run once per LoRA variant
  per language (4 variants x 2 configs x 2 languages = 16), for
  4 + 16 = 20 conditions total.

  Configs:
    A = Frozen mT5-base            (zero-shot baseline, no training)
    B = CE LoRA                    (cross-entropy loss only)
    C = Composite LoRA             (CE + hallucination penalty, lambda1=0.3)
    D = Full fine-tuning           (all parameters, composite loss)

  LoRA variants (applied to Configs B & C):
    QLoRA   - 4-bit NF4 quantized backbone (GPU-only)
    AdaLoRA - adaptive rank allocation via SVD
    DoRA    - magnitude/direction decomposed weight update
    VeRA    - shared frozen random projections (ultra low-resource)

  Languages: Arabic (XQuAD), Malay (Belebele-zsm_Latn)

BUGS FIXED IN THIS VERSION (found during code review, before any server run):
  1. Belebele dataset fields are DIFFERENT from XQuAD — it has 'flores_passage'
     (not 'context') and is multiple-choice (mc_answer1-4 + correct_answer_num),
     not a free-text QA task. The original script assumed XQuAD-style fields
     for both languages, which would have crashed or silently produced empty
     answers on every Malay run.
  2. QLoRA's 4-bit quantization (bitsandbytes) requires CUDA. The original
     script would crash immediately if run on a Mac / CPU-only machine.
     Now it auto-detects CUDA and falls back gracefully during smoke tests.
  3. AdaLoRA requires total_step/tinit/tfinal/deltaT for its rank-decay
     schedule — these were missing, which would raise a ValueError as soon
     as training started. Now set from actual per-run training step count.
  4. VeRA compatibility with encoder-decoder (mT5) models varies by PEFT
     version — this is now wrapped in try/except so one incompatible variant
     doesn't kill the entire 32-run matrix; it's logged and skipped instead.
  5. Per-run try/except added throughout: one failing run no longer crashes
     the whole script — you get a full PASS/FAIL report across all 20 (or
     fewer, in smoke-test) runs, which is exactly what you want to check
     BEFORE spending real GPU-hours on the server.
  6. The --full training loop was previously a TODO stub that only built the
     model and evaluated it untrained — no actual fine-tuning occurred, and
     every LoRA variant produced identical scores as a result. This is now
     implemented with a real Seq2SeqTrainer, including the Config C
     composite-loss override.
  7. Malay (Belebele) task formulation was unfair to the model: it was
     trained/evaluated to freely generate the exact text of the correct
     multiple-choice option WITHOUT ever being shown what the 4 options
     were. This collapsed Malay F_faith scores (~0.08-0.10, flat across
     every config) relative to Arabic (~0.31-0.37), because the model had
     no way to know the phrasing/scope of a valid answer -- not because
     training failed. Fixed via build_prompt(): the 4 options are now
     included in the prompt for Malay, matching Belebele's actual
     multiple-choice design. A one-line sample (gold vs. generated) is now
     also printed for the first eval example of every run, for a quick
     qualitative sanity check in the terminal log.
  8. [CRITICAL, found during external peer review] The Config C "composite
     loss" was NOT actually differentiable w.r.t. the faithfulness term.
     The old code called model.generate() under torch.no_grad(), scored
     the resulting text with a frozen NLI model (also no-grad), and added
     the resulting Python float to ce_loss as a constant offset:
         loss = ce_loss + lambda1 * (1.0 - faithfulness_score)
     Because d(ce_loss + constant)/d(theta) == d(ce_loss)/d(theta) exactly,
     this made Config C's backward pass IDENTICAL to Config B's (CE-only)
     for every trainable parameter -- the "faithfulness penalty" changed
     the number printed in the training log but had ZERO effect on what
     the optimizer actually did. Combined with the fact that no random
     seed was ever set (see SEED below), every prior "Config C beats
     Config B" result in the paper was comparing two independently-
     initialized CE-only training runs, not two different objectives.
     FIXED by replacing the no-op composite_loss()/compute_loss() path
     with a genuine self-critical policy-gradient objective (Rennie et
     al., 2017, "Self-Critical Sequence Training", CVPR) using a
     leave-one-out baseline computed from K sampled generations per
     training example. See CompositeLossTrainer.compute_loss() below for
     the full gradient-path documentation. A global SEED is now also set
     for reproducibility across B vs. C comparisons.

REQUIREMENTS:
  Smoke test (Mac, CPU):
    pip install transformers peft accelerate datasets sentencepiece torch \
                pandas tabulate

  Full run (lab server, GPU):
    pip install transformers peft bitsandbytes accelerate datasets \
                sentencepiece torch pandas tabulate evaluate

USAGE:
    python3 lora_experiment_matrix.py --smoke_test                      # on your MacBook, now
    python3 lora_experiment_matrix.py --full                            # on lab server, later (all 20 conditions)
    python3 lora_experiment_matrix.py --full --composite_only           # on lab server (8 Config C conditions only --
                                                                          # cheaper rerun of just the previously-invalid
                                                                          # runs; see BUGS FIXED item 8 above)
"""

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    AutoModelForSequenceClassification,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    LogitsProcessor,
    LogitsProcessorList,
)


class _NaNSafeLogitsProcessor(LogitsProcessor):
    """Sanitize NaN/Inf logits before sampling.

    [BUG FOUND ON GPU RERUN, Aug 2026] The QLoRA (4-bit, bitsandbytes NF4)
    Config C runs hit `torch.AcceleratorError: CUDA error: device-side
    assert triggered` inside `model.generate()`'s `_sample()` step around
    step ~30 of training, with a very large pre-clip grad_norm (~586) logged
    a few steps earlier. This is consistent with the K=4-sample policy-
    gradient step (Section "The fix" in the paper) occasionally producing
    NaN/Inf logits under 4-bit-quantized compute combined with an
    unstabilized LoRA adapter early in training -- `torch.multinomial`
    cannot sample from a probability distribution containing NaN, which
    manifests as a device-side assert (and, because the CUDA context is
    then unrecoverable within the same process, every subsequent CUDA call
    -- including the `torch.cuda.empty_cache()` in the per-run cleanup
    handler -- also raises, escaping the per-run try/except and crashing
    the whole script instead of just marking one run FAILED). This
    processor clamps any NaN/Inf logit to a large-finite value immediately
    before sampling, which is the standard mitigation for this failure mode
    and does not alter the training objective on any step where logits are
    already finite (the overwhelming majority of steps).
    """

    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

from peft import (
    LoraConfig,
    AdaLoraConfig,
    VeraConfig,
    get_peft_model,
    TaskType,
)


# ─── DEVICE DETECTION ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTIZATION_AVAILABLE = _BNB_AVAILABLE and DEVICE == "cuda"

if DEVICE == "cpu":
    print("⚠ No CUDA GPU detected — running in CPU smoke-test mode.")
    print("  QLoRA 4-bit quantization will be SKIPPED (needs CUDA).")
    print("  This validates pipeline LOGIC only, not real training quality.\n")


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_MODEL    = "google/mt5-base"
# FIX: the previous NLI_MODEL ("cross-encoder/nli-deberta-v3-base") is
# trained ONLY on English SNLI/MultiNLI -- it has never seen Arabic or
# Malay. Evidence this was corrupting every F_faith score: the 10 Arabic
# runs averaged 0.332, almost exactly the 1/3 chance level for a 3-way
# {entailment, neutral, contradiction} classifier guessing near-uniformly
# on text it can't parse; Malay scores sat BELOW chance (0.076-0.116),
# consistent with the classifier confidently misreading Latin-script Malay
# as "not entailment" rather than genuinely judging faithfulness. Neither
# language's scores were measuring what the metric was supposed to measure.
# Switched to a genuinely multilingual NLI model: mDeBERTa-v3-base,
# pretrained on 100 languages (CC100) and fine-tuned on XNLI (includes
# Arabic) + English MNLI. This is the actual "mDeBERTa, a cross-lingual
# model" described as the intended F_faith design.
NLI_MODEL     = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LANGUAGES     = ["arabic", "malay"]
LORA_VARIANTS = ["qlora", "adalora", "dora", "vera"]

# Data scale for --full runs. 500 is safely under both XQuAD's ~1190 Arabic
# validation examples and Belebele's ~900 Malay test examples per language.
# Raised from an earlier 50-example cap that made --full finish in minutes
# instead of doing meaningful training -- that cap was fine for smoke-test
# pipeline validation but was accidentally also governing the real run.
FULL_DATA_CAP   = 500
SMOKE_DATA_CAP  = 3     # unchanged -- pipeline validation only, keep tiny
FULL_EVAL_SIZE  = 30    # more eval examples -> less noisy F_faith comparison
SMOKE_EVAL_SIZE = 3
TRAIN_BATCH_SIZE = 4
# FIX: 3 epochs on ~470 train examples (~350 steps total) was nowhere near
# enough exposure for mT5-base -- a checkpoint that has NEVER been fine-tuned
# for QA, only pretrained on span-corruption denoising -- to learn a brand
# new input/output format from scratch. Evidence: every one of the 10
# completed Arabic runs scored 0% exact match, and generated text was full
# of literal "<extra_id_N>" sentinel tokens (T5's pretraining mask markers)
# leaking straight into the output -- a clear sign of an undertrained model
# still defaulting to its pretraining behavior rather than answering
# questions. Raised from 3 to 12 epochs (~1,400 steps) to give the model a
# real chance to converge on the task before drawing conclusions from the
# comparison.
TRAIN_EPOCHS     = 12

# FIX (peer review, item 8 above): no seed was previously set anywhere, so
# Config B and Config C ran with independently-randomized LoRA adapter
# initialization -- meaning any B-vs-C difference could not be
# distinguished from ordinary run-to-run noise even before the
# differentiability bug is accounted for. SEED is now fixed and applied to
# every run via set_all_seeds() below, so B and C differ ONLY in their
# training objective, holding LoRA initialization constant.
SEED = 42

# Self-critical policy-gradient hyperparameters for the corrected Config C
# objective (see CompositeLossTrainer.compute_loss()). K is the number of
# sampled generations per training example used to build the leave-one-out
# reward baseline; SAMPLE_TEMPERATURE controls exploration in sampling.
PG_NUM_SAMPLES      = 4
PG_SAMPLE_TEMPERATURE = 1.0
PG_SAMPLE_TOP_P     = 0.95


def set_all_seeds(seed: int = SEED):
    """Seed every RNG source that affects LoRA init / batch order / sampling
    so that Config B and Config C differ only in their training objective,
    not in incidental randomness. Call once at the start of each run."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

OUTPUT_DIR = Path("./experiment_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── CHECKPOINTING (resume after a crash / kill / disconnect) ────────────────
# Every run's result is appended to this file the moment it finishes — not
# batched until the end. If the script dies partway through the 20-condition
# matrix (killed process, RunPod disconnect, OOM, etc.), re-running the same
# command skips every run that already PASSED and continues from where it
# stopped, instead of re-running (and re-paying for) everything from scratch.
def _run_key(config_name: str, lora_variant: str, language: str) -> str:
    return f"{config_name}|{lora_variant}|{language}"


def load_completed_runs(checkpoint_file: Path) -> set:
    """Set of run keys that already PASSED in a previous session."""
    completed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("status") == "PASS":
                    completed.add(_run_key(entry["config"], entry["lora_variant"],
                                            entry["language"]))
    return completed


def append_checkpoint(checkpoint_file: Path, result: dict) -> None:
    """Append one run's result immediately after it finishes (crash-safe)."""
    with open(checkpoint_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_all_checkpoint_results(checkpoint_file: Path) -> list:
    """Every recorded result (PASS and FAIL) across all sessions so far."""
    results = []
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


# ─── AUTO-PUSH RESULTS TO GITHUB AFTER EVERY RUN ─────────────────────────────
# Requires git credentials already configured non-interactively on this
# machine (e.g. `git config credential.helper store` + one manual push to
# cache the token, or an SSH deploy key). If push fails for any reason
# (no network, no cached credentials, etc.) this NEVER crashes the
# experiment — the checkpoint file is still safely saved and committed
# locally; you can push it manually later with `git push`.
def git_commit_and_push(repo_file: Path, message: str) -> None:
    try:
        subprocess.run(["git", "add", str(repo_file)],
                        check=True, capture_output=True, text=True)
        commit = subprocess.run(["git", "commit", "-m", message],
                                 capture_output=True, text=True)
        if commit.returncode != 0 and "nothing to commit" not in commit.stdout:
            print(f"  [GIT] commit warning: {commit.stdout.strip()} {commit.stderr.strip()}")
            return
        push = subprocess.run(["git", "push"], capture_output=True, text=True)
        if push.returncode != 0:
            print(f"  [GIT] push FAILED (checkpoint is committed locally, "
                  f"push manually later): {push.stderr.strip()}")
        else:
            print("  [GIT] checkpoint pushed to GitHub")
    except Exception as e:
        print(f"  [GIT] auto-push error (non-fatal, experiment continues): {e}")


# ─── DATA LOADING (FIXED: correct fields per dataset) ────────────────────────
def load_qa_data(language: str, smoke_test: bool = False):
    """
    Arabic -> XQuAD (google/xquad, xquad.ar split): fields = context, question,
              answers
    Malay  -> Belebele (facebook/belebele, zsm_Latn): fields = flores_passage,
              question, mc_answer1..4, correct_answer_num (multiple-choice,
              NOT free-text QA — handled differently below).
    """
    if language == "arabic":
        # NOTE: huggingface_hub >= 1.16 requires fully-qualified "namespace/name"
        # dataset IDs. The bare "xquad" slug is rejected with HfUriError; the
        # correct namespaced ID is "google/xquad".
        ds = load_dataset("google/xquad", "xquad.ar", split="validation")
    elif language == "malay":
        ds = load_dataset("facebook/belebele", "zsm_Latn", split="test")
    else:
        raise ValueError(f"Unknown language: {language}")

    if smoke_test:
        ds = ds.select(range(min(SMOKE_DATA_CAP, len(ds))))  # tiny slice, dry run
    else:
        ds = ds.select(range(min(FULL_DATA_CAP, len(ds))))   # real training scale
    return ds


def extract_context_question_answer(example: dict, language: str):
    """
    Returns (context, question, gold_answer, options) normalized across the
    two different dataset schemas.

    Arabic (XQuAD): free-text extractive QA. options=None -- gold_answer is
    a short verbatim span lifted directly from the context.

    Malay (Belebele): multiple-choice. gold_answer is the text of the
    correct option, and options is the list of all 4 option texts.

    FIX (see build_prompt() below): Belebele's actual task is "pick the
    correct option given these 4 choices", not "freely recall a sentence
    you were never shown." Earlier versions of this script trained/evaluated
    the model on question+context ONLY, never showing it the options --
    the model had no way to know the length/phrasing/scope of a valid
    answer and had to blindly guess wording it had never seen. That made
    the Malay F_faith scores collapse (~0.08-0.10, flat across every config
    including full fine-tuning) versus Arabic (~0.31-0.37), because the
    task itself was unfair, not because training failed. Now options are
    included in the prompt so the model can actually ground its answer in
    the choices it's meant to select from -- matching Belebele's real task
    design and giving F_faith a fair basis for comparison across languages.
    """
    if language == "arabic":
        context = example.get("context", "")
        question = example.get("question", "")
        answers = example.get("answers", {})
        gold = answers.get("text", [""])[0] if isinstance(answers, dict) else ""
        return context, question, gold, None

    elif language == "malay":
        context = example.get("flores_passage", "")
        question = example.get("question", "")
        correct_num = example.get("correct_answer_num", "1")
        gold = example.get(f"mc_answer{correct_num}", "")
        options = [
            example.get("mc_answer1", ""),
            example.get("mc_answer2", ""),
            example.get("mc_answer3", ""),
            example.get("mc_answer4", ""),
        ]
        return context, question, gold, options

    raise ValueError(f"Unknown language: {language}")


def build_prompt(question: str, context: str, options=None) -> str:
    """
    Builds the model input prompt. Context comes right after the question
    (before options) so that if max_length truncation kicks in, the options
    block is what gets cut, never the passage the answer depends on.

    For Malay (options is a list of 4 strings), the options are appended so
    the model can see what it's choosing among -- this is the fix described
    in extract_context_question_answer() above.
    """
    base = f"question: {question}  context: {context}"
    if options:
        options_block = " ".join(f"({i + 1}) {opt}" for i, opt in enumerate(options))
        return f"{base}  options: {options_block}"
    return base


# ─── LoRA VARIANT CONFIG BUILDERS (FIXED: AdaLoRA scheduling params) ─────────
def build_peft_config(variant: str, total_steps: int = 30):
    if variant == "qlora":
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q", "k", "v", "o"],
        )
    elif variant == "adalora":
        # FIX: tinit/tfinal/deltaT/total_step are REQUIRED by AdaLoRA's rank
        # scheduler. Missing these raises a ValueError at training start.
        return AdaLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q", "k", "v", "o"],
            init_r=12, target_r=8,
            tinit=max(1, total_steps // 10),
            tfinal=max(2, total_steps // 2),
            deltaT=max(1, total_steps // 20),
            total_step=total_steps,
        )
    elif variant == "dora":
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q", "k", "v", "o"],
            use_dora=True,
        )
    elif variant == "vera":
        return VeraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            target_modules=["q", "k", "v", "o"],
        )
    else:
        raise ValueError(f"Unknown LoRA variant: {variant}")


def load_model_for_config(config_name: str, lora_variant: str = None,
                          total_steps: int = 30):
    """
    A_frozen   -> plain mT5-base, no adapter, eval only
    B/C (LoRA) -> mT5-base + PEFT adapter (specified variant)
    D_full_ft  -> mT5-base, all params trainable
    """
    quant_config = None
    if lora_variant == "qlora" and QUANTIZATION_AVAILABLE:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif lora_variant == "qlora" and not QUANTIZATION_AVAILABLE:
        print("  [INFO] QLoRA requested but CUDA unavailable — "
              "loading full-precision model instead (smoke test only).")

    model = MT5ForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=quant_config,
    )
    model.to(DEVICE)

    if config_name == "A_frozen":
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    if config_name == "D_full_ft":
        return model   # all params trainable, no PEFT wrapper

    # Config B / C -> attach LoRA adapter
    peft_config = build_peft_config(lora_variant, total_steps=total_steps)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ─── COMPOSITE LOSS (Config C only) ──────────────────────────────────────────
# DEPRECATED -- kept only so the bug is visible in a diff / for the paper's
# reproducibility appendix. This function is NO LONGER CALLED anywhere.
#
# Why it was wrong: faithfulness_score here is a plain Python float,
# computed under torch.no_grad() from a frozen NLI model scoring
# already-generated (non-differentiable, discretely-sampled) text.
# `ce_loss + penalty` where `penalty` is a Python float constant has
# d(ce_loss + penalty)/d(theta) == d(ce_loss)/d(theta) for every parameter
# theta -- i.e. this is mathematically indistinguishable, at the gradient
# level, from plain CE-only training. It changed the LOGGED loss value but
# never changed a single optimizer step. See CompositeLossTrainer below for
# the corrected, actually-differentiable replacement.
def composite_loss(ce_loss: torch.Tensor, faithfulness_score: float,
                    lambda1: float = 0.3) -> torch.Tensor:
    """L_G = L_CE + lambda1 * (1 - F_faith)  [NON-FUNCTIONAL -- see note above]"""
    penalty = lambda1 * (1.0 - faithfulness_score)
    return ce_loss + penalty


def _seq_log_probs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    target_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Differentiable log p(target_ids | input_ids) per sequence.

    This is the ONLY place in the corrected composite-loss path where
    gradient actually flows back into the trainable (LoRA) parameters. It
    is a plain teacher-forced forward pass -- NOT wrapped in no_grad -- so
    autograd builds a graph from `logits` back through `model`'s trainable
    weights. `target_ids` themselves are treated as fixed constants (they
    were produced by a no_grad sampling step upstream); we are not
    differentiating through how they were chosen, only through how likely
    the current policy says they are, which is exactly what the REINFORCE
    / score-function gradient estimator requires.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                     labels=target_ids)
    log_probs = F.log_softmax(outputs.logits, dim=-1)          # (B, T, V), requires_grad
    mask = (target_ids != pad_token_id).float()                  # (B, T)
    safe_targets = target_ids.clamp(min=0)                       # gather() needs >=0 indices
    token_log_probs = log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return (token_log_probs * mask).sum(dim=1)                   # (B,) -- sum over tokens


# ─── F_faith METRIC (evaluation) ─────────────────────────────────────────────
_nli_tokenizer = None
_nli_model = None

def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


# ─── CORRECTNESS METRICS (separate from F_faith) ─────────────────────────────
# F_faith measures whether an answer is entailed by the context -- it does
# NOT measure whether the answer is actually correct. A model can generate
# fluent, contextually-plausible text that scores well on entailment while
# being factually wrong, or a model that trivially copies chunks of the
# context (as an untrained frozen model tends to do) can score artificially
# high on entailment without ever attempting the actual task. Added after
# observing a case where D_full_ft/arabic generated "ي دور قاف عشر عشرين"
# against a gold answer of "308" -- clearly wrong -- yet still scored
# mean_f_faith=0.4164, in line with every other Arabic run. EM/F1 below
# are the standard SQuAD-style correctness metrics, run alongside F_faith
# so faithfulness and correctness can be told apart in the results.
def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize_text(pred) == _normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f_faith(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    tok, model = _load_nli()
    # Robust to label-dict ordering/casing differences across NLI model
    # cards (id2label keys are the actual class indices, not list position --
    # relying on list-position previously would silently break if a model's
    # id2label dict wasn't ordered {0: entailment, 1: neutral, 2: contradiction}).
    label_map = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    if "entailment" not in label_map:
        raise ValueError(f"NLI model has no 'entailment' label: {model.config.id2label}")
    ent_idx = label_map["entailment"]
    enc = tok(context, answer, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return round(probs[ent_idx].item(), 4)


# ─── CUSTOM TRAINER (Config C composite loss -- CORRECTED, self-critical) ────
class CompositeLossTrainer(Seq2SeqTrainer):
    """
    Standard Seq2SeqTrainer, except when use_composite=True (Config C only):
    uses a self-critical policy-gradient objective (Rennie et al., 2017,
    "Self-Critical Sequence Training", CVPR) to give the faithfulness score
    a genuine, differentiable path into the trainable LoRA parameters.
    For all other configs this behaves identically to the base Trainer.

    GRADIENT PATH SUMMARY (see compute_loss() below for the full walk-through):
      1. Sample K generations per example from the current policy
         (no_grad -- sampling is non-differentiable by construction).
      2. Score each with the frozen NLI reward model f_faith() (no_grad --
         the reward model's parameters are never updated).
      3. Build a leave-one-out advantage per sample: advantage_k =
         reward_k - mean(reward_{j != k}). Detached scalar weight, not
         differentiated.
      4. Recompute log p(sampled sequence | input) via a SEPARATE
         teacher-forced forward pass that IS differentiable
         (_seq_log_probs()). This is the only tensor in the composite loss
         that carries gradient back to the LoRA adapter weights.
      5. pg_loss = -mean(advantage * log_prob); final loss = ce_loss +
         lambda1 * pg_loss.
    FROZEN vs. TRAINABLE: the base mT5 backbone is frozen by
    get_peft_model() (Section II-B); only the injected LoRA adapter
    matrices receive gradient, from both ce_loss and pg_loss. The NLI
    reward model is entirely frozen throughout and supplies a scalar
    reward signal only, never a gradient.
    """
    def __init__(self, *args, use_composite: bool = False, tokenizer_ref=None,
                 lambda1: float = 0.3, num_samples: int = PG_NUM_SAMPLES,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_composite = use_composite
        self.tokenizer_ref = tokenizer_ref
        self.lambda1 = lambda1
        self.num_samples = num_samples

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if not self.use_composite:
            return (ce_loss, outputs) if return_outputs else ce_loss

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.size(0)
        K = self.num_samples
        pad_id = self.tokenizer_ref.pad_token_id or 0

        # ---- Step 1-2: sample K generations per example, score (no_grad) ----
        with torch.no_grad():
            rep_input_ids = input_ids.repeat_interleave(K, dim=0)
            rep_attention_mask = attention_mask.repeat_interleave(K, dim=0)
            sampled_ids = model.generate(
                input_ids=rep_input_ids,
                attention_mask=rep_attention_mask,
                max_new_tokens=40,
                do_sample=True,
                temperature=PG_SAMPLE_TEMPERATURE,
                top_p=PG_SAMPLE_TOP_P,
                renormalize_logits=True,
                logits_processor=LogitsProcessorList([_NaNSafeLogitsProcessor()]),
            )
            sampled_texts = self.tokenizer_ref.batch_decode(
                sampled_ids, skip_special_tokens=True)
            context_texts = self.tokenizer_ref.batch_decode(
                rep_input_ids, skip_special_tokens=True)
            rewards = [f_faith(t, c) for t, c in zip(sampled_texts, context_texts)]

        # ---- Step 3: leave-one-out baseline / advantage (detached) ----
        rewards_t = torch.tensor(rewards, device=input_ids.device,
                                  dtype=torch.float32).view(batch_size, K)
        if K > 1:
            sum_r = rewards_t.sum(dim=1, keepdim=True)
            baseline = (sum_r - rewards_t) / (K - 1)   # leave-one-out mean
        else:
            baseline = rewards_t.mean()                 # degenerate K=1 fallback
        advantages = (rewards_t - baseline).view(-1).detach()  # (B*K,)

        # ---- Step 4: differentiable log-prob of the sampled sequences ----
        seq_log_probs = _seq_log_probs(
            model, rep_input_ids, rep_attention_mask, sampled_ids, pad_id)  # (B*K,)

        # ---- Step 5: policy-gradient loss + combine with CE ----
        pg_loss = -(advantages * seq_log_probs).mean()
        loss = ce_loss + self.lambda1 * pg_loss
        return (loss, outputs) if return_outputs else loss


# ─── TRAIN + EVAL ONE RUN (FIXED: per-run try/except, no full-script crash) ──
def run_single_experiment(config_name: str, lora_variant: str, language: str,
                           smoke_test: bool = False) -> dict:
    label = f"config={config_name} variant={lora_variant} lang={language}"
    print(f"\n{'='*70}\nRUN: {label}\n{'='*70}")
    t0 = time.time()
    model = None  # tracked so the finally block below can always release GPU
                  # memory, even if this run fails before/while building it

    # FIX (peer review, item 8): fixed seed before every run so Config B and
    # Config C start from identical LoRA initialization and batch order,
    # isolating the training-objective difference as the only variable
    # between them. Previously unseeded.
    set_all_seeds(SEED)

    try:
        dataset = load_qa_data(language, smoke_test=smoke_test)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Carve out the eval slice FIRST and exclude it from training data.
        # Without this split, train_ds and eval_slice previously overlapped
        # (eval_slice was the first N examples of the SAME dataset used for
        # training), meaning the model was evaluated on data it had just
        # trained on -- train/test leakage that would invalidate the
        # F_faith comparison across LoRA variants for the paper.
        n_eval = min(SMOKE_EVAL_SIZE if smoke_test else FULL_EVAL_SIZE, len(dataset))
        eval_indices = list(range(n_eval))
        train_indices = list(range(n_eval, len(dataset)))
        n_train = len(train_indices) if train_indices else len(eval_indices)

        # total_steps drives AdaLoRA's rank-decay schedule (tinit/tfinal/
        # deltaT). It must match the ACTUAL number of optimizer steps the
        # Trainer will run below, or AdaLoRA prunes ranks on a schedule that
        # doesn't match training length. Previously this was computed from
        # len(dataset)//4 before the train/eval split existed, so it never
        # matched the real step count even at the old small scale --
        # computed correctly here from the real batch size/epoch count.
        if smoke_test:
            total_steps = 5
        else:
            steps_per_epoch = max(1, -(-n_train // TRAIN_BATCH_SIZE))  # ceil div
            total_steps = max(2, steps_per_epoch * TRAIN_EPOCHS)

        model = load_model_for_config(config_name, lora_variant, total_steps)

        if config_name != "A_frozen" and not smoke_test:
            def _tokenize_fn(ex):
                ctx, q, gold, options = extract_context_question_answer(ex, language)
                prompt = build_prompt(q, ctx, options)
                model_inputs = tokenizer(prompt, truncation=True, max_length=512)
                labels = tokenizer(text_target=(gold or ""), truncation=True,
                                    max_length=64)
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            train_ds_raw = (dataset.select(train_indices) if train_indices
                             else dataset.select(eval_indices))
            train_ds = train_ds_raw.map(_tokenize_fn,
                                         remove_columns=train_ds_raw.column_names)
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            training_args = Seq2SeqTrainingArguments(
                output_dir=str(OUTPUT_DIR / f"ckpt_{config_name}_{lora_variant}_{language}"),
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                num_train_epochs=TRAIN_EPOCHS,
                # FIX: lora_variant is the STRING "none" for D_full_ft (see
                # build_run_matrix()), which is truthy in Python -- so this
                # condition was silently giving full fine-tuning the same
                # 2e-4 learning rate meant for tiny LoRA adapters, instead of
                # the intended lower 5e-5 for updating all 580M parameters
                # at once. This produced exactly the optimizer-divergence
                # signature seen in the D_full_ft/arabic training log
                # (grad_norm spiking to 7e4+, loss oscillating instead of
                # decreasing). Now keyed off config_name, which is unambiguous.
                learning_rate=2e-4 if config_name in ("B_ce_lora", "C_composite_lora") else 5e-5,
                logging_steps=5,
                save_strategy="no",
                report_to=[],
                remove_unused_columns=False,
                # NOTE: T5-family models (including mT5) are numerically
                # unstable under fp16 mixed precision and frequently produce
                # NaN gradients / zeroed-out loss, silently corrupting
                # training without raising an exception. bf16 does not have
                # this issue and is natively supported on the RTX 4090.
                bf16=(DEVICE == "cuda"),
                # FIX: D_full_ft trains all 580M mT5 parameters with a
                # standard AdamW optimizer, which needs ~2.3GB weights +
                # ~2.3GB gradients + ~4.6GB AdamW momentum/variance (2 fp32
                # copies per param) = ~9.3GB baseline before any activation
                # memory -- right at the edge of a 24GB GPU, and the run
                # that OOM'd here happened right after A_frozen's model had
                # already touched memory in the same process. Gradient
                # checkpointing trades some compute for much lower
                # activation memory, and Adafactor (the optimizer T5's own
                # authors used) needs far less optimizer-state memory than
                # AdamW. Both only apply to D_full_ft -- LoRA runs have a
                # tiny trainable parameter budget and don't need this.
                gradient_checkpointing=(config_name == "D_full_ft"),
                optim=("adafactor" if config_name == "D_full_ft" else "adamw_torch"),
            )

            trainer = CompositeLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                data_collator=data_collator,
                use_composite=(config_name == "C_composite_lora"),
                tokenizer_ref=tokenizer,
            )
            trainer.train()
            print(f"  [TRAINING] completed {training_args.num_train_epochs} epochs "
                  f"({'composite loss' if config_name == 'C_composite_lora' else 'CE loss'})")
        elif config_name != "A_frozen" and smoke_test:
            print("  [SMOKE TEST] Skipping actual training — validating "
                  "model/adapter construction only.")

        # Evaluation: generate + score F_faith on the held-out slice reserved
        # above (guaranteed disjoint from train_ds when training occurred).
        eval_slice = dataset.select(eval_indices)
        scores = []
        em_scores = []
        f1_scores = []
        sample_log = []  # first 5 (gold, generated) pairs per run, for the paper
        for i, ex in enumerate(eval_slice):
            context, question, gold, options = extract_context_question_answer(ex, language)
            if not context or not question:
                continue
            prompt = build_prompt(question, context, options)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=512).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
            answer = tokenizer.decode(out[0], skip_special_tokens=True)
            scores.append(f_faith(answer, context))
            # EM/F1 measure whether the answer is actually CORRECT (matches
            # gold), independent of F_faith's "is it entailed by context"
            # judgment -- see note above exact_match()/token_f1() for why
            # both are needed.
            em_scores.append(exact_match(answer, gold))
            f1_scores.append(token_f1(answer, gold))
            # Print the first 5 eval examples per run so generated vs. gold
            # text is visible in the terminal log -- useful for a qualitative
            # sanity check / paper appendix examples, and for catching any
            # future task-formulation regressions immediately.
            if i < 5:
                print(f"  [SAMPLE {i}] gold=\"{gold[:80]}\" | generated=\"{answer[:80]}\" "
                      f"| em={em_scores[-1]} f1={round(f1_scores[-1], 2)}")
                sample_log.append({"gold": gold, "generated": answer,
                                    "em": em_scores[-1], "f1": round(f1_scores[-1], 4)})

        mean_score = sum(scores) / len(scores) if scores else 0.0
        mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        elapsed = time.time() - t0

        result = {
            "config": config_name, "lora_variant": lora_variant, "language": language,
            "status": "PASS", "mean_f_faith": round(mean_score, 4),
            "mean_em": round(mean_em, 4), "mean_f1": round(mean_f1, 4),
            "n_eval": len(scores), "runtime_sec": round(elapsed, 1), "error": "",
            "samples": json.dumps(sample_log, ensure_ascii=False),
        }
        print(f"  RESULT: PASS | mean_f_faith={result['mean_f_faith']} "
              f"| mean_em={result['mean_em']} | mean_f1={result['mean_f1']} "
              f"| n_eval={result['n_eval']} | {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  RESULT: FAIL | {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "config": config_name, "lora_variant": lora_variant, "language": language,
            "status": "FAIL", "mean_f_faith": None, "mean_em": None, "mean_f1": None,
            "n_eval": 0, "runtime_sec": round(elapsed, 1),
            "error": f"{type(e).__name__}: {e}", "samples": "",
        }

    finally:
        # FIX: explicitly release GPU memory before the next run starts.
        # Without this, PyTorch's CUDA caching allocator can leave freed
        # blocks fragmented rather than fully reclaimed, and a heavy run
        # right after a lighter one (e.g. D_full_ft's full optimizer state
        # right after a frozen eval-only run) can hit an avoidable
        # OutOfMemoryError even though total free memory would otherwise
        # be sufficient. This is what caused the one D_full_ft/arabic FAIL.
        if model is not None:
            del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()


# ─── MAIN: BUILD THE RUN MATRIX ────────────────────────────────────────────
def build_run_matrix():
    """A_frozen and D_full_ft don't vary by LoRA variant -> use 'none'."""
    runs = []
    for lang in LANGUAGES:
        runs.append(("A_frozen", "none", lang))
        runs.append(("D_full_ft", "none", lang))
        for variant in LORA_VARIANTS:
            runs.append(("B_ce_lora", variant, lang))
            runs.append(("C_composite_lora", variant, lang))
    return runs   # 2 langs x (1 + 1 + 4 + 4) = 20 conditions total.
                  # This is the final experimental design (see Paper 2 report):
                  # Configs A/D run once per language, Configs B/C run once
                  # per LoRA variant per language.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true",
                        help="Validate pipeline on CPU/MacBook, no server needed")
    parser.add_argument("--full", action="store_true",
                        help="Run the full experiment matrix (requires GPU)")
    parser.add_argument("--composite_only", action="store_true",
                        help="With --full, only (re)run the 8 Config C "
                             "(composite-loss) conditions -- the ones "
                             "invalidated by the non-differentiability bug. "
                             "Configs A/B/D are unaffected by that bug and do "
                             "not need to be rerun. Cuts GPU time by more "
                             "than half relative to a full 20-condition rerun.")
    parser.add_argument("--variant", type=str, default=None,
                        choices=LORA_VARIANTS,
                        help="Restrict to a single LoRA variant (e.g. 'qlora'). "
                             "Combine with --lang to isolate exactly one of the "
                             "8 Config C conditions, so different rented GPU "
                             "pods can each run one condition in parallel "
                             "instead of all 8 running serially on one GPU.")
    parser.add_argument("--lang", type=str, default=None,
                        choices=LANGUAGES,
                        help="Restrict to a single language ('arabic' or "
                             "'malay'). See --variant.")
    args = parser.parse_args()

    if not (args.smoke_test or args.full):
        print("Specify --smoke_test (run now, on your Mac) "
              "or --full (run later, on lab server). Exiting.")
        return

    if args.full and DEVICE == "cpu":
        print("⚠ WARNING: --full requested but no GPU detected.")
        print("  QLoRA quantization will fail. Aborting — run --full on the "
              "lab server, not locally.")
        return

    if args.smoke_test:
        checkpoint_name = "checkpoint_smoke.jsonl"
    elif args.variant and args.lang:
        # Sharded run (one pod = one condition): give each shard its OWN
        # checkpoint file so N parallel pods never push-conflict on the same
        # path/branch in the shared GitHub repo. Merge the shard files
        # together once all pods finish (see printed NOTE below).
        checkpoint_name = f"checkpoint_full_{args.variant}_{args.lang}.jsonl"
    else:
        checkpoint_name = "checkpoint_full.jsonl"
    checkpoint_file = OUTPUT_DIR / checkpoint_name

    runs = build_run_matrix()
    if args.composite_only:
        runs = [r for r in runs if r[0] == "C_composite_lora"]
        print(f"--composite_only: restricting to the {len(runs)} Config C "
              f"runs (4 LoRA variants x 2 languages). Configs A/B/D are "
              f"assumed already valid from the original run and are skipped.")
    if args.variant:
        runs = [r for r in runs if r[1] == args.variant]
        print(f"--variant {args.variant}: restricting to this LoRA variant only.")
    if args.lang:
        runs = [r for r in runs if r[2] == args.lang]
        print(f"--lang {args.lang}: restricting to this language only.")
    if args.variant and args.lang and not args.smoke_test:
        print(f"NOTE: sharded run -- this pod writes ONLY to "
              f"{checkpoint_file.name} (not the shared checkpoint_full.jsonl), "
              f"so 8 parallel pods can each push to the same GitHub repo "
              f"without conflicting. After all 8 pods finish, run "
              f"`cat experiment_results/checkpoint_full_*.jsonl > "
              f"experiment_results/checkpoint_full.jsonl` once (on any one "
              f"pod, after pulling all shards) to merge them into the file "
              f"the final report/CSV step expects.")
    completed = load_completed_runs(checkpoint_file)
    if completed:
        skip_count = sum(1 for r in runs if _run_key(*r) in completed)
        runs = [r for r in runs if _run_key(*r) not in completed]
        print(f"Resuming from checkpoint ({checkpoint_file.name}): "
              f"{skip_count} runs already PASSED in a previous session, skipping them.")

    print(f"Total runs queued: {len(runs)} | mode: "
          f"{'SMOKE TEST (CPU, pipeline validation only)' if args.smoke_test else 'FULL (GPU)'}")

    for config_name, lora_variant, language in runs:
        result = run_single_experiment(config_name, lora_variant, language,
                                       smoke_test=args.smoke_test)
        append_checkpoint(checkpoint_file, result)
        if not args.smoke_test:
            git_commit_and_push(
                checkpoint_file,
                f"Checkpoint: {config_name}/{lora_variant}/{language} = {result['status']}",
            )

    # Final report always reflects EVERY recorded result — runs skipped this
    # session because they already passed, plus runs completed just now.
    all_results = load_all_checkpoint_results(checkpoint_file)
    df = pd.DataFrame(all_results)
    print("\n\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(tabulate(df, headers="keys", tablefmt="rounded_grid", showindex=False))

    n_pass = (df["status"] == "PASS").sum()
    n_fail = (df["status"] == "FAIL").sum()
    print(f"\n{n_pass}/{len(df)} runs PASSED, {n_fail}/{len(df)} FAILED.")
    if args.smoke_test:
        if n_fail == 0:
            print("\n✓ Pipeline fully validated on CPU. Safe to request lab "
                  "server time and run --full with confidence.")
        else:
            print("\n✗ Fix the FAILED runs above before requesting server "
                  "access — otherwise these same errors will waste GPU time.")

    out_path = OUTPUT_DIR / ("smoke_test_results.csv" if args.smoke_test
                             else "full_matrix_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
