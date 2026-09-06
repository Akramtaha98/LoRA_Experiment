"""
Qwen Experiment Matrix — Cross-Architecture Generalization Check for Paper 2
=============================================================================
Author : Akram Taha Zeyad
Thesis : Formalizing RAG Failure Modes and LoRA-Based Mitigation

WHY THIS SCRIPT EXISTS:
  lora_experiment_matrix.py (mT5-base, encoder-decoder) found a NULL effect
  for the composite/faithfulness-penalized objective at seed 42 under the
  corrected pipeline (sign test p=0.289, Wilcoxon p=0.383, mean F_faith
  LOWER under Config C than Config B), with weak absolute Arabic generation
  quality across every condition including the untrained frozen baseline.
  This script re-runs the SAME 20-condition design on a different backbone
  to test whether that null (and the weak-Arabic-generation pattern)
  generalizes across architecture, or is specific to mT5-base.

  BACKBONE CHOICE, AND WHY (see chat log / paper Discussion for full
  reasoning -- summarized here for anyone reading only this file):
    - Qwen3-0.6B-Base (Alibaba, Apache 2.0, released 2025). "Base" variant
      specifically -- NOT the instruction-tuned checkpoint -- to match
      mT5-base's role as a pretrained-but-not-task-tuned backbone; Config A
      (frozen, zero-shot) is only a fair "what does an untrained model do"
      baseline if the checkpoint itself was never instruction-tuned.
    - Its technical report explicitly lists Arabic and Malay among its
      trained/benchmarked languages (Qwen3 Technical Report,
      arXiv:2505.09388) -- a stronger, citable claim than Gemma 4's
      unpublished "35+ out-of-the-box" list, and unlike Cohere's Aya
      Expanse, which lists Indonesian but not Malay specifically.
    - 0.6B params is almost exactly mT5-base's 580M. This is deliberate:
      picking a much larger backbone (e.g. a multi-billion-parameter Gemma
      variant) would confound scale, architecture, AND language-coverage
      differences all at once. Matching scale isolates architecture +
      pretraining-language-composition as the variable under test.
    - Full parity (all 20 conditions, INCLUDING Config D full fine-tuning)
      is feasible on the same 24-32GB rented-GPU budget used for the mT5
      runs. A larger backbone (e.g. Gemma-4 E2B, ~5.1B total params) would
      need ~80GB+ for a standard full-AdamW fine-tune and force dropping
      Config D or renting fundamentally different hardware.

  IMPORTANT SCIENTIFIC CAVEAT (read before writing this up): this backbone
  was chosen for principled reasons (matched scale, confirmed language
  coverage, license, compute feasibility) BEFORE running anything -- it was
  not selected, and must not be reported, as "the backbone that gave a
  positive result." If this script's results also show a null effect for
  the composite objective, that is itself a stronger, cross-architecture
  finding (the null replicates beyond mT5) and should be reported as such,
  not discarded in favor of continuing to search for a backbone that
  flips the sign.

WHAT CHANGES FROM lora_experiment_matrix.py, AND WHY:
  1. Model class: AutoModelForCausalLM instead of MT5ForConditionalGeneration.
     Qwen3 is a decoder-only causal LM, not an encoder-decoder seq2seq model.
  2. Prompt format: causal-LM completion style ("Question: ... Context: ...
     Answer:") instead of T5's "question: ... context: ..." span-infilling
     style. Same information content per example (question, context,
     Malay's 4 options), different surface format required by the
     architecture.
  3. No sentinel-token suppression. T5's "<extra_id_N>" span-corruption
     leakage (56% of pre-fix Arabic generations, see main paper) is a T5-
     pretraining-objective-specific failure mode. Qwen3 has no equivalent
     sentinel vocabulary, so there is nothing analogous to suppress. If
     Qwen3 shows its OWN distinct degenerate-generation failure mode, that
     is a genuine new finding to report, not something to paper over by
     inventing a suppression list.
  4. target_modules for LoRA/AdaLoRA/DoRA/VeRA cover BOTH attention
     (q_proj/k_proj/v_proj/o_proj) AND the SwiGLU MLP
     (gate_proj/up_proj/down_proj) from the start -- i.e. the FFN-coverage
     fix that had to be discovered the hard way for mT5 (readiness review
     item B1) is applied here from the first run, not after finding the
     same bug twice.
  5. Custom SFT data collator (QwenSFTCollator below) instead of
     DataCollatorForSeq2Seq: causal-LM SFT needs prompt-token loss masking
     (labels=-100 on the prompt span) that seq2seq collators don't do.
  6. SCST composite loss (CompositeLossTrainer.compute_loss below) is
     rewritten for causal-LM generation semantics: sampling continues from
     a left-padded prompt-only tensor (required for batched generate() on
     a decoder-only model), and the differentiable log-prob step
     explicitly computes position_ids from the attention_mask before the
     teacher-forced forward pass. This matters and is easy to get silently
     wrong: a bare forward() call on a left-padded batch does NOT
     automatically recover correct RoPE position indices from
     attention_mask the way generate() does internally -- if you skip this
     and let the default arange position_ids apply, every left-padded
     sequence in the batch silently gets WRONG position embeddings for its
     real (non-pad) tokens, corrupting the policy-gradient log-probs
     without raising any error. See _seq_log_probs_causal() below.
  7. KNOWN PARITY QUIRK, CARRIED OVER DELIBERATELY: the original mT5
     script's SCST reward step scores f_faith(sampled_answer, DECODED_PROMPT)
     -- i.e. the reward signal during TRAINING is computed against the
     full "question: ... context: ..." prompt text, not the bare passage
     -- while the final EVAL metric scores f_faith(answer, bare_context).
     This is arguably an inconsistency in the original script (training
     reward and reported metric are not scored against identical text).
     It is reproduced here EXACTLY, on purpose, so any difference between
     the mT5 and Qwen results is attributable to the backbone and not to
     a simultaneous, unannounced pipeline fix. Flagged here (and in the
     handoff to Akram) as a real candidate fix for a THIRD script version,
     applied identically to both backbones at the same time, not silently
     patched in only one.
  8. Everything backbone-agnostic is UNCHANGED and copied verbatim from
     lora_experiment_matrix.py: dataset loading (XQuAD/Belebele), F_faith/
     EM/F1 metric functions, GPU-compatibility check, git auto-push-with-
     retry, checkpoint/resume logic, CLI flag surface (--smoke_test,
     --full, --composite_only, --bc_only, --ad_only, --variant, --lang,
     --seed), TRAIN_SIZE/TRAIN_EPOCHS/SEED/PG_* hyperparameters. Keeping
     these identical is deliberate: it isolates the backbone as the only
     varied factor between this script's results and the mT5 script's.

REQUIREMENTS:
  Smoke test (Mac, CPU):
    pip install transformers peft accelerate datasets sentencepiece torch \
                pandas tabulate

  Full run (lab server, GPU):
    pip install transformers peft bitsandbytes accelerate datasets \
                sentencepiece torch pandas tabulate evaluate

  Same RTX 5090 / Blackwell torch-build caveat as lora_experiment_matrix.py
  applies here unchanged -- see check_gpu_compatibility() below, which is
  copied verbatim.

USAGE (mirrors lora_experiment_matrix.py exactly -- same runbook, new
script name and BASE_MODEL):
    python3 qwen_experiment_matrix.py --smoke_test
    python3 qwen_experiment_matrix.py --full                         # all 20 conditions
    python3 qwen_experiment_matrix.py --full --ad_only               # Config A/D only (2 langs)
    python3 qwen_experiment_matrix.py --full --variant qlora --lang arabic   # one sharded condition
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
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    LogitsProcessor,
    LogitsProcessorList,
)

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


class _NaNSafeLogitsProcessor(LogitsProcessor):
    """Sanitize NaN/Inf logits before sampling/argmax. Copied verbatim from
    lora_experiment_matrix.py -- the QLoRA (4-bit) NaN-logit failure mode
    that motivated this is a bitsandbytes/4-bit-quantization risk, not a
    T5-specific one, so it applies here too. Cheap defensive measure kept
    on generically rather than waiting to rediscover the same crash."""

    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


# ─── DEVICE DETECTION (copied verbatim from lora_experiment_matrix.py) ───────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTIZATION_AVAILABLE = _BNB_AVAILABLE and DEVICE == "cuda"

if DEVICE == "cpu":
    print("⚠ No CUDA GPU detected — running in CPU smoke-test mode.")
    print("  QLoRA 4-bit quantization will be SKIPPED (needs CUDA).")
    print("  This validates pipeline LOGIC only, not real training quality.\n")


def check_gpu_compatibility() -> tuple:
    """Returns (is_compatible: bool, message: str). Copied verbatim from
    lora_experiment_matrix.py -- a real CUDA matmul is the only reliable
    way to know whether the installed torch build can execute on this GPU
    (see that file's docstring for the RTX 4090/5090 false-positive and
    false-negative history this replaced). Entirely backbone-agnostic."""
    if DEVICE != "cuda":
        return True, ""
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    sm_tag = f"sm_{cap_major}{cap_minor}"
    gpu_name = torch.cuda.get_device_name(0)
    arch_list = torch.cuda.get_arch_list()
    base_msg = (f"GPU: {gpu_name} | compute capability {cap_major}.{cap_minor} "
                f"({sm_tag}) | torch {torch.__version__} | torch built for: "
                f"{', '.join(arch_list) if arch_list else '(none reported)'}")
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        (a @ b).sum().item()
        torch.cuda.synchronize()
        return True, base_msg
    except RuntimeError as e:
        msg = (
            f"{base_msg}\n"
            f"  ✗ A real CUDA matmul on this GPU FAILED: {e}\n"
            f"  This torch build cannot actually execute kernels on {sm_tag}. Fix: "
            f"pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
        return False, msg


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_MODEL    = "Qwen/Qwen3-0.6B-Base"   # pretrained-only checkpoint, NOT
                                          # instruction-tuned -- see docstring
NLI_MODEL     = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # unchanged, same
                                                            # reward model as mT5 script
LANGUAGES     = ["arabic", "malay"]
LORA_VARIANTS = ["qlora", "adalora", "dora", "vera"]

# Identical to lora_experiment_matrix.py, kept identical on purpose so the
# only varied factor between the two scripts' results is the backbone.
TRAIN_SIZE       = 470
SMOKE_DATA_CAP   = 3
SMOKE_EVAL_SIZE  = 3
TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS     = 12
SEED             = 42

PG_NUM_SAMPLES        = 4
PG_SAMPLE_TEMPERATURE = 1.0
PG_SAMPLE_TOP_P       = 0.95

MAX_PROMPT_LEN = 512   # matches mT5 script's tokenizer truncation length
MAX_ANSWER_LEN = 40    # matches mT5 script's max_new_tokens / label length


def set_all_seeds(seed: int = SEED):
    """Copied verbatim from lora_experiment_matrix.py."""
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


OUTPUT_DIR = Path("./experiment_results_qwen")   # SEPARATE from the mT5
                                                  # script's ./experiment_results
                                                  # so checkpoint files never
                                                  # collide in the shared repo.
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── CHECKPOINTING (copied verbatim from lora_experiment_matrix.py) ──────────
def _run_key(config_name: str, lora_variant: str, language: str, seed: int) -> str:
    return f"{config_name}|{lora_variant}|{language}|seed={seed}"


def load_completed_runs(checkpoint_file: Path) -> set:
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
                                            entry["language"],
                                            entry.get("seed", SEED)))
    return completed


def append_checkpoint(checkpoint_file: Path, result: dict) -> None:
    with open(checkpoint_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_all_checkpoint_results(checkpoint_file: Path) -> list:
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


# ─── AUTO-PUSH TO GITHUB (copied verbatim from lora_experiment_matrix.py) ────
def git_commit_and_push(repo_file: Path, message: str, max_retries: int = 3) -> None:
    try:
        subprocess.run(["git", "add", str(repo_file)],
                        check=True, capture_output=True, text=True)
        commit = subprocess.run(["git", "commit", "-m", message],
                                 capture_output=True, text=True)
        if commit.returncode != 0 and "nothing to commit" not in commit.stdout:
            print(f"  [GIT] commit warning: {commit.stdout.strip()} {commit.stderr.strip()}")
            return

        for attempt in range(1, max_retries + 1):
            push = subprocess.run(["git", "push"], capture_output=True, text=True)
            if push.returncode == 0:
                print("  [GIT] checkpoint pushed to GitHub")
                return

            rejected = ("fetch first" in push.stderr or
                        "non-fast-forward" in push.stderr or
                        "rejected" in push.stderr)
            if not rejected or attempt == max_retries:
                print(f"  [GIT] push FAILED after {attempt} attempt(s) "
                      f"(checkpoint is committed locally, push manually "
                      f"later): {push.stderr.strip()}")
                return

            print(f"  [GIT] push rejected (another pod pushed first) -- "
                  f"pulling + retrying, attempt {attempt}/{max_retries}")
            pull = subprocess.run(["git", "pull", "--rebase", "--autostash"],
                                   capture_output=True, text=True)
            if pull.returncode != 0:
                print(f"  [GIT] pull --rebase failed, aborting retry loop: "
                      f"{pull.stderr.strip()}")
                subprocess.run(["git", "rebase", "--abort"],
                                capture_output=True, text=True)
                return
            time.sleep(2)
    except Exception as e:
        print(f"  [GIT] auto-push error (non-fatal, experiment continues): {e}")


# ─── DATA LOADING (copied verbatim from lora_experiment_matrix.py) ───────────
def load_qa_data(language: str, smoke_test: bool = False):
    if language == "arabic":
        ds = load_dataset("google/xquad", "xquad.ar", split="validation")
    elif language == "malay":
        ds = load_dataset("facebook/belebele", "zsm_Latn", split="test")
    else:
        raise ValueError(f"Unknown language: {language}")

    if smoke_test:
        ds = ds.select(range(min(SMOKE_DATA_CAP, len(ds))))
    return ds


def extract_context_question_answer(example: dict, language: str):
    """Copied verbatim from lora_experiment_matrix.py -- dataset schema is
    backbone-agnostic."""
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
    """Causal-LM completion-style prompt, adapted from the mT5 script's
    'question: ... context: ...[ options: ...]' format. Same information
    content per example (question, context, and Malay's 4 options), but
    phrased as a completion cue ('Answer:') rather than T5's span-infilling
    style, since Qwen3-0.6B-Base is a plain decoder-only LM, not trained on
    T5's sentinel/span-corruption format."""
    prompt = f"Question: {question}\nContext: {context}"
    if options:
        options_block = " ".join(f"({i + 1}) {opt}" for i, opt in enumerate(options))
        prompt += f"\nOptions: {options_block}"
    prompt += "\nAnswer:"
    return prompt


# ─── LoRA VARIANT CONFIG BUILDERS (causal-LM target modules from the start) ──
# Qwen3's decoder block (like Llama/Mistral-family models) has attention
# projections q_proj/k_proj/v_proj/o_proj and a SwiGLU MLP with
# gate_proj/up_proj/down_proj. Both are targeted from the FIRST run here --
# unlike the mT5 script, which initially targeted only q/k/v/o and had to
# add the gated-FFN modules (wi_0/wi_1/wo) after readiness review flagged it
# as a likely cause of sentinel-leakage/undertraining. Same lesson, applied
# up front instead of rediscovered.
QWEN_ATTN_AND_MLP_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"]


def build_peft_config(variant: str, total_steps: int = 30):
    modules = QWEN_ATTN_AND_MLP_MODULES
    if variant == "qlora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=modules,
        )
    elif variant == "adalora":
        return AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=modules,
            init_r=12, target_r=8,
            tinit=max(1, total_steps // 10),
            tfinal=max(2, total_steps // 2),
            deltaT=max(1, total_steps // 20),
            total_step=total_steps,
        )
    elif variant == "dora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=modules,
            use_dora=True,
        )
    elif variant == "vera":
        return VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            target_modules=modules,
        )
    else:
        raise ValueError(f"Unknown LoRA variant: {variant}")


_TOKENIZER_CACHE = None


def get_tokenizer():
    """Qwen3 tokenizer may not define a pad token by default (common for
    causal-LM checkpoints whose pretraining never batched variable-length
    sequences with padding). Fall back to eos_token as pad_token if unset --
    standard, safe practice for causal-LM SFT; the pad token is never a
    valid label anyway since it's masked out (-100) or attention-masked
    everywhere it appears."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER_CACHE = tok
    return _TOKENIZER_CACHE


def load_model_for_config(config_name: str, lora_variant: str = None,
                          total_steps: int = 30):
    """
    A_frozen   -> plain Qwen3-0.6B-Base, no adapter, eval only
    B/C (LoRA) -> Qwen3-0.6B-Base + PEFT adapter (specified variant)
    D_full_ft  -> Qwen3-0.6B-Base, all params trainable
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

    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    if config_name == "A_frozen":
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    if config_name == "D_full_ft":
        return model

    peft_config = build_peft_config(lora_variant, total_steps=total_steps)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    return model


# ─── CAUSAL-LM SFT COLLATOR (loss-masked prompt, NEW for this script) ────────
class QwenSFTCollator:
    """Builds two separate views of each training example, both needed by
    CompositeLossTrainer below:

      1. "Full" fields (input_ids/attention_mask/labels), RIGHT-padded,
         used for the standard CE-loss forward pass. Prompt-token positions
         in `labels` are set to -100 so cross-entropy is only computed on
         the answer span -- the model is not penalized for "failing to
         predict" tokens it was only ever supposed to condition on.
         Right-padding here is deliberate and safe: the default
         torch.arange position_ids that AutoModelForCausalLM.forward()
         uses when position_ids is omitted are CORRECT for right-padded
         batches (every real token still starts at position 0), so no
         extra position_ids bookkeeping is needed for this path.

      2. "Prompt-only" fields (prompt_input_ids/prompt_attention_mask),
         LEFT-padded, used to seed model.generate() calls (both eval-time
         and SCST training-time sampling). Batched generate() on a
         decoder-only model requires left-padding so that new tokens are
         appended in the same column for every sequence in the batch;
         transformers' generate() internally derives correct position_ids
         from a left-padded attention_mask, so this half needs no special
         handling either -- the tricky part is the *teacher-forced replay*
         of a left-padded sequence for the SCST log-prob step, handled in
         _seq_log_probs_causal() below, not here.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, examples: list) -> dict:
        pad_id = self.tok.pad_token_id

        # ---- 1. full (prompt + answer) sequences, right-padded ----
        full_ids_list, full_labels_list = [], []
        for ex in examples:
            prompt_ids = ex["prompt_ids"]
            answer_ids = ex["answer_ids"]  # already includes a trailing eos
            full = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + list(answer_ids)
            full_ids_list.append(full)
            full_labels_list.append(labels)
        max_full = max(len(x) for x in full_ids_list)

        input_ids = torch.full((len(examples), max_full), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_full), dtype=torch.long)
        labels = torch.full((len(examples), max_full), -100, dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(full_ids_list, full_labels_list)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
            labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)

        # ---- 2. prompt-only sequences, LEFT-padded (for generate()) ----
        prompt_lens = [len(ex["prompt_ids"]) for ex in examples]
        max_prompt = max(prompt_lens)
        prompt_input_ids = torch.full((len(examples), max_prompt), pad_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((len(examples), max_prompt), dtype=torch.long)
        for i, ex in enumerate(examples):
            p = ex["prompt_ids"]
            prompt_input_ids[i, max_prompt - len(p):] = torch.tensor(p, dtype=torch.long)
            prompt_attention_mask[i, max_prompt - len(p):] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


def build_sft_dataset(dataset, indices: list, language: str, tokenizer):
    """Pre-tokenizes prompt and answer SEPARATELY (never concatenated as raw
    text and re-tokenized together), so the exact prompt/answer token
    boundary is known for label masking -- concatenating strings first and
    re-tokenizing the combined text can shift subword boundaries at the
    seam and silently mis-align the mask by one or more tokens."""
    examples = []
    for idx in indices:
        ex = dataset[idx]
        context, question, gold, options = extract_context_question_answer(ex, language)
        if not context or not question:
            continue
        prompt = build_prompt(question, context, options)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_PROMPT_LEN,
                               add_special_tokens=False)["input_ids"]
        answer_text = f" {gold or ''}"
        answer_ids = tokenizer(answer_text, truncation=True, max_length=MAX_ANSWER_LEN,
                               add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        examples.append({"prompt_ids": prompt_ids, "answer_ids": answer_ids})
    return examples


# ─── F_faith METRIC (copied verbatim from lora_experiment_matrix.py) ─────────
_nli_tokenizer = None
_nli_model = None

def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


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
    label_map = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    if "entailment" not in label_map:
        raise ValueError(f"NLI model has no 'entailment' label: {model.config.id2label}")
    ent_idx = label_map["entailment"]
    enc = tok(context, answer, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return round(probs[ent_idx].item(), 4)


def _truncate_at_newline(text: str) -> str:
    """Base (non-instruction-tuned) causal LMs frequently keep generating
    past the answer -- e.g. hallucinating a fresh 'Question:' for a next
    turn it was never asked, mimicking a multi-example pretraining pattern.
    Cutting at the first newline isolates just the answer span. This is a
    standard practice for zero/few-shot base-LM completion evaluation, not
    a metric-gaming trick: no gold XQuAD/Belebele answer contains a
    newline, so this never removes part of a genuinely correct answer."""
    return text.split("\n")[0].strip()


# ─── DIFFERENTIABLE LOG-PROB OF A SAMPLED CONTINUATION (causal-LM version) ──
def _seq_log_probs_causal(model, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
                          continuation_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Differentiable log p(continuation_ids | prompt_ids) per sequence, for
    a decoder-only causal LM. This is the causal-LM counterpart of the mT5
    script's _seq_log_probs() -- the only place gradient flows back into
    the trainable (LoRA) parameters from the SCST objective.

    CRITICAL DETAIL: prompt_ids is LEFT-padded (see QwenSFTCollator above).
    A bare model(input_ids=..., attention_mask=...) forward call does NOT
    automatically recompute RoPE position_ids from a left-padded
    attention_mask the way generate() does internally -- if position_ids
    is left as the default None here, every sequence's real tokens get
    position indices as if there were no left-padding at all, silently
    misaligning every rotary position embedding for any example whose
    prompt was shorter than the batch max. We compute position_ids
    explicitly from the concatenated attention_mask (cumulative sum minus
    1, clamped at the pad positions), matching exactly what
    generate() uses internally for left-padded batches.
    """
    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    cont_mask = (continuation_ids != pad_id).long()
    full_attention_mask = torch.cat([prompt_mask, cont_mask], dim=1)

    position_ids = full_attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(full_attention_mask == 0, 1)

    outputs = model(input_ids=full_ids, attention_mask=full_attention_mask,
                     position_ids=position_ids)
    logits = outputs.logits  # (B, Lp+Lc, V)

    Lp = prompt_ids.size(1)
    Lc = continuation_ids.size(1)
    # logits[:, t] predicts token at position t+1 (standard causal-LM shift).
    # Position Lp-1 predicts continuation_ids[:, 0]; position Lp+Lc-2
    # predicts continuation_ids[:, Lc-1].
    continuation_logits = logits[:, Lp - 1: Lp + Lc - 1, :]  # (B, Lc, V)
    log_probs = F.log_softmax(continuation_logits, dim=-1)
    mask = cont_mask.float()  # (B, Lc)
    safe_targets = continuation_ids.clamp(min=0)
    token_log_probs = log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=1)  # (B,)


# ─── CUSTOM TRAINER (Config C composite loss, causal-LM SCST) ───────────────
class CompositeLossTrainer(Trainer):
    """Causal-LM counterpart of lora_experiment_matrix.py's
    CompositeLossTrainer. Same self-critical policy-gradient objective
    (Rennie et al., 2017) and same leave-one-out advantage baseline; the
    sampling and log-prob steps are rewritten for decoder-only generation
    semantics (see module docstring items 6-7 above for what changed and
    why, including the deliberately-reproduced training-reward-vs-
    eval-metric text mismatch inherited from the mT5 script)."""

    def __init__(self, *args, use_composite: bool = False, tokenizer_ref=None,
                 lambda1: float = 0.3, num_samples: int = PG_NUM_SAMPLES,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_composite = use_composite
        self.tokenizer_ref = tokenizer_ref
        self.lambda1 = lambda1
        self.num_samples = num_samples

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_input_ids = inputs.pop("prompt_input_ids")
        prompt_attention_mask = inputs.pop("prompt_attention_mask")

        outputs = model(**inputs)
        ce_loss = outputs.loss

        if not self.use_composite:
            return (ce_loss, outputs) if return_outputs else ce_loss

        K = self.num_samples
        pad_id = self.tokenizer_ref.pad_token_id
        batch_size = prompt_input_ids.size(0)

        # ---- Step 1-2: sample K generations per example, score (no_grad) ----
        with torch.no_grad():
            rep_prompt_ids = prompt_input_ids.repeat_interleave(K, dim=0)
            rep_prompt_mask = prompt_attention_mask.repeat_interleave(K, dim=0)
            gen_out = model.generate(
                input_ids=rep_prompt_ids,
                attention_mask=rep_prompt_mask,
                max_new_tokens=MAX_ANSWER_LEN,
                do_sample=True,
                temperature=PG_SAMPLE_TEMPERATURE,
                top_p=PG_SAMPLE_TOP_P,
                renormalize_logits=True,
                logits_processor=LogitsProcessorList([_NaNSafeLogitsProcessor()]),
                pad_token_id=pad_id,
            )
            # gen_out includes the (left-padded) prompt followed by new
            # tokens -- slice off exactly the newly generated portion.
            continuation_ids = gen_out[:, rep_prompt_ids.size(1):]
            sampled_texts = [
                _truncate_at_newline(t) for t in
                self.tokenizer_ref.batch_decode(continuation_ids, skip_special_tokens=True)
            ]
            # KNOWN PARITY QUIRK (module docstring item 7): reward is scored
            # against the DECODED PROMPT text, matching the mT5 script's
            # existing behavior exactly, not against the bare passage.
            context_texts = self.tokenizer_ref.batch_decode(
                rep_prompt_ids, skip_special_tokens=True)
            rewards = [f_faith(t, c) for t, c in zip(sampled_texts, context_texts)]

        # ---- Step 3: leave-one-out baseline / advantage (detached) ----
        rewards_t = torch.tensor(rewards, device=prompt_input_ids.device,
                                  dtype=torch.float32).view(batch_size, K)
        if K > 1:
            sum_r = rewards_t.sum(dim=1, keepdim=True)
            baseline = (sum_r - rewards_t) / (K - 1)
        else:
            baseline = rewards_t.mean()
        advantages = (rewards_t - baseline).view(-1).detach()

        # ---- Step 4: differentiable log-prob of the sampled continuation ----
        seq_log_probs = _seq_log_probs_causal(
            model, rep_prompt_ids, rep_prompt_mask, continuation_ids, pad_id)

        # ---- Step 5: policy-gradient loss + combine with CE ----
        pg_loss = -(advantages * seq_log_probs).mean()
        loss = ce_loss + self.lambda1 * pg_loss
        return (loss, outputs) if return_outputs else loss


# ─── TRAIN + EVAL ONE RUN ────────────────────────────────────────────────────
def run_single_experiment(config_name: str, lora_variant: str, language: str,
                           smoke_test: bool = False, seed: int = SEED) -> dict:
    label = f"config={config_name} variant={lora_variant} lang={language} seed={seed}"
    print(f"\n{'='*70}\nRUN (Qwen3-0.6B-Base): {label}\n{'='*70}")
    t0 = time.time()
    model = None
    set_all_seeds(seed)

    try:
        dataset = load_qa_data(language, smoke_test=smoke_test)
        tokenizer = get_tokenizer()

        if smoke_test:
            n_train = min(SMOKE_DATA_CAP, len(dataset))
        else:
            n_train = min(TRAIN_SIZE, len(dataset))
        train_indices = list(range(n_train))
        eval_indices = list(range(n_train, len(dataset))) or list(range(n_train))

        if smoke_test:
            total_steps = 5
        else:
            steps_per_epoch = max(1, -(-n_train // TRAIN_BATCH_SIZE))
            total_steps = max(2, steps_per_epoch * TRAIN_EPOCHS)

        model = load_model_for_config(config_name, lora_variant, total_steps)

        if config_name != "A_frozen":
            # FIX vs. the mT5 script's smoke-test design: lora_experiment_
            # matrix.py's --smoke_test SKIPS real training entirely for
            # non-frozen configs ("validating model/adapter construction
            # only"), which means it never actually exercises the
            # composite-loss (SCST) training step -- exactly the code path
            # that had the silent, non-differentiable-loss bug for months
            # before an external readiness review caught it, not a smoke
            # test. Here, --smoke_test instead runs ONE real (tiny) training
            # step for every config, B/C/D included, at a scale small
            # enough to finish in well under a minute on a CPU-only laptop:
            # this is deliberately a stronger smoke test than the original
            # script's, because "does the composite loss path run and
            # produce a finite, backward()-able loss on 2-3 examples" is
            # exactly the class of bug (a broken gradient path) this
            # project has already been burned by once.
            n_smoke_train = min(len(train_indices), 3) if smoke_test else None
            smoke_batch = min(2, n_smoke_train) if smoke_test else None
            smoke_pg_samples = 2 if smoke_test else PG_NUM_SAMPLES

            train_examples = build_sft_dataset(
                dataset, train_indices[:n_smoke_train] if smoke_test else train_indices,
                language, tokenizer)
            collator = QwenSFTCollator(tokenizer)

            training_args = TrainingArguments(
                output_dir=str(OUTPUT_DIR / f"ckpt_{config_name}_{lora_variant}_{language}"),
                per_device_train_batch_size=(smoke_batch if smoke_test else TRAIN_BATCH_SIZE),
                num_train_epochs=(1 if smoke_test else TRAIN_EPOCHS),
                learning_rate=2e-4 if config_name in ("B_ce_lora", "C_composite_lora") else 5e-5,
                logging_steps=5,
                save_strategy="no",
                report_to=[],
                remove_unused_columns=False,
                bf16=(DEVICE == "cuda"),
                # Same memory-conscious defaults as the mT5 script: gradient
                # checkpointing for full-FT and for non-QLoRA Config C
                # (SCST's K=4-sample forward pass is the real memory driver
                # for Config C on a non-4-bit backbone, not the LoRA adapter
                # size -- see lora_experiment_matrix.py's note on this).
                gradient_checkpointing=(
                    (config_name == "D_full_ft"
                     or (config_name == "C_composite_lora" and lora_variant != "qlora"))
                    and not smoke_test  # keep the smoke path simple/fast
                ),
                optim=("adafactor" if config_name == "D_full_ft" else "adamw_torch"),
            )

            class _ListDataset(torch.utils.data.Dataset):
                def __init__(self, items):
                    self.items = items

                def __len__(self):
                    return len(self.items)

                def __getitem__(self, idx):
                    return self.items[idx]

            trainer = CompositeLossTrainer(
                model=model,
                args=training_args,
                train_dataset=_ListDataset(train_examples),
                data_collator=collator,
                use_composite=(config_name == "C_composite_lora"),
                tokenizer_ref=tokenizer,
                num_samples=smoke_pg_samples,
            )
            trainer.train()
            mode_label = "SMOKE-TEST mini-training (1 tiny step)" if smoke_test else \
                f"{training_args.num_train_epochs} epochs"
            print(f"  [TRAINING] completed {mode_label} "
                  f"({'composite loss' if config_name == 'C_composite_lora' else 'CE loss'})")

        # ---- Evaluation (single-example loop, matches mT5 script style) ----
        eval_slice = dataset.select(eval_indices)
        scores, em_scores, f1_scores, sample_log = [], [], [], []
        model.eval()
        for i, ex in enumerate(eval_slice):
            context, question, gold, options = extract_context_question_answer(ex, language)
            if not context or not question:
                continue
            prompt = build_prompt(question, context, options)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=MAX_PROMPT_LEN,
                               add_special_tokens=False).to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_ANSWER_LEN,
                    renormalize_logits=True,
                    logits_processor=LogitsProcessorList([_NaNSafeLogitsProcessor()]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_ids = out[0, inputs["input_ids"].size(1):]
            answer = _truncate_at_newline(
                tokenizer.decode(generated_ids, skip_special_tokens=True))
            scores.append(f_faith(answer, context))
            em_scores.append(exact_match(answer, gold))
            f1_scores.append(token_f1(answer, gold))
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
            "seed": seed, "backbone": "qwen3-0.6b-base",
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
            "seed": seed, "backbone": "qwen3-0.6b-base",
            "status": "FAIL", "mean_f_faith": None, "mean_em": None, "mean_f1": None,
            "n_eval": 0, "runtime_sec": round(elapsed, 1),
            "error": f"{type(e).__name__}: {e}", "samples": "",
        }

    finally:
        if model is not None:
            del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()


# ─── MAIN: BUILD THE RUN MATRIX (copied verbatim from lora_experiment_matrix.py) ──
def build_run_matrix():
    runs = []
    for lang in LANGUAGES:
        runs.append(("A_frozen", "none", lang))
        runs.append(("D_full_ft", "none", lang))
        for variant in LORA_VARIANTS:
            runs.append(("B_ce_lora", variant, lang))
            runs.append(("C_composite_lora", variant, lang))
    return runs   # 20 conditions, same design as lora_experiment_matrix.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--composite_only", action="store_true")
    parser.add_argument("--bc_only", action="store_true")
    parser.add_argument("--ad_only", action="store_true")
    parser.add_argument("--variant", type=str, default=None, choices=LORA_VARIANTS)
    parser.add_argument("--lang", type=str, default=None, choices=LANGUAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not (args.smoke_test or args.full):
        print("Specify --smoke_test (run now, on your Mac) "
              "or --full (run later, on lab server). Exiting.")
        return

    if args.full and DEVICE == "cpu":
        print("⚠ WARNING: --full requested but no GPU detected. Aborting.")
        return

    if args.full:
        gpu_compatible, gpu_msg = check_gpu_compatibility()
        print(gpu_msg)
        if not gpu_compatible:
            print("\nAborting --full run: incompatible torch/GPU combination "
                  "(see message above for the fix).")
            return

    seed_suffix = "" if args.seed == SEED else f"_seed{args.seed}"
    if args.smoke_test:
        checkpoint_name = "checkpoint_smoke_qwen.jsonl"
    elif args.ad_only:
        checkpoint_name = f"checkpoint_full_qwen_ad{seed_suffix}.jsonl"
    elif args.variant and args.lang:
        checkpoint_name = f"checkpoint_full_qwen_{args.variant}_{args.lang}{seed_suffix}.jsonl"
    else:
        checkpoint_name = f"checkpoint_full_qwen{seed_suffix}.jsonl"
    checkpoint_file = OUTPUT_DIR / checkpoint_name

    runs = build_run_matrix()
    if args.composite_only:
        runs = [r for r in runs if r[0] == "C_composite_lora"]
    if args.bc_only:
        runs = [r for r in runs if r[0] in ("B_ce_lora", "C_composite_lora")]
    if args.ad_only:
        runs = [r for r in runs if r[0] in ("A_frozen", "D_full_ft")]
    if args.variant:
        runs = [r for r in runs if r[1] == args.variant]
    if args.lang:
        runs = [r for r in runs if r[2] == args.lang]
    if args.variant and args.lang and not args.smoke_test:
        print(f"NOTE: sharded run -- this pod writes ONLY to "
              f"{checkpoint_file.name}. Merge shards with "
              f"`cat experiment_results_qwen/checkpoint_full_qwen_*.jsonl > "
              f"experiment_results_qwen/checkpoint_full_qwen.jsonl` once all "
              f"pods finish.")

    completed = load_completed_runs(checkpoint_file)
    if completed:
        skip_count = sum(1 for r in runs if _run_key(*r, args.seed) in completed)
        runs = [r for r in runs if _run_key(*r, args.seed) not in completed]
        print(f"Resuming from checkpoint ({checkpoint_file.name}): "
              f"{skip_count} runs already PASSED in a previous session, skipping them.")

    print(f"Total runs queued: {len(runs)} | seed: {args.seed} | backbone: "
          f"{BASE_MODEL} | mode: "
          f"{'SMOKE TEST (CPU, pipeline validation only)' if args.smoke_test else 'FULL (GPU)'}")

    for config_name, lora_variant, language in runs:
        result = run_single_experiment(config_name, lora_variant, language,
                                       smoke_test=args.smoke_test, seed=args.seed)
        append_checkpoint(checkpoint_file, result)
        if not args.smoke_test:
            git_commit_and_push(
                checkpoint_file,
                f"Qwen checkpoint: {config_name}/{lora_variant}/{language} "
                f"seed={args.seed} = {result['status']}",
            )

    all_results = load_all_checkpoint_results(checkpoint_file)
    df = pd.DataFrame(all_results)
    print("\n\n" + "=" * 70)
    print("FINAL REPORT (Qwen3-0.6B-Base)")
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

    out_path = OUTPUT_DIR / ("smoke_test_results_qwen.csv" if args.smoke_test
                             else "full_matrix_results_qwen.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
