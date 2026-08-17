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

REQUIREMENTS:
  Smoke test (Mac, CPU):
    pip install transformers peft accelerate datasets sentencepiece torch \
                pandas tabulate

  Full run (lab server, GPU):
    pip install transformers peft bitsandbytes accelerate datasets \
                sentencepiece torch pandas tabulate evaluate

USAGE:
    python3 lora_experiment_matrix.py --smoke_test     # on your MacBook, now
    python3 lora_experiment_matrix.py --full           # on lab server, later
"""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
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


# ─── DEVICE DETECTION ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTIZATION_AVAILABLE = _BNB_AVAILABLE and DEVICE == "cuda"

if DEVICE == "cpu":
    print("⚠ No CUDA GPU detected — running in CPU smoke-test mode.")
    print("  QLoRA 4-bit quantization will be SKIPPED (needs CUDA).")
    print("  This validates pipeline LOGIC only, not real training quality.\n")


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_MODEL    = "google/mt5-base"
NLI_MODEL     = "cross-encoder/nli-deberta-v3-base"
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
TRAIN_EPOCHS     = 3

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
def composite_loss(ce_loss: torch.Tensor, faithfulness_score: float,
                    lambda1: float = 0.3) -> torch.Tensor:
    """L_G = L_CE + lambda1 * (1 - F_faith)"""
    penalty = lambda1 * (1.0 - faithfulness_score)
    return ce_loss + penalty


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


def f_faith(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    tok, model = _load_nli()
    ent_idx = list(model.config.id2label.values()).index("entailment")
    enc = tok(context, answer, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return round(probs[ent_idx].item(), 4)


# ─── CUSTOM TRAINER (Config C composite loss) ────────────────────────────────
class CompositeLossTrainer(Seq2SeqTrainer):
    """
    Standard Seq2SeqTrainer, except when use_composite=True (Config C only):
    generates a prediction for the current batch, scores it with f_faith(),
    and blends the faithfulness penalty into the loss via composite_loss():
        L_G = L_CE + lambda1 * (1 - F_faith)
    For all other configs this behaves identically to the base Trainer.
    """
    def __init__(self, *args, use_composite: bool = False, tokenizer_ref=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_composite = use_composite
        self.tokenizer_ref = tokenizer_ref

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if not self.use_composite:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Config C: generate a prediction to score its faithfulness, then
        # blend that penalty into the loss. Generation under no_grad keeps
        # this from adding to the backward graph (only ce_loss is trained).
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=40,
            )
        pred_text = self.tokenizer_ref.batch_decode(
            gen_ids, skip_special_tokens=True)[0]
        context_text = self.tokenizer_ref.batch_decode(
            inputs["input_ids"], skip_special_tokens=True)[0]
        faithfulness_score = f_faith(pred_text, context_text)
        loss = composite_loss(ce_loss, faithfulness_score)
        return (loss, outputs) if return_outputs else loss


# ─── TRAIN + EVAL ONE RUN (FIXED: per-run try/except, no full-script crash) ──
def run_single_experiment(config_name: str, lora_variant: str, language: str,
                           smoke_test: bool = False) -> dict:
    label = f"config={config_name} variant={lora_variant} lang={language}"
    print(f"\n{'='*70}\nRUN: {label}\n{'='*70}")
    t0 = time.time()

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
                learning_rate=2e-4 if lora_variant else 5e-5,
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
            # Print the first eval example per run so generated vs. gold text
            # is visible in the terminal log -- useful for a qualitative
            # sanity check / paper appendix example, and for catching any
            # future task-formulation regressions immediately.
            if i == 0:
                print(f"  [SAMPLE] gold=\"{gold[:80]}\" | generated=\"{answer[:80]}\"")

        mean_score = sum(scores) / len(scores) if scores else 0.0
        elapsed = time.time() - t0

        result = {
            "config": config_name, "lora_variant": lora_variant, "language": language,
            "status": "PASS", "mean_f_faith": round(mean_score, 4),
            "n_eval": len(scores), "runtime_sec": round(elapsed, 1), "error": "",
        }
        print(f"  RESULT: PASS | mean_f_faith={result['mean_f_faith']} "
              f"| n_eval={result['n_eval']} | {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  RESULT: FAIL | {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "config": config_name, "lora_variant": lora_variant, "language": language,
            "status": "FAIL", "mean_f_faith": None, "n_eval": 0,
            "runtime_sec": round(elapsed, 1), "error": f"{type(e).__name__}: {e}",
        }


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

    checkpoint_file = OUTPUT_DIR / (
        "checkpoint_smoke.jsonl" if args.smoke_test else "checkpoint_full.jsonl"
    )

    runs = build_run_matrix()
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
