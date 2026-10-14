"""lora_nli: NLI-faithfulness-reward LoRA study (mT5-base; Arabic XQuAD, Malay Belebele).

Modules
  matrix              training/evaluation pipeline (set LORA_V2=1 for the corrected path)
  qwen_matrix         Qwen3-0.6B-Base variant (original protocol)
  t5gemma2_matrix     T5Gemma 2 variant (original protocol)
  analyze             pre-specified seed-paired analysis of the corrected study
  independent_eval    independent NLI scorer (xlm-roberta-large-xnli)
  human_eval          human-validation package builder and scorer
"""
__version__ = "2.0.0"
