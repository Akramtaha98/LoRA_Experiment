# Human validation of the NLI faithfulness score

`human_validation_labels.csv`: 350 mT5-base generations (175 Arabic, 175 Malay; seed 42; QLoRA and DoRA; Config B and C).
Annotators saw only passage, question and generated answer (condition, variant and scores hidden). Columns: `q1_rater1/2` = faithfulness
label from the passage (2 supported, 1 partly, 0 unsupported, X empty/unreadable); `q1_final` = agreed label, or adjudicated label where the
raters disagreed; `q2_*` = correctness against the reference. Rater identities are not recorded. Reproduce the paper's tables with
`python analyze_human_validation.py human_validation_labels.csv`.
