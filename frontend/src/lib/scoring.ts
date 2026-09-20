export type ModelScores = {
  answer: string;
  faithfulness: number;
  exact_match: number;
  f1: number;
  latency_ms?: number;
};

export type BestModel = "mt5" | "qwen" | "tie";

/** Higher faithfulness first, then EM, then F1. Equal on all three → tie. */
export function pickBestModel(
  mt5: Pick<ModelScores, "faithfulness" | "exact_match" | "f1">,
  qwen: Pick<ModelScores, "faithfulness" | "exact_match" | "f1">,
): BestModel {
  const mt = [mt5.faithfulness, mt5.exact_match, mt5.f1] as const;
  const qw = [qwen.faithfulness, qwen.exact_match, qwen.f1] as const;
  for (let i = 0; i < 3; i++) {
    if (mt[i] > qw[i]) return "mt5";
    if (qw[i] > mt[i]) return "qwen";
  }
  return "tie";
}

export const SCORING_RULE_LABEL =
  "Best model: higher faithfulness, then Exact Match, then Token F1. Equal on all three → tie.";
