export type ModelScores = {
  answer: string;
  faithfulness: number;
  exact_match: number;
  f1: number;
  latency_ms?: number;
};

export type BestModel = "mt5" | "qwen" | "tie";

/**
 * Best answer vs the reference: higher Exact Match first, then Token F1,
 * then faithfulness. Equal on all three is a tie.
 *
 * Correctness is prioritized so a wrong high-faithfulness answer cannot beat
 * an exact match to the gold reference.
 */
export function pickBestModel(
  mt5: Pick<ModelScores, "faithfulness" | "exact_match" | "f1">,
  qwen: Pick<ModelScores, "faithfulness" | "exact_match" | "f1">,
): BestModel {
  const mt = [mt5.exact_match, mt5.f1, mt5.faithfulness] as const;
  const qw = [qwen.exact_match, qwen.f1, qwen.faithfulness] as const;
  for (let i = 0; i < 3; i++) {
    if (mt[i] > qw[i]) return "mt5";
    if (qw[i] > mt[i]) return "qwen";
  }
  return "tie";
}

export const SCORING_RULE_LABEL =
  "Best answer: higher Exact Match vs reference, then Token F1, then faithfulness. Equal on all three is a tie.";
