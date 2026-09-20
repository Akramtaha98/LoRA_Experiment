export type ModelScores = {
  answer: string;
  faithfulness: number;
  exact_match: number;
  f1: number;
  latency_ms?: number;
};

export type BestModel = "mt5" | "qwen" | "tie" | "incomplete";

/**
 * Best answer vs the reference: higher Exact Match first, then Token F1,
 * then faithfulness. Equal on all three is a tie.
 *
 * Requires both model predictions. If either side is missing, returns
 * "incomplete" so a solo logged answer is never crowned Best.
 */
export function pickBestModel(
  mt5: Pick<ModelScores, "faithfulness" | "exact_match" | "f1"> | null | undefined,
  qwen: Pick<ModelScores, "faithfulness" | "exact_match" | "f1"> | null | undefined,
): BestModel {
  if (!mt5 || !qwen) return "incomplete";
  const mt = [mt5.exact_match, mt5.f1, mt5.faithfulness] as const;
  const qw = [qwen.exact_match, qwen.f1, qwen.faithfulness] as const;
  for (let i = 0; i < 3; i++) {
    if (mt[i] > qw[i]) return "mt5";
    if (qw[i] > mt[i]) return "qwen";
  }
  return "tie";
}

export const SCORING_RULE_LABEL =
  "Best answer: higher Exact Match vs reference, then Token F1, then faithfulness. Compared only when both models are logged.";
