export type ModelScores = {
  answer: string;
  faithfulness: number;
  exact_match: number;
  f1: number;
  latency_ms?: number;
};

export type BestModel = "mt5" | "qwen" | "tie" | "incomplete";

/**
 * Honest best match to the gold reference:
 * 1) higher Exact Match
 * 2) then higher Token F1
 * 3) faithfulness only if EM/F1 are tied and at least one model has F1 > 0
 *
 * If both models have EM=0 and F1=0, returns "tie" (neither matched gold).
 * Missing either prediction => "incomplete".
 */
export function pickBestModel(
  mt5: Pick<ModelScores, "faithfulness" | "exact_match" | "f1"> | null | undefined,
  qwen: Pick<ModelScores, "faithfulness" | "exact_match" | "f1"> | null | undefined,
): BestModel {
  if (!mt5 || !qwen) return "incomplete";

  if (mt5.exact_match !== qwen.exact_match) {
    return mt5.exact_match > qwen.exact_match ? "mt5" : "qwen";
  }
  if (mt5.f1 !== qwen.f1) {
    return mt5.f1 > qwen.f1 ? "mt5" : "qwen";
  }
  // Both wrong on EM and F1: do not let faithfulness invent a "correct" winner.
  if (mt5.exact_match === 0 && mt5.f1 === 0) {
    return "tie";
  }
  if (mt5.faithfulness !== qwen.faithfulness) {
    return mt5.faithfulness > qwen.faithfulness ? "mt5" : "qwen";
  }
  return "tie";
}

export function neitherMatchedGold(
  mt5: Pick<ModelScores, "exact_match" | "f1"> | null | undefined,
  qwen: Pick<ModelScores, "exact_match" | "f1"> | null | undefined,
): boolean {
  return Boolean(
    mt5 &&
      qwen &&
      mt5.exact_match === 0 &&
      qwen.exact_match === 0 &&
      mt5.f1 === 0 &&
      qwen.f1 === 0,
  );
}

export const SCORING_RULE_LABEL =
  "Best vs gold: higher Exact Match, then Token F1. Faithfulness is only a tie-break when at least one model has F1 > 0. If both miss gold (EM=0, F1=0), neither is Best.";
