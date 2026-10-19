export interface MetricSummary { mean: number; sd: number; n: number }
export interface SavedAnswer { answer: string; faith: number | null; faith_2nd: number | null; f1: number | null; em: number | null; n_words: number | null }
export interface HeroExample {
  id: number; question: string; context: string; gold: string; gold_option: string | null; options: string[] | null;
  answers: Record<"qwen3" | "t5gemma2", Record<"B" | "C", SavedAnswer>>;
}
export interface HeroData {
  meta: { title: string; models: string[]; languages: string[]; conditions: string[]; seeds: number[]; source: string; protocol: string };
  cells: Record<string, Record<string, Record<string, { seeds: number[]; metrics: Record<string, MetricSummary>; n_eval: number[] }>>>;
  contrasts: Record<string, Record<string, Record<string, Record<string, { mean: number; lo: number; hi: number; n_examples: number; seeds_positive: number; n_seeds: number }>>>>;
  length: { bins: Array<Record<string, string | number | null>>; contrasts: Array<Record<string, string | number | null>> };
  trainingDiagnostics: Record<string, Record<string, Array<{ quarter: number; inactive_share: number; mean_pg_abs: number; mean_reward: number; mean_ce: number; logged_steps: number }>>>;
  examples: Record<string, HeroExample[]>;
}
