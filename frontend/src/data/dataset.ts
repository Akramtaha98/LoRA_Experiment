import type { BestModel, ModelScores } from "../lib/scoring";

export type DatasetLanguage = "arabic" | "malay";

export type ConfigId =
  | "A_frozen"
  | "B_ce_lora"
  | "C_composite_lora"
  | "D_full_ft";

export type ConfigPrediction = {
  mt5: (ModelScores & { lora_variant?: string }) | null;
  qwen: (ModelScores & { lora_variant?: string }) | null;
  best_model: BestModel;
};

export type DatasetExample = {
  id: string;
  language: DatasetLanguage;
  example_index: number;
  context: string;
  question: string;
  reference: string;
  configs: Record<ConfigId, ConfigPrediction>;
};

export type ConfigDef = {
  id: ConfigId;
  label: string;
  description: string;
};

export type AggregateMetrics = {
  mean_faithfulness: number;
  mean_em: number;
  mean_f1: number;
  n_eval: number;
  lora_variant: string;
  seed: number;
};

export type DatasetFile = {
  version: number;
  description: string;
  notes?: string[];
  scoring_rule: {
    order: string[];
    description: string;
  };
  config_defs: ConfigDef[];
  aggregates: Record<
    string,
    Record<string, Partial<Record<"mt5" | "qwen", AggregateMetrics>>>
  >;
  examples: DatasetExample[];
};

let cache: DatasetFile | null = null;

export async function loadDatasetExamples(): Promise<DatasetFile> {
  if (cache) return cache;
  const response = await fetch(
    `${import.meta.env.BASE_URL}data/dataset-examples.json`,
  );
  if (!response.ok) {
    throw new Error(`Failed to load dataset examples (${response.status})`);
  }
  cache = (await response.json()) as DatasetFile;
  return cache;
}
