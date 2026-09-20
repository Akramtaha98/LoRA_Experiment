import type { BestModel, ModelScores } from "../lib/scoring";

export type DatasetLanguage = "arabic" | "malay";

export type DatasetExample = {
  id: string;
  language: DatasetLanguage;
  example_index: number;
  curation_tag: string;
  context: string;
  question: string;
  reference: string;
  mt5: ModelScores;
  qwen: ModelScores;
  best_model: BestModel;
  source: {
    mt5_file: string;
    qwen_file: string;
    mt5_config: string;
    mt5_lora_variant: string;
    qwen_config: string;
    qwen_lora_variant: string;
    seed: number;
  };
};

export type DatasetFile = {
  version: number;
  description: string;
  scoring_rule: {
    order: string[];
    description: string;
  };
  sources: Record<string, unknown>;
  examples: DatasetExample[];
};

let cache: DatasetFile | null = null;

export async function loadDatasetExamples(): Promise<DatasetFile> {
  if (cache) return cache;
  const response = await fetch("/data/dataset-examples.json");
  if (!response.ok) {
    throw new Error(`Failed to load dataset examples (${response.status})`);
  }
  cache = (await response.json()) as DatasetFile;
  return cache;
}
