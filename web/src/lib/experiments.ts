export type Metric = "em" | "f1" | "nli";
export interface Experiment {
  id: string;
  protocol: "original" | "corrected";
  model: string;
  config: string;
  adapter: string;
  language: string;
  seeds: number[];
  nRuns: number;
  nEval: number[];
  em: number;
  f1: number;
  nli: number;
  sampleCount: number;
  exampleFile: string | null;
  sources: { path: string; line?: number; seed?: number }[];
}
export interface Sample {
  seed: number;
  index: number;
  question: string | null;
  context: string | null;
  gold: string;
  generated: string;
  em: number;
  f1: number;
  source: string;
  line: number;
}
export const MODEL_NAMES: Record<string, string> = {
  mt5: "mT5-base",
  qwen3: "Qwen3-0.6B-Base",
  t5gemma2: "T5Gemma 2 (270M)",
};
export const ADAPTER_NAMES: Record<string, string> = {
  none: "No adapter",
  lora: "LoRA",
  qlora: "QLoRA",
  adalora: "AdaLoRA",
  dora: "DoRA",
  vera: "VeRA",
};
export const CONFIG_NAMES: Record<string, string> = {
  A: "A · Frozen model",
  B: "B · Standard training",
  C: "C · NLI-reward training",
  D: "D · Full fine-tuning",
  C_l0: "Control · Zero reward weight",
  C_shuf: "Control · Shuffled reward",
  B_ckpt: "Check · B with checkpointing",
  C_ckpt: "Check · C with checkpointing",
};
export const PROTOCOL_NAMES = {
  corrected: "Corrected study · mT5",
  original: "Original study · historical",
};
export function filterExperiments(
  rows: Experiment[],
  filters: Record<string, string>,
) {
  return rows.filter((row) =>
    Object.entries(filters).every(
      ([key, value]) =>
        value === "all" || row[key as keyof Experiment] === value,
    ),
  );
}
export function compatible(a: Experiment, b: Experiment) {
  return (
    a.seeds.length > 0 &&
    b.seeds.length > 0 &&
    a.nEval.length > 0 &&
    b.nEval.length > 0 &&
    a.protocol === b.protocol &&
    a.language === b.language &&
    a.nRuns === b.nRuns &&
    a.seeds.join(",") === b.seeds.join(",") &&
    a.nEval.join(",") === b.nEval.join(",")
  );
}
export function matchedSamples(a: Sample[], b: Sample[]) {
  // Only full question + passage identity can establish a shared item. Gold or position alone cannot.
  const key = (s: Sample) =>
    JSON.stringify([s.seed, s.question, s.context, s.gold]);
  const counts = (rows: Sample[]) => {
    const found = new Map<string, number>();
    for (const s of rows) found.set(key(s), (found.get(key(s)) || 0) + 1);
    return found;
  };
  const ca = counts(a),
    cb = counts(b);
  const lookup = new Map(
    b.filter((s) => s.question && s.context).map((s) => [key(s), s]),
  );
  return a
    .filter(
      (s) =>
        s.question &&
        s.context &&
        lookup.has(key(s)) &&
        ca.get(key(s)) === 1 &&
        cb.get(key(s)) === 1,
    )
    .map((s) => ({ before: s, after: lookup.get(key(s))! }));
}
export function score(value: number, metric: Metric = "em") {
  return metric === "nli" ? value.toFixed(3) : `${(value * 100).toFixed(2)}%`;
}
export function rankExperiments(rows: Experiment[], metric: Metric) {
  return [...rows].sort(
    (a, b) => b[metric] - a[metric] || a.id.localeCompare(b.id),
  );
}
const sampleCache = new Map<string, Promise<Sample[]>>();
export function loadSamples(row: Experiment): Promise<Sample[]> {
  if (!row.exampleFile) return Promise.resolve([]);
  const file = row.exampleFile;
  if (!sampleCache.has(file))
    sampleCache.set(
      file,
      fetch(`${import.meta.env.BASE_URL}data/${file}`)
        .then((r) => {
          if (!r.ok) throw new Error("Saved examples could not be loaded.");
          return r.json();
        })
        .catch((e) => {
          sampleCache.delete(file);
          throw e;
        }),
    );
  return sampleCache.get(file)!;
}
