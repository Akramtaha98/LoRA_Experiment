import { useEffect, useState } from "react";
import { loadResults, type Cell, type ResultsFile } from "../data/results";

export function useResults() {
  const [data, setData] = useState<ResultsFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let off = false;
    loadResults().then((d) => !off && setData(d)).catch((e: unknown) => !off && setError(e instanceof Error ? e.message : "Failed to load results"));
    return () => { off = true; };
  }, []);
  return { data, error };
}

/** Mean of a metric over the three adapters for one condition x language (equal seeds per cell). */
export function condMean(cells: Cell[], label: string, language: string, metric: "mean_f_faith" | "mean_em" | "mean_f1"): number {
  const c = cells.filter((x) => x.label === label && x.language === language);
  return c.reduce((s, x) => s + x[metric], 0) / Math.max(1, c.length);
}
