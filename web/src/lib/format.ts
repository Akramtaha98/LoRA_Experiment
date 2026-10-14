export const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(d);
export const signed = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
export const ci = (lo: number | null | undefined, hi: number | null | undefined, d = 3) =>
  lo === null || lo === undefined || hi === null || hi === undefined ? "" : `[${signed(lo, d)}, ${signed(hi, d)}]`;
export const pval = (p: number | null | undefined) =>
  p === null || p === undefined ? "n/a" : p < 0.001 ? "<0.001" : p.toFixed(3);
export const LABELS: Record<string, string> = {
  B: "B · cross-entropy", C: "C · NLI reward", C_l0: "C · λ₁=0 (control)", C_shuf: "C · shuffled reward (control)",
  B_ckpt: "B · checkpointing on", C_ckpt: "C · checkpointing on",
};
export const ADAPTER = (a: string) => (a === "qlora" ? "QLoRA" : a === "dora" ? "DoRA" : a === "lora" ? "LoRA" : a === "adalora" ? "AdaLoRA" : a === "vera" ? "VeRA" : a);
export const LANG = (l: string) => (l === "arabic" ? "Arabic" : l === "malay" ? "Malay" : l);
