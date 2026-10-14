import { useMemo, useState } from "react";
import type { ResultsFile, SeedRow } from "../data/results";
import { BarList, Callout, Card, Section, StripPlot, type StripRow } from "../components/ui";
import { ADAPTER, LABELS, LANG, ci, fmt, pval, signed } from "../lib/format";

type Metric = "mean_f_faith" | "mean_em" | "mean_f1";
const METRICS: { id: Metric; label: string }[] = [
  { id: "mean_f_faith", label: "NLI score (F_faith)" }, { id: "mean_em", label: "Exact match" }, { id: "mean_f1", label: "Token F1" },
];
const ADAPTERS = ["lora", "dora", "qlora"] as const;

function tci(x: number[]) {
  const n = x.length, m = x.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1));
  const t = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306][n - 1] ?? 2.0; // t(0.975, n-1) for n <= 9
  const h = (t * sd) / Math.sqrt(n);
  return { mean: m, lo: m - h, hi: m + h };
}

export default function Corrected({ d }: { d: ResultsFile }) {
  const c = d.corrected;
  const [metric, setMetric] = useState<Metric>("mean_f_faith");
  const [lang, setLang] = useState<"arabic" | "malay">("arabic");

  const rows: StripRow[] = useMemo(() => {
    const out: StripRow[] = [];
    for (const lg of ["arabic", "malay"]) for (const ad of ADAPTERS) {
      const sr = c.per_seed.filter((r: SeedRow) => r.language === lg && r.adapter === ad);
      const v = sr.map((r) => (r[`${metric}|C`] as number) - (r[`${metric}|B`] as number)).filter((x) => Number.isFinite(x));
      if (v.length < 2) continue;
      const t = tci(v);
      out.push({ label: `${LANG(lg)} · ${ADAPTER(ad)}`, values: v, mean: t.mean, lo: t.lo, hi: t.hi });
    }
    return out;
  }, [c.per_seed, metric]);

  const cells = c.cells.filter((x) => x.language === lang && ["B", "C_l0", "C_shuf", "C"].includes(x.label));
  const get = (ad: string, lab: string) => cells.find((x) => x.adapter === ad && x.label === lab);

  const primary = c.contrasts.filter((r) => r.scope === "pooled(6 cells)" && r.metric === "mean_f_faith");
  const secondary = c.contrasts.filter((r) => r.contrast === "C-B" && r.metric !== "mean_f_faith" && r.scope === "pooled(6 cells)");
  const jb = c.joint_bootstrap as unknown as Record<string, Record<string, { point: number; ci95: [number, number] }>>;
  const legacy = c.diagnostics.legacy_eval_state_minus_headline_by_label;

  return (
    <>
      <section className="hero small">
        <h1>Corrected study · mT5-base</h1>
        <p>Eight pre-specified seeds per cell (42, 123, 2026, 7, 11, 13, 17, 19), seed-paired t intervals with 7 degrees of freedom, equivalence margin ±{d.meta.margin}.</p>
      </section>

      <Section title="Per-seed effect of the reward (C − B)" kicker="Each dot is one seed; bar = mean; whisker = 95% CI">
        <div className="seg" role="tablist" aria-label="Metric">
          {METRICS.map((m) => (
            <button key={m.id} className={metric === m.id ? "seg-btn active" : "seg-btn"} onClick={() => setMetric(m.id)}>{m.label}</button>
          ))}
        </div>
        <Card>
          <StripPlot rows={rows} unit={`C − B (${METRICS.find((m) => m.id === metric)?.label})`} />
          <p className="hint">All eight seeds agree in sign for the NLI score in every Arabic cell; the Malay cells straddle zero. Exact match and F1 move the other way.</p>
        </Card>
      </Section>

      <Section title="Condition means by adapter" kicker="Mean ± SD over 8 seeds">
        <div className="seg" role="tablist" aria-label="Language">
          {(["arabic", "malay"] as const).map((l) => (
            <button key={l} className={lang === l ? "seg-btn active" : "seg-btn"} onClick={() => setLang(l)}>{LANG(l)}</button>
          ))}
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Adapter</th>{["B", "C_l0", "C_shuf", "C"].map((k) => <th key={k}>{LABELS[k]}</th>)}</tr></thead>
            <tbody>
              {ADAPTERS.map((ad) => (
                <tr key={ad}>
                  <td>{ADAPTER(ad)}</td>
                  {["B", "C_l0", "C_shuf", "C"].map((k) => {
                    const x = get(ad, k);
                    const sd = x ? (x[`${metric}_sd` as "mean_f_faith_sd"] as number | null) : null;
                    return <td key={k} className="mono">{x ? fmt(x[metric], 3) : "n/a"}{sd != null ? <em> ±{sd.toFixed(3)}</em> : null}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Pre-specified contrasts" kicker="NLI score, seed-paired">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Contrast</th><th>Δ</th><th>95% CI</th><th>p</th><th>Within ±{d.meta.margin}?</th><th>Seeds positive</th></tr></thead>
            <tbody>
              {primary.map((r) => (
                <tr key={r.contrast}>
                  <td>{r.contrast.replace("C_l0", "λ₁=0").replace("C_shuf", "shuffled")}</td>
                  <td className="mono">{signed(r.mean, 4)}</td>
                  <td className="mono">{ci(r.ci95_lo, r.ci95_hi, 4)}</td>
                  <td className="mono">{pval(r.p)}</td>
                  <td>{r.ci90_lo == null ? "—" : r.equivalent_within_margin ? "equivalent (90% CI)" : "not shown"}</td>
                  <td>{r.cells_positive ?? "—"}</td>
                </tr>
              ))}
              {secondary.map((r) => (
                <tr key={r.metric}>
                  <td>C − B · {r.metric.replace("mean_", "").replace("_", " ")}</td>
                  <td className="mono">{signed(r.mean, 4)}</td>
                  <td className="mono">{ci(r.ci95_lo, r.ci95_hi, 4)}</td>
                  <td className="mono">{pval(r.p)}</td><td>—</td><td>{r.cells_positive ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hint">The shuffled and zero-weight controls differ from B by about −0.005; they fall inside the ±0.01 margin, so they are treated as equivalent to cross-entropy training. The Malay exact-match difference (−0.0156) is exploratory and uncorrected.</p>
      </Section>

      <Section title="Item and passage bootstrap" kicker="Items resampled jointly across adapters; passages as clusters">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Scope</th><th>Item bootstrap</th><th>Passage-cluster bootstrap</th></tr></thead>
            <tbody>
              {["pooled", "arabic", "malay"].map((s) => {
                const a = jb["C-B|f_faith|item"]?.[s], b = jb["C-B|f_faith|passage"]?.[s];
                return (
                  <tr key={s}>
                    <td>{s === "pooled" ? "Pooled" : LANG(s)}</td>
                    <td className="mono">{a ? `${signed(a.point)} ${ci(a.ci95[0], a.ci95[1])}` : "n/a"}</td>
                    <td className="mono">{b ? `${signed(b.point)} ${ci(b.ci95[0], b.ci95[1])}` : "n/a"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Checkpointing and the evaluation-state artefact" kicker="Why the first study was wrong about LoRA and DoRA">
        <div className="two">
          <Card title="Checkpointing is neutral in eval mode" note="Gradient checkpointing on vs off, same seeds, evaluation in eval mode.">
            <div className="kv"><span>B with checkpointing − B</span><strong className="mono">{signed(c.headline.ckpt_B.mean, 4)}</strong></div>
            <div className="kv"><span>C with checkpointing − C</span><strong className="mono">{signed(c.headline.ckpt_C.mean, 4)}</strong></div>
          </Card>
          <Card title="Scoring a checkpointed model left in train mode" note="Change in NLI score when the same adapter is scored in the legacy (train-mode) state.">
            <BarList
              items={Object.entries(legacy).map(([k, v]) => ({ label: LABELS[k] ?? k, value: Math.abs(v), sub: ` (${signed(v, 3)})`, tone: Math.abs(v) > 0.1 ? "bad" : "" }))}
              max={0.55}
            />
          </Card>
        </div>
        <Callout kind="info" title="Mechanism">
          With checkpointing enabled, T5 generate() in train mode decodes from the last token only. In the first study this lowered the score of checkpointed runs
          (LoRA and DoRA); zero-weight and shuffled-reward controls showed it was not caused by the reward.
        </Callout>
      </Section>
    </>
  );
}
