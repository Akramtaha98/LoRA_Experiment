import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FirstRow, ResultsFile } from "../data/results";
import { Callout, Card, Section, Stat, StripPlot, type StripRow } from "../components/ui";
import { ADAPTER, LANG, ci, fmt, pval, signed } from "../lib/format";

type BB = "t5gemma2" | "mt5" | "qwen3";
const BB_INFO: Record<BB, { name: string; model: string; blurb: string }> = {
  t5gemma2: { name: "T5Gemma 2", model: "google/t5gemma-2-270m-270m", blurb: "Encoder–decoder, LoRA / DoRA / QLoRA, seeds 42, 123, 2026." },
  mt5: { name: "mT5-base (first study)", model: "google/mt5-base", blurb: "First, uncorrected mT5 study. Superseded by the corrected study for every mT5 claim." },
  qwen3: { name: "Qwen3-0.6B-Base", model: "Qwen/Qwen3-0.6B-Base", blurb: "Decoder-only, QLoRA / DoRA / AdaLoRA / VeRA, seeds 42, 123, 2026." },
};

export default function Backbones({ d }: { d: ResultsFile }) {
  const [bb, setBb] = useState<BB>("t5gemma2");
  const fs = d.first_study;
  const rows = fs.rows.filter((r: FirstRow) => r.backbone === bb && r.adapter !== "vera");
  const sum = fs.summary[bb]?.pooled;
  const strip: StripRow[] = rows.map((r) => ({
    label: `${LANG(r.language)} · ${ADAPTER(r.adapter)}`, values: r.delta_per_seed, mean: r.delta,
  }));
  const chart = rows.map((r) => ({ name: `${LANG(r.language).slice(0, 2)} · ${ADAPTER(r.adapter)}`, "B · cross-entropy": +r.B.toFixed(4), "C · NLI reward": +r.C.toFixed(4) }));
  const re = fs.reanalysis as unknown as Record<string, Record<string, { delta: number; lo: number; hi: number; p: number; p_welch: number; seeds_same_sign: number }>>;
  const planned = bb === "t5gemma2" ? re.t5gemma2 : bb === "qwen3" ? re.qwen3 : re.mt5_planned;
  const cross = re.contrasts as unknown as Record<string, { diff: number; lo: number; hi: number; p: number }>;
  const corrArabic = d.corrected.headline.arabic;

  return (
    <>
      <section className="hero small">
        <h1>Other backbones · original protocol</h1>
        <p>T5Gemma 2 and Qwen3 were run with the first-study training and evaluation code. They are shown for completeness and are not comparable with the corrected mT5 study.</p>
      </section>

      <Callout kind="warn" title="Not comparable with the corrected study">
        These runs used the original path (prompt rather than passage as the NLI premise during training, evaluation outside eval mode) and three seeds per cell.
        The corrected mT5 study found an Arabic gain of {signed(corrArabic.mean)} {ci(corrArabic.lo, corrArabic.hi)}; no corrected run of T5Gemma 2 or Qwen3 exists, so this page cannot say whether the reward would help them under the corrected protocol.
      </Callout>

      <div className="seg seg-wrap" role="tablist" aria-label="Backbone">
        {(Object.keys(BB_INFO) as BB[]).map((k) => (
          <button key={k} className={bb === k ? "seg-btn active" : "seg-btn"} onClick={() => setBb(k)}>{BB_INFO[k].name}</button>
        ))}
      </div>
      <p className="hint"><code>{BB_INFO[bb].model}</code> · {BB_INFO[bb].blurb}</p>

      {sum ? (
        <div className="stat-grid">
          <Stat label="Pooled C − B (NLI score)" value={signed(sum.mean, 4)} sub={`95% CI ${ci(sum.lo, sum.hi, 4)} · ${sum.n} seeds`} tone={bb === "mt5" ? "warn" : "neutral"} />
          <Stat label="Runs" value={String(fs.summary[bb].n_runs)} sub="B and C records, original protocol" />
          {bb === "t5gemma2" ? <Stat label="Heterogeneity across adapters" value={`p = ${fmt((re.t5g_heterogeneity as unknown as { p: number }).p, 2)}`} sub="chi-square, 2 df" /> : null}
        </div>
      ) : null}

      <Section title="NLI score by cell" kicker={`${BB_INFO[bb].name} · mean of 3 seeds`}>
        <Card>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
              <XAxis dataKey="name" stroke="var(--muted)" interval={0} angle={-20} textAnchor="end" height={60} />
              <YAxis stroke="var(--muted)" />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Bar isAnimationActive={false} dataKey="B · cross-entropy" fill="var(--muted)" radius={[4, 4, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="C · NLI reward" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </Section>

      <Section title="Per-seed effect (C − B)" kicker="Each dot is one seed">
        <Card><StripPlot rows={strip} /></Card>
      </Section>

      <Section title="All metrics" kicker="Means over 3 seeds">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Cell</th><th>NLI B</th><th>NLI C</th><th>Δ</th><th>EM B</th><th>EM C</th><th>F1 B</th><th>F1 C</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.language}${r.adapter}`}>
                  <td>{LANG(r.language)} · {ADAPTER(r.adapter)}</td>
                  <td className="mono">{fmt(r.B)}</td><td className="mono">{fmt(r.C)}</td><td className="mono">{signed(r.delta)}</td>
                  <td className="mono">{fmt(r.em_B)}</td><td className="mono">{fmt(r.em_C)}</td>
                  <td className="mono">{fmt(r.f1_B)}</td><td className="mono">{fmt(r.f1_C)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {planned ? (
        <Section title="Planned contrast per adapter" kicker="Pooled-variance t, as in the original analysis">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Adapter</th><th>Δ</th><th>95% CI</th><th>p</th><th>p (Welch)</th><th>Seeds same sign</th></tr></thead>
              <tbody>
                {Object.entries(planned).filter(([, v]) => typeof v === "object" && v && "delta" in v).map(([k, v]) => (
                  <tr key={k}>
                    <td>{ADAPTER(k)}</td><td className="mono">{signed(v.delta, 4)}</td><td className="mono">{ci(v.lo, v.hi, 4)}</td>
                    <td className="mono">{pval(v.p)}</td><td className="mono">{pval(v.p_welch)}</td><td>{v.seeds_same_sign}/3</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {bb === "t5gemma2" ? (
            <p className="hint">T5Gemma 2 shows no detectable reward effect on the NLI score in this protocol; the intervals are about ±0.014 wide, so effects below roughly 0.02 are not excluded.</p>
          ) : null}
        </Section>
      ) : null}

      <Section title="Cross-backbone contrasts" kicker="Original protocol, QLoRA and DoRA">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Contrast</th><th>Difference in Δ</th><th>95% CI</th><th>p</th></tr></thead>
            <tbody>
              {Object.entries(cross).map(([k, v]) => (
                <tr key={k}><td>{k.replace(/_/g, " ").replace("t5g", "T5Gemma 2").replace("qwen3", "Qwen3")}</td><td className="mono">{signed(v.diff, 4)}</td><td className="mono">{ci(v.lo, v.hi, 4)}</td><td className="mono">{pval(v.p)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hint">The mT5 differences here reflect the first-study artefact (checkpointed LoRA and DoRA runs scored in train mode) and should not be read as evidence about the reward.</p>
      </Section>
    </>
  );
}
