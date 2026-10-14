import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ResultsFile } from "../data/results";
import { Callout, Card, Section, Stat } from "../components/ui";
import { ci, fmt, signed } from "../lib/format";
import { condMean } from "../lib/useResults";

const COND = [
  ["B", "Cross-entropy (B)", "var(--muted)"],
  ["C_l0", "Reward weight 0 (control)", "var(--accent-2)"],
  ["C_shuf", "Shuffled reward (control)", "var(--gold)"],
  ["C", "NLI reward (C)", "var(--accent)"],
] as const;

export default function Overview({ d, go }: { d: ResultsFile; go: (p: string) => void }) {
  const c = d.corrected, h = c.headline;
  const ar = c.answer_length["arabic|B"], arC = c.answer_length["arabic|C"];
  const indep = d.independent as Record<string, { mean: number; ci95: number[] }>;
  const kappa = d.human.summary.kappa_supported_quadratic;
  const chart = (metric: "mean_f_faith" | "mean_em") =>
    ["arabic", "malay"].map((lang) => ({
      language: lang === "arabic" ? "Arabic" : "Malay",
      ...Object.fromEntries(COND.map(([k]) => [k, +condMean(c.cells, k, lang, metric).toFixed(4)])),
    }));
  return (
    <>
      <section className="hero">
        <h1>An NLI-faithfulness reward raises the score it optimises — in Arabic, and not much else</h1>
        <p>
          Corrected, pre-specified study of a leave-one-out REINFORCE objective with an NLI entailment reward for LoRA, QLoRA and
          DoRA on mT5-base: {c.n_runs} run records, {c.seeds.length} seeds per cell, Arabic XQuAD and Malay Belebele. All figures
          below are read from the released run records.
        </p>
      </section>

      <div className="stat-grid">
        <Stat label="Arabic · NLI score C − B" value={signed(h.arabic.mean)} sub={`95% CI ${ci(h.arabic.lo, h.arabic.hi)}`} tone="good" />
        <Stat label="Malay · NLI score C − B" value={signed(h.malay.mean)} sub={`95% CI ${ci(h.malay.lo, h.malay.hi)}`} tone="neutral" />
        <Stat label="Arabic · exact match" value={`${fmt(condMean(c.cells, "B", "arabic", "mean_em"), 3)} → ${fmt(condMean(c.cells, "C", "arabic", "mean_em"), 3)}`} sub="B → C, mean of 3 adapters" tone="warn" />
        <Stat label="Arabic · answer length" value={`${ar.mean_words.toFixed(1)} → ${arC.mean_words.toFixed(1)} words`} sub={`${(arC.share_gt_3x_gold * 100).toFixed(1)}% longer than 3× gold`} tone="warn" />
        <Stat label="Independent NLI scorer" value={signed(indep["f_indep:C-B"]?.mean)} sub={`xlm-roberta-large-xnli · ${ci(indep["f_indep:C-B"]?.ci95[0], indep["f_indep:C-B"]?.ci95[1])}`} tone="good" />
        <Stat label="Human agreement" value={`κ = ${fmt(kappa, 2)}`} sub="two annotators, 60 items per language" tone="neutral" />
      </div>

      <Callout kind="warn" title="How to read these results">
        The NLI score rises, but exact match falls, answers become about five times longer, and blinded human ratings do not show a
        gain (one annotator rated reward-trained answers as less supported, the other saw no difference). This is consistent with
        over-optimisation of the score; it is not proof that the answers are less faithful, and the human evidence is limited.
      </Callout>

      <Section title="The effect is specific to the reward" kicker="Mean over LoRA, DoRA and QLoRA · 8 seeds">
        <div className="two">
          <Card title="NLI entailment score (F_faith)" note="Controls stay within 0.01 of cross-entropy training.">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chart("mean_f_faith")} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                <XAxis dataKey="language" stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" domain={[0, 1]} />
                <Tooltip />
                <Legend />
                {COND.map(([k, name, fill]) => <Bar isAnimationActive={false} key={k} dataKey={k} name={name} fill={fill} radius={[4, 4, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Exact match" note="Exact match falls to almost zero in Arabic under the reward.">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chart("mean_em")} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                <XAxis dataKey="language" stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" />
                <Tooltip />
                <Legend />
                {COND.map(([k, name, fill]) => <Bar isAnimationActive={false} key={k} dataKey={k} name={name} fill={fill} radius={[4, 4, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </Section>

      <Section title="What is on this site" kicker="Study status">
        <div className="status-grid">
          <button className="status-card main" onClick={() => go("corrected")}>
            <span className="pill">Main result</span>
            <h3>Corrected study · mT5-base</h3>
            <p>Pre-specified, seed-paired, controls and checkpointing factorial. Headline numbers above.</p>
          </button>
          <button className="status-card" onClick={() => go("backbones")}>
            <span className="pill warn">Original protocol</span>
            <h3>T5Gemma 2 · Qwen3 · first mT5 study</h3>
            <p>Three seeds each, run before the training and evaluation fixes. Not comparable with the corrected study.</p>
          </button>
          <button className="status-card" onClick={() => go("validation")}>
            <span className="pill">Validation</span>
            <h3>Independent scorer, humans, length</h3>
            <p>Does the gain survive another NLI model, human judgement and an answer-length adjustment?</p>
          </button>
          <button className="status-card" onClick={() => go("examples")}>
            <span className="pill">Browse</span>
            <h3>Worked examples and first-study logs</h3>
            <p>Side-by-side answers with their scores, and the seed-42 prediction browser.</p>
          </button>
        </div>
      </Section>
    </>
  );
}
