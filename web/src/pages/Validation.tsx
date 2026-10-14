import type { ResultsFile } from "../data/results";
import { BarList, Callout, Card, Section } from "../components/ui";
import { LANG, ci, fmt, signed } from "../lib/format";

type CI = { mean: number; ci95: number[] };

export default function Validation({ d }: { d: ResultsFile }) {
  const c = d.corrected, lm = c.length_matched, al = c.answer_length;
  const ind = d.independent as unknown as Record<string, CI | number | string>;
  const hs = d.human.sensitivity as Record<string, { delta?: number; ci95?: number[]; n?: number; C_lower_equal_higher?: number[] } | number>;
  const bins = Object.entries(lm.bin_means);
  const maxScore = 1;
  const human = (k: string) => hs[k] as { delta: number; ci95: number[]; n: number; C_lower_equal_higher?: number[] } | undefined;
  return (
    <>
      <section className="hero small">
        <h1>Does the gain survive other measures?</h1>
        <p>Independent NLI scorer, blinded human judgements, answer-length adjustment and run diagnostics.</p>
      </section>

      <Section title="Independent NLI scorer" kicker="xlm-roberta-large-xnli, not used in training">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Contrast</th><th>Training scorer (mDeBERTa)</th><th>Independent scorer (XLM-R large)</th></tr></thead>
            <tbody>
              {([["C-B", "C − B"], ["C-C_l0", "C − λ₁=0 control"], ["C-C_shuf", "C − shuffled control"]] as const).map(([k, name]) => {
                const a = ind[`f_faith:${k}`] as CI, b = ind[`f_indep:${k}`] as CI;
                return (
                  <tr key={k}><td>{name}</td>
                    <td className="mono">{signed(a?.mean)} {ci(a?.ci95[0], a?.ci95[1])}</td>
                    <td className="mono">{signed(b?.mean)} {ci(b?.ci95[0], b?.ci95[1])}</td></tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="hint">Spearman correlation between the two scorers over answers: {fmt(ind.spearman_reward_vs_indep as number, 2)}. An independent model of the same family reproduces the gain, so it is not specific to one scorer, but both share the NLI formulation and cross-lingual transfer.</p>
      </Section>

      <Section title="Blinded human judgements" kicker="Two annotators · 60 items per language · support and correctness">
        <div className="stat-grid">
          <div className="stat"><span className="stat-label">Weighted κ (support)</span><strong className="stat-value">{fmt(d.human.summary.kappa_supported_quadratic, 2)}</strong><span className="stat-sub">quadratic weights</span></div>
          <div className="stat"><span className="stat-label">κ (correctness)</span><strong className="stat-value">{fmt(d.human.summary.kappa_correct, 2)}</strong><span className="stat-sub">low agreement</span></div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Arabic, C − B (0–2 scale)</th><th>Δ</th><th>95% CI</th><th>C lower / equal / higher</th></tr></thead>
            <tbody>
              {[["arabic|s1", "Annotator 1 · support"], ["arabic|s2", "Annotator 2 · support"], ["arabic|mean", "Mean of annotators"], ["arabic|c1", "Annotator 1 · correctness"], ["arabic|c2", "Annotator 2 · correctness"]].map(([k, name]) => {
                const h = human(k); if (!h) return null;
                return <tr key={k}><td>{name}</td><td className="mono">{signed(h.delta, 3)}</td><td className="mono">{ci(h.ci95?.[0], h.ci95?.[1], 2)}</td><td className="mono">{h.C_lower_equal_higher ? h.C_lower_equal_higher.join(" / ") : "—"}</td></tr>;
              })}
            </tbody>
          </table>
        </div>
        <Callout kind="warn" title="Limits of the human evidence">
          Two annotators disagree on direction: annotator 1 rates reward-trained answers as clearly less supported, annotator 2 sees no difference. Malay shows no difference. The study cannot show that reward-trained answers are less faithful, only that the human ratings do not follow the NLI score.
        </Callout>
      </Section>

      <Section title="Answer length" kicker="Most of the Arabic gain tracks length">
        <div className="two">
          <Card title="What the reward changed in Arabic answers" note="Pooled over adapters and seeds; headline evaluation set.">
            <table>
              <thead><tr><th>Condition</th><th>Words</th><th>&gt;3× gold</th><th>Verbatim in passage</th></tr></thead>
              <tbody>
                {(["B", "C_l0", "C_shuf", "C"] as const).map((k) => {
                  const r = al[`arabic|${k}`]; if (!r) return null;
                  return <tr key={k}><td>{k === "B" ? "Cross-entropy" : k === "C" ? "NLI reward" : k === "C_l0" ? "λ₁=0" : "Shuffled"}</td><td className="mono">{r.mean_words.toFixed(1)}</td><td className="mono">{(r.share_gt_3x_gold * 100).toFixed(1)}%</td><td className="mono">{(r.in_context_rate * 100).toFixed(1)}%</td></tr>;
                })}
              </tbody>
            </table>
            <p className="hint">Malay is unchanged: about 4.0 words and 0.8–1.0% long answers in every condition.</p>
          </Card>
          <Card title="NLI score by answer-length bin" note="Within every bin reward-trained answers score higher, but the two conditions rarely produce answers of the same length.">
            <BarList items={bins.filter(([, v]) => v.B.n > 0 && v.C.n > 0).flatMap(([b, v]) => [
              { label: `${b} words · B (n=${v.B.n})`, value: v.B.mean ?? 0, tone: "muted" },
              { label: `${b} words · C (n=${v.C.n})`, value: v.C.mean ?? 0 },
            ])} max={maxScore} digits={2} />
          </Card>
        </div>
        <p className="hint">
          Only {(lm.matched_overall.share_of_all_pairs * 100).toFixed(0)}% of B/C answer pairs for the same item, adapter and seed fall in the same bin ({lm.pairs_same_bin} of {lm.pairs_total}); in those pairs the reward-trained answer scores {signed(lm.matched_overall.mean_diff, 2)} higher {ci(lm.matched_overall.ci95[0], lm.matched_overall.ci95[1], 2)}.
          A regression with length-bin fixed effects gives a reward coefficient of {signed(lm.fe_regression_C_coef.estimate, 2)} {ci(lm.fe_regression_C_coef.ci95[0], lm.fe_regression_C_coef.ci95[1], 2)}, against {signed(c.headline.arabic.mean, 2)} without length terms. This is a descriptive decomposition, not an intervention; a length-penalised reward was not run.
        </p>
      </Section>

      <Section title="Run diagnostics" kicker={`${c.diagnostics.n_runs} runs`}>
        <div className="table-wrap">
          <table>
            <tbody>
              <tr><td>Failed runs</td><td className="mono">{c.diagnostics.n_failed_runs}</td></tr>
              <tr><td>Runs with non-finite logits</td><td className="mono">{c.diagnostics.runs_with_nonfinite_logits}</td></tr>
              <tr><td>Generation calls / non-finite</td><td className="mono">{c.diagnostics.sanitize_total.gen_calls.toLocaleString()} / {c.diagnostics.sanitize_total.gen_nan + c.diagnostics.sanitize_total.gen_posinf}</td></tr>
              <tr><td>Scoring calls / non-finite</td><td className="mono">{c.diagnostics.sanitize_total.score_calls.toLocaleString()} / {c.diagnostics.sanitize_total.score_nan + c.diagnostics.sanitize_total.score_posinf}</td></tr>
              <tr><td>Evaluation prompts truncated (mean / max)</td><td className="mono">{(c.diagnostics.frac_eval_prompt_truncated.mean * 100).toFixed(1)}% / {(c.diagnostics.frac_eval_prompt_truncated.max * 100).toFixed(1)}%</td></tr>
            </tbody>
          </table>
        </div>
        <p className="hint">Languages: {LANG("arabic")} (716 headline items) and {LANG("malay")} (430).</p>
      </Section>
    </>
  );
}
