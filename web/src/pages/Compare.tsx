import { useState } from "react";
import { Trophy } from "lucide-react";
import type { ResultsFile } from "../data/results";
import { Callout, Card, Section } from "../components/ui";
import { ci, fmt, pval, signed } from "../lib/format";

type Cfg = "B" | "C";
type Lang = "all" | "arabic" | "malay";
const METRICS = [
  { id: "em", label: "Exact match", hint: "Answer equals the gold answer" },
  { id: "f1", label: "Token F1", hint: "Word overlap with the gold answer" },
  { id: "f_faith", label: "NLI score", hint: "Entailment score; also the training reward" },
] as const;
const COLORS: Record<string, string> = { qwen3: "var(--accent)", t5gemma2: "var(--accent-2)", mt5: "var(--gold)" };

export default function Compare({ d }: { d: ResultsFile }) {
  const cmp = d.compare, names = cmp.models;
  const [cfg, setCfg] = useState<Cfg>("B");
  const [lang, setLang] = useState<Lang>("all");
  const R = (m: string, l: Lang = lang) => cmp.results[`${cfg}|${l}|${m}`];
  const win = R("em"), f1w = R("f1");
  const sameWinner = win.winner === f1w.winner;
  const langName = lang === "all" ? "Arabic and Malay combined" : lang === "arabic" ? "Arabic" : "Malay";

  return (
    <>
      <section className="hero">
        <h1>Which model answers best?</h1>
        <p>
          mT5-base, Qwen3-0.6B-Base and T5Gemma 2 fine-tuned with the same adapters (DoRA, QLoRA) and the same three seeds on Arabic
          (XQuAD) and Malay (Belebele) closed-passage questions. The winner is decided on answer accuracy against the gold answer.
        </p>
      </section>

      <div className="control-card compact">
        <div className="control-block">
          <span className="control-label">Training</span>
          <div className="seg">
            <button className={cfg === "B" ? "seg-btn active" : "seg-btn"} onClick={() => setCfg("B")}>Standard fine-tuning (B)</button>
            <button className={cfg === "C" ? "seg-btn active" : "seg-btn"} onClick={() => setCfg("C")}>With NLI reward (C)</button>
          </div>
        </div>
        <div className="control-block">
          <span className="control-label">Language</span>
          <div className="seg">
            {(["all", "arabic", "malay"] as const).map((l) => (
              <button key={l} className={lang === l ? "seg-btn active" : "seg-btn"} onClick={() => setLang(l)}>{l === "all" ? "Both" : l === "arabic" ? "Arabic" : "Malay"}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="winner-banner">
        <Trophy size={28} />
        <div>
          <span className="kicker">Winner · {langName}</span>
          <h2>{names[win.winner]}</h2>
          <p>
            {sameWinner ? "Best on both exact match and F1." : `Best exact match; ${names[f1w.winner]} has the best F1.`}{" "}
            Exact match {fmt(win.ranking[0].mean, 3)} versus {fmt(win.ranking[1].mean, 3)} for {names[win.runner_up]}
            {" "}({signed(win.margin, 3)}, 95% CI {ci(win.lo, win.hi, 3)}) — {win.clear ? "a clear win" : "not a clear win: the interval includes zero"}.
          </p>
        </div>
      </div>

      <Section title="Ranking by metric" kicker={`${cfg === "B" ? "Standard fine-tuning" : "NLI-reward training"} · ${langName}`}>
        <div className="three">
          {METRICS.map((m) => {
            const r = R(m.id);
            const top = Math.max(...r.ranking.map((x) => x.mean), 1e-9);
            return (
              <Card key={m.id} title={m.label} note={m.hint}>
                <div className="rank">
                  {r.ranking.map((x, i) => (
                    <div className="rank-row" key={x.model}>
                      <span className="rank-pos">{i === 0 ? <Trophy size={14} /> : i + 1}</span>
                      <div className="rank-main">
                        <div className="rank-name"><strong>{x.name}</strong><span className="mono">{fmt(x.mean, 3)}</span></div>
                        <div className="bartrack"><div className="barfill" style={{ width: `${(x.mean / top) * 100}%`, background: COLORS[x.model] }} /></div>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="hint">
                  Lead over runner-up: {signed(r.margin, 3)} {ci(r.lo, r.hi, 3)}, p {pval(r.p)}. {r.clear ? "Clear." : "Not clear."}
                </p>
              </Card>
            );
          })}
        </div>
      </Section>

      {cfg === "C" ? (
        <Callout kind="warn" title="Why mT5 'wins' the NLI score here">
          With the reward, mT5's NLI score rises sharply (Arabic {fmt(cmp.results["C|arabic|f_faith"].ranking[0].mean, 2)}), but its exact match collapses
          ({fmt(cmp.results["C|arabic|em"].ranking.find((x) => x.model === "mt5")?.mean, 3)}) and its answers grow about five times longer. The
          score was optimised directly, so a high value is not evidence of a better answer. Accuracy decides the winner.
        </Callout>
      ) : (
        <Callout kind="info" title="How to read the NLI score">
          The NLI score is nearly identical across models (about 0.14 in Arabic) when no reward is used. Accuracy (exact match and F1) separates the models.
        </Callout>
      )}

      <Section title="All numbers" kicker="Mean ± SD over adapter × seed runs">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>Language</th><th>Exact match</th><th>F1</th><th>NLI score</th></tr></thead>
            <tbody>
              {(["arabic", "malay", "all"] as const).flatMap((l) =>
                ["qwen3", "t5gemma2", "mt5"].map((m) => {
                  const cell = (k: string) => { const x = cmp.results[`${cfg}|${l}|${k}`].ranking.find((r) => r.model === m)!; const best = cmp.results[`${cfg}|${l}|${k}`].winner === m; return <td className={best ? "mono best" : "mono"}>{fmt(x.mean, 3)}<em> ±{x.sd.toFixed(3)}</em></td>; };
                  return <tr key={`${l}${m}`}><td>{names[m]}</td><td>{l === "all" ? "Both (mean)" : l === "arabic" ? "Arabic" : "Malay"}</td>{cell("em")}{cell("f1")}{cell("f_faith")}</tr>;
                }),
              )}
            </tbody>
          </table>
        </div>
        <p className="hint">Bold cells mark the best model in each column. {cmp.protocol_note}</p>
      </Section>

      <Callout kind="warn" title="What this comparison can and cannot say">
        The three models differ in size, architecture and pre-training, so this ranks the fine-tuned systems, not the architectures. Qwen3 and T5Gemma 2 were run
        with the first-study code (three seeds); mT5 uses the corrected study. Standard fine-tuning (B) does not use the reward, so it is the cleanest comparison.
        Only DoRA and QLoRA are used because all three models share them.
      </Callout>
    </>
  );
}
