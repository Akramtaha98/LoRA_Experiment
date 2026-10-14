import { useState } from "react";
import type { ExampleAnswer, ResultsFile } from "../data/results";
import { Callout, Section } from "../components/ui";
import { LANG, fmt } from "../lib/format";

function Answer({ title, a, tone }: { title: string; a: ExampleAnswer; tone: string }) {
  return (
    <div className={`answer ${tone}`}>
      <span className="pill">{title}</span>
      <p dir="auto" className="answer-text">{a.answer}</p>
      <dl className="scores">
        <div><dt>NLI score</dt><dd>{fmt(a.f_faith, 3)}</dd></div>
        <div><dt>Independent</dt><dd>{fmt(a.f_indep, 3)}</dd></div>
        <div><dt>Support (A1 / A2)</dt><dd>{a.sup1} / {a.sup2}</dd></div>
        <div><dt>Correct (A1 / A2)</dt><dd>{a.cor1} / {a.cor2}</dd></div>
      </dl>
    </div>
  );
}

export default function Worked({ d }: { d: ResultsFile }) {
  const ex = d.corrected.examples;
  const [open, setOpen] = useState<string | null>(null);
  return (
    <Section title="Worked examples" kicker="Same item, cross-entropy answer versus reward-trained answer">
      <Callout kind="info" title="How these were chosen">
        Five items from the human-validation sample are shown with the training scorer, the independent scorer and both annotators' blinded ratings (support 0–2, correctness 0/1). They illustrate the mechanism; they are not a random sample of the results.
      </Callout>
      <div className="examples-list">
        {ex.map((e) => (
          <article className="card example" key={e.src_id}>
            <header><span className="pill">{LANG(e.language)}</span><code>{e.src_id}</code></header>
            <h3 dir="auto">{e.question}</h3>
            <p className="gold" dir="auto"><strong>Gold:</strong> {e.gold}</p>
            <div className="answers">
              <Answer title="B · cross-entropy" a={e.B} tone="b" />
              <Answer title="C · NLI reward" a={e.C} tone="c" />
            </div>
            <button className="ghost-btn" onClick={() => setOpen(open === e.src_id ? null : e.src_id)}>{open === e.src_id ? "Hide passage" : "Show passage"}</button>
            {open === e.src_id ? <div className="passage" dir="auto">{e.passage}</div> : null}
          </article>
        ))}
      </div>
    </Section>
  );
}
