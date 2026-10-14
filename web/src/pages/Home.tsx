import { useState } from "react";
import { Check, ChevronLeft, ChevronRight, Trophy, X } from "lucide-react";
import type { OutputItem, ResultsFile } from "../data/results";

type Lang = "all" | "arabic" | "malay";
const LANG_LABEL: Record<Lang, string> = { all: "Both languages", arabic: "Arabic", malay: "Malay" };
const ORDER = ["qwen3", "t5gemma2", "mt5"];
const COLOR: Record<string, string> = { qwen3: "var(--accent)", t5gemma2: "var(--accent-2)", mt5: "var(--gold)" };
const pct = (x: number) => `${Math.round(x * 100)}%`;

function Answers({ items, names }: { items: OutputItem[]; names: Record<string, string> }) {
  const [i, setI] = useState(0);
  const [showPassage, setShowPassage] = useState(false);
  const it = items[Math.min(i, items.length - 1)];
  if (!it) return null;
  const get = (m: string) => it.answers[m]?.qlora?.B_ce_lora;
  const best = Math.max(...ORDER.map((m) => (get(m) ? get(m)!.em * 10 + get(m)!.f1 : 0)));
  const go = (n: number) => { setI((n + items.length) % items.length); setShowPassage(false); };

  return (
    <div className="s-card">
      <div className="s-qnav">
        <button className="s-round" onClick={() => go(i - 1)} aria-label="Previous question"><ChevronLeft size={18} /></button>
        <div className="s-dots">{items.map((_, n) => <button key={n} className={n === i ? "s-dot on" : "s-dot"} onClick={() => go(n)} aria-label={`Question ${n + 1}`}>{n + 1}</button>)}</div>
        <button className="s-round" onClick={() => go(i + 1)} aria-label="Next question"><ChevronRight size={18} /></button>
      </div>
      <h3 className="s-q" dir="auto">{it.question}</h3>
      <p className="s-gold" dir="auto"><span>Correct answer</span><strong>{it.gold}</strong></p>
      <div className="s-answers">
        {ORDER.map((m) => {
          const a = get(m);
          const isBest = a && best > 0 && a.em * 10 + a.f1 === best;
          return (
            <div key={m} className={isBest ? "s-ans best" : "s-ans"}>
              <div className="s-ans-head"><i style={{ background: COLOR[m] }} />{names[m]}{isBest ? <Trophy size={14} /> : null}</div>
              <p dir="auto" className="s-ans-text">{a?.generated || "(no answer)"}</p>
              <span className={a?.em === 1 ? "s-tag ok" : "s-tag no"}>{a?.em === 1 ? <><Check size={13} /> Correct</> : a && a.f1 > 0 ? <>Partly right</> : <><X size={13} /> Wrong</>}</span>
            </div>
          );
        })}
      </div>
      <button className="s-link" onClick={() => setShowPassage((v) => !v)}>{showPassage ? "Hide the passage" : "Show the passage the question is about"}</button>
      {showPassage ? <div className="s-passage" dir="auto">{it.context}</div> : null}
    </div>
  );
}

export default function Home({ d }: { d: ResultsFile }) {
  const cmp = d.compare, names = cmp.models;
  const [lang, setLang] = useState<Lang>("all");
  const [qlang, setQlang] = useState<"arabic" | "malay">("arabic");
  const em = cmp.results[`B|${lang}|em`];
  const f1 = (m: string) => cmp.results[`B|${lang}|f1`].ranking.find((r) => r.model === m)!.mean;
  const top = em.ranking[0].mean;
  const winner = em.winner;

  return (
    <>
      <section className="s-hero">
        <span className="s-eyebrow">Arabic and Malay question answering</span>
        <h1><span className="s-grad">{names[winner]}</span> is the best model</h1>
        <p>Three small language models were fine-tuned the same way and asked the same questions. The one with the most correct answers wins.</p>
      </section>

      <div className="s-seg" role="tablist" aria-label="Language">
        {(Object.keys(LANG_LABEL) as Lang[]).map((l) => <button key={l} className={lang === l ? "on" : ""} onClick={() => setLang(l)}>{LANG_LABEL[l]}</button>)}
      </div>

      <section className="s-podium" aria-label="Ranking">
        {em.ranking.map((r, n) => (
          <article key={r.model} className={n === 0 ? "s-model first" : "s-model"}>
            <div className="s-rank">{n === 0 ? <><Trophy size={15} /> Winner</> : `#${n + 1}`}</div>
            <h2>{names[r.model]}</h2>
            <div className="s-big">{pct(r.mean)}</div>
            <div className="s-big-l">of answers exactly right</div>
            <div className="s-bar"><div style={{ width: `${(r.mean / top) * 100}%`, background: COLOR[r.model] }} /></div>
            <div className="s-sub">Word overlap with the correct answer: <strong>{pct(f1(r.model))}</strong></div>
          </article>
        ))}
      </section>
      <p className="s-note">{names[winner]} leads {names[em.runner_up]} by {Math.round(em.margin * 100)} points{em.clear ? " — a clear win, not a fluke of random seeds" : ""}. It also has the best word overlap.</p>

      <section className="s-section">
        <h2>See the actual answers</h2>
        <p className="s-lead">The same question, answered by each model.</p>
        <div className="s-seg small" role="tablist" aria-label="Question language">
          {(["arabic", "malay"] as const).map((l) => <button key={l} className={qlang === l ? "on" : ""} onClick={() => setQlang(l)}>{l === "arabic" ? "Arabic" : "Malay"}</button>)}
        </div>
        <Answers key={qlang} items={cmp.outputs.filter((o) => o.language === qlang)} names={names} />
      </section>

      <details className="s-fine">
        <summary>How was this tested?</summary>
        <ul>
          <li>Each model was fine-tuned with the same method (QLoRA and DoRA adapters, standard training) and the same three random seeds, then scored on held-out Arabic (716) and Malay (430) questions.</li>
          <li>“Exactly right” means the answer equals the correct answer after normalisation. Scores are averages over adapters and seeds.</li>
          <li>The models differ in size and design, so this ranks the fine-tuned systems, not the architectures.</li>
          <li>Only five questions per language are shown above because that is all three models have in common in the saved logs; the ranking uses every question.</li>
        </ul>
        <p><a href="https://github.com/Akramtaha98/LoRA_Experiment" target="_blank" rel="noreferrer">Code, data and full results on GitHub</a></p>
      </details>
    </>
  );
}
