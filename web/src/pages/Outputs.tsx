import { useState } from "react";
import { Check, Trophy, X } from "lucide-react";
import type { OutputItem, ResultsFile } from "../data/results";
import { Callout, Section } from "../components/ui";
import { ADAPTER, fmt } from "../lib/format";

const ORDER = ["qwen3", "t5gemma2", "mt5"];

function bestOf(it: OutputItem, ad: string, cfg: string): string[] {
  const rows = ORDER.map((m) => ({ m, a: it.answers[m]?.[ad]?.[cfg] })).filter((r) => r.a);
  const top = Math.max(...rows.map((r) => r.a!.em * 10 + r.a!.f1));
  if (top <= 0) return [];
  return rows.filter((r) => r.a!.em * 10 + r.a!.f1 === top).map((r) => r.m);
}

export default function Outputs({ d }: { d: ResultsFile }) {
  const cmp = d.compare, names = cmp.models;
  const [lang, setLang] = useState<"arabic" | "malay">("arabic");
  const [ad, setAd] = useState("qlora");
  const [cfg, setCfg] = useState("B_ce_lora");
  const [open, setOpen] = useState<string | null>(null);
  const items = cmp.outputs.filter((i) => i.language === lang);
  const tally = ORDER.map((m) => ({ m, em: items.filter((i) => i.answers[m]?.[ad]?.[cfg]?.em === 1).length }));

  return (
    <>
      <section className="hero small">
        <h1>What each model actually answered</h1>
        <p>The same held-out questions, answered by each fine-tuned model. A tick means the answer equals the gold answer exactly.</p>
      </section>

      <div className="control-card compact">
        <div className="control-block"><span className="control-label">Language</span>
          <div className="seg">{(["arabic", "malay"] as const).map((l) => <button key={l} className={lang === l ? "seg-btn active" : "seg-btn"} onClick={() => setLang(l)}>{l === "arabic" ? "Arabic" : "Malay"}</button>)}</div></div>
        <div className="control-block"><span className="control-label">Adapter</span>
          <div className="seg">{cmp.adapters.map((a) => <button key={a} className={ad === a ? "seg-btn active" : "seg-btn"} onClick={() => setAd(a)}>{ADAPTER(a)}</button>)}</div></div>
        <div className="control-block"><span className="control-label">Training</span>
          <div className="seg"><button className={cfg === "B_ce_lora" ? "seg-btn active" : "seg-btn"} onClick={() => setCfg("B_ce_lora")}>Standard (B)</button>
            <button className={cfg === "C_composite_lora" ? "seg-btn active" : "seg-btn"} onClick={() => setCfg("C_composite_lora")}>NLI reward (C)</button></div></div>
      </div>

      <div className="tally">
        {tally.map((t) => (
          <div key={t.m} className="tally-item" style={{ borderColor: t.em === Math.max(...tally.map((x) => x.em)) ? "var(--accent)" : undefined }}>
            <span>{names[t.m]}</span><strong>{t.em}/{items.length}</strong><small>exact matches</small>
          </div>
        ))}
      </div>

      <Section title={`${items.length} questions`} kicker="Same questions for every model">
        <div className="examples-list">
          {items.map((it, n) => {
            const best = bestOf(it, ad, cfg);
            return (
              <article className="card example" key={it.id}>
                <header><span className="pill">Question {n + 1}</span>
                  {best.length ? <span className="pill good"><Trophy size={12} /> {best.map((m) => names[m]).join(" · ")}</span> : <span className="pill warn">No model matched</span>}</header>
                <h3 dir="auto">{it.question}</h3>
                <p className="gold" dir="auto"><strong>Gold answer:</strong> {it.gold}</p>
                <div className="answers three">
                  {ORDER.map((m) => {
                    const a = it.answers[m]?.[ad]?.[cfg];
                    return (
                      <div key={m} className={`answer ${best.includes(m) ? "c" : "b"}`}>
                        <span className="pill">{names[m]}</span>
                        {a ? (
                          <>
                            <p dir="auto" className="answer-text">{a.generated || "(empty)"}</p>
                            <p className="verdict">{a.em === 1 ? <span className="ok"><Check size={14} /> exact match</span> : <span className="no"><X size={14} /> not exact</span>} · F1 {fmt(a.f1, 2)}</p>
                          </>
                        ) : <p className="hint">no answer logged</p>}
                      </div>
                    );
                  })}
                </div>
                <button className="ghost-btn" onClick={() => setOpen(open === it.id ? null : it.id)}>{open === it.id ? "Hide passage" : "Show passage"}</button>
                {open === it.id ? <div className="passage" dir="auto">{it.context}</div> : null}
              </article>
            );
          })}
        </div>
      </Section>

      <Callout kind="info" title="Why only five questions per language">
        T5Gemma 2's run records keep answers for the first five held-out questions only, so these are the questions all three models share. Five questions are an
        illustration, not a ranking: the ranking on the previous tab uses every held-out question (716 Arabic, 430 Malay) across all runs. Qwen3 and T5Gemma 2 answers are from the original protocol (seed 42); mT5 is from the corrected study (seed 42).
      </Callout>
    </>
  );
}
