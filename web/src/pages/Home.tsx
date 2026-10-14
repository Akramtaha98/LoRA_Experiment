import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, Search, Trophy, X } from "lucide-react";
import type { ResultsFile } from "../data/results";

type Lang = "all" | "arabic" | "malay";
const LANG_LABEL: Record<Lang, string> = { all: "Both languages", arabic: "Arabic", malay: "Malay" };
const COLOR: Record<string, string> = { qwen3: "var(--accent)", t5gemma2: "var(--accent-2)", mt5: "var(--gold)" };
const pct = (x: number) => Math.round(x * 100);

interface Ans { a: string; em: number; f1: number }
interface Ex { id: string; language: "arabic" | "malay"; question: string; context: string; gold: string; answers: Record<string, Ans> }

/** Counts up from 0 to `to` when mounted or when `to` changes. */
function CountUp({ to, suffix = "" }: { to: number; suffix?: string }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setV(to); return; }
    let raf = 0; const t0 = performance.now(), dur = 900;
    const tick = (t: number) => { const p = Math.min(1, (t - t0) / dur); setV(Math.round(to * (1 - Math.pow(1 - p, 3)))); if (p < 1) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [to]);
  return <>{v}{suffix}</>;
}

/** Fades/slides children in when they scroll into view. */
function Reveal({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null); const [on, setOn] = useState(false);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setOn(true); io.disconnect(); } }, { threshold: 0.12 });
    io.observe(el); return () => io.disconnect();
  }, []);
  return <div ref={ref} className={`reveal ${on ? "in" : ""} ${className}`} style={{ transitionDelay: `${delay}ms` }}>{children}</div>;
}

function useExamples() {
  const [ex, setEx] = useState<Ex[] | null>(null);
  useEffect(() => { fetch(`${import.meta.env.BASE_URL}data/compare-examples.json`).then((r) => r.json()).then(setEx).catch(() => setEx([])); }, []);
  return ex;
}

const FILTERS = [
  { id: "all", label: "All" },
  { id: "qwen", label: "Qwen3 right" },
  { id: "mt5", label: "mT5 right" },
  { id: "none", label: "Both wrong" },
] as const;
type Filter = (typeof FILTERS)[number]["id"];

function Browser({ names }: { names: Record<string, string> }) {
  const all = useExamples();
  const [lang, setLang] = useState<"arabic" | "malay">("arabic");
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [shown, setShown] = useState(10);
  const [open, setOpen] = useState<string | null>(null);

  const inLang = useMemo(() => (all ?? []).filter((e) => e.language === lang), [all, lang]);
  const list = useMemo(() => inLang.filter((e) => {
    const qw = e.answers.qwen3.em === 1, m = e.answers.mt5.em === 1;
    if (filter === "qwen" && !qw) return false;
    if (filter === "mt5" && !m) return false;
    if (filter === "none" && (qw || m)) return false;
    return !q || (e.question + e.gold + Object.values(e.answers).map((a) => a.a).join(" ")).toLowerCase().includes(q.toLowerCase());
  }), [inLang, filter, q]);
  const score = (m: string) => inLang.filter((e) => e.answers[m]?.em === 1).length;
  useEffect(() => setShown(10), [lang, filter, q]);

  if (!all) return <p className="hint">Loading questions…</p>;
  const order = ["qwen3", "mt5", "t5gemma2"];

  return (
    <>
      <div className="s-tools">
        <div className="s-seg small">{(["arabic", "malay"] as const).map((l) => <button key={l} className={lang === l ? "on" : ""} onClick={() => setLang(l)}>{l === "arabic" ? "Arabic" : "Malay"} · 50</button>)}</div>
        <label className="s-search"><Search size={15} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search questions or answers" /></label>
      </div>
      <div className="s-score">
        <div><b style={{ color: COLOR.qwen3 }}><CountUp key={lang + "q"} to={score("qwen3")} />/50</b><span>{names.qwen3} correct</span></div>
        <div><b style={{ color: COLOR.mt5 }}><CountUp key={lang + "m"} to={score("mt5")} />/50</b><span>{names.mt5} correct</span></div>
      </div>
      <div className="s-chips">{FILTERS.map((f) => <button key={f.id} className={filter === f.id ? "s-chip on" : "s-chip"} onClick={() => setFilter(f.id)}>{f.label}</button>)}</div>

      <div className="s-list" key={`${lang}-${filter}-${q}`}>
        {list.slice(0, shown).map((e, n) => (
          <article className="s-ex" key={e.id} style={{ animationDelay: `${Math.min(n, 9) * 45}ms` }}>
            <div className="s-ex-top"><span className="s-num">#{Number(e.id.split("-")[1]) + 1}</span><h3 dir="auto">{e.question}</h3></div>
            <p className="s-gold" dir="auto"><span>Correct answer</span><strong>{e.gold}</strong></p>
            <div className="s-answers">
              {order.filter((m) => e.answers[m]).map((m) => {
                const a = e.answers[m];
                return (
                  <div key={m} className={a.em ? "s-ans best" : "s-ans"}>
                    <div className="s-ans-head"><i style={{ background: COLOR[m] }} />{names[m]}{a.em ? <Trophy size={14} /> : null}</div>
                    <p dir="auto" className="s-ans-text">{a.a || "(no answer)"}</p>
                    <span className={a.em ? "s-tag ok" : a.f1 > 0 ? "s-tag part" : "s-tag no"}>{a.em ? <><Check size={13} /> Correct</> : a.f1 > 0 ? "Partly right" : <><X size={13} /> Wrong</>}</span>
                  </div>
                );
              })}
            </div>
            <button className="s-link" onClick={() => setOpen(open === e.id ? null : e.id)}>{open === e.id ? "Hide passage" : "Show passage"} <ChevronDown size={14} className={open === e.id ? "flip" : ""} /></button>
            {open === e.id ? <div className="s-passage" dir="auto">{e.context}</div> : null}
          </article>
        ))}
        {list.length === 0 ? <p className="hint">No questions match.</p> : null}
      </div>
      {shown < list.length ? <button className="s-more" onClick={() => setShown((s) => s + 10)}>Show 10 more ({list.length - shown} left)</button> : null}
    </>
  );
}

export default function Home({ d }: { d: ResultsFile }) {
  const cmp = d.compare, names = cmp.models;
  const [lang, setLang] = useState<Lang>("all");
  const em = cmp.results[`B|${lang}|em`];
  const f1 = (m: string) => cmp.results[`B|${lang}|f1`].ranking.find((r) => r.model === m)!.mean;
  const top = em.ranking[0].mean;
  const winner = em.winner;

  return (
    <>
      <section className="s-hero">
        <span className="s-eyebrow s-in" style={{ animationDelay: "0ms" }}>Arabic and Malay question answering</span>
        <h1 className="s-in" style={{ animationDelay: "90ms" }}><span className="s-grad">{names[winner]}</span> is the best model</h1>
        <p className="s-in" style={{ animationDelay: "180ms" }}>Three small language models were fine-tuned the same way and asked the same questions. The one with the most correct answers wins.</p>
      </section>

      <div className="s-seg s-in" style={{ animationDelay: "260ms" }} role="tablist" aria-label="Language">
        {(Object.keys(LANG_LABEL) as Lang[]).map((l) => <button key={l} className={lang === l ? "on" : ""} onClick={() => setLang(l)}>{LANG_LABEL[l]}</button>)}
      </div>

      <section className="s-podium" aria-label="Ranking" key={lang}>
        {em.ranking.map((r, n) => (
          <article key={r.model} className={n === 0 ? "s-model first s-in" : "s-model s-in"} style={{ animationDelay: `${320 + n * 110}ms` }}>
            <div className="s-rank">{n === 0 ? <><Trophy size={15} className="s-trophy" /> Winner</> : `#${n + 1}`}</div>
            <h2>{names[r.model]}</h2>
            <div className="s-big"><CountUp to={pct(r.mean)} suffix="%" /></div>
            <div className="s-big-l">of answers exactly right</div>
            <div className="s-bar"><div className="s-fill" style={{ ["--w" as string]: `${(r.mean / top) * 100}%`, background: COLOR[r.model], animationDelay: `${500 + n * 110}ms` }} /></div>
            <div className="s-sub">Word overlap with the correct answer: <strong>{pct(f1(r.model))}%</strong></div>
          </article>
        ))}
      </section>
      <p className="s-note">{names[winner]} leads {names[em.runner_up]} by {Math.round(em.margin * 100)} points{em.clear ? " — a clear win, not a fluke of random seeds" : ""}. It also has the best word overlap.</p>

      <section className="s-section">
        <Reveal>
          <h2>Browse 100 real questions</h2>
          <p className="s-lead">The first 50 Arabic and 50 Malay test questions, with each model's answer. Nothing is cherry-picked.</p>
          <div className="s-info"><b>Which runs are these?</b> Only Qwen3 and mT5 saved every answer, and only for the NLI-reward training, so these answers come from that setup (QLoRA, seed 42). T5Gemma 2 saved just its first 5 answers per language, so it appears on those only. The ranking above uses standard training and every question.</div>
        </Reveal>
        <Browser names={names} />
      </section>

      <Reveal>
        <details className="s-fine">
          <summary>How was this tested?</summary>
          <ul>
            <li>Ranking: each model was fine-tuned the standard way (QLoRA and DoRA adapters) with the same three random seeds, then scored on held-out Arabic (716) and Malay (430) questions.</li>
            <li>“Exactly right” means the answer equals the correct answer after normalisation. Scores are averages over adapters and seeds.</li>
            <li>The models differ in size and design, so this ranks the fine-tuned systems, not the architectures.</li>
          </ul>
          <p><a href="https://github.com/Akramtaha98/LoRA_Experiment" target="_blank" rel="noreferrer">Code, data and full results on GitHub</a></p>
        </details>
      </Reveal>
    </>
  );
}
