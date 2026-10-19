import { useMemo, useState } from "react";
import { ArrowDownRight, ArrowRight, BookOpen, ChevronDown, CircleHelp, Sparkles } from "lucide-react";
import type { HeroData } from "../data/hero";

type Language = "arabic" | "malay";
type Metric = "f1" | "em" | "faith";
const models = ["qwen3", "t5gemma2"] as const;
const modelName: Record<string, string> = { qwen3: "Qwen3-0.6B-Base", t5gemma2: "T5Gemma 2" };
const languageName: Record<Language, string> = { arabic: "Arabic", malay: "Malay" };
const metricInfo: Record<Metric, { name: string; help: string; format: (v: number) => string }> = {
  f1: { name: "Answer overlap", help: "Compares answer words with the reference answer. Higher is better.", format: (v) => v.toFixed(3) },
  em: { name: "Exact match", help: "Share of answers that exactly match the reference. Higher is better.", format: (v) => v.toFixed(3) },
  faith: { name: "Passage support", help: "Estimates whether the passage supports an answer. It does not measure answer correctness.", format: (v) => v.toFixed(3) },
};
const pct = (v: number) => `${Math.round(v * 100)}%`;

function Segment<T extends string>({ value, options, onChange, label }: { value: T; options: Array<{ value: T; label: string }>; onChange: (v: T) => void; label: string }) {
  return <div className="hero-segment" role="group" aria-label={label}>{options.map((item) => <button key={item.value} type="button" aria-pressed={value === item.value} className={value === item.value ? "selected" : ""} onClick={() => onChange(item.value)}>{item.label}</button>)}</div>;
}

function Score({ value, sd }: { value: number; sd?: number }) {
  return <span className="hero-score">{value.toFixed(3)}{sd !== undefined ? <small> ± {sd.toFixed(3)}</small> : null}</span>;
}

export default function HeroDashboard({ data }: { data: HeroData }) {
  const [section, setSection] = useState<"verdict" | "compare" | "why" | "examples">("verdict");
  const [language, setLanguage] = useState<Language>("arabic");
  const [metric, setMetric] = useState<Metric>("f1");
  const [condition, setCondition] = useState<"B" | "C">("B");
  const [exampleId, setExampleId] = useState<number | null>(null);
  const examples = data.examples[language] ?? [];
  const example = examples.find((item) => item.id === exampleId) ?? examples[0];
  const winner = (lang: Language, cond: "B" | "C" = "B", score: "f1" | "em" = "f1") => {
    const a = data.cells.qwen3[lang][cond].metrics[score].mean;
    const b = data.cells.t5gemma2[lang][cond].metrics[score].mean;
    return a >= b ? "qwen3" : "t5gemma2";
  };
  const chartRows = useMemo(() => models.map((model) => {
    const current = data.cells[model][language][condition].metrics[metric];
    const scale = metric === "faith" ? 0.4 : 1;
    return { model, ...current, width: Math.min(100, current.mean / scale * 100) };
  }), [data, language, metric, condition]);
  const controlRecords = ["C_l0", "C_shuf"];
  const frozen = models.flatMap((model) => {
    const row = data.cells[model][language];
    if (!row.A_frozen || !row.B) return [];
    return [{ model, a: row.A_frozen.metrics.n_words.mean, b: row.B.metrics.n_words.mean }];
  });
  const lateInactive = models.map((model) => ({ model, value: data.trainingDiagnostics[model][language].at(-1)?.inactive_share ?? 0 }));

  return <div className="hero-dashboard">
    <section className="hero-intro">
      <div className="hero-kicker"><Sparkles size={15} /> Current study · 5 runs</div>
      <h1>Which model answers better?</h1>
      <p>Compare Qwen3 and T5Gemma 2 in Arabic and Malay.</p>
      <div className="hero-verdict"><div className="hero-verdict-icon"><ArrowDownRight size={22} /></div><div><strong>Qwen3 scores highest in both languages.</strong><span>Adding the reward makes small, mixed changes.</span></div></div>
      <div className="hero-limits-inline">The models differ in size, so treat this ranking as a guide.</div>
    </section>

    <nav className="hero-tabs" role="tablist" aria-label="Study sections">{[
      ["verdict", "Results"], ["compare", "Compare"], ["why", "Why reward?"], ["examples", "Examples"],
    ].map(([key, label]) => <button role="tab" aria-selected={section === key} key={key} onClick={() => setSection(key as typeof section)}>{label}</button>)}</nav>

    {section === "verdict" && <section className="hero-section" aria-label="Result summary">
      <div className="hero-section-heading"><div><span className="hero-kicker">THE RESULT</span><h2>Qwen3 leads in Arabic and Malay</h2><p>Average across five runs · ± shows variation.</p></div></div>
      <div className="hero-card-grid">{models.flatMap((model) => (["arabic", "malay"] as Language[]).map((lang) => {
        const b = data.cells[model][lang].B.metrics, c = data.cells[model][lang].C.metrics;
        return <article className={`hero-result-card ${winner(lang) === model ? "winner" : ""}`} key={`${model}-${lang}`}>
          <div className="hero-card-top"><span>{languageName[lang]}</span>{winner(lang) === model && <span className="hero-winner">Best accuracy</span>}</div>
          <h3>{modelName[model]}</h3><div className="hero-card-metrics">
            <div><span>{metricInfo.f1.name} <MetricHelp text={metricInfo.f1.help}/></span><Score value={b.f1.mean} sd={b.f1.sd}/></div>
            <div><span>{metricInfo.em.name} <MetricHelp text={metricInfo.em.help}/></span><Score value={b.em.mean} sd={b.em.sd}/></div>
          </div>
          <div className="hero-nli"><span>{metricInfo.faith.name} <MetricHelp text={metricInfo.faith.help}/></span><strong>{pct(b.faith.mean)} <small>standard</small><ArrowRight size={14}/> {pct(c.faith.mean)} <small>reward</small></strong></div>
        </article>;
      }))}</div>
      <p className="hero-footnote">NLI estimates passage support. It is not an accuracy score.</p>
    </section>}

    {section === "compare" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">COMPARE</span><h2>Which score do you want to see?</h2><p>Choose a language and training method.</p></div></div>
      <div className="hero-controls"><div><span className="hero-control-label">Score</span><Segment value={metric} label="Score" onChange={setMetric} options={[{value:"f1",label:"Answer overlap"},{value:"em",label:"Exact match"},{value:"faith",label:"Passage support"}]}/></div><div><span className="hero-control-label">Language</span><Segment value={language} label="Language" onChange={setLanguage} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div><div><span className="hero-control-label">Training</span><Segment value={condition} label="Training condition" onChange={setCondition} options={[{value:"B",label:"Standard"},{value:"C",label:"With reward"}]}/></div></div>
      <div className="hero-chart-card"><div className="hero-chart-title"><div><strong>{metricInfo[metric].name} · {languageName[language]}</strong><span>{condition === "B" ? "Standard training" : "Training with NLI reward"}</span></div><MetricHelp text={metricInfo[metric].help}/></div>
        <div className="hero-bars">{chartRows.map((row) => {const scale=metric === "faith" ? 0.4 : 1;return <div className="hero-bar-row" key={row.model}><div className="hero-bar-label"><strong>{modelName[row.model]}</strong>{metric !== "faith" && row.model === winner(language, condition, metric) && <span>Leading model</span>}</div><div className="hero-bar-track"><div className={`hero-bar-fill ${row.model === "qwen3" ? "qwen" : "t5"}`} style={{ width: `${row.width}%` }}/><i className="hero-error" style={{ left: `${Math.max(0,(row.mean-row.sd)/scale*100)}%`, width: `${Math.min(100,row.sd*2/scale*100)}%` }}/></div><strong className="hero-bar-value"><Score value={row.mean} sd={row.sd}/></strong></div>;})}</div>
        <div className="hero-axis"><span>0</span><span>{metric === "faith" ? "0.2" : "0.5"}</span><span>{metric === "faith" ? "0.4" : "1.0"}</span></div>
        <p className="hero-footnote">Score ± variation across five runs.</p>
      </div>
      <details className="hero-contrast-note"><summary>Did the reward change the score?</summary><span>{data.contrasts.qwen3[language]["C-B"]?.[metric] ? models.map((model) => {const item=data.contrasts[model][language]["C-B"][metric];return `${modelName[model]}: ${item.mean >= 0 ? "+" : ""}${item.mean.toFixed(3)} (range ${item.lo.toFixed(3)} to ${item.hi.toFixed(3)}; same direction in ${item.seeds_positive}/${item.n_seeds} runs)`;}).join(" · ") : "Not available."} Reward score minus standard score; ranges are estimates across examples.</span></details>
    </section>}

    {section === "why" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">WHY THE REWARD?</span><h2>Why is the reward not a clear win?</h2><p>Three quick checks from the experiment.</p></div></div>
      <div className="hero-why-grid">
        <article className="hero-why-card"><span className="hero-step">1 · CONTROL TESTS</span><h3>Does reward beat the controls?</h3><p>We tested a zero reward and a shuffled reward.</p>{controlRecords.map((key) => {const item=data.contrasts.qwen3[language][`C-${key}`]?.f1;return <div className="hero-control-result" key={key}><span>{key === "C_l0" ? "Zero reward" : "Shuffled reward"}</span><strong>{item ? `${item.mean >= 0 ? "+" : ""}${item.mean.toFixed(3)}` : "—"}</strong><small>Qwen3 answer overlap · range {item?.lo.toFixed(3) ?? "—"} to {item?.hi.toFixed(3) ?? "—"}</small></div>;})}<small className="hero-caveat">{languageName[language]} · change versus each control.</small></article>
        <article className="hero-why-card"><span className="hero-step">2 · ANSWER LENGTH</span><h3>Fine-tuning made answers shorter</h3><p>Average words before → after fine-tuning.</p>{frozen.map((row) => <div className="hero-length-row" key={row.model}><span>{modelName[row.model]}</span><strong>{row.a.toFixed(1)} <ArrowRight size={13}/> {row.b.toFixed(1)} <small>words</small></strong></div>)}<small className="hero-caveat">Shorter answers may affect passage-support scores.</small></article>
        <article className="hero-why-card"><span className="hero-step">3 · TRAINING</span><h3>Reward signal fades near the end</h3><p>Share of late steps with almost no reward update.</p>{lateInactive.map((row) => <div className="hero-length-row" key={row.model}><span>{modelName[row.model]}</span><strong>{pct(row.value)} <small>of steps</small></strong></div>)}<small className="hero-caveat">The models still trained on the answers. “Almost none” means the reward update was near zero.</small></article>
      </div>
      <div className="hero-lang-filter"><span>Control and training summaries show:</span><Segment value={language} label="Diagnostic language" onChange={setLanguage} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div>
    </section>}

    {section === "examples" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">EXAMPLES</span><h2>Compare the answers</h2><p>Choose a language and question.</p></div></div>
      <div className="hero-controls hero-example-controls"><div><span className="hero-control-label">Language</span><Segment value={language} label="Example language" onChange={(next) => {setLanguage(next);setExampleId(null);}} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div><label className="hero-select-wrap"><span className="hero-control-label">Question</span><span className="hero-select"><select aria-label="Choose example question" value={example?.id ?? ""} onChange={(e) => setExampleId(Number(e.target.value))}>{examples.map((item,index)=><option value={item.id} key={item.id}>Example {index+1} · {item.question.slice(0,100)}</option>)}</select><ChevronDown size={16}/></span></label></div>
      {example && <><article className="hero-question"><span>QUESTION · #{example.id}</span><p dir="auto">{example.question}</p><details><summary><BookOpen size={15}/> Show passage</summary><p dir="auto">{example.context}</p></details></article><div className="hero-gold"><span>REFERENCE ANSWER</span><p dir="auto">{example.gold}</p></div><div className="hero-answer-grid">{models.map((model) => <article className="hero-answer-card" key={model}><header><div><span>{model === "qwen3" ? "MODEL 1" : "MODEL 2"}</span><h3>{modelName[model]}</h3></div>{winner(language) === model && <span className="hero-winner">Higher average accuracy</span>}</header>{(["B","C"] as const).map((cond) => {const saved=example.answers[model][cond];return <div className={`hero-output ${cond === "C" ? "reward-output" : ""}`} key={cond}><div className="hero-output-label"><strong>{cond === "B" ? "Standard" : "With reward"}</strong><span>Passage support {saved.faith?.toFixed(3) ?? "—"} · Overlap {saved.f1?.toFixed(3) ?? "—"}</span></div><p dir="auto">{saved.answer || "(empty answer)"}</p></div>;})}</article>)}</div><p className="hero-footnote">Examples show one run; summary scores average five runs.</p></>}
    </section>}

    <footer className="hero-footer"><details><summary><CircleHelp size={16}/> About these results</summary><p>Models differ in size (Qwen3 0.6B; T5Gemma 2 270M), so treat the ranking as a guide. Reward tests are exploratory. Human ratings are from the earlier mT5 study. The paper is under review.</p></details><span>Source: experiment results · {data.meta.seeds.length} runs</span></footer>
  </div>;
}

function MetricHelp({ text }: { text: string }) { return <span className="hero-help" title={text} aria-label={text} tabIndex={0}>?</span>; }
