import { useMemo, useState } from "react";
import { ArrowDownRight, ArrowRight, BookOpen, ChevronDown, CircleHelp, Sparkles } from "lucide-react";
import type { HeroData } from "../data/hero";

type Language = "arabic" | "malay";
type Metric = "f1" | "em" | "faith";
const models = ["qwen3", "t5gemma2"] as const;
const modelName: Record<string, string> = { qwen3: "Qwen3-0.6B-Base", t5gemma2: "T5Gemma 2" };
const languageName: Record<Language, string> = { arabic: "Arabic", malay: "Malay" };
const metricInfo: Record<Metric, { name: string; help: string; format: (v: number) => string }> = {
  f1: { name: "Token F1", help: "Measures word overlap with the reference answer. Higher is better; it is not a human rating.", format: (v) => v.toFixed(3) },
  em: { name: "Exact match", help: "Share of answers that exactly match the reference after normalization. Higher is better.", format: (v) => v.toFixed(3) },
  faith: { name: "NLI score", help: "The experiment's entailment-model score for support from the passage. It is not answer accuracy.", format: (v) => v.toFixed(3) },
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
  const controlRecords = ["C_l0", "C_shuf"].map((key) => ({ key, label: key === "C_l0" ? "Reward weight set to zero" : "Shuffled reward", result: data.contrasts.qwen3[language][`C-${key}`]?.f1 }));
  const frozen = models.flatMap((model) => {
    const row = data.cells[model][language];
    if (!row.A_frozen || !row.B) return [];
    return [{ model, a: row.A_frozen.metrics.n_words.mean, b: row.B.metrics.n_words.mean }];
  });
  const lateInactive = models.map((model) => ({ model, value: data.trainingDiagnostics[model][language].at(-1)?.inactive_share ?? 0 }));

  return <div className="hero-dashboard">
    <section className="hero-intro">
      <div className="hero-kicker"><Sparkles size={15} /> Current study · 5 training seeds</div>
      <h1>Which model answers best?</h1>
      <p>Compare answer quality in Arabic and Malay, then see whether NLI-reward training changes the result.</p>
      <div className="hero-verdict"><div className="hero-verdict-icon"><ArrowDownRight size={22} /></div><div><strong>Qwen3-0.6B-Base leads on answer accuracy.</strong><span>The NLI reward does not explain that lead consistently: its effects are small and mixed across languages.</span></div></div>
      <div className="hero-limits-inline">This is an exploratory comparison: the models differ in size and architecture.</div>
    </section>

    <nav className="hero-tabs" role="tablist" aria-label="Study sections">{[
      ["verdict", "The result"], ["compare", "Compare scores"], ["why", "Reward analysis"], ["examples", "See answers"],
    ].map(([key, label]) => <button role="tab" aria-selected={section === key} key={key} onClick={() => setSection(key as typeof section)}>{label}</button>)}</nav>

    {section === "verdict" && <section className="hero-section" aria-label="Result summary">
      <div className="hero-section-heading"><div><span className="hero-kicker">RESULT AT A GLANCE</span><h2>Qwen3 leads in both languages</h2><p>Each card shows averages across five seeds. The shaded value is the NLI score before and after reward training.</p></div></div>
      <div className="hero-card-grid">{models.flatMap((model) => (["arabic", "malay"] as Language[]).map((lang) => {
        const b = data.cells[model][lang].B.metrics, c = data.cells[model][lang].C.metrics;
        return <article className={`hero-result-card ${winner(lang) === model ? "winner" : ""}`} key={`${model}-${lang}`}>
          <div className="hero-card-top"><span>{languageName[lang]}</span>{winner(lang) === model && <span className="hero-winner">Best accuracy</span>}</div>
          <h3>{modelName[model]}</h3><div className="hero-card-metrics">
            <div><span>Token F1 <MetricHelp text={metricInfo.f1.help}/></span><Score value={b.f1.mean} sd={b.f1.sd}/></div>
            <div><span>Exact match <MetricHelp text={metricInfo.em.help}/></span><Score value={b.em.mean} sd={b.em.sd}/></div>
          </div>
          <div className="hero-nli"><span>NLI score <MetricHelp text={metricInfo.faith.help}/></span><strong>{pct(b.faith.mean)} <small>standard</small><ArrowRight size={14}/> {pct(c.faith.mean)} <small>reward</small></strong></div>
        </article>;
      }))}</div>
      <p className="hero-footnote">Accuracy measures are compared with the dataset’s reference answers. A higher NLI score does not necessarily mean a more correct answer.</p>
    </section>}

    {section === "compare" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">SIDE-BY-SIDE</span><h2>Compare model scores</h2><p>Change the score, language, and training condition. Bars include the standard deviation across seeds.</p></div></div>
      <div className="hero-controls"><div><span className="hero-control-label">Score</span><Segment value={metric} label="Score" onChange={setMetric} options={[{value:"f1",label:"Token F1"},{value:"em",label:"Exact match"},{value:"faith",label:"NLI score"}]}/></div><div><span className="hero-control-label">Language</span><Segment value={language} label="Language" onChange={setLanguage} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div><div><span className="hero-control-label">Training</span><Segment value={condition} label="Training condition" onChange={setCondition} options={[{value:"B",label:"Standard"},{value:"C",label:"NLI reward"}]}/></div></div>
      <div className="hero-chart-card"><div className="hero-chart-title"><div><strong>{metricInfo[metric].name} · {languageName[language]}</strong><span>{condition === "B" ? "Standard training" : "Training with NLI reward"}</span></div><MetricHelp text={metricInfo[metric].help}/></div>
        <div className="hero-bars">{chartRows.map((row) => {const scale=metric === "faith" ? 0.4 : 1;return <div className="hero-bar-row" key={row.model}><div className="hero-bar-label"><strong>{modelName[row.model]}</strong>{metric !== "faith" && row.model === winner(language, condition, metric) && <span>Leading model</span>}</div><div className="hero-bar-track"><div className={`hero-bar-fill ${row.model === "qwen3" ? "qwen" : "t5"}`} style={{ width: `${row.width}%` }}/><i className="hero-error" style={{ left: `${Math.max(0,(row.mean-row.sd)/scale*100)}%`, width: `${Math.min(100,row.sd*2/scale*100)}%` }}/></div><strong className="hero-bar-value"><Score value={row.mean} sd={row.sd}/></strong></div>;})}</div>
        <div className="hero-axis"><span>0</span><span>{metric === "faith" ? "0.2" : "0.5"}</span><span>{metric === "faith" ? "0.4" : "1.0"}</span></div>
        <p className="hero-footnote">Error range: ±1 standard deviation across five seeds. Compare B and C within each model; the model sizes differ, so the ranking is indicative.</p>
      </div>
      <div className="hero-contrast-note"><strong>What changes with reward training?</strong><span>{data.contrasts.qwen3[language]["C-B"]?.[metric] ? models.map((model) => {const item=data.contrasts[model][language]["C-B"][metric];return `${modelName[model]}: ${item.mean >= 0 ? "+" : ""}${item.mean.toFixed(3)} (95% interval ${item.lo.toFixed(3)} to ${item.hi.toFixed(3)}; same direction in ${item.seeds_positive}/${item.n_seeds} seeds)`;}).join(" · ") : "The selected comparison is not available."} Difference = reward-trained score minus standard score; intervals are bootstrapped over examples.</span><button type="button" className="hero-text-button" onClick={() => setSection("why")}>See the evidence <ArrowRight size={15}/></button></div>
    </section>}

    {section === "why" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">WHAT THE RUNS SHOW</span><h2>Why the reward gives no clear win</h2><p>Three checks help explain why adding an NLI reward did not consistently improve answers.</p></div></div>
      <div className="hero-why-grid">
        <article className="hero-why-card"><span className="hero-step">01 · CONTROL RUNS</span><h3>Does the reward itself help?</h3><p>We compare reward training with a zero-weight reward and a shuffled reward. These checks help separate a useful reward signal from other training changes.</p>{controlRecords.map(({key,label}) => {const item=data.contrasts.qwen3[language][`C-${key}`]?.f1;return <div className="hero-control-result" key={key}><span>{label}</span><strong>{item ? `${item.mean >= 0 ? "+" : ""}${item.mean.toFixed(3)}` : "—"}</strong><small>Qwen3 F1 difference · reward vs control · 95% interval [{item?.lo.toFixed(3) ?? "—"}, {item?.hi.toFixed(3) ?? "—"}]</small></div>;})}<small className="hero-caveat">Shown for Qwen3 · {languageName[language]}; contrast intervals cover examples and are exploratory.</small></article>
        <article className="hero-why-card"><span className="hero-step">02 · ANSWER LENGTH</span><h3>Fine-tuning shortens answers</h3><p>Average words per answer, frozen base output compared with standard fine-tuning. Short answers can affect passage-support scores.</p>{frozen.map((row) => <div className="hero-length-row" key={row.model}><span>{modelName[row.model]}</span><strong>{row.a.toFixed(1)} <ArrowRight size={13}/> {row.b.toFixed(1)} <small>words</small></strong></div>)}<small className="hero-caveat">This is a before/after association; it does not prove length caused the score change.</small></article>
        <article className="hero-why-card"><span className="hero-step">03 · TRAINING SIGNAL</span><h3>The reward signal is often inactive late in training</h3><p>Share of logged late-stage steps where the policy-gradient value was near zero. Standard cross-entropy training continued.</p>{lateInactive.map((row) => <div className="hero-length-row" key={row.model}><span>{modelName[row.model]}</span><strong>{pct(row.value)} <small>inactive steps</small></strong></div>)}<small className="hero-caveat">Late stage = final quarter of logged steps; “inactive” means |policy-gradient value| &lt; 0.001.</small></article>
      </div>
      <div className="hero-lang-filter"><span>Control and training summaries show:</span><Segment value={language} label="Diagnostic language" onChange={setLanguage} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div>
    </section>}

    {section === "examples" && <section className="hero-section">
      <div className="hero-section-heading"><div><span className="hero-kicker">SAVED MODEL OUTPUTS</span><h2>See how the models answered</h2><p>Same held-out question, reference answer, and saved outputs from standard and reward training.</p></div></div>
      <div className="hero-controls hero-example-controls"><div><span className="hero-control-label">Language</span><Segment value={language} label="Example language" onChange={(next) => {setLanguage(next);setExampleId(null);}} options={[{value:"arabic",label:"Arabic"},{value:"malay",label:"Malay"}]}/></div><label className="hero-select-wrap"><span className="hero-control-label">Question</span><span className="hero-select"><select aria-label="Choose example question" value={example?.id ?? ""} onChange={(e) => setExampleId(Number(e.target.value))}>{examples.map((item,index)=><option value={item.id} key={item.id}>Example {index+1} · {item.question.slice(0,100)}</option>)}</select><ChevronDown size={16}/></span></label></div>
      {example && <><article className="hero-question"><span>QUESTION · #{example.id}</span><p dir="auto">{example.question}</p><details><summary><BookOpen size={15}/> Show passage</summary><p dir="auto">{example.context}</p></details></article><div className="hero-gold"><span>REFERENCE ANSWER</span><p dir="auto">{example.gold}</p></div><div className="hero-answer-grid">{models.map((model) => <article className="hero-answer-card" key={model}><header><div><span>{model === "qwen3" ? "MODEL 1" : "MODEL 2"}</span><h3>{modelName[model]}</h3></div>{winner(language) === model && <span className="hero-winner">Higher average accuracy</span>}</header>{(["B","C"] as const).map((cond) => {const saved=example.answers[model][cond];return <div className={`hero-output ${cond === "C" ? "reward-output" : ""}`} key={cond}><div className="hero-output-label"><strong>{cond === "B" ? "Standard training" : "NLI reward"}</strong><span>NLI {saved.faith?.toFixed(3) ?? "—"} · F1 {saved.f1?.toFixed(3) ?? "—"}</span></div><p dir="auto">{saved.answer || "(empty answer)"}</p></div>;})}</article>)}</div><p className="hero-footnote">Examples are saved outputs from seed 42, selected from held-out items where at least one answer differs. The overall scores average five seeds.</p></>}
    </section>}

    <footer className="hero-footer"><div><CircleHelp size={17}/><strong>How to read this study</strong></div><p>Models differ in size and architecture (Qwen3 0.6B; T5Gemma 2 270M), so the model ranking is indicative. Reward contrasts are exploratory. Human evaluation covers the earlier mT5 study only. The paper is under review.</p><span>Source: saved outputs and analysis tables in results/hero_v2 · {data.meta.seeds.length} seeds</span></footer>
  </div>;
}

function MetricHelp({ text }: { text: string }) { return <span className="hero-help" title={text} aria-label={text} tabIndex={0}>?</span>; }
