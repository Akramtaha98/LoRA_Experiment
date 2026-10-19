import { useState } from "react";
import { ArrowRight, ChevronLeft, ChevronRight, Info } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HeroData } from "../data/hero";

type Lang = "arabic" | "malay";
type Model = "qwen3" | "t5gemma2";
type Metric = "f1" | "em" | "faith";
const models: Model[] = ["qwen3", "t5gemma2"];
const names = { qwen3: "Qwen3", t5gemma2: "T5Gemma 2" };
const fullNames = { qwen3: "Qwen3-0.6B-Base", t5gemma2: "T5Gemma 2 270M-270M" };
const metrics = {
  f1: { label: "Answer overlap", short: "F1", help: "How closely answer words match the reference. Higher is better." },
  em: { label: "Exact answers", short: "EM", help: "Answers that exactly match the reference after normalization. Higher is better." },
  faith: { label: "Passage support", short: "NLI", help: "An automatic estimate of support from the passage. This is not answer correctness." },
};
const format = (v: number) => v.toFixed(3);
function Help({ text }: { text: string }) { return <details className="qa-help"><summary aria-label={text}><Info size={14}/></summary><span>{text}</span></details>; }
function Choice<T extends string>({ value, choices, onChange, label }: { value: T; choices: { id: T; name: string }[]; onChange: (v: T) => void; label: string }) {
  return <div className="qa-choice" role="group" aria-label={label}>{choices.map(c => <button key={c.id} aria-pressed={value === c.id} onClick={() => onChange(c.id)}>{c.name}</button>)}</div>;
}
export default function HeroDashboard({ data }: { data: HeroData }) {
  const [page, setPage] = useState("results");
  const [lang, setLang] = useState<Lang>("arabic");
  const [metric, setMetric] = useState<Metric>("em");
  const [training, setTraining] = useState<"B" | "C" | "both">("both");
  const [exampleIndex, setExampleIndex] = useState(0);
  const [diagnosticModel, setDiagnosticModel] = useState<Model>("qwen3");
  const examples = data.examples[lang];
  const example = examples[exampleIndex % examples.length];
  const cell = (model: Model, condition = "B") => data.cells[model][lang][condition].metrics;
  const winner = [...models].sort((a,b) => cell(b).em.mean - cell(a).em.mean)[0];
  const chart = models.map(model => ({ name: names[model], standard: cell(model)[metric].mean, reward: cell(model,"C")[metric].mean, standardError: cell(model)[metric].sd, rewardError: cell(model,"C")[metric].sd }));
  const selected = data.cells[diagnosticModel][lang];
  const diagnostic = data.trainingDiagnostics[diagnosticModel][lang].at(-1);
  return <div className="qa-dashboard">
    <section className="qa-intro"><span className="qa-eyebrow">ARABIC & MALAY QUESTION ANSWERING</span><h1>Better answers.<br/><span>Does a reward help?</span></h1><p>Two models. The same questions. Standard training vs NLI reward.</p></section>
    <div className="qa-workspace"><nav className="qa-tabs" aria-label="Explore the study">{[["results","Results"],["compare","Compare"],["examples","Answers"],["reward","Reward checks"]].map(([id,name])=><button key={id} aria-current={page===id ? "page" : undefined} onClick={()=>setPage(id)}>{name}</button>)}</nav>
      <div className="qa-language"><span>Question language</span><Choice value={lang} onChange={v=>{setLang(v);setExampleIndex(0);}} label="Question language" choices={[{id:"arabic",name:"Arabic"},{id:"malay",name:"Malay"}]}/></div>
      <section className="qa-content" key={page}>
      {page === "results" && <>
        <div className="qa-verdict"><span className="qa-eyebrow">HIGHEST ANSWER ACCURACY</span><h2>{names[winner]} leads in {lang === "arabic" ? "Arabic" : "Malay"}.</h2><p>The reward adds small, mixed changes.</p><small>Models differ in size and design. Treat this ranking as a guide.</small></div>
        <div className="qa-model-grid">{models.map(model=><article className={`qa-model ${winner===model ? "qa-best" : ""}`} key={model}><header><div><h3>{names[model]}</h3><small>{fullNames[model]}</small></div>{model===winner && <span className="qa-badge">Best accuracy</span>}</header><div className="qa-scores">{(["em","f1"] as Metric[]).map(key=><div key={key}><label>{metrics[key].label}<Help text={metrics[key].help}/></label><strong>{format(cell(model)[key].mean)}</strong><small>{metrics[key].short} · standard training</small></div>)}</div><div className="qa-support"><span>Passage support <Help text={metrics.faith.help}/></span><strong>{format(cell(model).faith.mean)} <ArrowRight size={14}/> {format(cell(model,"C").faith.mean)}</strong><small>Standard → reward</small></div></article>)}</div>
        <div className="qa-next"><span>Want to see what the models said?</span><button onClick={()=>setPage("examples")}>Read the answers <ArrowRight size={16}/></button></div>
      </>}
      {page === "compare" && <>
        <div className="qa-heading"><h2>Compare the scores</h2><p>Higher is better. Small lines show variation across runs.</p></div>
        <div className="qa-filters"><div>Measure<Choice value={metric} onChange={setMetric} label="Measure" choices={(Object.keys(metrics) as Metric[]).map(id=>({id,name:metrics[id].short}))}/></div><div>Training<Choice value={training} onChange={setTraining} label="Training" choices={[{id:"both",name:"Both"},{id:"B",name:"Standard"},{id:"C",name:"Reward"}]}/></div></div>
        <div className="qa-chart-title"><strong>{metrics[metric].label}</strong><Help text={metrics[metric].help}/><span><i className="qa-dot standard"/> Standard <i className="qa-dot reward"/> Reward</span></div>
        <div className="qa-chart" role="img" aria-label={`${metrics[metric].label} comparing standard and reward training in ${lang}`}><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} margin={{top:20,right:18,left:0,bottom:0}} barGap={10}><CartesianGrid vertical={false} stroke="var(--line)"/><XAxis dataKey="name" tick={{fill:"var(--muted)",fontSize:13}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{fill:"var(--muted)",fontSize:11}} axisLine={false} tickLine={false} width={38}/><Tooltip formatter={value=>typeof value === "number" ? format(value) : String(value)} contentStyle={{background:"var(--card-solid)",border:"1px solid var(--line)",borderRadius:12,color:"var(--ink)"}}/>{training!=="C" && <Bar dataKey="standard" name="Standard" fill="#0f766e" radius={[6,6,0,0]} maxBarSize={70}><ErrorBar dataKey="standardError" direction="y" width={5} stroke="var(--ink)"/></Bar>}{training!=="B" && <Bar dataKey="reward" name="Reward" fill="#d09038" radius={[6,6,0,0]} maxBarSize={70}><ErrorBar dataKey="rewardError" direction="y" width={5} stroke="var(--ink)"/></Bar>}</BarChart></ResponsiveContainer></div>
        <table className="qa-table"><caption className="qa-sr">Exact values, mean ± standard deviation</caption><thead><tr><th>Model</th><th>Standard</th><th>With reward</th></tr></thead><tbody>{models.map(model=><tr key={model}><th>{names[model]}</th><td>{format(cell(model)[metric].mean)} <small>± {format(cell(model)[metric].sd)}</small></td><td>{format(cell(model,"C")[metric].mean)} <small>± {format(cell(model,"C")[metric].sd)}</small></td></tr>)}</tbody></table>
        <details className="qa-details"><summary>More details: reward effect</summary><p>Reward minus standard. 95% intervals are bootstrapped over examples; these tests are exploratory.</p>{models.map(model=>{const c=data.contrasts[model][lang]["C-B"][metric];return <p key={model}><b>{names[model]}</b>: {c.mean>=0?"+":""}{format(c.mean)} · interval {format(c.lo)} to {format(c.hi)} · positive in {c.seeds_positive}/{c.n_seeds} seeds.</p>;})}</details>
      </>}
      {page === "examples" && <>
        <div className="qa-heading"><h2>Read the answers</h2><p>One question. Both models. Before and after reward.</p></div>
        <div className="qa-example-nav"><button aria-label="Previous example" disabled={exampleIndex===0} onClick={()=>setExampleIndex(i=>i-1)}><ChevronLeft size={18}/></button><label><span>Example {exampleIndex+1} of {examples.length}</span><select aria-label="Choose a question" value={exampleIndex} onChange={e=>setExampleIndex(Number(e.target.value))}>{examples.map((e,i)=><option key={e.id} value={i}>{i+1}. {e.question}</option>)}</select></label><button aria-label="Next example" disabled={exampleIndex===examples.length-1} onClick={()=>setExampleIndex(i=>i+1)}><ChevronRight size={18}/></button></div>
        <article className="qa-question"><span className="qa-eyebrow">QUESTION</span><h3 dir="auto">{example.question}</h3><details><summary>Read the passage</summary><p dir="auto">{example.context}</p></details></article><div className="qa-reference"><span>Reference answer</span><strong dir="auto">{example.gold}</strong></div>
        <div className="qa-model-grid">{models.map(model=><article className="qa-answer" key={model}><h3>{names[model]}</h3>{(["B","C"] as const).map(cond=>{const answer=example.answers[model][cond];return <section key={cond}><header><b>{cond==="B"?"Standard":"With reward"}</b>{answer.em===1 && <span className="qa-badge">Exact match</span>}</header><p dir="auto">{answer.answer || "No answer"}</p><small>F1 {answer.f1?.toFixed(3)??"—"} · NLI {answer.faith?.toFixed(3)??"—"}</small></section>;})}<footer>{example.answers[model].B.answer===example.answers[model].C.answer ? "Same answer with reward." : "Reward changed this answer."}</footer></article>)}</div><details className="qa-details"><summary>About these examples</summary><p>Five fixed held-out questions per language from seed 42. Each is shared by both models; at least one answer differs across models or training conditions. Summary scores average five seeds.</p></details>
      </>}
      {page === "reward" && <>
        <div className="qa-heading"><h2>Why no clear reward benefit?</h2><p>Three checks help explain the result.</p></div><Choice value={diagnosticModel} onChange={setDiagnosticModel} label="Model for reward checks" choices={models.map(id=>({id,name:names[id]}))}/>
        <div className="qa-checks"><article><span className="qa-check-number">01</span><h3>Compare with controls</h3><p>Does a real reward beat zero or shuffled reward?</p>{["C_l0","C_shuf"].map(key=>{const c=data.contrasts[diagnosticModel][lang][`C-${key}`].faith;return <div className="qa-check-row" key={key}><span>{key==="C_l0"?"vs zero reward":"vs shuffled reward"}</span><strong>{c.mean>=0?"+":""}{format(c.mean)}</strong><small>NLI difference · 95% interval {format(c.lo)} to {format(c.hi)}</small></div>;})}</article><article><span className="qa-check-number">02</span><h3>Answers get shorter</h3><p>Average words before and after standard fine-tuning.</p><div className="qa-big-flow"><b>{selected.A_frozen.metrics.n_words.mean.toFixed(1)}</b><ArrowRight/><b>{selected.B.metrics.n_words.mean.toFixed(1)}</b></div><small>Answer length can affect NLI scores.</small><details className="qa-details"><summary>Length-matched results</summary><p>NLI change for answers in the same length group. This helps check the role of length; it does not prove causation.</p>{data.length.contrasts.filter(r=>r.model===diagnosticModel && r.lang===lang && r.kind==="matched" && r.a==="C" && r.ref==="B").map((r,i)=><p key={i}>{String(r.bin)} words: {Number(r.difference)>=0?"+":""}{format(Number(r.difference))} · interval {format(Number(r.lo))} to {format(Number(r.hi))}</p>)}</details></article><article><span className="qa-check-number">03</span><h3>Reward updates fade</h3><p>Late steps with a near-zero reward-training term.</p><strong className="qa-big-stat">{Math.round((diagnostic?.inactive_share??0)*100)}<small>%</small></strong><small>Standard answer training continued.</small><details className="qa-details"><summary>How this is measured</summary><p>Final quarter of logged training steps, pooled across seeds. Near zero means |policy-gradient term| &lt; 0.001. Memorisation is an interpretation supported by these diagnostics, not a direct causal test.</p></details></article></div>
      </>}
      </section>
    </div>
    <footer className="qa-study-notes"><details><summary>Study details & limits</summary><p>Five seeds per trained condition. Standard = cross-entropy (B); reward = cross-entropy + NLI reward (C). Both use LoRA.</p><p>The models differ in size and architecture. Comparisons are exploratory. Human evaluation covers the earlier mT5 study only. The manuscript is under review.</p><p>Source: results/hero_v2. Scores are computed from saved runs; mean ± sample standard deviation.</p></details><a href="data/hero.json" download>Download data</a></footer>
  </div>;
}
