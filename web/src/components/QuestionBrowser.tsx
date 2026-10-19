import { useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, ChevronDown, Search, X } from "lucide-react";
import type { HeroExample } from "../data/hero";

export default function QuestionBrowser({ examples, index, onChange }: { examples: HeroExample[]; index: number; onChange: (index: number) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const trigger = useRef<HTMLButtonElement>(null);
  const filtered = examples.map((example, i) => ({ ...example, index: i })).filter(example =>
    `${example.index + 1} ${example.question} ${example.gold}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
  );
  function close() { setOpen(false); setQuery(""); trigger.current?.focus(); }
  return <div className="question-browser" onKeyDown={event => { if (event.key === "Escape" && open) { event.stopPropagation(); close(); } }}>
    <div className="question-toolbar">
      <div className="question-pagination">
        <button className="question-step" aria-label="Previous example" disabled={index === 0} onClick={() => onChange(index - 1)}><ArrowLeft size={17}/><span>Previous</span></button>
        <span className="question-counter" aria-live="polite"><strong>{String(index + 1).padStart(2, "0")}</strong><span>/ {examples.length}</span></span>
        <button className="question-step" aria-label="Next example" disabled={index === examples.length - 1} onClick={() => onChange(index + 1)}><span>Next</span><ArrowRight size={17}/></button>
      </div>
      <button ref={trigger} className="question-browse" aria-expanded={open} aria-controls="question-picker" onClick={() => {setOpen(value => !value); setQuery("");}}><Search size={16}/><span>Browse questions</span><ChevronDown size={16} className={open ? "rotated" : ""}/></button>
    </div>
    <div className="question-progress" aria-hidden="true"><span style={{ width: `${(index + 1) / examples.length * 100}%` }}/></div>
    {open && <section id="question-picker" className="question-picker" aria-label="Choose a question">
      <header><div><strong>Choose a question</strong><span>{examples.length} saved examples</span></div><button aria-label="Close question picker" onClick={close}><X size={18}/></button></header>
      <label className="question-search"><Search size={17}/><input autoFocus aria-label="Search questions" placeholder="Search a question or answer…" value={query} onChange={event => setQuery(event.target.value)}/>{query && <button aria-label="Clear search" onClick={() => setQuery("")}><X size={15}/></button>}</label>
      <div className="question-options">{filtered.length ? filtered.map(example => <button key={example.id} className={example.index === index ? "active" : ""} aria-current={example.index === index ? "true" : undefined} onClick={() => {onChange(example.index); close();}}><span className="question-number">{String(example.index + 1).padStart(2, "0")}</span><span dir="auto">{example.question}</span>{example.index === index && <Check size={16}/>}</button>) : <p className="question-empty" role="status">No matching questions. Try another word.</p>}</div>
      <footer>{filtered.length} of {examples.length} questions</footer>
    </section>}
  </div>;
}
