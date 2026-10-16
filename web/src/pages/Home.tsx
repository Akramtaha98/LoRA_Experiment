import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Layers,
  Play,
  BookOpen,
  Check,
  ChevronDown,
  Download,
  Search,
} from "lucide-react";
import type { OutputAnswer, OutputItem, ResultsFile } from "../data/results";
import {
  MODELS,
  englishGuide,
  evidenceExcerpt,
  explainAnswer,
  guidedExample,
  selectExamples,
  selectedExampleIndex,
  type ExampleLanguage,
  highlightParts,
  trainingChange,
  trainingPairs,
  trainingSummary,
  type AnswerFilter,
  type Model,
  type SavedAnswer,
  type SavedExample,
} from "../lib/evidence";

const COLOR: Record<Model, string> = {
  qwen3: "var(--accent)",
  t5gemma2: "var(--accent-2)",
  mt5: "var(--gold)",
};
const CONFIG: Record<string, string> = {
  B_ce_lora: "Standard training (B)",
  C_composite_lora: "NLI-reward training (C)",
};
const percent = (n: number) => `${(n * 100).toFixed(2)}%`;
type Tab = "start" | "examples" | "training";
const TABS: { id: Tab; label: string }[] = [
  { id: "start", label: "Start here" },
  { id: "examples", label: "100 examples" },
  { id: "training", label: "Training comparison" },
];
const FILTERS: { id: AnswerFilter; label: string }[] = [
  { id: "all", label: "Every question" },
  { id: "disagree", label: "Different answers" },
  { id: "matched", label: "All three match" },
  { id: "missed", label: "None match" },
  { id: "qwen3", label: "Qwen3 matches" },
  { id: "mt5", label: "mT5 matches" },
  { id: "t5gemma2", label: "T5Gemma matches" },
];

function Highlight({
  text,
  reference,
  answer = "",
}: {
  text: string;
  reference: string;
  answer?: string;
}) {
  return (
    <>
      {highlightParts(text, reference, answer).map((p, i) =>
        p.kind ? (
          <mark key={i} className={`e-mark ${p.kind}`}>
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

function Evidence({ example }: { example: SavedExample }) {
  const [model, setModel] = useState<Model>("qwen3");
  const answer = example.answers[model]?.a ?? "";
  const excerpt = evidenceExcerpt(example.context, example.gold, answer);
  return (
    <section className="e-evidence" aria-label="Evidence for this question">
      <div className="e-evidence-head">
        <h4>
          <BookOpen size={17} /> Evidence from the passage
        </h4>
        <label>
          Highlight a model’s wording
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as Model)}
          >
            <option value="qwen3">Qwen3</option>
            <option value="mt5">mT5</option>
            <option value="t5gemma2">T5Gemma 2</option>
          </select>
        </label>
      </div>
      <p className="e-legend">
        <span>
          <i className="e-mark reference" /> Reference wording
        </span>
        <span>
          <i className="e-mark answer" /> Selected model’s wording
        </span>
      </p>
      <blockquote dir="auto">
        <Highlight text={excerpt} reference={example.gold} answer={answer} />
      </blockquote>
      {!example.context.includes(example.gold) && (
        <p className="e-caption">
          The reference does not appear verbatim in this passage. The excerpt is
          context, not a verified supporting span.
        </p>
      )}
      {answer && !example.context.includes(answer) && (
        <p className="e-caption">
          This model’s complete output does not appear verbatim in the passage.
          That alone does not establish whether its meaning is supported.
        </p>
      )}
      <details>
        <summary>
          Read full passage <ChevronDown size={14} />
        </summary>
        <p className="e-full-passage" dir="auto">
          <Highlight
            text={example.context}
            reference={example.gold}
            answer={answer}
          />
        </p>
      </details>
    </section>
  );
}

function AnswerCard({
  example,
  answer,
  model,
  name,
}: {
  example: SavedExample;
  answer: SavedAnswer | undefined;
  model: Model;
  name: string;
}) {
  return (
    <article className={`s-ans ${answer?.em === 1 ? "best" : ""}`}>
      <div className="s-ans-head">
        <i style={{ background: COLOR[model] }} />
        {name}
      </div>
      {answer ? (
        <>
          <p dir="auto" className="s-ans-text">
            {answer.a || "(Empty saved output)"}
          </p>
          <span className={`s-tag ${answer.em === 1 ? "ok" : "neutral"}`}>
            {answer.em === 1 && <Check size={13} />}
            {answer.em === 1
              ? "Matches reference"
              : answer.f1 > 0
                ? "Some shared words"
                : "Different answer"}
          </span>
          <details className="simple-answer-detail">
            <summary>Why this result?</summary>
            <p>{explainAnswer(example, answer)}</p>
            <dl>
              <div>
                <dt>Exact reference match</dt>
                <dd>{answer.em === 1 ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Word overlap (token F1)</dt>
                <dd>{answer.f1.toFixed(3)}</dd>
              </div>
            </dl>
            <p className="e-caption">
              Shared words alone do not establish a correct meaning.
            </p>
          </details>
        </>
      ) : (
        <p className="e-caption">
          This model’s answer is not available in the saved export.
        </p>
      )}
    </article>
  );
}

export function ExampleCard({
  example,
  names,
  featured = false,
}: {
  example: SavedExample;
  names: Record<string, string>;
  featured?: boolean;
}) {
  const guide = englishGuide(example);
  return (
    <article
      className={`s-ex ${featured ? "e-featured" : ""}`}
      data-example-id={example.id}
    >
      <div className="s-ex-top">
        <span className="s-num">
          {featured
            ? "Guided example"
            : example.language === "arabic"
              ? "Arabic"
              : "Malay"}
        </span>
        <h3 dir="auto">{example.question}</h3>
      </div>
      <p className="s-gold">
        <span>Reference answer</span>
        <strong dir="auto">{example.gold}</strong>
      </p>
      {guide && (
        <details className="e-english">
          <summary>English guide to this example</summary>
          <p>
            <strong>Question:</strong> {guide.question}
          </p>
          <p>
            <strong>Reference:</strong> {guide.reference}
          </p>
          <p>
            <strong>Passage summary:</strong> {guide.evidence}
          </p>
          <p className="e-caption">
            Human-written English guide. Original questions, outputs and scores
            remain unchanged.
          </p>
        </details>
      )}
      <div className="s-answers">
        {MODELS.map((model) => (
          <AnswerCard
            key={model}
            example={example}
            answer={example.answers[model]}
            model={model}
            name={names[model]}
          />
        ))}
      </div>
      <details className="simple-evidence">
        <summary>
          <BookOpen size={16} /> Show passage evidence
        </summary>
        <Evidence example={example} />
      </details>
      <details className="e-provenance">
        <summary>Example source and saved record</summary>
        <p>
          Record: <code>{example.id}</code> ·{" "}
          {CONFIG[example.config ?? ""] ??
            "Training configuration not specified in this export"}
          .
        </p>
        <p>
          Exported example setup: QLoRA, seed 42. The original export does not
          include per-answer run IDs or source-log paths. Consult the build
          script for lineage.
        </p>
        <a
          href="https://github.com/Akramtaha98/LoRA_Experiment/blob/main/scripts/build_compare_examples.py"
          target="_blank"
          rel="noreferrer"
        >
          Example export script
        </a>
        <pre>{JSON.stringify(example, null, 2)}</pre>
      </details>
    </article>
  );
}

function TrainingComparison({
  items,
  language,
  names,
}: {
  items: OutputItem[];
  language: ExampleLanguage;
  names: Record<string, string>;
}) {
  const [model, setModel] = useState<Model>("t5gemma2"),
    [adapter, setAdapter] = useState("dora"),
    [changedOnly, setChangedOnly] = useState(true),
    [index, setIndex] = useState(0);
  const inLanguage = items.filter(
    (i) => language === "all" || i.language === language,
  );
  const summary = trainingSummary(inLanguage, model, adapter);
  const overall = MODELS.flatMap((m) =>
    ["qlora", "dora"].map((a) => trainingSummary(items, m, a)),
  );
  const total = overall.reduce((n, s) => n + s.total, 0);
  const changed = overall.reduce((n, s) => n + s.changed, 0);
  const available = trainingPairs(inLanguage, model, adapter, changedOnly);
  const item = available[Math.min(index, Math.max(0, available.length - 1))];
  const pair = item?.answers[model]?.[adapter];
  const before = pair?.B_ce_lora,
    after = pair?.C_composite_lora;
  function result(answer: OutputAnswer | undefined, title: string) {
    return (
      <article className="e-training-answer">
        <h3>{title}</h3>
        {answer ? (
          <>
            <p dir="auto" className="e-output">
              {answer.generated || "(Empty saved output)"}
            </p>
            <dl>
              <div>
                <dt>Exact match</dt>
                <dd>{answer.em}</dd>
              </div>
              <div>
                <dt>Token F1</dt>
                <dd>{answer.f1.toFixed(3)}</dd>
              </div>
            </dl>
          </>
        ) : (
          <p>No matching saved output is available.</p>
        )}
      </article>
    );
  }
  return (
    <section id="training" className="simple-training">
      <h2>What changed with NLI reward?</h2>
      <p className="s-lead">
        The same model and question, two training methods. A different answer
        can be better, worse, or just different wording.
      </p>
      <div className="training-overview">
        <div>
          <strong>
            {changed}
            <span> / {total}</span>
          </strong>
          <span>saved comparisons changed</span>
        </div>
        <p>
          Some answers stay identical. This sample contains 10 questions × 3
          models × 2 adapters, using seed 42. It does not establish a general
          training benefit.
        </p>
      </div>
      <div className="training-pickers">
        <fieldset className="choice-group">
          <legend>1. Choose a model</legend>
          <div>
            {MODELS.map((m) => (
              <button
                type="button"
                key={m}
                aria-pressed={model === m}
                onClick={() => {
                  setModel(m);
                  setIndex(0);
                }}
              >
                {names[m]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset className="choice-group">
          <legend>2. Choose an adapter</legend>
          <div>
            {["qlora", "dora"].map((a) => (
              <button
                type="button"
                key={a}
                aria-pressed={adapter === a}
                onClick={() => {
                  setAdapter(a);
                  setIndex(0);
                }}
              >
                {a === "qlora" ? "QLoRA" : "DoRA"}
              </button>
            ))}
          </div>
        </fieldset>
      </div>
      <div className="training-selection-summary" role="status">
        <strong>
          {summary.changed} of {summary.total} answers changed
        </strong>
        <span>
          {" "}
          {summary.unchanged} stayed the same for {names[model]} +{" "}
          {adapter === "qlora" ? "QLoRA" : "DoRA"} in the selected language.
        </span>
      </div>
      <fieldset className="choice-group training-view">
        <legend>3. Choose what to explore</legend>
        <div>
          <button
            type="button"
            aria-pressed={changedOnly}
            onClick={() => {
              setChangedOnly(true);
              setIndex(0);
            }}
          >
            Changed answers · {summary.changed}
          </button>
          <button
            type="button"
            aria-pressed={!changedOnly}
            onClick={() => {
              setChangedOnly(false);
              setIndex(0);
            }}
          >
            All saved pairs · {summary.total}
          </button>
        </div>
      </fieldset>
      {available.length > 0 && (
        <div className="e-controls training-question-control">
          <label className="e-question-select">
            Saved question
            <select
              value={Math.min(index, Math.max(0, available.length - 1))}
              onChange={(e) => setIndex(Number(e.target.value))}
            >
              {available.map((x, i) => (
                <option key={x.id} value={i}>
                  {i + 1}. {x.language === "arabic" ? "Arabic" : "Malay"} ·{" "}
                  {x.question}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <p className="e-caption">
        {available.length} {changedOnly ? "changed" : "saved"} questions · Seed
        42 ·{" "}
        {model === "mt5"
          ? "Corrected mT5 protocol"
          : "Original protocol; these runs are not a corrected-protocol rerun"}
        . Claims about the reward stay within each protocol.
      </p>
      {item ? (
        <>
          <p className="s-gold">
            <span>Reference answer</span>
            <strong dir="auto">{item.gold}</strong>
          </p>
          <div className="e-training-grid">
            {result(before, "Standard training")}
            {result(after, "With NLI reward")}
          </div>
          <p className="e-change" role="status">
            {trainingChange(before, after)}
          </p>
          <details className="e-provenance">
            <summary>Training pair source</summary>
            <p>
              <code>
                results.json → compare.outputs → {item.id} → {model} → {adapter}
              </code>
              . These records use seed 42 according to{" "}
              <a
                href="https://github.com/Akramtaha98/LoRA_Experiment/blob/main/scripts/build_compare_outputs.py"
                target="_blank"
                rel="noreferrer"
              >
                the export script
              </a>
              .
            </p>
            <pre>
              {JSON.stringify(
                { id: item.id, model, adapter, seed: 42, before, after },
                null,
                2,
              )}
            </pre>
          </details>
        </>
      ) : (
        <div className="training-empty">
          <h3>No changed answers in this selection</h3>
          <p>
            All {summary.total} saved pairs have identical wording. That is a
            result of these saved runs, rather than a display error. Choose
            another model or adapter to explore differences.
          </p>
          <button
            className="simple-secondary"
            onClick={() => {
              setChangedOnly(false);
              setIndex(0);
            }}
          >
            View the unchanged pairs
          </button>
        </div>
      )}
    </section>
  );
}

function ResearchDetails({ d }: { d: ResultsFile }) {
  return (
    <details className="simple-research">
      <summary>Overall results and research details</summary>
      <p>
        These scores use the full held-out tests. The 100 browsable examples are
        a smaller saved sample and can use a different training objective.
      </p>
      <p>
        {d.compare.protocol_note} The recorded protocols differ, so these
        results do not isolate architecture or a training effect.
      </p>
      <div className="simple-table">
        <table>
          <thead>
            <tr>
              <th>Language</th>
              <th>Model</th>
              <th>Exact match</th>
              <th>Token F1</th>
            </tr>
          </thead>
          <tbody>
            {(["arabic", "malay"] as const).flatMap((language) =>
              d.compare.results[`B|${language}|em`].ranking.map((row) => (
                <tr key={language + row.model}>
                  <td>{language === "arabic" ? "Arabic" : "Malay"}</td>
                  <td>{d.compare.models[row.model]}</td>
                  <td>{percent(row.mean)}</td>
                  <td>
                    {percent(
                      d.compare.results[`B|${language}|f1`].ranking.find(
                        (r) => r.model === row.model,
                      )!.mean,
                    )}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p>
        Arabic: 716 XQuAD questions. Malay: 430 Belebele questions.
        Standard-training scores average QLoRA and DoRA across three seeds.
        Exact match compares normalized wording; token F1 measures word overlap.
      </p>
      <p>
        The separate corrected mT5 study contains {d.corrected.n_runs} run
        records, with {d.corrected.seeds.length} seeds in the main training
        comparisons. An increased NLI score alone does not prove improved
        correctness.
      </p>
      <p>
        <a
          href={`${import.meta.env.BASE_URL}data/results.json`}
          target="_blank"
          rel="noreferrer"
        >
          Full scores, intervals and seed values
        </a>{" "}
        ·{" "}
        <a
          href={`${import.meta.env.BASE_URL}data/compare-examples.json`}
          target="_blank"
          rel="noreferrer"
        >
          Original saved examples
        </a>
      </p>
    </details>
  );
}

export default function Home({ d }: { d: ResultsFile }) {
  const [tab, setTab] = useState<Tab>(() =>
      window.location.hash === "#examples"
        ? "examples"
        : window.location.hash === "#training"
          ? "training"
          : "start",
    ),
    [language, setLanguage] = useState<ExampleLanguage>("all"),
    [query, setQuery] = useState(""),
    [filter, setFilter] = useState<AnswerFilter>("all"),
    [selectedId, setSelectedId] = useState("");
  const [examples, setExamples] = useState<SavedExample[] | null>(null),
    [error, setError] = useState<string | null>(null),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}data/compare-examples.json`, {
      signal: controller.signal,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((value: SavedExample[]) => {
        setExamples(value);
        setError(null);
      })
      .catch((e: Error) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [attempt]);
  const filtered = useMemo(
    () => selectExamples(examples ?? [], language, query, filter),
    [examples, language, query, filter],
  );
  const index = selectedExampleIndex(filtered, selectedId),
    example = filtered[index];
  const counts = {
    all: examples?.length ?? 0,
    arabic: examples?.filter((e) => e.language === "arabic").length ?? 0,
    malay: examples?.filter((e) => e.language === "malay").length ?? 0,
  };
  function navigateTab(next: Tab) {
    setTab(next);
    window.history.replaceState(null, "", `#${next}`);
  }
  function openExamples(featured = false) {
    navigateTab("examples");
    if (featured && examples) {
      const item = guidedExample(
        examples,
        language === "malay" ? "malay" : "arabic",
      );
      setQuery("");
      setFilter("all");
      if (item) setSelectedId(item.id);
    }
  }
  function download() {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], {
        type: "application/json",
      }),
      url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = "saved-model-answers.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function move(amount: number) {
    const next = filtered[index + amount];
    if (next) {
      setSelectedId(next.id);
      document.getElementById("question-navigation")?.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      });
    }
  }
  return (
    <div className="simple-app">
      <header className="simple-heading">
        <div>
          <span className="e-kicker">Real questions. Saved answers.</span>
          <h1>Compare three AI models</h1>
          <p>Choose a question and read each model’s answer.</p>
        </div>
        <span className="simple-count">
          <Layers size={16} />
          {examples ? `${examples.length} examples` : "Loading examples…"}
        </span>
      </header>
      <div className="simple-tabs" role="tablist" aria-label="Website sections">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            onClick={() => navigateTab(t.id)}
            onKeyDown={(e) => {
              let next: number | undefined;
              if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
              if (e.key === "ArrowLeft")
                next = (i + TABS.length - 1) % TABS.length;
              if (e.key === "Home") next = 0;
              if (e.key === "End") next = TABS.length - 1;
              if (next !== undefined) {
                e.preventDefault();
                navigateTab(TABS[next].id);
                document.getElementById(`tab-${TABS[next].id}`)?.focus();
              }
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <section
        id="panel-start"
        role="tabpanel"
        aria-labelledby="tab-start"
        hidden={tab !== "start"}
        className="simple-panel"
      >
        <div className="simple-welcome">
          <div className="simple-play" aria-hidden>
            <Play size={23} />
          </div>
          <h2>Start with one question</h2>
          <p>
            Every example shows the reference answer and the answers saved from
            Qwen3, mT5 and T5Gemma 2.
          </p>
          <ol className="simple-steps">
            <li>
              <span>1</span>Choose a question
            </li>
            <li>
              <span>2</span>Compare the three answers
            </li>
            <li>
              <span>3</span>Open the passage if you want to check
            </li>
          </ol>
          <div className="simple-actions">
            <button className="simple-primary" onClick={() => openExamples()}>
              Explore all {counts.all || 100} examples <ArrowRight size={17} />
            </button>
            <button
              className="simple-secondary"
              onClick={() => openExamples(true)}
            >
              Show me a guided example
            </button>
          </div>
        </div>
        <details className="simple-help">
          <summary>What do the answer labels mean?</summary>
          <p>
            <strong>Matches reference:</strong> the saved answer matches the
            expected wording after normalization.
          </p>
          <p>
            <strong>Some shared words:</strong> it overlaps the reference, but
            its meaning may still be wrong.
          </p>
          <p>
            <strong>Different answer:</strong> it does not match the reference
            under the recorded scoring rule.
          </p>
        </details>
        <details className="simple-help">
          <summary>What is the training comparison?</summary>
          <p>
            Standard training learns from reference answers. NLI-reward training
            also rewards answers judged to follow from the passage. The training
            tab shows saved examples where you can compare the two objectives
            for the same model, question, adapter and seed.
          </p>
        </details>
        <ResearchDetails d={d} />
      </section>
      <section
        id="panel-examples"
        role="tabpanel"
        aria-labelledby="tab-examples"
        hidden={tab !== "examples"}
        className="simple-panel"
      >
        <div className="simple-toolbar">
          <label>
            Language
            <select
              aria-label="Example language"
              value={language}
              onChange={(e) => {
                setLanguage(e.target.value as ExampleLanguage);
                setSelectedId("");
              }}
            >
              <option value="all">All {counts.all} examples</option>
              <option value="arabic">Arabic · {counts.arabic}</option>
              <option value="malay">Malay · {counts.malay}</option>
            </select>
          </label>
          <label className="simple-search">
            <span>Find a question</span>
            <div>
              <Search size={16} />
              <input
                type="search"
                placeholder="Search questions or answers"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setSelectedId("");
                }}
              />
            </div>
          </label>
          <label>
            Show
            <select
              aria-label="Example filter"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value as AnswerFilter);
                setSelectedId("");
              }}
            >
              {FILTERS.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        {error ? (
          <div className="e-load-error" role="alert">
            <p>We couldn’t load the examples: {error}</p>
            <button
              className="simple-secondary"
              onClick={() => {
                setError(null);
                setAttempt((n) => n + 1);
              }}
            >
              Try again
            </button>
          </div>
        ) : !examples ? (
          <p role="status">Loading saved answers…</p>
        ) : example ? (
          <>
            <div id="question-navigation" className="simple-question-nav">
              <button
                type="button"
                aria-label="Previous example"
                disabled={index === 0}
                onClick={() => move(-1)}
              >
                <ChevronLeft size={18} />
                <span>Previous</span>
              </button>
              <label>
                <span role="status">
                  Question {index + 1} of {filtered.length}
                </span>
                <select
                  aria-label="Choose a question"
                  value={example.id}
                  onChange={(e) => setSelectedId(e.target.value)}
                >
                  {filtered.map((item, i) => (
                    <option key={item.id} value={item.id}>
                      {i + 1}. {item.language === "arabic" ? "Arabic" : "Malay"}{" "}
                      · {item.question}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                aria-label="Next example"
                disabled={index === filtered.length - 1}
                onClick={() => move(1)}
              >
                <span>Next</span>
                <ChevronRight size={18} />
              </button>
            </div>
            <div className="simple-progress" aria-hidden>
              <div
                style={{ width: `${((index + 1) / filtered.length) * 100}%` }}
              />
            </div>
            <div key={example.id} className="simple-question-stage">
              <ExampleCard example={example} names={d.compare.models} />
            </div>
            <nav className="simple-bottom-nav" aria-label="Continue browsing">
              <button
                className="simple-secondary"
                aria-label="Previous question"
                disabled={index === 0}
                onClick={() => move(-1)}
              >
                <ChevronLeft size={16} /> Previous
              </button>
              <span>
                {index + 1} / {filtered.length}
              </span>
              <button
                className="simple-primary"
                aria-label="Next question"
                disabled={index === filtered.length - 1}
                onClick={() => move(1)}
              >
                Next question <ChevronRight size={16} />
              </button>
            </nav>
            <p className="simple-footnote">
              {example.language === "arabic" ? "Arabic" : "Malay"} ·{" "}
              {CONFIG[example.config ?? ""] ?? "Saved training run"} · All
              answers are the original saved outputs.
            </p>
            <details className="simple-help">
              <summary>Sample details and download</summary>
              <p>
                This export contains {examples.length} saved questions, not the
                complete evaluation. The example export uses QLoRA and seed 42
                according to its build script. Different objectives and
                protocols should be compared within matching conditions.
              </p>
              <button className="simple-secondary" onClick={download}>
                <Download size={15} /> Download these {filtered.length} records
              </button>
            </details>
          </>
        ) : (
          <div className="simple-empty">
            <h2>No questions found</h2>
            <p>Try a different search or show every question.</p>
            <button
              className="simple-primary"
              onClick={() => {
                setQuery("");
                setFilter("all");
                setLanguage("all");
              }}
            >
              Show all examples
            </button>
          </div>
        )}
      </section>
      <section
        id="panel-training"
        role="tabpanel"
        aria-labelledby="tab-training"
        hidden={tab !== "training"}
        className="simple-panel"
      >
        <label className="simple-training-language">
          Question language
          <select
            aria-label="Training language"
            value={language}
            onChange={(e) => setLanguage(e.target.value as ExampleLanguage)}
          >
            <option value="all">Arabic and Malay</option>
            <option value="arabic">Arabic</option>
            <option value="malay">Malay</option>
          </select>
        </label>
        <TrainingComparison
          items={d.compare.outputs}
          language={language}
          names={d.compare.models}
        />
      </section>
    </div>
  );
}
