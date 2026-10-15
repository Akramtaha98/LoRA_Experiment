import { useEffect, useMemo, useState } from "react";
import {
  ArrowDown,
  BookOpen,
  Check,
  ChevronDown,
  Download,
  Search,
} from "lucide-react";
import type { OutputAnswer, OutputItem, ResultsFile } from "../data/results";
import {
  MODELS,
  answerVerdict,
  englishGuide,
  evidenceExcerpt,
  explainAnswer,
  guidedExample,
  highlightParts,
  matchesFilter,
  trainingChange,
  type AnswerFilter,
  type Language,
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
const signed = (n: number | null | undefined) =>
  n == null ? "Unavailable" : `${n >= 0 ? "+" : ""}${n.toFixed(4)}`;
const FILTERS: { id: AnswerFilter; label: string }[] = [
  { id: "all", label: "All examples" },
  { id: "matched", label: "All exact matches" },
  { id: "missed", label: "All miss the reference" },
  { id: "disagree", label: "Different answers" },
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
            {answerVerdict(answer)}
          </span>
          <div className="e-explanation">
            <h4>What happened?</h4>
            <p>{explainAnswer(example, answer)}</p>
          </div>
          <details className="e-answer-scores">
            <summary>Recorded scores</summary>
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
            <p className="e-caption">
              Token F1 measures word overlap, not semantic correctness or
              factual support.
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

function ExampleCard({
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
          {featured ? "Guided example" : example.id}
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
      <Evidence example={example} />
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
  language: Language;
  names: Record<string, string>;
}) {
  const [model, setModel] = useState<Model>("qwen3"),
    [adapter, setAdapter] = useState("qlora"),
    [index, setIndex] = useState(0);
  const available = items.filter((i) => i.language === language);
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
    <section id="training" className="s-section e-training">
      <span className="e-kicker">Training comparison</span>
      <h2>What changes with the NLI reward?</h2>
      <p className="s-lead">
        The same model, question, adapter and seed. These are saved answers, not
        a live generation demo.
      </p>
      <div className="e-controls">
        <label>
          Model
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as Model)}
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {names[m]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Adapter
          <select value={adapter} onChange={(e) => setAdapter(e.target.value)}>
            <option value="qlora">QLoRA</option>
            <option value="dora">DoRA</option>
          </select>
        </label>
        <label className="e-question-select">
          Saved question
          <select
            value={Math.min(index, Math.max(0, available.length - 1))}
            onChange={(e) => setIndex(Number(e.target.value))}
          >
            {available.map((x, i) => (
              <option key={x.id} value={i}>
                {i + 1}. {x.question}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="e-caption">
        {available.length} exported pairs of questions for this language · Seed
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
            {result(before, "B · Standard training")}
            {result(after, "C · NLI-reward training")}
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
        <p>No saved comparison is available for this language.</p>
      )}
    </section>
  );
}

export default function Home({ d }: { d: ResultsFile }) {
  const [language, setLanguage] = useState<Language>("arabic"),
    [filter, setFilter] = useState<AnswerFilter>("all"),
    [query, setQuery] = useState(""),
    [page, setPage] = useState(1);
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
  const names = d.compare.models;
  const inLanguage = useMemo(
    () => (examples ?? []).filter((e) => e.language === language),
    [examples, language],
  );
  const filtered = useMemo(
    () =>
      inLanguage.filter(
        (e) =>
          matchesFilter(e, filter) &&
          (!query.trim() ||
            [
              e.question,
              e.gold,
              e.context,
              ...Object.values(e.answers).map((a) => a?.a ?? ""),
            ]
              .join(" ")
              .toLocaleLowerCase()
              .includes(query.trim().toLocaleLowerCase())),
      ),
    [inLanguage, filter, query],
  );
  const featured = guidedExample(examples ?? [], language);
  const rank = d.compare.results[`B|${language}|em`];
  const pages = Math.max(1, Math.ceil(filtered.length / 5)),
    currentPage = Math.min(page, pages);
  function changeLanguage(value: Language) {
    setLanguage(value);
    setPage(1);
  }
  function download() {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = `saved-answers-${language}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
  return (
    <>
      <section className="s-hero e-hero">
        <span className="s-eyebrow">Arabic and Malay question answering</span>
        <h1>
          One question.
          <br />
          <span className="s-grad">Three model answers.</span>
        </h1>
        <p>
          Read what each model actually answered, find the evidence in the
          passage, and explore what changes with an NLI training reward.
        </p>
        <nav className="e-hero-links" aria-label="Page sections">
          <a href="#guided">
            Explore an example <ArrowDown size={16} />
          </a>
          <a href="#training">Compare training</a>
          <a href="#results">Overall results</a>
        </nav>
      </section>
      <div className="e-language">
        <span>Language for every section</span>
        <div className="s-seg" role="group" aria-label="Language">
          {(["arabic", "malay"] as const).map((l) => (
            <button
              key={l}
              type="button"
              aria-pressed={language === l}
              className={language === l ? "on" : ""}
              onClick={() => changeLanguage(l)}
            >
              {l === "arabic" ? "Arabic" : "Malay"}
            </button>
          ))}
        </div>
      </div>
      <section id="guided" className="s-section e-guide">
        <span className="e-kicker">Start with the evidence</span>
        <h2>What did each model do?</h2>
        <p className="s-lead">
          The explanations describe visible outputs, not the model’s hidden
          reasoning. Exact match and word overlap are separate from human
          judgments of meaning.
        </p>
        {error ? (
          <div className="e-load-error" role="alert">
            <p>Saved examples could not be loaded: {error}</p>
            <button
              onClick={() => {
                setError(null);
                setAttempt((a) => a + 1);
              }}
            >
              Retry loading examples
            </button>
          </div>
        ) : !examples ? (
          <p role="status">Loading saved answers…</p>
        ) : featured ? (
          <ExampleCard
            key={featured.id}
            example={featured}
            names={names}
            featured
          />
        ) : (
          <p>No saved examples are available for this language.</p>
        )}
      </section>
      <section
        className="s-section e-how"
        aria-label="How the experiment works"
      >
        <h2>How does this work?</h2>
        <ol>
          <li>
            <strong>A question and a passage</strong>
            <p>
              Every example supplies the source text. These are closed-passage
              tests, so the model does not browse the web here.
            </p>
          </li>
          <li>
            <strong>Two training objectives</strong>
            <p>
              Standard training rewards the reference text. The composite
              objective adds an NLI entailment reward. LoRA-family adapters
              train a small set of model parameters.
            </p>
          </li>
          <li>
            <strong>Saved outputs, separate checks</strong>
            <p>
              Exact match checks the normalized reference. Token F1 checks word
              overlap. NLI checks entailment according to a scorer; a higher
              score alone does not prove a better answer.
            </p>
          </li>
        </ol>
      </section>
      <TrainingComparison
        key={language}
        items={d.compare.outputs}
        language={language}
        names={names}
      />
      <section id="examples" className="s-section">
        <span className="e-kicker">Explore the saved sample</span>
        <h2>Browse every exported example</h2>
        <p className="s-lead">
          {examples?.length ?? "…"} saved questions across both languages. The
          example export is a subset of the held-out evaluation, not the full
          benchmark.
        </p>
        <p className="e-caption">
          {[
            ...new Set(
              inLanguage.map(
                (e) => CONFIG[e.config ?? ""] ?? "Unspecified configuration",
              ),
            ),
          ].join(", ")}{" "}
          · QLoRA · seed 42. The overall ranking below uses standard-training
          averages over adapters and seeds.
        </p>
        <div className="s-tools">
          <label className="s-search">
            <Search size={15} aria-hidden />
            <span className="e-sr-only">
              Search saved questions, answers or passages
            </span>
            <input
              type="search"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              placeholder="Search questions, answers or evidence"
            />
          </label>
          <button
            className="e-secondary"
            disabled={!filtered.length}
            onClick={download}
          >
            <Download size={15} /> Download filtered records
          </button>
        </div>
        <div className="s-score">
          {MODELS.map((m) => {
            const available = inLanguage.filter((e) => e.answers[m]);
            return (
              <div key={m}>
                <b style={{ color: COLOR[m] }}>
                  {available.filter((e) => e.answers[m]?.em === 1).length}/
                  {available.length}
                </b>
                <span>
                  {names[m]} exact matches in this language’s exported sample
                </span>
              </div>
            );
          })}
        </div>
        <div className="s-chips" role="group" aria-label="Filter model matches">
          {FILTERS.map((f) => (
            <button
              type="button"
              key={f.id}
              aria-pressed={filter === f.id}
              className={`s-chip ${filter === f.id ? "on" : ""}`}
              onClick={() => {
                setFilter(f.id);
                setPage(1);
              }}
            >
              {f.label}
            </button>
          ))}
        </div>
        <p role="status" className="e-caption">
          {filtered.length} matching examples · counts above describe the full{" "}
          {language === "arabic" ? "Arabic" : "Malay"} exported sample.
        </p>
        <div className="s-list">
          {filtered.slice((currentPage - 1) * 5, currentPage * 5).map((e) => (
            <ExampleCard key={e.id} example={e} names={names} />
          ))}
        </div>
        {examples && !filtered.length && (
          <p className="e-empty">
            No examples match these filters.{" "}
            <button
              onClick={() => {
                setFilter("all");
                setQuery("");
                setPage(1);
              }}
            >
              Clear filters
            </button>
          </p>
        )}
        {pages > 1 && (
          <nav className="e-pagination" aria-label="Example pages">
            <button
              disabled={currentPage === 1}
              onClick={() => setPage(currentPage - 1)}
            >
              Previous
            </button>
            <span>
              Page {currentPage} of {pages}
            </span>
            <button
              disabled={currentPage === pages}
              onClick={() => setPage(currentPage + 1)}
            >
              Next
            </button>
          </nav>
        )}
      </section>
      <section id="results" className="s-section">
        <span className="e-kicker">Full held-out evaluation</span>
        <h2>Overall exact-match results</h2>
        <p className="s-lead">
          {language === "arabic"
            ? "Arabic · XQuAD · 716"
            : "Malay · Belebele · 430"}{" "}
          held-out questions. Standard training, averaged over QLoRA and DoRA
          and three seeds.
        </p>
        <p className="e-protocol-note">
          {d.compare.protocol_note} These scores describe the tested systems
          under their recorded protocols; they do not isolate architecture or
          the training reward.
        </p>
        <section className="s-podium" aria-label="Overall exact-match scores">
          {rank.ranking.map((r, i) => (
            <article key={r.model} className="s-model">
              <div className="s-rank">#{i + 1} in this evaluation</div>
              <h3>{names[r.model]}</h3>
              <div className="s-big">{percent(r.mean)}</div>
              <p className="s-big-l">Exact reference match</p>
              <div className="s-bar" aria-hidden>
                <div
                  style={{
                    width: `${r.mean * 100}%`,
                    background: COLOR[r.model as Model],
                  }}
                />
              </div>
              <p className="e-caption">Common scale: 0–100%</p>
              <p className="s-sub">
                Token F1:{" "}
                <strong>
                  {percent(
                    d.compare.results[`B|${language}|f1`].ranking.find(
                      (x) => x.model === r.model,
                    )!.mean,
                  )}
                </strong>
              </p>
            </article>
          ))}
        </section>
        <details className="s-fine">
          <summary>Research details, uncertainty and data sources</summary>
          <p>
            Scores below retain the stored numeric precision. The
            combined-language results in the source use equal language
            weighting; they are not a pooled question count.
          </p>
          <pre>{JSON.stringify(rank, null, 2)}</pre>
          <h3>Separate corrected mT5 study</h3>
          <p>
            {d.corrected.n_runs} run records. Primary B-versus-C comparisons use{" "}
            {d.corrected.seeds.length} seeds. These contrasts measure the NLI
            reward, rather than a comparison between model architectures.
          </p>
          <p>
            NLI score change: {signed(d.corrected.headline[language].mean)} (95%
            CI {signed(d.corrected.headline[language].lo)} to{" "}
            {signed(d.corrected.headline[language].hi)}). Exact-match change:{" "}
            {signed(d.corrected.headline[`em_${language}`].mean)}. Both are on a
            0–1 scale.
          </p>
          <p>
            Arabic is extractive QA. Malay uses Belebele multiple-choice
            references, so the tasks differ. The NLI reward scorer can be
            over-optimized; correctness and independent evaluation must be
            examined separately.
          </p>
          <p>
            <a
              href={`${import.meta.env.BASE_URL}data/results.json`}
              target="_blank"
              rel="noreferrer"
            >
              Full stored results JSON
            </a>{" "}
            ·{" "}
            <a
              href={`${import.meta.env.BASE_URL}data/compare-examples.json`}
              target="_blank"
              rel="noreferrer"
            >
              All exported answers
            </a>{" "}
            ·{" "}
            <a
              href="https://github.com/Akramtaha98/LoRA_Experiment/tree/main/results/corrected_v2"
              target="_blank"
              rel="noreferrer"
            >
              Corrected-study source files
            </a>
          </p>
        </details>
      </section>
    </>
  );
}
