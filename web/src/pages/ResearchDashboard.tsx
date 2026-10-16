import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  ArrowLeft,
  Search,
  Download,
  FlaskConical,
  Check,
  ChevronRight,
} from "lucide-react";
import type { ResultsFile } from "../data/results";
import {
  ADAPTER_NAMES,
  CONFIG_NAMES,
  MODEL_NAMES,
  PROTOCOL_NAMES,
  compatible,
  filterExperiments,
  loadSamples,
  matchedSamples,
  rankExperiments,
  score,
  type Experiment,
  type Metric,
  type Sample,
} from "../lib/experiments";
import { ExampleCard } from "./Home";
import {
  selectExamples,
  type SavedExample,
  type ExampleLanguage,
} from "../lib/evidence";

const GITHUB = "https://github.com/Akramtaha98/LoRA_Experiment/blob/main/";
type Tab = "overview" | "explore" | "compare" | "leaderboard";
const TABS: { id: Tab; name: string }[] = [
  { id: "overview", name: "Overview" },
  { id: "explore", name: "Experiments" },
  { id: "compare", name: "Compare answers" },
  { id: "leaderboard", name: "Leaderboard" },
];
const title = (r: Experiment) =>
  `${MODEL_NAMES[r.model]} · ${CONFIG_NAMES[r.config]} · ${ADAPTER_NAMES[r.adapter]} · ${r.language}`;
const language = (s: string) => (s === "arabic" ? "Arabic" : "Malay");
function Source({ row }: { row: Experiment }) {
  return (
    <details className="research-detail">
      <summary>Sources and run details</summary>
      <p>
        Scores are averages of {row.nRuns} saved seed runs.{" "}
        {row.nEval.length
          ? `Evaluation items per run: ${row.nEval.join(", ")}.`
          : "Evaluation counts are recorded in the linked conditions file."}{" "}
        {row.seeds.length
          ? `Seeds: ${row.seeds.join(", ")}.`
          : "This cell is aggregated; individual seed identities are in the source file."}
      </p>
      {row.sources.map((s, i) => (
        <a
          key={i}
          href={`${GITHUB}${s.path}${s.line ? `#L${s.line}` : ""}`}
          target="_blank"
          rel="noreferrer"
        >
          {s.path}
          {s.seed !== undefined ? ` · seed ${s.seed}` : ""}
        </a>
      ))}
    </details>
  );
}
function Scores({ row }: { row: Experiment }) {
  return (
    <div className="research-scores">
      <div>
        <span>Exact match</span>
        <strong>{score(row.em)}</strong>
        <small>Matches the reference</small>
      </div>
      <div>
        <span>Word overlap · F1</span>
        <strong>{score(row.f1)}</strong>
        <small>Shared answer words</small>
      </div>
      <div>
        <span>NLI reward score</span>
        <strong>{score(row.nli, "nli")}</strong>
        <small>Training reward, not accuracy</small>
      </div>
    </div>
  );
}
function useSamples(row: Experiment | undefined) {
  const [state, setState] = useState<{
    id: string;
    data: Sample[];
    error: string;
  }>({ id: "", data: [], error: "" });
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let live = true;
    if (row)
      loadSamples(row)
        .then((data) => {
          if (live) setState({ id: row.id, data, error: "" });
        })
        .catch((e) => {
          if (live)
            setState({ id: row.id, data: [], error: String(e.message) });
        });
    return () => {
      live = false;
    };
  }, [row, retry]);
  return {
    data: state.id === row?.id ? state.data : [],
    error: state.id === row?.id ? state.error : "",
    loading: !!row && state.id !== row.id,
    retry: () => setRetry((n) => n + 1),
  };
}
function Output({ sample, label }: { sample: Sample; label: string }) {
  return (
    <article className="research-output">
      <h4>{label}</h4>
      <p dir="auto">{sample.generated || "(Empty saved output)"}</p>
      <div className="research-output-score">
        <span>
          {sample.em === 1 ? (
            <>
              <Check size={14} /> Exact reference match
            </>
          ) : (
            "No exact reference match"
          )}
        </span>
        <span>Word overlap {score(sample.f1)}</span>
      </div>
    </article>
  );
}
function SampleBrowser({ row }: { row: Experiment }) {
  const { data, error, loading, retry } = useSamples(row);
  const [seed, setSeed] = useState(String(row.seeds[0] ?? "all")),
    [index, setIndex] = useState(0);
  const samples = data.filter((s) => seed === "all" || String(s.seed) === seed);
  const sample = samples[Math.min(index, Math.max(0, samples.length - 1))];
  return (
    <div className="research-samples">
      <div className="research-section-title">
        <div>
          <h3>Read the saved answers</h3>
          <p>
            {row.sampleCount} saved outputs. These are examples, not the full
            evaluation score above.
          </p>
        </div>
      </div>
      {loading ? (
        <p role="status">Loading saved answers…</p>
      ) : error ? (
        <p role="alert">
          {error} <button onClick={retry}>Try again</button>
        </p>
      ) : !sample ? (
        <div className="research-empty">
          <h4>Scores available; individual answers not exported</h4>
          <p>
            The corrected-study cell has measured scores, but its per-run
            prediction archive is not included in this repository. The five
            worked examples are available under Overview → Study evidence, with
            their own provenance. No answer has been filled in for this cell.
          </p>
        </div>
      ) : (
        <>
          <div className="research-sample-nav">
            <label>
              Seed
              <select
                value={seed}
                onChange={(e) => {
                  setSeed(e.target.value);
                  setIndex(0);
                }}
              >
                <option value="all">All saved seeds</option>
                {[...new Set(data.map((s) => s.seed))]
                  .sort((a, b) => a - b)
                  .map((s) => (
                    <option key={s} value={s}>
                      Seed {s}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              Saved answer
              <select
                value={Math.min(index, samples.length - 1)}
                onChange={(e) => setIndex(Number(e.target.value))}
              >
                {samples.map((s, i) => (
                  <option key={i} value={i}>
                    {i + 1}.{" "}
                    {s.question ||
                      `Question text not recorded · reference: ${s.gold}`}{" "}
                    · seed {s.seed}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="research-question" key={row.id + seed + index}>
            <span className="research-eyebrow">
              {language(row.language)} · seed {sample.seed} · saved item{" "}
              {sample.index + 1}
            </span>
            <h3 dir="auto">
              {sample.question ||
                "Question text was not stored in this run log"}
            </h3>
            <div className="s-gold">
              <span>Reference answer</span>
              <strong dir="auto">{sample.gold}</strong>
            </div>
            <Output sample={sample} label="Saved model answer" />
            {sample.context ? (
              <details className="research-detail">
                <summary>Read the passage</summary>
                <p dir="auto">{sample.context}</p>
              </details>
            ) : (
              <p className="research-note">
                The source recorded the answer and scores, but did not include
                its question or passage. They have not been inferred from
                another experiment.
              </p>
            )}
            <a
              className="research-source"
              href={`${GITHUB}${sample.source}#L${sample.line}`}
              target="_blank"
              rel="noreferrer"
            >
              Open this answer’s source record <ChevronRight size={14} />
            </a>
          </div>
          <div className="research-pagination">
            <button
              disabled={index === 0}
              onClick={() => setIndex((n) => n - 1)}
            >
              <ArrowLeft size={16} /> Previous
            </button>
            <span role="status">
              {Math.min(index + 1, samples.length)} / {samples.length}
            </span>
            <button
              disabled={index >= samples.length - 1}
              onClick={() => setIndex((n) => n + 1)}
            >
              Next <ArrowRight size={16} />
            </button>
          </div>
        </>
      )}
    </div>
  );
}
const EMPTY_FILTERS = {
  protocol: "corrected",
  model: "all",
  config: "all",
  adapter: "all",
  language: "arabic",
};
const START_FILTERS = {
  protocol: "original",
  model: "qwen3",
  config: "C",
  adapter: "qlora",
  language: "arabic",
};
function Filters({
  value,
  onChange,
  ranking = false,
}: {
  value: Record<string, string>;
  onChange: (v: Record<string, string>) => void;
  ranking?: boolean;
}) {
  const update = (key: string, next: string) => {
    let v = { ...value, [key]: next };
    if (key === "protocol") {
      if (next === "corrected") {
        v = {
          ...v,
          model: "mt5",
          adapter: "all",
          config: ["A", "D"].includes(v.config) ? "B" : v.config,
        };
      } else {
        v = {
          ...v,
          model: "all",
          adapter: "all",
          config: ["A", "B", "C", "D", "all"].includes(v.config)
            ? v.config
            : "B",
        };
      }
    }
    if (key === "config") {
      if (["A", "D"].includes(next)) v.adapter = "none";
      else if (v.adapter === "none") v.adapter = "all";
    }
    onChange(v);
  };
  const field = (
    key: string,
    label: string,
    choices: Record<string, string>,
    all = true,
  ) => (
    <label>
      {label}
      <select
        aria-label={`${ranking ? "Leaderboard" : "Experiment"} ${label.toLowerCase()}`}
        value={value[key]}
        disabled={key === "adapter" && ["A", "D"].includes(value.config)}
        onChange={(e) => update(key, e.target.value)}
      >
        {all && <option value="all">All {label.toLowerCase()}s</option>}
        {Object.entries(choices).map(([k, v]) => (
          <option key={k} value={k}>
            {v}
          </option>
        ))}
      </select>
    </label>
  );
  return (
    <div className="research-filters">
      {field("protocol", "Study", PROTOCOL_NAMES, false)}
      {field(
        "model",
        "Model",
        value.protocol === "corrected" ? { mt5: MODEL_NAMES.mt5 } : MODEL_NAMES,
      )}
      {field(
        "config",
        "Configuration",
        Object.fromEntries(
          Object.entries(CONFIG_NAMES).filter(([k]) =>
            value.protocol === "corrected"
              ? !["A", "D"].includes(k)
              : ["A", "B", "C", "D"].includes(k),
          ),
        ),
      )}
      {field(
        "adapter",
        "Adapter",
        Object.fromEntries(
          Object.entries(ADAPTER_NAMES).filter(
            ([k]) =>
              value.protocol !== "corrected" ||
              ["lora", "qlora", "dora"].includes(k),
          ),
        ),
      )}
      {field(
        "language",
        "Language",
        { arabic: "Arabic", malay: "Malay" },
        false,
      )}
    </div>
  );
}
function ResultTable({
  rows,
  onSelect,
  metric,
}: {
  rows: Experiment[];
  onSelect: (row: Experiment) => void;
  metric?: Metric;
}) {
  return (
    <div className="research-table-wrap">
      <p className="research-table-hint">
        Swipe sideways for all scores and details →
      </p>
      <table className="research-table">
        <thead>
          <tr>
            {metric && <th>Order</th>}
            <th>Model / configuration</th>
            <th>Adapter</th>
            <th>Exact match</th>
            <th>Word overlap</th>
            <th>NLI reward</th>
            <th>Seeds</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.id}>
              {metric && <td>{i + 1}</td>}
              <td>
                <strong>{MODEL_NAMES[row.model]}</strong>
                <span>{CONFIG_NAMES[row.config]}</span>
              </td>
              <td>{ADAPTER_NAMES[row.adapter]}</td>
              <td className={metric === "em" ? "metric-active" : ""}>
                {score(row.em)}
              </td>
              <td className={metric === "f1" ? "metric-active" : ""}>
                {score(row.f1)}
              </td>
              <td className={metric === "nli" ? "metric-active" : ""}>
                {score(row.nli, "nli")}
              </td>
              <td>{row.nRuns}</td>
              <td>
                <button
                  aria-label={`Explore ${title(row)}`}
                  onClick={() => onSelect(row)}
                >
                  Explore <ChevronRight size={15} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function EvidenceLibrary({ d }: { d: ResultsFile }) {
  return (
    <details className="research-library">
      <summary>Study evidence, controls and validation</summary>
      <p>
        The corrected study supersedes the original mT5 reward-effect analysis.
        The NLI reward can increase while exact match falls; a larger reward
        score alone does not establish better answers.
      </p>
      <h3>Corrected study · recorded contrasts</h3>
      <div className="research-table-wrap">
        <table className="research-table">
          <thead>
            <tr>
              <th>Contrast</th>
              <th>Mean change</th>
              <th>95% interval</th>
              <th>Seed pairs</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(d.corrected.headline).map(([key, v]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{v.mean?.toFixed(4) ?? "Unavailable"}</td>
                <td>
                  {v.lo?.toFixed(4)} to {v.hi?.toFixed(4)}
                </td>
                <td>{v.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="research-note">
        em_* contrasts measure exact match; f1_* measure token F1; other
        headline contrasts measure NLI reward. Zero-weight, shuffled-reward and
        checkpointing cells can be explored in Experiments.
      </p>
      <h3>Five worked examples · corrected-study evidence</h3>
      <p>
        These records have no adapter/seed assignment in this export, so they
        are shown here rather than attached to an experiment cell.
      </p>
      {d.corrected.examples.map((e) => (
        <details className="research-detail" key={e.src_id}>
          <summary>
            {language(e.language)} · {e.question}
          </summary>
          <p dir="auto">{e.passage}</p>
          <div className="s-gold">
            <span>Reference</span>
            <strong dir="auto">{e.gold}</strong>
          </div>
          <div className="research-compare-grid">
            {(["B", "C"] as const).map((k) => (
              <article className="research-output" key={k}>
                <h4>{CONFIG_NAMES[k]}</h4>
                <p dir="auto">{e[k].answer}</p>
                <small>
                  NLI reward {e[k].f_faith.toFixed(3)} · independent score{" "}
                  {e[k].f_indep.toFixed(3)}
                </small>
              </article>
            ))}
          </div>
          <small>Source item: {e.src_id}</small>
        </details>
      ))}
      <h3>Independent and human checks</h3>
      <div className="research-table-wrap">
        <table className="research-table">
          <thead>
            <tr>
              <th>Measure</th>
              <th>Recorded result</th>
              <th>Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Independent NLI scorer · C − B</td>
              <td>
                {String(
                  (
                    d.independent["f_indep:C-B"] as { mean: number }
                  ).mean.toFixed(3),
                )}
              </td>
              <td>
                A separate scorer also records a higher score; this is not
                exact-match accuracy.
              </td>
            </tr>
            <tr>
              <td>Human support agreement</td>
              <td>
                κ = {d.human.summary.kappa_supported_quadratic?.toFixed(3)}
              </td>
              <td>Two annotators; agreement is limited.</td>
            </tr>
            <tr>
              <td>Length-matched NLI change</td>
              <td>
                {d.corrected.length_matched.matched_overall.mean_diff.toFixed(
                  3,
                )}
              </td>
              <td>
                Descriptive analysis holding answer-length bins fixed; not
                causal.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <h3>Validation and full analysis downloads</h3>
      <p>
        Human validation used two blinded annotators and 60 items per language.
        Support agreement κ ={" "}
        {d.human.summary.kappa_supported_quadratic?.toFixed(2)}. The limited
        human evidence does not establish a broad correctness gain.
      </p>
      {Object.entries({
        corrected: d.corrected,
        independent: d.independent,
        human: d.human,
        original: d.first_study,
      }).map(([key, data]) => (
        <details className="research-detail" key={key}>
          <summary>
            {
              {
                corrected:
                  "Corrected scores, seed results, uncertainty, length analysis and diagnostics",
                independent: "Independent scorer results",
                human: "Human ratings, agreement and sensitivity analysis",
                original: "Original study, adapter effects and reanalysis",
              }[key]
            }
          </summary>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </details>
      ))}
      <a
        className="research-download"
        href={`${import.meta.env.BASE_URL}data/results.json`}
        download
      >
        <Download size={16} /> Download complete analysis JSON
      </a>
      <a
        className="research-download"
        href={`${import.meta.env.BASE_URL}data/experiments.json`}
        download
      >
        <Download size={16} /> Download experiment index
      </a>
    </details>
  );
}
function SharedExamples({ d }: { d: ResultsFile }) {
  const [data, setData] = useState<SavedExample[] | null>(null),
    [error, setError] = useState(""),
    [query, setQuery] = useState(""),
    [lang, setLang] = useState<ExampleLanguage>("all"),
    [index, setIndex] = useState(0),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/compare-examples.json`)
      .then((r) => {
        if (!r.ok) throw Error("Saved examples could not be loaded");
        return r.json();
      })
      .then((x) => {
        if (live) {
          setData(x);
          setError("");
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [attempt]);
  const rows = useMemo(
    () => selectExamples(data || [], lang, query),
    [data, lang, query],
  );
  const selected = rows[Math.min(index, Math.max(0, rows.length - 1))];
  return (
    <section className="research-shared">
      <h2>100 shared questions</h2>
      <p className="research-lead">
        Read the same question answered by all three models. This saved set uses
        configuration C, QLoRA, seed 42, original protocol; it does not cover
        other combinations.
      </p>
      <div className="research-shared-filters">
        <label>
          Language
          <select
            value={lang}
            onChange={(e) => {
              setLang(e.target.value as ExampleLanguage);
              setIndex(0);
            }}
          >
            <option value="all">All 100 questions</option>
            <option value="arabic">Arabic · 50</option>
            <option value="malay">Malay · 50</option>
          </select>
        </label>
        <label>
          Find a question
          <div className="research-search">
            <Search size={16} />
            <input
              placeholder="Search questions or answers"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setIndex(0);
              }}
            />
          </div>
        </label>
      </div>
      {error ? (
        <p role="alert">
          {error}{" "}
          <button onClick={() => setAttempt((n) => n + 1)}>Try again</button>
        </p>
      ) : !data ? (
        <p>Loading…</p>
      ) : selected ? (
        <>
          <label>
            Question
            <select
              value={Math.min(index, rows.length - 1)}
              onChange={(e) => setIndex(Number(e.target.value))}
            >
              {rows.map((s, i) => (
                <option value={i} key={s.id}>
                  {i + 1}. {language(s.language)} · {s.question}
                </option>
              ))}
            </select>
          </label>
          <ExampleCard
            key={selected.id}
            example={selected}
            names={d.compare.models}
          />
          <div className="research-pagination">
            <button
              disabled={index === 0}
              onClick={() => setIndex((n) => n - 1)}
            >
              <ArrowLeft size={16} /> Previous
            </button>
            <span role="status">
              Question {index + 1} of {rows.length}
            </span>
            <button
              disabled={index >= rows.length - 1}
              onClick={() => setIndex((n) => n + 1)}
            >
              Next <ArrowRight size={16} />
            </button>
          </div>
        </>
      ) : (
        <p role="status">
          No questions match. Clear the search or choose another language.
        </p>
      )}
    </section>
  );
}
function PairComparison({
  rows,
  initial,
}: {
  rows: Experiment[];
  initial: string;
}) {
  const [left, setLeft] = useState(initial || rows[0]?.id || ""),
    [right, setRight] = useState(""),
    [differentOnly, setDifferentOnly] = useState(true),
    [index, setIndex] = useState(0);
  const a = rows.find((r) => r.id === left) || rows[0];
  const options = rows.filter(
    (r) =>
      a &&
      r.id !== a.id &&
      r.protocol === a.protocol &&
      r.language === a.language,
  );
  const b =
    options.find((r) => r.id === right) ||
    options.find(
      (r) =>
        r.model === a?.model && r.config === a?.config && r.adapter === "dora",
    ) ||
    options.find(
      (r) =>
        r.model === a?.model && r.adapter === a?.adapter && r.config === "C",
    ) ||
    options[0];
  const sa = useSamples(a),
    sb = useSamples(b);
  const allPairs = matchedSamples(sa.data, sb.data);
  const pairs = allPairs.filter(
    (p) => !differentOnly || p.before.generated !== p.after.generated,
  );
  const pair = pairs[Math.min(index, Math.max(0, pairs.length - 1))];
  const isCompatible = !!a && !!b && compatible(a, b);
  return (
    <>
      <h2>Compare two experiments</h2>
      <p className="research-lead">
        Choose two saved experiments. The second selection stays within the
        first study and language. Examples are paired only when the question,
        passage, reference and seed match.
      </p>
      <div className="research-compare-grid">
        <label>
          First experiment
          <select
            value={a?.id || ""}
            onChange={(e) => {
              setLeft(e.target.value);
              setRight("");
              setIndex(0);
            }}
          >
            {rows.map((r) => (
              <option key={r.id} value={r.id}>
                {PROTOCOL_NAMES[r.protocol]} · {title(r)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Second experiment
          <select
            value={b?.id || ""}
            onChange={(e) => {
              setRight(e.target.value);
              setIndex(0);
            }}
          >
            {options.map((r) => (
              <option key={r.id} value={r.id}>
                {title(r)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {a && b && (
        <>
          <div className="research-compare-grid">
            <article className="research-card">
              <h3>{title(a)}</h3>
              <Scores row={a} />
            </article>
            <article className="research-card">
              <h3>{title(b)}</h3>
              <Scores row={b} />
            </article>
          </div>
          <p className="research-callout">
            {isCompatible
              ? `Same recorded study, language, seed count and evaluation size. Second − first exact match: ${((b.em - a.em) * 100).toFixed(2)} percentage points. This is a descriptive score difference, not a significance test.`
              : "Different seed coverage or evaluation sizes. The scores are shown separately; a like-for-like change is not calculated."}
          </p>
          <fieldset className="choice-group research-mode">
            <legend>
              Saved answer view · {allPairs.length} verified pairs
            </legend>
            <div>
              <button
                aria-pressed={differentOnly}
                onClick={() => {
                  setDifferentOnly(true);
                  setIndex(0);
                }}
              >
                Different answers ·{" "}
                {
                  allPairs.filter(
                    (p) => p.before.generated !== p.after.generated,
                  ).length
                }
              </button>
              <button
                aria-pressed={!differentOnly}
                onClick={() => {
                  setDifferentOnly(false);
                  setIndex(0);
                }}
              >
                All shared answers
              </button>
            </div>
          </fieldset>
          {sa.loading || sb.loading ? (
            <p role="status">Loading shared examples…</p>
          ) : sa.error || sb.error ? (
            <p role="alert">
              {sa.error || sb.error}
              <button
                onClick={() => {
                  sa.retry();
                  sb.retry();
                }}
              >
                Try again
              </button>
            </p>
          ) : pair ? (
            <>
              <div className="research-section-title">
                <h3>{pairs.length} verified shared saved answers</h3>
              </div>
              <label>
                Shared question
                <select
                  value={Math.min(index, pairs.length - 1)}
                  onChange={(e) => setIndex(Number(e.target.value))}
                >
                  {pairs.map((p, i) => (
                    <option value={i} key={i}>
                      {i + 1}. {p.before.question} · seed {p.before.seed}
                    </option>
                  ))}
                </select>
              </label>
              <div className="research-question">
                <h3 dir="auto">{pair.before.question}</h3>
                <div className="s-gold">
                  <span>Reference answer</span>
                  <strong dir="auto">{pair.before.gold}</strong>
                </div>
                <div className="research-compare-grid">
                  <Output sample={pair.before} label="First experiment" />
                  <Output sample={pair.after} label="Second experiment" />
                </div>
                <p className="research-callout">
                  {pair.before.generated === pair.after.generated
                    ? "The saved wording is identical."
                    : "The saved wording changed. Check the reference and passage before treating it as an improvement."}
                </p>
                <details className="research-detail">
                  <summary>Read the shared passage</summary>
                  <p dir="auto">{pair.before.context}</p>
                </details>
              </div>
            </>
          ) : (
            <div className="research-empty">
              <h3>
                {allPairs.length
                  ? "The saved answers are identical"
                  : "No verifiable shared question records"}
              </h3>
              <p>
                {allPairs.length
                  ? "No wording changes were found. Choose All shared answers to inspect the identical outputs."
                  : "Scores exist, but these selections do not have saved answers with matching question and passage identity. Select other experiments or use the 100 shared questions."}
              </p>
            </div>
          )}
          <Source row={a} />
          <Source row={b} />
        </>
      )}
    </>
  );
}
export default function ResearchDashboard({ d }: { d: ResultsFile }) {
  const [tab, setTab] = useState<Tab>(
    () => TABS.find((t) => `#${t.id}` === location.hash)?.id || "overview",
  );
  const [rows, setRows] = useState<Experiment[] | null>(null),
    [error, setError] = useState(""),
    [attempt, setAttempt] = useState(0);
  const [filters, setFilters] = useState<Record<string, string>>(START_FILTERS),
    [selected, setSelected] = useState(""),
    [rankFilters, setRankFilters] = useState<Record<string, string>>({
      ...EMPTY_FILTERS,
      config: "B",
    }),
    [metric, setMetric] = useState<Metric>("em"),
    [compareMode, setCompareMode] = useState("experiments"),
    [comparison, setComparison] = useState("");
  const scrollTo = (id: string) =>
    requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      }),
    );
  const changeTab = (t: Tab) => {
    setTab(t);
    history.replaceState(null, "", `#${t}`);
    scrollTo(`research-panel-${t}`);
  };
  useEffect(() => {
    const sync = () =>
      setTab(TABS.find((t) => `#${t.id}` === location.hash)?.id || "overview");
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  useEffect(() => {
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/experiments.json`)
      .then((r) => {
        if (!r.ok) throw Error("Experiment index could not be loaded");
        return r.json();
      })
      .then((x) => {
        if (live) {
          setRows(x.experiments);
          setError("");
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [attempt]);
  const visible = filterExperiments(rows || [], filters);
  const current = visible.find((r) => r.id === selected) || visible[0];
  const ranking = rankExperiments(
    filterExperiments(rows || [], rankFilters),
    metric,
  );
  const open = (row: Experiment) => {
    setFilters({
      protocol: row.protocol,
      model: row.model,
      config: row.config,
      adapter: row.adapter,
      language: row.language,
    });
    setSelected(row.id);
    changeTab("explore");
  };
  const totalRuns = (rows || []).reduce((n, r) => n + r.nRuns, 0);
  const original = (rows || []).filter((r) => r.protocol === "original");
  return (
    <div className="research-app simple-app">
      <div className="research-hero">
        <div>
          <span className="research-eyebrow">
            <FlaskConical size={14} /> THE COMPLETE EXPERIMENT
          </span>
          <h1>
            Understand the results.
            <br />
            <span>One experiment at a time.</span>
          </h1>
          <p>
            Explore three models, four configurations and the saved adapter
            methods. See the scores, then read the answers behind them.
          </p>
        </div>
        <div className="research-hero-stat">
          <strong>{rows?.length ?? "—"}</strong>
          <span>saved experiment cells</span>
          <small>
            {rows ? `${totalRuns} seed-run results` : "Loading results…"}
          </small>
        </div>
      </div>
      <nav
        className="simple-tabs research-tabs"
        role="tablist"
        aria-label="Results sections"
      >
        {TABS.map((t, i) => (
          <button
            key={t.id}
            role="tab"
            id={`research-tab-${t.id}`}
            aria-controls={`research-panel-${t.id}`}
            aria-selected={tab === t.id}
            tabIndex={tab === t.id ? 0 : -1}
            onClick={() => changeTab(t.id)}
            onKeyDown={(e) => {
              let n = i;
              if (e.key === "ArrowRight") n = (i + 1) % TABS.length;
              else if (e.key === "ArrowLeft")
                n = (i + TABS.length - 1) % TABS.length;
              else if (e.key === "Home") n = 0;
              else if (e.key === "End") n = TABS.length - 1;
              else return;
              e.preventDefault();
              changeTab(TABS[n].id);
              document.getElementById(`research-tab-${TABS[n].id}`)?.focus();
            }}
          >
            {t.name}
          </button>
        ))}
      </nav>
      {error ? (
        <div className="research-empty" role="alert">
          <p>{error}</p>
          <button onClick={() => setAttempt((n) => n + 1)}>Try again</button>
        </div>
      ) : !rows ? (
        <p role="status">Loading the complete experiment index…</p>
      ) : (
        <>
          <section
            role="tabpanel"
            id="research-panel-overview"
            aria-labelledby="research-tab-overview"
            hidden={tab !== "overview"}
            className="simple-panel"
          >
            <div className="research-section-title">
              <div>
                <span className="research-eyebrow">START HERE</span>
                <h2>What was tested?</h2>
              </div>
              <button
                className="simple-primary"
                onClick={() => changeTab("explore")}
              >
                Explore experiments <ArrowRight size={16} />
              </button>
            </div>
            <div className="research-config-grid">
              {Object.entries({
                A: [
                  "Frozen model",
                  "No fine-tuning. A baseline for comparison.",
                ],
                B: [
                  "Standard training",
                  "Train an adapter using reference answers.",
                ],
                C: [
                  "NLI reward",
                  "Add a reward for answers judged to follow from the passage.",
                ],
                D: [
                  "Full fine-tuning",
                  "Update the whole model without an adapter.",
                ],
              }).map(([key, [name, desc]]) => (
                <button
                  key={key}
                  className="research-config"
                  onClick={() => {
                    setFilters({
                      protocol: "original",
                      model: "all",
                      config: key,
                      adapter: "all",
                      language: "arabic",
                    });
                    changeTab("explore");
                  }}
                >
                  <span>{key}</span>
                  <h3>{name}</h3>
                  <p>{desc}</p>
                  <small>
                    View saved results <ChevronRight size={13} />
                  </small>
                </button>
              ))}
            </div>
            <p className="research-note">
              B and C use adapter methods. A and D do not. The repository
              includes QLoRA, AdaLoRA, DoRA, VeRA, and additional plain LoRA
              runs. Not every model has every combination saved.
            </p>
            <div className="research-overview-grid">
              <article className="research-card">
                <span className="research-eyebrow">WHICH MODEL LEADS?</span>
                <h2>
                  {d.compare.models[d.compare.results["B|all|em"].winner]}
                </h2>
                <p>
                  Leads the saved standard-training model comparison by exact
                  match.
                </p>
                {d.compare.results["B|all|em"].ranking.map((r, i) => (
                  <div className="research-rank-line" key={r.model}>
                    <span>
                      {i + 1}. {r.name}
                    </span>
                    <strong>{score(r.mean)}</strong>
                  </div>
                ))}
                <p className="research-note">
                  B only · DoRA + QLoRA · seeds 42, 123, 2026 · Arabic and Malay
                  averaged equally. mT5 uses the corrected protocol; Qwen3 and
                  T5Gemma use the original protocol. This ranks these recorded
                  systems, not all architectures or configurations.
                </p>
                <button
                  className="research-link"
                  onClick={() => changeTab("leaderboard")}
                >
                  Explore scoped rankings <ArrowRight size={15} />
                </button>
              </article>
              <article className="research-card">
                <span className="research-eyebrow">DOES THE REWARD HELP?</span>
                <h2>A higher reward is not always a better answer.</h2>
                <p>
                  In the corrected mT5 study, Arabic NLI reward increased while
                  exact match decreased.
                </p>
                <div className="research-findings">
                  <div>
                    <span>Arabic NLI reward change</span>
                    <strong>
                      +{d.corrected.headline.arabic.mean?.toFixed(3)}
                    </strong>
                  </div>
                  <div>
                    <span>Arabic exact-match change</span>
                    <strong>
                      {(
                        (d.corrected.headline.em_arabic.mean || 0) * 100
                      ).toFixed(2)}{" "}
                      points
                    </strong>
                  </div>
                </div>
                <p className="research-note">
                  C − B · 8 paired seeds. Reward was used during training.
                  Independent and human checks, controls, and uncertainty are in
                  Study evidence below.
                </p>
                <button
                  className="research-link"
                  onClick={() => {
                    setFilters(EMPTY_FILTERS);
                    changeTab("explore");
                  }}
                >
                  Explore corrected results <ArrowRight size={15} />
                </button>
              </article>
            </div>
            <details className="research-library">
              <summary>Which combinations are available?</summary>
              <p>
                Original full-evaluation records: {original.length} cells.
                Corrected mT5 records: {rows.length - original.length} cells. A
                cell is one study × model × configuration × adapter × language,
                averaged across its recorded seeds.
              </p>
              <div className="research-table-wrap">
                <table className="research-table">
                  <thead>
                    <tr>
                      <th>Original study</th>
                      <th>A / D</th>
                      <th>B / C adapter methods</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(MODEL_NAMES).map((model) => (
                      <tr key={model}>
                        <td>{MODEL_NAMES[model]}</td>
                        <td>
                          {[
                            ...new Set(
                              original
                                .filter(
                                  (r) =>
                                    r.model === model &&
                                    ["A", "D"].includes(r.config),
                                )
                                .map((r) => r.config),
                            ),
                          ].join(" / ") || "Not saved"}
                        </td>
                        <td>
                          {[
                            ...new Set(
                              original
                                .filter(
                                  (r) =>
                                    r.model === model &&
                                    ["B", "C"].includes(r.config),
                                )
                                .map((r) => ADAPTER_NAMES[r.adapter]),
                            ),
                          ].join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>
                Corrected study: mT5 with LoRA, QLoRA and DoRA; B, C,
                zero-weight control, shuffled-reward control, and checkpointing
                checks. Corrected per-cell predictions are not exported here.
                Missing is different from zero.
              </p>
            </details>
            <EvidenceLibrary d={d} />
          </section>
          <section
            role="tabpanel"
            id="research-panel-explore"
            aria-labelledby="research-tab-explore"
            hidden={tab !== "explore"}
            className="simple-panel"
          >
            <div className="research-section-title">
              <div>
                <h2>Find your experiment</h2>
                <p>
                  Choose a study and language, then narrow down the model,
                  configuration or adapter.
                </p>
              </div>
            </div>
            <Filters
              value={filters}
              onChange={(v) => {
                setFilters(v);
                setSelected("");
              }}
            />
            <p className="research-note">
              {filters.protocol === "corrected"
                ? "Corrected protocol: mT5 results supersede its original reward-effect analysis."
                : "Historical protocol: the corrected study supersedes original mT5 reward-effect findings. These scores are retained for inspection, not combined with corrected runs."}{" "}
              A and D use “No adapter”.
            </p>
            <p className="research-result-count" role="status">
              {visible.length} matching experiment cells
            </p>
            {visible.length ? (
              <>
                <ResultTable
                  rows={visible}
                  onSelect={(r) => {
                    setSelected(r.id);
                    scrollTo("selected-experiment");
                  }}
                />
                {current && tab === "explore" && (
                  <article
                    id="selected-experiment"
                    className="research-selected"
                    key={current.id}
                  >
                    <span className="research-eyebrow">
                      SELECTED EXPERIMENT · {PROTOCOL_NAMES[current.protocol]}
                    </span>
                    <h2>{title(current)}</h2>
                    <Scores row={current} />
                    <button
                      className="research-link"
                      onClick={() => {
                        setComparison(current.id);
                        setCompareMode("experiments");
                        changeTab("compare");
                      }}
                    >
                      Compare with another experiment <ArrowRight size={16} />
                    </button>
                    <SampleBrowser row={current} />
                    <Source row={current} />
                  </article>
                )}
              </>
            ) : (
              <div className="research-empty">
                <h3>No saved result for this combination</h3>
                <p>
                  This does not mean a score of zero. Choose another
                  configuration or adapter, or show all results in this study
                  and language.
                </p>
                <button
                  className="simple-secondary"
                  onClick={() =>
                    setFilters({
                      ...filters,
                      model: "all",
                      config: "all",
                      adapter: "all",
                    })
                  }
                >
                  Show available results
                </button>
              </div>
            )}
          </section>
          <section
            role="tabpanel"
            id="research-panel-compare"
            aria-labelledby="research-tab-compare"
            hidden={tab !== "compare"}
            className="simple-panel"
          >
            <div className="choice-group research-mode">
              <div>
                <button
                  aria-pressed={compareMode === "experiments"}
                  onClick={() => setCompareMode("experiments")}
                >
                  Two experiments
                </button>
                <button
                  aria-pressed={compareMode === "shared"}
                  onClick={() => setCompareMode("shared")}
                >
                  100 shared questions
                </button>
              </div>
            </div>
            {tab === "compare" &&
              (compareMode === "shared" ? (
                <SharedExamples d={d} />
              ) : (
                <PairComparison
                  key={comparison}
                  rows={rows}
                  initial={comparison || "original:qwen3:C:qlora:arabic"}
                />
              ))}
          </section>
          <section
            role="tabpanel"
            id="research-panel-leaderboard"
            aria-labelledby="research-tab-leaderboard"
            hidden={tab !== "leaderboard"}
            className="simple-panel"
          >
            <h2>Best within the selected results</h2>
            <p className="research-lead">
              There is no universal winner across incomplete experiments. Choose
              one study, language and metric to see the highest recorded scores.
            </p>
            <Filters value={rankFilters} onChange={setRankFilters} ranking />
            <fieldset className="choice-group research-mode">
              <legend>Rank by</legend>
              <div>
                {(["em", "f1", "nli"] as Metric[]).map((m) => (
                  <button
                    key={m}
                    aria-pressed={metric === m}
                    onClick={() => setMetric(m)}
                  >
                    {
                      {
                        em: "Exact match",
                        f1: "Word overlap",
                        nli: "NLI reward",
                      }[m]
                    }
                  </button>
                ))}
              </div>
            </fieldset>
            <p className="research-callout">
              {metric === "nli"
                ? "NLI is the training reward. This orders reward scores and must not be read as an accuracy or human-quality ranking."
                : "Ordered by observed mean score. Different seed counts or configurations are not a controlled comparison; this table does not establish statistical superiority."}
            </p>
            {ranking.length ? (
              <>
                <div className="research-leader">
                  <span className="research-eyebrow">
                    HIGHEST RECORDED{" "}
                    {metric === "em"
                      ? "EXACT MATCH"
                      : metric === "f1"
                        ? "WORD OVERLAP"
                        : "REWARD SCORE"}
                  </span>
                  <h3>{title(ranking[0])}</h3>
                  <strong>{score(ranking[0][metric], metric)}</strong>
                  <p>
                    {PROTOCOL_NAMES[ranking[0].protocol]} ·{" "}
                    {language(ranking[0].language)} · {ranking[0].nRuns} seeds ·{" "}
                    {ranking[0].sampleCount} saved answer examples
                  </p>
                  <button
                    className="simple-primary"
                    onClick={() => open(ranking[0])}
                  >
                    Inspect this result <ArrowRight size={16} />
                  </button>
                </div>
                <ResultTable rows={ranking} onSelect={open} metric={metric} />
              </>
            ) : (
              <div className="research-empty">
                <h3>No saved results in this scope</h3>
                <button
                  onClick={() =>
                    setRankFilters({ ...EMPTY_FILTERS, config: "B" })
                  }
                >
                  Reset ranking filters
                </button>
              </div>
            )}
          </section>
        </>
      )}
      <div className="research-bottom">
        <span>
          Measured scores. Original saved outputs. Missing data labelled.
        </span>
        <a href={`${import.meta.env.BASE_URL}data/experiments.json`} download>
          <Download size={14} /> Experiment index
        </a>
      </div>
    </div>
  );
}
