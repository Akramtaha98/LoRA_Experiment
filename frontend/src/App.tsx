import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Moon, Sun } from "lucide-react";

import {
  loadDatasetExamples,
  type AggregateMetrics,
  type ConfigDef,
  type ConfigId,
  type DatasetExample,
  type DatasetFile,
  type DatasetLanguage,
} from "./data/dataset";
import {
  pickBestModel,
  SCORING_RULE_LABEL,
  type BestModel,
  type ModelScores,
} from "./lib/scoring";
import {
  applyTheme,
  getPreferredTheme,
  toggleTheme,
  type Theme,
} from "./lib/theme";

function fmt(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Number(value).toFixed(digits);
}

function AnswerCard({
  title,
  subtitle,
  answer,
  metrics,
  badge,
  variant = "model",
}: {
  title: string;
  subtitle: string;
  answer: string;
  metrics?: ModelScores | null;
  badge?: string;
  variant?: "gold" | "model";
}) {
  return (
    <article className={`answer-card ${variant}`}>
      <header>
        <div>
          <p>{subtitle}</p>
          <h3>{title}</h3>
        </div>
        {badge ? <span className="badge">{badge}</span> : null}
      </header>
      <div className="answer-body" dir="auto">
        {answer}
      </div>
      {metrics ? (
        <dl className="metric-dl">
          <div>
            <dt>EM</dt>
            <dd>{fmt(metrics.exact_match, 2)}</dd>
          </div>
          <div>
            <dt>F1</dt>
            <dd>{fmt(metrics.f1, 3)}</dd>
          </div>
          <div>
            <dt>Faith</dt>
            <dd>{fmt(metrics.faithfulness, 3)}</dd>
          </div>
        </dl>
      ) : (
        <p className="missing">No indexed prediction in the logs for this example.</p>
      )}
    </article>
  );
}

function AggregateTable({
  language,
  configDefs,
  aggregates,
}: {
  language: DatasetLanguage;
  configDefs: ConfigDef[];
  aggregates: DatasetFile["aggregates"];
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Config</th>
            <th>Model</th>
            <th>Adapter</th>
            <th>n</th>
            <th>Faith</th>
            <th>EM</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {configDefs.flatMap((cfg) => {
            const block = aggregates?.[language]?.[cfg.id] || {};
            return (["mt5", "qwen"] as const).map((model) => {
              const row = block[model] as AggregateMetrics | undefined;
              return (
                <tr key={`${cfg.id}-${model}`}>
                  <td>{cfg.label}</td>
                  <td>{model === "mt5" ? "mT5" : "Qwen"}</td>
                  <td>{row?.lora_variant ?? "n/a"}</td>
                  <td>{row?.n_eval ?? "n/a"}</td>
                  <td>{fmt(row?.mean_faithfulness)}</td>
                  <td>{fmt(row?.mean_em)}</td>
                  <td>{fmt(row?.mean_f1)}</td>
                </tr>
              );
            });
          })}
        </tbody>
      </table>
    </div>
  );
}

function bestLabel(best: BestModel) {
  if (best === "tie") return "Tie versus the correct answer";
  if (best === "mt5") return "Best match to the correct answer: mT5";
  return "Best match to the correct answer: Qwen";
}

export default function App() {
  const [theme, setTheme] = useState<Theme>("light");
  const [file, setFile] = useState<DatasetFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<DatasetLanguage>("arabic");
  const [configId, setConfigId] = useState<ConfigId>("C_composite_lora");
  const [exampleId, setExampleId] = useState<string>("");
  const [showContext, setShowContext] = useState(false);

  useEffect(() => {
    const initial = getPreferredTheme();
    applyTheme(initial);
    setTheme(initial);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadDatasetExamples()
      .then((data) => {
        if (cancelled) return;
        setFile(data);
        const first = data.examples.find((e) => e.language === "arabic");
        if (first) setExampleId(first.id);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const configDefs = file?.config_defs ?? [];
  const examples = file?.examples ?? [];

  const filtered = useMemo(
    () => examples.filter((e) => e.language === language),
    [examples, language],
  );

  const active: DatasetExample | null = useMemo(() => {
    if (!filtered.length) return null;
    return filtered.find((e) => e.id === exampleId) ?? filtered[0];
  }, [filtered, exampleId]);

  const configPred = active?.configs?.[configId];
  const mt5 = configPred?.mt5 ?? null;
  const qwen = configPred?.qwen ?? null;
  const best = useMemo(() => {
    if (!mt5 && !qwen) return "tie" as BestModel;
    if (!mt5) return "qwen" as BestModel;
    if (!qwen) return "mt5" as BestModel;
    return pickBestModel(mt5, qwen);
  }, [mt5, qwen]);

  const arabicCount = examples.filter((e) => e.language === "arabic").length;
  const malayCount = examples.filter((e) => e.language === "malay").length;

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-main">
          <p className="eyebrow">LoRA Experiment · Offline evaluation browser</p>
          <h1>RAG Faithfulness Lab</h1>
          <p className="lede">
            Compare logged mT5 and Qwen answers against the gold reference under
            training configurations A (Frozen), B (CE LoRA), C (Composite LoRA),
            and D (Full FT). No live GPU inference.
          </p>
        </div>
        <div className="masthead-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={() => setTheme((t) => toggleTheme(t))}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a
            className="icon-btn"
            href="https://github.com/Akramtaha98/LoRA_Experiment"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
          >
            <ExternalLink size={16} />
          </a>
        </div>
      </header>

      <section className="controls panel">
        <label>
          Language
          <select
            value={language}
            onChange={(e) => {
              const next = e.target.value as DatasetLanguage;
              setLanguage(next);
              const first = examples.find((ex) => ex.language === next);
              if (first) setExampleId(first.id);
              setShowContext(false);
            }}
          >
            <option value="arabic">Arabic ({arabicCount})</option>
            <option value="malay">Malay ({malayCount})</option>
          </select>
        </label>

        <label>
          Training configuration
          <select
            value={configId}
            onChange={(e) => setConfigId(e.target.value as ConfigId)}
          >
            {configDefs.map((cfg) => (
              <option key={cfg.id} value={cfg.id}>
                {cfg.label}
              </option>
            ))}
          </select>
        </label>

        <label className="menubox">
          Example
          <select
            value={active?.id ?? ""}
            onChange={(e) => {
              setExampleId(e.target.value);
              setShowContext(false);
            }}
          >
            {filtered.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.language === "arabic" ? "AR" : "MS"} #{ex.example_index}:{" "}
                {ex.question.length > 72
                  ? `${ex.question.slice(0, 72)}…`
                  : ex.question}
              </option>
            ))}
          </select>
        </label>
      </section>

      {configDefs.length ? (
        <p className="config-note">
          {configDefs.find((c) => c.id === configId)?.description}.{" "}
          {SCORING_RULE_LABEL}
        </p>
      ) : null}

      <section className="panel">
        <div className="section-title">
          <h2>Mean metrics by configuration</h2>
          <p>
            Aggregate scores from experiment logs ({language}; preferred seed 42,
            QLoRA when available).
          </p>
        </div>
        {file ? (
          <AggregateTable
            language={language}
            configDefs={configDefs}
            aggregates={file.aggregates}
          />
        ) : (
          <p className="muted">{error ?? "Loading…"}</p>
        )}
      </section>

      {active ? (
        <>
          <section className="panel">
            <div className="section-title row">
              <div>
                <h2>Selected example</h2>
                <p>
                  {active.language === "arabic" ? "Arabic" : "Malay"} example #
                  {active.example_index}
                </p>
              </div>
              <button
                type="button"
                className="text-btn"
                onClick={() => setShowContext((v) => !v)}
              >
                {showContext ? "Hide passage" : "Show passage"}
              </button>
            </div>
            <p className="question" dir="auto">
              {active.question}
            </p>
            {showContext ? (
              <div className="passage" dir="auto">
                {active.context}
              </div>
            ) : null}
          </section>

          <section className="best-strip">
            <strong>{bestLabel(best)}</strong>
            <span>{SCORING_RULE_LABEL}</span>
          </section>

          <section className="compare-grid">
            <AnswerCard
              title="Correct answer"
              subtitle="Gold reference"
              answer={active.reference}
              variant="gold"
              badge="Gold"
            />
            <AnswerCard
              title="mT5"
              subtitle="Encoder-decoder"
              answer={mt5?.answer || "Not logged"}
              metrics={mt5}
              badge={best === "mt5" ? "Best" : undefined}
            />
            <AnswerCard
              title="Qwen"
              subtitle="Decoder-only"
              answer={qwen?.answer || "Not logged"}
              metrics={qwen}
              badge={best === "qwen" ? "Best" : undefined}
            />
          </section>

          <section className="panel">
            <div className="section-title">
              <h2>Same example across configs</h2>
              <p>
                Quick view of logged answers for this question under A/B/C/D when
                present.
              </p>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Config</th>
                    <th>mT5 answer</th>
                    <th>Qwen answer</th>
                    <th>Best</th>
                  </tr>
                </thead>
                <tbody>
                  {configDefs.map((cfg) => {
                    const row = active.configs[cfg.id];
                    return (
                      <tr
                        key={cfg.id}
                        className={cfg.id === configId ? "is-active-row" : undefined}
                      >
                        <td>{cfg.label}</td>
                        <td dir="auto">{row?.mt5?.answer ?? "Not logged"}</td>
                        <td dir="auto">{row?.qwen?.answer ?? "Not logged"}</td>
                        <td>
                          {!row?.mt5 && !row?.qwen
                            ? "n/a"
                            : row.best_model === "tie"
                              ? "Tie"
                              : row.best_model === "mt5"
                                ? "mT5"
                                : "Qwen"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : (
        <p className="muted">{error ?? "Loading examples…"}</p>
      )}

      <footer className="site-footer">
        <span>
          {arabicCount} Arabic and {malayCount} Malay examples · static JSON from{" "}
          <code>experiment_results/</code>
        </span>
        <span>mT5 × Qwen RAG faithfulness study</span>
      </footer>
    </div>
  );
}
