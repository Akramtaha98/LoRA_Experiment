import { useEffect, useMemo, useState } from "react";
import {
  BookOpenText,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  Moon,
  Sun,
  Trophy,
} from "lucide-react";

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

function MetricBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="mbar">
      <div className="mbar-top">
        <span>{label}</span>
        <strong>{fmt(value, label === "EM" ? 2 : 3)}</strong>
      </div>
      <div className="mbar-track">
        <div className="mbar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
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
  const matched = metrics && metrics.exact_match >= 1;
  return (
    <article
      className={`answer-card ${variant}${badge === "Best" ? " is-best" : ""}`}
    >
      <header>
        <div>
          <p>{subtitle}</p>
          <h3>{title}</h3>
        </div>
        {badge ? (
          <span className={`badge ${badge === "Best" ? "badge-best" : ""}`}>
            {badge === "Best" ? <Trophy size={12} /> : null}
            {badge}
          </span>
        ) : null}
      </header>
      <div className="answer-body" dir="auto">
        {answer}
      </div>
      {metrics ? (
        <div className="metric-stack">
          {matched ? (
            <p className="match-ok">
              <CheckCircle2 size={14} /> Exact match to gold
            </p>
          ) : null}
          <MetricBar label="EM" value={metrics.exact_match} />
          <MetricBar label="F1" value={metrics.f1} />
          <MetricBar label="Faith" value={metrics.faithfulness} />
        </div>
      ) : variant === "gold" ? (
        <p className="missing">Use this gold answer to judge the models.</p>
      ) : (
        <p className="missing">
          No indexed prediction in the logs for this example under the selected
          config.
        </p>
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
  const [showMeans, setShowMeans] = useState(false);
  const [showAcross, setShowAcross] = useState(true);

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

  const activeConfig: ConfigDef | undefined = configDefs.find(
    (c) => c.id === configId,
  );
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

  function setLang(next: DatasetLanguage) {
    setLanguage(next);
    const first = examples.find((ex) => ex.language === next);
    if (first) setExampleId(first.id);
    setShowContext(false);
  }

  return (
    <div className="app">
      <div className="bg-orb bg-orb-a" aria-hidden />
      <div className="bg-orb bg-orb-b" aria-hidden />

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <BookOpenText size={18} />
          </span>
          <div>
            <strong>RAG Faithfulness Lab</strong>
            <small>mT5 × Qwen · offline logs</small>
          </div>
        </div>
        <div className="top-actions">
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

      <main className="shell">
        <section className="hero">
          <h1>Compare answers against gold evidence</h1>
          <p>
            Pick a language, training config, and question. See the correct
            answer beside mT5 and Qwen using logged experiment results.
          </p>
        </section>

        <section className="control-card">
          <div className="control-block">
            <span className="control-label">Language</span>
            <div className="seg" role="tablist" aria-label="Language">
              <button
                type="button"
                className={language === "arabic" ? "seg-btn active" : "seg-btn"}
                onClick={() => setLang("arabic")}
              >
                Arabic <em>{arabicCount}</em>
              </button>
              <button
                type="button"
                className={language === "malay" ? "seg-btn active" : "seg-btn"}
                onClick={() => setLang("malay")}
              >
                Malay <em>{malayCount}</em>
              </button>
            </div>
          </div>

          <div className="control-block">
            <span className="control-label">Training config</span>
            <div className="seg seg-wrap" role="tablist" aria-label="Config">
              {configDefs.map((cfg) => (
                <button
                  key={cfg.id}
                  type="button"
                  className={configId === cfg.id ? "seg-btn active" : "seg-btn"}
                  onClick={() => setConfigId(cfg.id)}
                  title={cfg.description}
                >
                  {cfg.label.replace(/^([A-D])\s+/, "$1 · ")}
                </button>
              ))}
            </div>
            {activeConfig ? (
              <p className="hint">{activeConfig.description}</p>
            ) : null}
          </div>

          <label className="control-block menubox">
            <span className="control-label">Example question</span>
            <div className="select-shell">
              <select
                value={active?.id ?? ""}
                onChange={(e) => {
                  setExampleId(e.target.value);
                  setShowContext(false);
                }}
              >
                {filtered.map((ex) => (
                  <option key={ex.id} value={ex.id}>
                    {ex.language === "arabic" ? "AR" : "MS"} #{ex.example_index}
                    :{" "}
                    {ex.question.length > 90
                      ? `${ex.question.slice(0, 90)}…`
                      : ex.question}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} className="select-caret" />
            </div>
          </label>
        </section>

        {active ? (
          <>
            <section className="question-card">
              <div className="question-top">
                <div>
                  <span className="pill">
                    {active.language === "arabic" ? "Arabic" : "Malay"} #
                    {active.example_index}
                  </span>
                  <h2 dir="auto">{active.question}</h2>
                </div>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setShowContext((v) => !v)}
                >
                  {showContext ? "Hide passage" : "Show passage"}
                </button>
              </div>
              {showContext ? (
                <div className="passage" dir="auto">
                  {active.context}
                </div>
              ) : null}
            </section>

            <section className={`winner winner-${best}`}>
              <Trophy size={18} />
              <div>
                <strong>{bestLabel(best)}</strong>
                <p>{SCORING_RULE_LABEL}</p>
              </div>
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

            <section className="fold panel">
              <button
                type="button"
                className="fold-head"
                onClick={() => setShowAcross((v) => !v)}
                aria-expanded={showAcross}
              >
                <div>
                  <h3>Same example across A / B / C / D</h3>
                  <p>Scan how answers change with training configuration.</p>
                </div>
                <ChevronDown
                  size={18}
                  className={showAcross ? "chev open" : "chev"}
                />
              </button>
              {showAcross ? (
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
                            className={
                              cfg.id === configId ? "is-active-row" : undefined
                            }
                          >
                            <td>
                              <button
                                type="button"
                                className="linkish"
                                onClick={() => setConfigId(cfg.id)}
                              >
                                {cfg.label}
                              </button>
                            </td>
                            <td dir="auto">{row?.mt5?.answer ?? "Not logged"}</td>
                            <td dir="auto">
                              {row?.qwen?.answer ?? "Not logged"}
                            </td>
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
              ) : null}
            </section>
          </>
        ) : (
          <p className="muted">{error ?? "Loading examples…"}</p>
        )}

        <section className="fold panel">
          <button
            type="button"
            className="fold-head"
            onClick={() => setShowMeans((v) => !v)}
            aria-expanded={showMeans}
          >
            <div>
              <h3>Mean metrics by configuration</h3>
              <p>
                Aggregate experiment scores for {language} (seed 42 / QLoRA when
                available).
              </p>
            </div>
            <ChevronDown size={18} className={showMeans ? "chev open" : "chev"} />
          </button>
          {showMeans && file ? (
            <AggregateTable
              language={language}
              configDefs={configDefs}
              aggregates={file.aggregates}
            />
          ) : null}
        </section>

        <footer className="footer">
          <span>
            {arabicCount} Arabic · {malayCount} Malay · static logs
          </span>
          <span>No GPU required</span>
        </footer>
      </main>
    </div>
  );
}
