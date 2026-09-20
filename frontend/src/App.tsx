import { useEffect, useMemo, useState } from "react";
import {
  ExternalLink,
  Moon,
  Play,
  Search,
  Sparkles,
  Sun,
  Trophy,
} from "lucide-react";

import { examples } from "./data/examples";
import {
  loadDatasetExamples,
  type DatasetExample,
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

type TabId = "dataset" | "compare";

function MetricBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="metric-bar">
      <div className="metric-bar-top">
        <span>{label}</span>
        <strong>{value.toFixed(value >= 0.01 || value === 0 ? 2 : 4)}</strong>
      </div>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ModelPanel({
  name,
  kind,
  result,
  isBest,
}: {
  name: string;
  kind: string;
  result: ModelScores;
  isBest: boolean;
}) {
  return (
    <article className={isBest ? "model-panel is-best" : "model-panel"}>
      <header>
        <div>
          <p>{kind}</p>
          <h3>{name}</h3>
        </div>
        {isBest ? (
          <span className="best-pill">
            <Trophy size={13} /> Best
          </span>
        ) : null}
      </header>
      <div className="answer" dir="auto">
        {result.answer || "—"}
      </div>
      <MetricBar label="Faithfulness" value={result.faithfulness} />
      <MetricBar label="Exact Match" value={result.exact_match} />
      <MetricBar label="Token F1" value={result.f1} />
      {typeof result.latency_ms === "number" ? (
        <p className="latency">{result.latency_ms} ms</p>
      ) : null}
    </article>
  );
}

function BestLine({ best }: { best: BestModel }) {
  const label =
    best === "tie"
      ? "Tie on faithfulness → EM → F1"
      : best === "mt5"
        ? "Best: mT5"
        : "Best: Qwen";
  return (
    <div className={`best-line best-${best}`}>
      <Trophy size={15} />
      <span>{label}</span>
      <small>{SCORING_RULE_LABEL}</small>
    </div>
  );
}

function truncate(text: string, max = 64) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  return cleaned.length <= max ? cleaned : `${cleaned.slice(0, max - 1)}…`;
}

export default function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [tab, setTab] = useState<TabId>("dataset");
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState<DatasetLanguage | "all">("arabic");
  const [dataset, setDataset] = useState<DatasetExample[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [demoIndex, setDemoIndex] = useState(3);
  const [demoResult, setDemoResult] = useState({
    mt5: examples[3].mt5,
    qwen: examples[3].qwen,
  });
  const [loading, setLoading] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);

  useEffect(() => {
    const initial = getPreferredTheme();
    applyTheme(initial);
    setTheme(initial);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadDatasetExamples()
      .then((file) => {
        if (cancelled) return;
        setDataset(file.examples);
        const first = file.examples.find((e) => e.language === "arabic");
        setSelectedId(first?.id ?? file.examples[0]?.id ?? null);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setDatasetError(
            error instanceof Error ? error.message : "Failed to load dataset",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const arabicCount = useMemo(
    () => dataset.filter((e) => e.language === "arabic").length,
    [dataset],
  );
  const malayCount = useMemo(
    () => dataset.filter((e) => e.language === "malay").length,
    [dataset],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return dataset.filter((example) => {
      if (language !== "all" && example.language !== language) return false;
      if (!q) return true;
      return (
        example.question.toLowerCase().includes(q) ||
        example.reference.toLowerCase().includes(q) ||
        example.id.toLowerCase().includes(q) ||
        String(example.example_index).includes(q)
      );
    });
  }, [dataset, language, query]);

  const active = useMemo(() => {
    if (!filtered.length) return null;
    return filtered.find((e) => e.id === selectedId) ?? filtered[0];
  }, [filtered, selectedId]);

  const datasetBest = useMemo(
    () => (active ? pickBestModel(active.mt5, active.qwen) : "tie"),
    [active],
  );

  const demo = examples[demoIndex];
  const compareBest = useMemo(
    () => pickBestModel(demoResult.mt5, demoResult.qwen),
    [demoResult],
  );

  function selectDemo(index: number) {
    setDemoIndex(index);
    setDemoResult({ mt5: examples[index].mt5, qwen: examples[index].qwen });
    setContextOpen(false);
  }

  async function runCompare() {
    setLoading(true);
    const fallback = { mt5: demo.mt5, qwen: demo.qwen };
    try {
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context: demo.context,
          question: demo.question,
          reference: demo.reference,
          mt5_configuration: "frozen",
          qwen_configuration: "frozen",
        }),
      });
      if (!response.ok) throw new Error("API unavailable");
      const data = (await response.json()) as {
        mt5: ModelScores;
        qwen: ModelScores;
      };
      setDemoResult(data);
    } catch {
      // Offline / Vercel: use the per-example mock so the button always updates.
      setDemoResult(fallback);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Sparkles size={16} />
          </span>
          <div>
            <strong>RAG Faithfulness Lab</strong>
            <small>mT5 × Qwen · offline results</small>
          </div>
        </div>

        <nav className="tabs" aria-label="Primary">
          <button
            type="button"
            className={tab === "dataset" ? "tab active" : "tab"}
            onClick={() => setTab("dataset")}
          >
            Dataset
            <em>{dataset.length || "…"}</em>
          </button>
          <button
            type="button"
            className={tab === "compare" ? "tab active" : "tab"}
            onClick={() => setTab("compare")}
          >
            Demos
            <em>{examples.length}</em>
          </button>
        </nav>

        <div className="top-actions">
          <button
            type="button"
            className="ghost-btn"
            onClick={() => setTheme((t) => toggleTheme(t))}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a
            className="ghost-btn"
            href="https://github.com/Akramtaha98/LoRA_Experiment"
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={16} />
          </a>
        </div>
      </header>

      {tab === "dataset" ? (
        <main className="lab-grid">
          <aside className="picker">
            <div className="picker-head">
              <h1>Examples</h1>
              <p>
                {arabicCount} Arabic · {malayCount} Malay
              </p>
            </div>

            <div className="lang-row">
              {(
                [
                  ["arabic", `Arabic ${arabicCount}`],
                  ["malay", `Malay ${malayCount}`],
                  ["all", `All ${dataset.length}`],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={language === value ? "chip active" : "chip"}
                  onClick={() => {
                    setLanguage(value);
                    setQuery("");
                    const next =
                      value === "all"
                        ? dataset[0]
                        : dataset.find((e) => e.language === value);
                    if (next) setSelectedId(next.id);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            <label className="search">
              <Search size={15} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search question or id…"
              />
            </label>

            <div className="pick-list" role="listbox">
              {datasetError ? (
                <p className="error">{datasetError}</p>
              ) : filtered.length === 0 ? (
                <p className="muted">No matches.</p>
              ) : (
                filtered.map((example) => {
                  const selected = active?.id === example.id;
                  return (
                    <button
                      key={example.id}
                      type="button"
                      role="option"
                      aria-selected={selected}
                      className={selected ? "pick active" : "pick"}
                      onClick={() => {
                        setSelectedId(example.id);
                        setContextOpen(false);
                      }}
                    >
                      <span className="pick-meta">
                        {example.language === "arabic" ? "AR" : "MS"} · #
                        {example.example_index}
                        <i className={`dot ${example.best_model}`} />
                      </span>
                      <strong dir="auto">{truncate(example.question, 70)}</strong>
                    </button>
                  );
                })
              )}
            </div>
          </aside>

          <section className="stage">
            {active ? (
              <>
                <div className="evidence">
                  <div className="evidence-top">
                    <div>
                      <small>Reference</small>
                      <h2 dir="auto">{active.reference}</h2>
                    </div>
                    <button
                      type="button"
                      className="text-btn"
                      onClick={() => setContextOpen((v) => !v)}
                    >
                      {contextOpen ? "Hide context" : "Show context"}
                    </button>
                  </div>
                  <p className="question" dir="auto">
                    {active.question}
                  </p>
                  {contextOpen ? (
                    <div className="context-box" dir="auto">
                      {active.context}
                    </div>
                  ) : null}
                  <p className="source-line">
                    Logged QLoRA · seed {active.source.seed} · mT5{" "}
                    {active.source.mt5_config} · Qwen{" "}
                    {active.source.qwen_config}
                  </p>
                </div>

                <BestLine best={datasetBest} />

                <div className="panels">
                  <ModelPanel
                    name="mT5"
                    kind="Encoder–decoder"
                    result={active.mt5}
                    isBest={datasetBest === "mt5"}
                  />
                  <ModelPanel
                    name="Qwen"
                    kind="Decoder-only"
                    result={active.qwen}
                    isBest={datasetBest === "qwen"}
                  />
                </div>
              </>
            ) : (
              <p className="muted">Loading examples…</p>
            )}
          </section>
        </main>
      ) : (
        <main className="lab-grid">
          <aside className="picker">
            <div className="picker-head">
              <h1>English demos</h1>
              <p>Offline mock results update instantly</p>
            </div>
            <div className="pick-list">
              {examples.map((example, index) => (
                <button
                  key={example.name}
                  type="button"
                  className={demoIndex === index ? "pick active" : "pick"}
                  onClick={() => selectDemo(index)}
                >
                  <span className="pick-meta">Demo</span>
                  <strong>{example.name}</strong>
                  <em>{example.description}</em>
                </button>
              ))}
            </div>
          </aside>

          <section className="stage">
            <div className="evidence">
              <div className="evidence-top">
                <div>
                  <small>Reference</small>
                  <h2>{demo.reference}</h2>
                </div>
                <button
                  type="button"
                  className="compare-btn"
                  onClick={runCompare}
                  disabled={loading}
                >
                  <Play size={15} fill="currentColor" />
                  {loading ? "Running…" : "Compare"}
                </button>
              </div>
              <p className="question">{demo.question}</p>
              <div className="context-box">{demo.context}</div>
              <p className="source-line">
                Selecting a demo updates answers below. On Vercel, Compare uses
                built-in mocks (no GPU API).
              </p>
            </div>

            <BestLine best={compareBest} />

            <div className="panels">
              <ModelPanel
                name="mT5"
                kind="Encoder–decoder"
                result={demoResult.mt5}
                isBest={compareBest === "mt5"}
              />
              <ModelPanel
                name="Qwen"
                kind="Decoder-only"
                result={demoResult.qwen}
                isBest={compareBest === "qwen"}
              />
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
