import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  ChevronRight,
  Clock3,
  ExternalLink,
  Layers3,
  Moon,
  Play,
  ShieldCheck,
  Sparkles,
  Sun,
  Trophy,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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

type ComparisonResponse = {
  mt5: ModelScores;
  qwen: ModelScores;
};

type TabId = "compare" | "dataset";

const demoResult: ComparisonResponse = {
  mt5: {
    answer: "Lyon",
    faithfulness: 0.97,
    exact_match: 1,
    f1: 1,
    latency_ms: 713,
  },
  qwen: {
    answer: "Paris",
    faithfulness: 0.31,
    exact_match: 0,
    f1: 0,
    latency_ms: 482,
  },
};

function Score({
  title,
  value,
  suffix = "",
}: {
  title: string;
  value: string | number;
  suffix?: string;
}) {
  return (
    <div className="metric-card">
      <span>{title}</span>
      <strong>
        {value}
        {suffix}
      </strong>
    </div>
  );
}

function ModelCard({
  title,
  subtitle,
  result,
  isBest,
  bestLabel,
}: {
  title: string;
  subtitle: string;
  result: ModelScores;
  isBest?: boolean;
  bestLabel?: string;
}) {
  return (
    <section className={isBest ? "model-card is-best" : "model-card"}>
      <div className="model-title">
        <div>
          <p>{subtitle}</p>
          <h2>{title}</h2>
        </div>

        <div className="model-title-right">
          {isBest ? (
            <span className="best-badge">
              <Trophy size={14} />
              {bestLabel ?? "Best answer"}
            </span>
          ) : null}
          <div className="model-icon">
            <BrainCircuit size={22} />
          </div>
        </div>
      </div>

      <div className="answer-box">
        <span>Generated answer</span>
        <p dir="auto">{result.answer || "—"}</p>
      </div>

      <div className="metric-grid">
        <Score title="Faithfulness" value={result.faithfulness.toFixed(4)} />
        <Score title="Exact Match" value={result.exact_match.toFixed(2)} />
        <Score title="Token F1" value={result.f1.toFixed(4)} />
        {typeof result.latency_ms === "number" ? (
          <Score title="Latency" value={result.latency_ms} suffix=" ms" />
        ) : (
          <Score title="Latency" value="—" />
        )}
      </div>
    </section>
  );
}

function BestBanner({ best }: { best: BestModel }) {
  const text =
    best === "tie"
      ? "Tie — mT5 and Qwen score equally on faithfulness, EM, and F1."
      : best === "mt5"
        ? "Best answer: mT5 (by faithfulness → EM → F1)."
        : "Best answer: Qwen (by faithfulness → EM → F1).";

  return (
    <div className={`best-banner best-${best}`}>
      <Trophy size={18} />
      <div>
        <strong>{text}</strong>
        <p>{SCORING_RULE_LABEL}</p>
      </div>
    </div>
  );
}

function truncateLabel(text: string, max = 72) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= max) return cleaned;
  return `${cleaned.slice(0, max - 1)}…`;
}

export default function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [tab, setTab] = useState<TabId>("dataset");

  const [selectedExample, setSelectedExample] = useState(3);
  const [context, setContext] = useState(examples[3].context);
  const [question, setQuestion] = useState(examples[3].question);
  const [reference, setReference] = useState(examples[3].reference);
  const [result, setResult] = useState<ComparisonResponse>(demoResult);
  const [loading, setLoading] = useState(false);

  const [language, setLanguage] = useState<DatasetLanguage | "all">("arabic");
  const [dataset, setDataset] = useState<DatasetExample[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    null,
  );
  const [scoringDescription, setScoringDescription] =
    useState(SCORING_RULE_LABEL);

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
        setScoringDescription(file.scoring_rule.description);
        const firstArabic = file.examples.find((e) => e.language === "arabic");
        setSelectedDatasetId(firstArabic?.id ?? file.examples[0]?.id ?? null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetError(
          error instanceof Error ? error.message : "Failed to load dataset",
        );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredDataset = useMemo(() => {
    if (language === "all") return dataset;
    return dataset.filter((example) => example.language === language);
  }, [dataset, language]);

  const activeDataset = useMemo(() => {
    if (!filteredDataset.length) return null;
    return (
      filteredDataset.find((example) => example.id === selectedDatasetId) ??
      filteredDataset[0]
    );
  }, [filteredDataset, selectedDatasetId]);

  const datasetBest = useMemo(() => {
    if (!activeDataset) return "tie" as BestModel;
    return pickBestModel(activeDataset.mt5, activeDataset.qwen);
  }, [activeDataset]);

  const compareBest = useMemo(
    () => pickBestModel(result.mt5, result.qwen),
    [result],
  );

  const chartData = useMemo(() => {
    const mt5 = activeDataset && tab === "dataset" ? activeDataset.mt5 : result.mt5;
    const qwen =
      activeDataset && tab === "dataset" ? activeDataset.qwen : result.qwen;
    return [
      { metric: "Faith", mT5: mt5.faithfulness, Qwen: qwen.faithfulness },
      { metric: "EM", mT5: mt5.exact_match, Qwen: qwen.exact_match },
      { metric: "F1", mT5: mt5.f1, Qwen: qwen.f1 },
    ];
  }, [activeDataset, result, tab]);

  const arabicCount = dataset.filter((e) => e.language === "arabic").length;
  const malayCount = dataset.filter((e) => e.language === "malay").length;

  function onThemeToggle() {
    setTheme((current) => toggleTheme(current));
  }

  function loadExample(index: number) {
    const example = examples[index];
    setSelectedExample(index);
    setContext(example.context);
    setQuestion(example.question);
    setReference(example.reference);
  }

  async function compareModels() {
    setLoading(true);
    try {
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context,
          question,
          reference,
          mt5_configuration: "frozen",
          qwen_configuration: "frozen",
        }),
      });

      if (!response.ok) throw new Error("Backend request failed");
      const data: ComparisonResponse = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setResult(demoResult);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <strong>RAG Faithfulness Lab</strong>
            <small>mT5 × Qwen</small>
          </div>
        </div>

        <div className="nav-actions">
          <button
            type="button"
            className={tab === "dataset" ? "nav-tab active" : "nav-tab"}
            onClick={() => setTab("dataset")}
          >
            Dataset
          </button>
          <button
            type="button"
            className={tab === "compare" ? "nav-tab active" : "nav-tab"}
            onClick={() => setTab("compare")}
          >
            Compare
          </button>

          <button
            type="button"
            className="theme-toggle"
            onClick={onThemeToggle}
            aria-label={
              theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
            }
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            {theme === "dark" ? "Light" : "Dark"}
          </button>

          <a
            href="https://github.com/Akramtaha98/LoRA_Experiment"
            target="_blank"
            rel="noreferrer"
            className="github-button"
          >
            <ExternalLink size={17} />
            GitHub
          </a>
        </div>
      </nav>

      <header className="hero">
        <div className="hero-badge">
          <ShieldCheck size={16} />
          Retrieval-grounded generation
        </div>

        <h1>
          RAG Faithfulness Lab
          <span> mT5 vs Qwen on real evidence.</span>
        </h1>

        <p>
          Browse logged Arabic and Malay evaluation examples, or run the English
          Compare demo. Scores come from offline experiment results — no GPU
          required on this site.
        </p>

        <div className="hero-features">
          <div>
            <Activity size={18} />
            Faithfulness
          </div>
          <div>
            <Layers3 size={18} />
            EM & F1
          </div>
          <div>
            <Clock3 size={18} />
            Offline results
          </div>
        </div>
      </header>

      {tab === "dataset" ? (
        <>
          <section className="workspace">
            <aside className="examples-panel">
              <div className="section-heading">
                <small>LOGGED EVAL</small>
                <h2>Dataset examples</h2>
              </div>

              <div className="language-filters">
                {(
                  [
                    ["arabic", `Arabic (${arabicCount})`],
                    ["malay", `Malay (${malayCount})`],
                    ["all", `All (${dataset.length})`],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={
                      language === value ? "filter-chip active" : "filter-chip"
                    }
                    onClick={() => {
                      setLanguage(value);
                      const next =
                        value === "all"
                          ? dataset[0]
                          : dataset.find((e) => e.language === value);
                      if (next) setSelectedDatasetId(next.id);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {datasetError ? (
                <p className="panel-error">{datasetError}</p>
              ) : (
                <div className="example-list">
                  {filteredDataset.map((example) => (
                    <button
                      key={example.id}
                      type="button"
                      onClick={() => setSelectedDatasetId(example.id)}
                      className={
                        activeDataset?.id === example.id
                          ? "example active"
                          : "example"
                      }
                    >
                      <div>
                        <strong>
                          {example.language === "arabic" ? "AR" : "MS"} · #
                          {example.example_index}
                        </strong>
                        <p dir="auto">{truncateLabel(example.question)}</p>
                      </div>
                      <ChevronRight size={18} />
                    </button>
                  ))}
                </div>
              )}
            </aside>

            <section className="input-panel">
              <div className="section-heading">
                <small>EVIDENCE</small>
                <h2>Context, question & reference</h2>
              </div>

              {activeDataset ? (
                <>
                  <div className="meta-row">
                    <span>
                      Source: mT5 {activeDataset.source.mt5_config}/
                      {activeDataset.source.mt5_lora_variant} · Qwen{" "}
                      {activeDataset.source.qwen_config}/
                      {activeDataset.source.qwen_lora_variant} · seed{" "}
                      {activeDataset.source.seed}
                    </span>
                  </div>

                  <label>
                    Context
                    <textarea
                      rows={8}
                      value={activeDataset.context}
                      readOnly
                      dir="auto"
                    />
                  </label>

                  <label>
                    Question
                    <input value={activeDataset.question} readOnly dir="auto" />
                  </label>

                  <label>
                    Reference / correct answer
                    <div className="reference-box" dir="auto">
                      {activeDataset.reference}
                    </div>
                  </label>

                  <p className="scoring-note">{scoringDescription}</p>
                </>
              ) : (
                <p className="panel-error">Loading curated examples…</p>
              )}
            </section>
          </section>

          {activeDataset ? (
            <section className="results-section">
              <div className="section-heading">
                <small>OUTPUT</small>
                <h2>Side-by-side comparison</h2>
              </div>

              <BestBanner best={datasetBest} />

              <div className="models-grid">
                <ModelCard
                  title="mT5"
                  subtitle="Encoder–decoder · logged"
                  result={activeDataset.mt5}
                  isBest={datasetBest === "mt5"}
                />
                <ModelCard
                  title="Qwen"
                  subtitle="Decoder-only · logged"
                  result={activeDataset.qwen}
                  isBest={datasetBest === "qwen"}
                />
              </div>

              <section className="chart-card">
                <div className="section-heading">
                  <small>METRICS</small>
                  <h2>Comparison profile</h2>
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height={290}>
                    <BarChart data={chartData}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        vertical={false}
                        opacity={0.15}
                      />
                      <XAxis dataKey="metric" axisLine={false} tickLine={false} />
                      <YAxis
                        domain={[0, 1]}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip />
                      <Bar dataKey="mT5" fill="var(--chart-mt5)" radius={[8, 8, 0, 0]} />
                      <Bar dataKey="Qwen" fill="var(--chart-qwen)" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="interpretation-card">
                <Sparkles size={22} />
                <div>
                  <strong>Offline logged predictions</strong>
                  <p>
                    Answers and faithfulness/EM/F1 are taken from existing
                    experiment JSONL files under <code>experiment_results/</code>{" "}
                    and <code>experiment_results_qwen/</code>. This path works on
                    static hosting (Vercel) without a FastAPI process.
                  </p>
                </div>
              </section>
            </section>
          ) : null}
        </>
      ) : (
        <>
          <section className="workspace">
            <aside className="examples-panel">
              <div className="section-heading">
                <small>TEST SUITE</small>
                <h2>English demos</h2>
              </div>

              <div className="example-list">
                {examples.map((example, index) => (
                  <button
                    key={example.name}
                    type="button"
                    onClick={() => loadExample(index)}
                    className={
                      selectedExample === index ? "example active" : "example"
                    }
                  >
                    <div>
                      <strong>{example.name}</strong>
                      <p>{example.description}</p>
                    </div>
                    <ChevronRight size={18} />
                  </button>
                ))}
              </div>
            </aside>

            <section className="input-panel">
              <div className="section-heading">
                <small>INPUT</small>
                <h2>Evidence & Question</h2>
              </div>

              <label>
                Context
                <textarea
                  rows={7}
                  value={context}
                  onChange={(event) => setContext(event.target.value)}
                />
              </label>

              <label>
                Question
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
              </label>

              <label>
                Reference answer
                <input
                  value={reference}
                  onChange={(event) => setReference(event.target.value)}
                />
              </label>

              <div className="configuration-row">
                <label>
                  mT5 configuration
                  <select defaultValue="Frozen">
                    <option>Frozen</option>
                    <option>QLoRA — CE</option>
                    <option>QLoRA — Composite</option>
                  </select>
                </label>
                <label>
                  Qwen configuration
                  <select defaultValue="Frozen">
                    <option>Frozen</option>
                    <option>QLoRA — CE</option>
                    <option>QLoRA — Composite</option>
                  </select>
                </label>
              </div>

              <button
                type="button"
                className="compare-button"
                onClick={compareModels}
                disabled={loading}
              >
                <Play size={18} fill="currentColor" />
                {loading ? "Comparing..." : "Compare models"}
              </button>
              <p className="scoring-note">
                Live compare needs the local FastAPI mock/API. On Vercel this
                demo falls back to the built-in Knowledge Conflict sample.
              </p>
            </section>
          </section>

          <section className="results-section">
            <div className="section-heading">
              <small>OUTPUT</small>
              <h2>Side-by-side comparison</h2>
            </div>

            <BestBanner best={compareBest} />

            <div className="models-grid">
              <ModelCard
                title="mT5"
                subtitle="Encoder–decoder"
                result={result.mt5}
                isBest={compareBest === "mt5"}
              />
              <ModelCard
                title="Qwen"
                subtitle="Decoder-only"
                result={result.qwen}
                isBest={compareBest === "qwen"}
              />
            </div>

            <section className="chart-card">
              <div className="section-heading">
                <small>METRICS</small>
                <h2>Comparison profile</h2>
              </div>
              <div className="chart">
                <ResponsiveContainer width="100%" height={290}>
                  <BarChart data={chartData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      opacity={0.15}
                    />
                    <XAxis dataKey="metric" axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 1]} axisLine={false} tickLine={false} />
                    <Tooltip />
                    <Bar dataKey="mT5" fill="var(--chart-mt5)" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="Qwen" fill="var(--chart-qwen)" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="interpretation-card">
              <Sparkles size={22} />
              <div>
                <strong>How to interpret this example</strong>
                <p>
                  The models receive identical evidence. A high faithfulness
                  score indicates that the generated answer is supported by the
                  supplied context. Exact Match and token F1 separately describe
                  agreement with the reference answer.
                </p>
              </div>
            </section>
          </section>
        </>
      )}

      <footer>
        <span>LoRA RAG Faithfulness Experiment</span>
        <span>
          Dataset: {arabicCount} Arabic · {malayCount} Malay
        </span>
      </footer>
    </main>
  );
}
