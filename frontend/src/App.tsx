import { useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  ChevronRight,
  Clock3,
  ExternalLink,
  Layers3,
  Play,
  ShieldCheck,
  Sparkles,
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

type ModelResult = {
  answer: string;
  faithfulness: number;
  exact_match: number;
  f1: number;
  latency_ms: number;
};

type ComparisonResponse = {
  mt5: ModelResult;
  qwen: ModelResult;
};

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
}: {
  title: string;
  subtitle: string;
  result: ModelResult;
}) {
  return (
    <section className="model-card">
      <div className="model-title">
        <div>
          <p>{subtitle}</p>
          <h2>{title}</h2>
        </div>

        <div className="model-icon">
          <BrainCircuit size={22} />
        </div>
      </div>

      <div className="answer-box">
        <span>Generated answer</span>
        <p>{result.answer}</p>
      </div>

      <div className="metric-grid">
        <Score title="Faithfulness" value={result.faithfulness.toFixed(2)} />
        <Score title="Exact Match" value={result.exact_match.toFixed(2)} />
        <Score title="Token F1" value={result.f1.toFixed(2)} />
        <Score title="Latency" value={result.latency_ms} suffix=" ms" />
      </div>
    </section>
  );
}

export default function App() {
  const [selectedExample, setSelectedExample] = useState(3);

  const [context, setContext] = useState(examples[3].context);
  const [question, setQuestion] = useState(examples[3].question);
  const [reference, setReference] = useState(examples[3].reference);

  const [result, setResult] = useState<ComparisonResponse>(demoResult);
  const [loading, setLoading] = useState(false);

  const chartData = useMemo(
    () => [
      {
        metric: "Faith",
        mT5: result.mt5.faithfulness,
        Qwen: result.qwen.faithfulness,
      },
      {
        metric: "EM",
        mT5: result.mt5.exact_match,
        Qwen: result.qwen.exact_match,
      },
      {
        metric: "F1",
        mT5: result.mt5.f1,
        Qwen: result.qwen.f1,
      },
    ],
    [result],
  );

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
      const response = await fetch("http://localhost:8000/api/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          context,
          question,
          reference,
          mt5_configuration: "frozen",
          qwen_configuration: "frozen",
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data: ComparisonResponse = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      // Keeps the UI demonstrable before the Python API is connected.
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
          <span>Compare</span>
          <span>Examples</span>
          <span>Results</span>

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
          See what changes when the
          <span> model architecture changes.</span>
        </h1>

        <p>
          Give mT5 and Qwen the same context and question. Compare what they
          generate, how faithfully they follow the supplied evidence, and where
          their behavior diverges.
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
            Latency
          </div>
        </div>
      </header>

      <section className="workspace">
        <aside className="examples-panel">
          <div className="section-heading">
            <small>TEST SUITE</small>
            <h2>Examples</h2>
          </div>

          <div className="example-list">
            {examples.map((example, index) => (
              <button
                key={example.name}
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
                <option>AdaLoRA — CE</option>
                <option>AdaLoRA — Composite</option>
                <option>DoRA — CE</option>
                <option>DoRA — Composite</option>
                <option>VeRA — CE</option>
                <option>VeRA — Composite</option>
              </select>
            </label>

            <label>
              Qwen configuration
              <select defaultValue="Frozen">
                <option>Frozen</option>
                <option>QLoRA — CE</option>
                <option>QLoRA — Composite</option>
                <option>AdaLoRA — CE</option>
                <option>AdaLoRA — Composite</option>
                <option>DoRA — CE</option>
                <option>DoRA — Composite</option>
                <option>VeRA — CE</option>
                <option>VeRA — Composite</option>
              </select>
            </label>
          </div>

          <button
            className="compare-button"
            onClick={compareModels}
            disabled={loading}
          >
            <Play size={18} fill="currentColor" />
            {loading ? "Comparing..." : "Compare models"}
          </button>
        </section>
      </section>

      <section className="results-section">
        <div className="section-heading">
          <small>OUTPUT</small>
          <h2>Side-by-side comparison</h2>
        </div>

        <div className="models-grid">
          <ModelCard
            title="mT5"
            subtitle="Encoder–decoder"
            result={result.mt5}
          />

          <ModelCard
            title="Qwen"
            subtitle="Decoder-only"
            result={result.qwen}
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
                <Bar dataKey="mT5" radius={[8, 8, 0, 0]} />
                <Bar dataKey="Qwen" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="interpretation-card">
          <Sparkles size={22} />

          <div>
            <strong>How to interpret this example</strong>
            <p>
              The models receive identical evidence. A high faithfulness score
              indicates that the generated answer is supported by the supplied
              context. Exact Match and token F1 separately describe agreement
              with the reference answer.
            </p>
          </div>
        </section>
      </section>

      <footer>
        <span>LoRA RAG Faithfulness Experiment</span>
        <span>mT5 × Qwen</span>
      </footer>
    </main>
  );
}
