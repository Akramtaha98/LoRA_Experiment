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
  type VariantDef,
  type VariantId,
} from "./data/dataset";
import {
  pickBestModel,
  neitherMatchedGold,
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

function MetricBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const width = value > 0 ? Math.max(pct, 3) : 0;
  return (
    <div className="mbar">
      <div className="mbar-top">
        <span>{label}</span>
        <strong>{fmt(value, label === "EM" ? 2 : 3)}</strong>
      </div>
      <div className="mbar-track">
        <div className="mbar-fill" style={{ width: `${width}%` }} />
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
  aggregate,
  missingHint,
}: {
  title: string;
  subtitle: string;
  answer: string;
  metrics?: ModelScores | null;
  badge?: string;
  variant?: "gold" | "model";
  aggregate?: AggregateMetrics | null;
  missingHint?: string;
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
        <div className="missing-block">
          <p className="missing">
            {missingHint ??
              "No per-example prediction was saved in the experiment logs for this selection."}
          </p>
          {aggregate ? (
            <div className="agg-fallback">
              <p className="agg-title">Dataset mean for this selection</p>
              <div className="metric-stack">
                <MetricBar label="EM" value={aggregate.mean_em} />
                <MetricBar label="F1" value={aggregate.mean_f1} />
                <MetricBar label="Faith" value={aggregate.mean_faithfulness} />
              </div>
              <p className="missing">
                n={aggregate.n_eval}, seed={aggregate.seed}
                {aggregate.source_file ? `, ${aggregate.source_file}` : ""}
              </p>
            </div>
          ) : null}
        </div>
      )}
    </article>
  );
}

function bestLabel(
  best: BestModel,
  mt5: ModelScores | null,
  qwen: ModelScores | null,
) {
  if (best === "incomplete") {
    return "Incomplete comparison: only one model is logged for this selection";
  }
  if (neitherMatchedGold(mt5, qwen)) {
    return "Neither model matched the correct answer";
  }
  if (best === "tie") return "Tie versus the correct answer";
  if (best === "mt5") return "Best match to the correct answer: mT5";
  return "Best match to the correct answer: Qwen";
}

export default function App() {
  const [theme, setTheme] = useState<Theme>("light");
  const [file, setFile] = useState<DatasetFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<DatasetLanguage>("arabic");
  const [variantId, setVariantId] = useState<VariantId>("qlora");
  const [configId, setConfigId] = useState<ConfigId>("C_composite_lora");
  const [exampleId, setExampleId] = useState<string>("");
  const [showContext, setShowContext] = useState(false);
  const [showMeans, setShowMeans] = useState(true);
  const [showAcross, setShowAcross] = useState(true);
  const [completeOnly, setCompleteOnly] = useState(true);

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
  const variantDefs = file?.variant_defs ?? [];
  const examples = file?.examples ?? [];

  const filtered = useMemo(
    () => examples.filter((e) => e.language === language),
    [examples, language],
  );

  function pairStatus(
    example: DatasetExample,
    cfg: ConfigId,
    variant: VariantId,
  ) {
    const row = example.configs?.[cfg]?.variants?.[variant];
    const hasMt5 = Boolean(row?.mt5);
    const hasQwen = Boolean(row?.qwen);
    return {
      hasMt5,
      hasQwen,
      both: hasMt5 && hasQwen,
      either: hasMt5 || hasQwen,
    };
  }

  const coverage = useMemo(() => {
    const map: Record<
      string,
      { both: number; mt5: number; qwen: number; total: number }
    > = {};
    for (const cfg of configDefs) {
      for (const variant of variantDefs) {
        const key = `${cfg.id}::${variant.id}`;
        const stats = { both: 0, mt5: 0, qwen: 0, total: filtered.length };
        for (const example of filtered) {
          const status = pairStatus(example, cfg.id, variant.id);
          if (status.hasMt5) stats.mt5 += 1;
          if (status.hasQwen) stats.qwen += 1;
          if (status.both) stats.both += 1;
        }
        map[key] = stats;
      }
    }
    return map;
  }, [filtered, configDefs, variantDefs]);

  function selectionStats(cfg: ConfigId, variant: VariantId) {
    return (
      coverage[`${cfg}::${variant}`] ?? {
        both: 0,
        mt5: 0,
        qwen: 0,
        total: filtered.length,
      }
    );
  }

  const usableExamples = useMemo(() => {
    if (!completeOnly) return filtered;
    return filtered.filter(
      (example) => pairStatus(example, configId, variantId).both,
    );
  }, [filtered, completeOnly, configId, variantId]);

  const active: DatasetExample | null = useMemo(() => {
    if (!usableExamples.length) return null;
    return usableExamples.find((e) => e.id === exampleId) ?? usableExamples[0];
  }, [usableExamples, exampleId]);

  // Keep selection on combinations that actually have logged answers.
  useEffect(() => {
    if (!filtered.length || !configDefs.length || !variantDefs.length) return;
    const current = selectionStats(configId, variantId);
    const ok = completeOnly ? current.both > 0 : current.mt5 + current.qwen > 0;
    if (ok) return;

    const preferred: Array<[ConfigId, VariantId]> = [
      ["C_composite_lora", "qlora"],
      ["C_composite_lora", "dora"],
      ["C_composite_lora", "adalora"],
      ["C_composite_lora", "vera"],
      ["B_ce_lora", "qlora"],
      ["B_ce_lora", "dora"],
    ];
    for (const [cfg, variant] of preferred) {
      const stats = selectionStats(cfg, variant);
      if (completeOnly ? stats.both > 0 : stats.mt5 + stats.qwen > 0) {
        setConfigId(cfg);
        setVariantId(variant);
        return;
      }
    }
  }, [
    filtered,
    configDefs,
    variantDefs,
    configId,
    variantId,
    completeOnly,
    coverage,
  ]);

  useEffect(() => {
    if (!usableExamples.length) return;
    if (!usableExamples.some((e) => e.id === exampleId)) {
      setExampleId(usableExamples[0].id);
    }
  }, [usableExamples, exampleId]);

  const activeConfig: ConfigDef | undefined = configDefs.find(
    (c) => c.id === configId,
  );
  const activeVariant: VariantDef | undefined = variantDefs.find(
    (v) => v.id === variantId,
  );

  const pred = active?.configs?.[configId]?.variants?.[variantId];
  const mt5 = pred?.mt5 ?? null;
  const qwen = pred?.qwen ?? null;
  const best = useMemo(() => pickBestModel(mt5, qwen), [mt5, qwen]);
  const noGoldMatch = neitherMatchedGold(mt5, qwen);

  const qwenAggregate =
    file?.aggregates?.[language]?.[configId]?.[variantId]?.qwen ?? null;
  const mt5Aggregate =
    file?.aggregates?.[language]?.[configId]?.[variantId]?.mt5 ?? null;

  const arabicCount = examples.filter((e) => e.language === "arabic").length;
  const malayCount = examples.filter((e) => e.language === "malay").length;

  const currentStats = selectionStats(configId, variantId);
  const completeCombos = useMemo(() => {
    const labels: string[] = [];
    for (const cfg of configDefs) {
      for (const variant of variantDefs) {
        const stats = selectionStats(cfg.id, variant.id);
        if (stats.both > 0) {
          labels.push(
            `${cfg.label.replace(/^([A-D])\s+/, "$1")} · ${variant.label} (${stats.both}/${stats.total})`,
          );
        }
      }
    }
    return labels;
  }, [configDefs, variantDefs, coverage]);

  const coverageBanner = useMemo(() => {
    if (!filtered.length) return "Loading coverage…";
    if (completeOnly) {
      if (!completeCombos.length) {
        return "No complete mT5+Qwen pairs in the static logs for this language. Turn off Complete pairs to browse partial runs, or re-export full per-question dumps.";
      }
      return `Full side-by-side logs exist only for: ${completeCombos.join("; ")}. Other procedures/variants show “Not logged” because those checkpoints never wrote per-question answers — not a UI bug.`;
    }
    const partial =
      currentStats.mt5 > 0 && currentStats.qwen === 0
        ? "Qwen answers were not saved for this combo (means may still appear below)."
        : currentStats.qwen > 0 && currentStats.mt5 === 0
          ? "mT5 answers were not saved for this combo (means may still appear below)."
          : currentStats.mt5 === 0 && currentStats.qwen === 0
            ? "Neither model has per-question logs here."
            : `${currentStats.both}/${currentStats.total} questions have both models.`;
    return partial;
  }, [
    filtered.length,
    completeOnly,
    completeCombos,
    currentStats,
  ]);

  function setLang(next: DatasetLanguage) {
    setLanguage(next);
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
            <small>mT5 × Qwen · offline seed-42 logs</small>
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
          <h1>Compare LoRA variants against gold answers</h1>
          <p>
            Paper axis: QLoRA / AdaLoRA / DoRA / VeRA. Secondary axis: training
            procedure A–D. Numbers prefer dedicated seed-42 checkpoint files.
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
            <span className="control-label">LoRA variant (paper comparison)</span>
            <div className="seg seg-wrap" role="tablist" aria-label="Variant">
              {variantDefs.map((v) => {
                const stats = selectionStats(configId, v.id);
                const available = completeOnly
                  ? stats.both > 0
                  : stats.mt5 + stats.qwen > 0;
                return (
                  <button
                    key={v.id}
                    type="button"
                    className={
                      variantId === v.id ? "seg-btn active" : "seg-btn"
                    }
                    disabled={!available}
                    title={
                      available
                        ? `${stats.both}/${stats.total} complete pairs`
                        : completeOnly
                          ? "No complete mT5+Qwen logs — turn off Complete pairs or pick another combo"
                          : "No per-example logs for this variant under the current procedure"
                    }
                    onClick={() => setVariantId(v.id)}
                  >
                    {v.label}
                    <em>
                      {completeOnly
                        ? `${stats.both}/${stats.total}`
                        : `${Math.max(stats.mt5, stats.qwen)}/${stats.total}`}
                    </em>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="control-block">
            <span className="control-label">Training procedure</span>
            <div className="seg seg-wrap" role="tablist" aria-label="Config">
              {configDefs.map((cfg) => {
                const stats = selectionStats(cfg.id, variantId);
                const available = completeOnly
                  ? stats.both > 0
                  : stats.mt5 + stats.qwen > 0;
                return (
                  <button
                    key={cfg.id}
                    type="button"
                    className={
                      configId === cfg.id ? "seg-btn active" : "seg-btn"
                    }
                    disabled={!available}
                    title={
                      available
                        ? cfg.description
                        : completeOnly
                          ? "No complete mT5+Qwen logs for this procedure with the current variant"
                          : "No per-example logs for this procedure with the current variant"
                    }
                    onClick={() => setConfigId(cfg.id)}
                  >
                    {cfg.label.replace(/^([A-D])\s+/, "$1 · ")}
                    <em>
                      {completeOnly
                        ? `${stats.both}/${stats.total}`
                        : `${Math.max(stats.mt5, stats.qwen)}/${stats.total}`}
                    </em>
                  </button>
                );
              })}
            </div>
            {activeConfig ? (
              <p className="hint">{activeConfig.description}</p>
            ) : null}
          </div>

          <div className="control-block coverage-row">
            <label className="toggle-line">
              <input
                type="checkbox"
                checked={completeOnly}
                onChange={(e) => setCompleteOnly(e.target.checked)}
              />
              <span>
                Complete pairs only
                <small>
                  Hide combos where mT5 or Qwen was not saved per question
                </small>
              </span>
            </label>
            <p className="coverage-banner" role="status">
              {coverageBanner}
            </p>
          </div>

          <label className="control-block menubox">
            <span className="control-label">
              Example question
              {completeOnly ? (
                <em className="count-inline">
                  {" "}
                  · {usableExamples.length} with both models
                </em>
              ) : null}
            </span>
            <div className="select-shell">
              <select
                value={active?.id ?? ""}
                onChange={(e) => {
                  setExampleId(e.target.value);
                  setShowContext(false);
                }}
                disabled={!usableExamples.length}
              >
                {usableExamples.map((ex) => (
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
                    {active.example_index} · {activeVariant?.label ?? variantId}{" "}
                    · {activeConfig?.label ?? configId}
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
                <strong>{bestLabel(best, mt5, qwen)}</strong>
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
                badge={best === "mt5" && !noGoldMatch ? "Best" : undefined}
                aggregate={!mt5 ? mt5Aggregate : null}
                missingHint={
                  mt5
                    ? undefined
                    : "mT5 per-question answers were never written for this procedure × variant. Re-run eval with answer dumps to fill this cell."
                }
              />
              <AnswerCard
                title="Qwen"
                subtitle="Decoder-only"
                answer={qwen?.answer || "Not logged"}
                metrics={qwen}
                badge={best === "qwen" && !noGoldMatch ? "Best" : undefined}
                aggregate={!qwen ? qwenAggregate : null}
                missingHint={
                  qwen
                    ? undefined
                    : "Qwen per-question answers were never written for this procedure × variant. Means below are from aggregate files only."
                }
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
                  <h3>Same example across LoRA variants</h3>
                  <p>
                    Under {activeConfig?.label ?? configId}: compare logged
                    variants for this question
                    {completeOnly ? " (complete pairs only)" : ""}.
                  </p>
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
                        <th>Variant</th>
                        <th>mT5 answer</th>
                        <th>Qwen answer</th>
                        <th>Best</th>
                      </tr>
                    </thead>
                    <tbody>
                      {variantDefs
                        .filter((v) => {
                          if (!completeOnly) return true;
                          return selectionStats(configId, v.id).both > 0;
                        })
                        .map((v) => {
                          const row =
                            active.configs[configId]?.variants?.[v.id];
                          const hasPair = Boolean(row?.mt5 && row?.qwen);
                          const canSelect = completeOnly
                            ? hasPair
                            : Boolean(row?.mt5 || row?.qwen);
                          return (
                            <tr
                              key={v.id}
                              className={
                                v.id === variantId ? "is-active-row" : undefined
                              }
                            >
                              <td>
                                <button
                                  type="button"
                                  className="linkish"
                                  disabled={!canSelect}
                                  onClick={() => setVariantId(v.id)}
                                >
                                  {v.label}
                                </button>
                              </td>
                              <td dir="auto">
                                {row?.mt5?.answer ?? "Not logged"}
                              </td>
                              <td dir="auto">
                                {row?.qwen?.answer ?? "Not logged"}
                              </td>
                              <td>
                                {row?.best_model === "incomplete" ||
                                (!row?.mt5 && !row?.qwen)
                                  ? "n/a"
                                  : row.best_model === "tie"
                                    ? neitherMatchedGold(row.mt5, row.qwen)
                                      ? "Neither"
                                      : "Tie"
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
          <div className="empty-state">
            <p className="muted">
              {error ??
                (file
                  ? completeOnly
                    ? "No complete mT5+Qwen pairs for this selection. Pick C · QLoRA or C · DoRA, or turn off Complete pairs."
                    : "No logged answers for this selection."
                  : "Loading examples…")}
            </p>
          </div>
        )}

        <section className="fold panel">
          <button
            type="button"
            className="fold-head"
            onClick={() => setShowMeans((v) => !v)}
            aria-expanded={showMeans}
          >
            <div>
              <h3>Mean metrics by variant ({activeConfig?.label})</h3>
              <p>
                Aggregate seed-42 scores for {language}. Prefer dedicated
                checkpoint files (paper QLoRA Arabic Qwen: 0.1331 / 0.3931 /
                0.5568).
              </p>
            </div>
            <ChevronDown size={18} className={showMeans ? "chev open" : "chev"} />
          </button>
          {showMeans && file ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variant</th>
                    <th>Model</th>
                    <th>Source file</th>
                    <th>n</th>
                    <th>Faith</th>
                    <th>EM</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {variantDefs.flatMap((v) => {
                    const block =
                      file.aggregates?.[language]?.[configId]?.[v.id] || {};
                    return (["mt5", "qwen"] as const).map((model) => {
                      const row = block[model] as AggregateMetrics | undefined;
                      return (
                        <tr
                          key={`${v.id}-${model}`}
                          className={
                            v.id === variantId ? "is-active-row" : undefined
                          }
                        >
                          <td>{v.label}</td>
                          <td>{model === "mt5" ? "mT5" : "Qwen"}</td>
                          <td className="mono">
                            {row?.source_file ?? "n/a"}
                          </td>
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
          ) : null}
        </section>

        <footer className="footer">
          <span>
            {arabicCount} Arabic · {malayCount} Malay · static logs
            {file?.generated_at ? ` · data ${file.generated_at}` : ""}
          </span>
          <span>No GPU required</span>
        </footer>
      </main>
    </div>
  );
}
