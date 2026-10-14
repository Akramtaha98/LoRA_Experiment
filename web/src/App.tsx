import { useEffect, useState } from "react";
import { BookOpenText, ExternalLink, Moon, Sun } from "lucide-react";
import { applyTheme, getPreferredTheme, toggleTheme, type Theme } from "./lib/theme";
import { useResults } from "./lib/useResults";
import Overview from "./pages/Overview";
import Corrected from "./pages/Corrected";
import Backbones from "./pages/Backbones";
import Validation from "./pages/Validation";
import Worked from "./pages/Worked";
import ExamplesPage from "./pages/Examples";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "corrected", label: "Corrected study" },
  { id: "backbones", label: "T5Gemma 2 & other backbones" },
  { id: "validation", label: "Validation" },
  { id: "worked", label: "Worked examples" },
  { id: "examples", label: "First-study logs" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const fromHash = (): TabId => {
  const h = window.location.hash.replace(/^#\/?/, "");
  return (TABS.find((t) => t.id === h)?.id ?? "overview") as TabId;
};

export default function App() {
  const [theme, setTheme] = useState<Theme>("light");
  const [tab, setTab] = useState<TabId>(fromHash());
  const { data, error } = useResults();

  useEffect(() => { const t = getPreferredTheme(); applyTheme(t); setTheme(t); }, []);
  useEffect(() => {
    const on = () => setTab(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = (id: string) => { window.location.hash = `/${id}`; window.scrollTo({ top: 0 }); };

  return (
    <div className="app">
      <div className="bg-orb bg-orb-a" aria-hidden />
      <div className="bg-orb bg-orb-b" aria-hidden />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><BookOpenText size={18} /></span>
          <div><strong>NLI-Faithfulness Reward Lab</strong><small>LoRA · QLoRA · DoRA · Arabic and Malay QA</small></div>
        </div>
        <div className="top-actions">
          <button type="button" className="icon-btn" onClick={() => setTheme((t) => toggleTheme(t))} aria-label="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a className="icon-btn" href="https://github.com/Akramtaha98/LoRA_Experiment" target="_blank" rel="noreferrer" aria-label="GitHub repository"><ExternalLink size={16} /></a>
        </div>
      </header>
      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => go(t.id)} aria-current={tab === t.id ? "page" : undefined}>{t.label}</button>
        ))}
      </nav>
      <main className="shell">
        {error ? <p className="error">Could not load results: {error}</p> : null}
        {tab === "examples" ? <ExamplesPage /> : null}
        {data && tab === "overview" ? <Overview d={data} go={go} /> : null}
        {data && tab === "corrected" ? <Corrected d={data} /> : null}
        {data && tab === "backbones" ? <Backbones d={data} /> : null}
        {data && tab === "validation" ? <Validation d={data} /> : null}
        {data && tab === "worked" ? <Worked d={data} /> : null}
        {!data && !error && tab !== "examples" ? <p className="hint">Loading results…</p> : null}
        <footer className="footer">
          <span>Results are read from <code>data/results.json</code>, generated from the repository's <code>results/</code> folder.</span>
          <span>Static site · no GPU required</span>
        </footer>
      </main>
    </div>
  );
}
