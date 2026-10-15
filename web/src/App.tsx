import { useEffect, useState } from "react";
import { BookOpenText, ExternalLink, Moon, Sun } from "lucide-react";
import { applyTheme, getPreferredTheme, toggleTheme, type Theme } from "./lib/theme";
import Home from "./pages/Home";
import { useResults } from "./lib/useResults";

export default function App() {
  const [theme, setTheme] = useState<Theme>(getPreferredTheme);
  const { data, error } = useResults();

  useEffect(() => { applyTheme(theme); }, [theme]);

  return (
    <div className="app">
      <div className="bg-orb bg-orb-a" aria-hidden />
      <div className="bg-orb bg-orb-b" aria-hidden />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><BookOpenText size={18} /></span>
          <div><strong>Explore model answers</strong><small>mT5 · Qwen3 · T5Gemma 2</small></div>
        </div>
        <div className="top-actions">
          <button type="button" className="icon-btn" onClick={() => setTheme((t) => toggleTheme(t))} aria-label="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a className="icon-btn" href="https://github.com/Akramtaha98/LoRA_Experiment" target="_blank" rel="noreferrer" aria-label="GitHub repository"><ExternalLink size={16} /></a>
        </div>
      </header>
      <main className="shell">
        {error ? <p className="error">Could not load results: {error}</p> : null}
        {data ? <Home d={data} /> : !error ? <p className="hint">Loading…</p> : null}
        <footer className="footer"><span>Saved experimental outputs and scores from the repository.</span><span>Static site</span></footer>
      </main>
    </div>
  );
}
