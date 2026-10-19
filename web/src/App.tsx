import { useEffect, useState } from "react";
import { BookOpenText, ExternalLink, Moon, Sun } from "lucide-react";
import { applyTheme, getPreferredTheme, toggleTheme, type Theme } from "./lib/theme";
import Home from "./pages/Home";
import { useResults } from "./lib/useResults";
import HeroDashboard from "./pages/HeroDashboard";
import type { HeroData } from "./data/hero";
import "./hero.css";

export default function App() {
  const [theme, setTheme] = useState<Theme>("light");
  const [study, setStudy] = useState<"current" | "earlier">("current");
  const { data, error } = useResults();
  const [hero, setHero] = useState<HeroData | null>(null);
  const [heroError, setHeroError] = useState<string | null>(null);

  useEffect(() => { const t = getPreferredTheme(); applyTheme(t); setTheme(t); }, []);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/hero.json`).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json() as Promise<HeroData>;
    }).then(setHero).catch((reason: unknown) => setHeroError(reason instanceof Error ? reason.message : "Unknown error"));
  }, []);

  return (
    <div className="app">
      <div className="bg-orb bg-orb-a" aria-hidden />
      <div className="bg-orb bg-orb-b" aria-hidden />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><BookOpenText size={18} /></span>
          <div><strong>The QA study</strong><small>Arabic & Malay</small></div>
        </div>
        <div className="top-actions">
          <button type="button" className="icon-btn" onClick={() => setTheme((t) => toggleTheme(t))} aria-label="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a className="icon-btn" href="https://github.com/Akramtaha98/LoRA_Experiment" target="_blank" rel="noreferrer" aria-label="GitHub repository"><ExternalLink size={16} /></a>
        </div>
      </header>
      <main className="shell">
        <div className="study-switch" role="tablist" aria-label="Study"><button role="tab" aria-selected={study === "current"} onClick={() => setStudy("current")}>Current results <span>Qwen3 · T5Gemma 2</span></button><button role="tab" aria-selected={study === "earlier"} onClick={() => setStudy("earlier")}>Earlier study <span>mT5 baseline</span></button></div>
        {study === "current" ? (hero ? <HeroDashboard data={hero}/> : heroError ? <p className="error">Could not load current results: {heroError}</p> : <p className="hint">Loading current results…</p>) : <div className="earlier-study"><div className="earlier-note"><strong>Earlier study · mT5 baseline</strong><span>Earlier experiments with a different setup. Compare these separately.</span></div>{error ? <p className="error">Could not load earlier results: {error}</p> : data ? <Home d={data}/> : <p className="hint">Loading earlier results…</p>}</div>}
        <footer className="footer"><span>Numbers come from saved experiment results.</span><span>Saved experiment results</span></footer>
      </main>
    </div>
  );
}
