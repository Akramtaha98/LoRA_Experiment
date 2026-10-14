import type { ReactNode } from "react";
import { AlertTriangle, Info } from "lucide-react";

export function Section({ title, kicker, children }: { title: string; kicker?: string; children: ReactNode }) {
  return (
    <section className="block">
      <header className="block-head">
        {kicker ? <span className="kicker">{kicker}</span> : null}
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "good" | "warn" | "neutral" }) {
  return (
    <div className={`stat stat-${tone ?? "neutral"}`}>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {sub ? <span className="stat-sub">{sub}</span> : null}
    </div>
  );
}

export function Callout({ kind = "info", title, children }: { kind?: "info" | "warn"; title?: string; children: ReactNode }) {
  const Icon = kind === "warn" ? AlertTriangle : Info;
  return (
    <div className={`callout callout-${kind}`} role="note">
      <Icon size={18} />
      <div>
        {title ? <strong>{title}</strong> : null}
        <p>{children}</p>
      </div>
    </div>
  );
}

export function Card({ title, note, children }: { title?: string; note?: string; children: ReactNode }) {
  return (
    <div className="card">
      {title ? <h3>{title}</h3> : null}
      {note ? <p className="hint">{note}</p> : null}
      {children}
    </div>
  );
}

export interface StripRow { label: string; values: number[]; mean: number; lo?: number | null; hi?: number | null }

/** Paired-seed strip plot: one row per cell, one dot per seed, mean marker and optional CI whisker. */
export function StripPlot({ rows, unit = "C − B", domain }: { rows: StripRow[]; unit?: string; domain?: [number, number] }) {
  const all = rows.flatMap((r) => [...r.values, ...(r.lo != null ? [r.lo] : []), ...(r.hi != null ? [r.hi] : [])]);
  const lo = domain ? domain[0] : Math.min(0, ...all);
  const hi = domain ? domain[1] : Math.max(0, ...all);
  const pad = (hi - lo) * 0.06 || 0.01;
  const a = lo - pad, b = hi + pad;
  const W = 640, L = 150, R = 16, rowH = 30, H = rows.length * rowH + 34;
  const x = (v: number) => L + ((v - a) / (b - a)) * (W - L - R);
  const ticks = Array.from({ length: 5 }, (_, i) => a + ((b - a) * i) / 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="strip" role="img" aria-label={`Per-seed ${unit} by cell`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={6} y2={H - 24} className="grid" />
          <text x={x(t)} y={H - 8} textAnchor="middle" className="axis">{t.toFixed(Math.abs(b - a) < 0.1 ? 3 : 2)}</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={6} y2={H - 24} className="zero" />
      {rows.map((r, i) => {
        const y = 20 + i * rowH;
        return (
          <g key={r.label}>
            <text x={L - 10} y={y + 4} textAnchor="end" className="rowlabel">{r.label}</text>
            {r.lo != null && r.hi != null ? <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} className="whisker" /> : null}
            {r.values.map((v, k) => <circle key={k} cx={x(v)} cy={y} r={3.2} className="dot" />)}
            <rect x={x(r.mean) - 1.5} y={y - 8} width={3} height={16} className="mean" rx={1} />
          </g>
        );
      })}
    </svg>
  );
}

/** Horizontal bar list with optional error whiskers; values share one scale. */
export function BarList({ items, max, digits = 3 }: { items: { label: string; value: number; sub?: string; tone?: string; lo?: number; hi?: number }[]; max?: number; digits?: number }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.hi ?? i.value)), 1e-9);
  return (
    <div className="barlist">
      {items.map((i) => (
        <div className="barrow" key={i.label}>
          <span className="barlabel">{i.label}</span>
          <div className="bartrack">
            <div className={`barfill ${i.tone ?? ""}`} style={{ width: `${Math.max(0, (i.value / m) * 100)}%` }} />
            {i.lo != null && i.hi != null ? (
              <div className="barci" style={{ left: `${Math.max(0, (i.lo / m) * 100)}%`, width: `${Math.max(0, ((i.hi - i.lo) / m) * 100)}%` }} />
            ) : null}
          </div>
          <span className="barval">{i.value.toFixed(digits)}{i.sub ? <em>{i.sub}</em> : null}</span>
        </div>
      ))}
    </div>
  );
}
