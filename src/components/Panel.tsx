import type { ReactNode } from "react";

export function Panel({
  kicker,
  title,
  children,
  side = "left",
  wide = false,
}: {
  kicker: string;
  title?: string;
  children: ReactNode;
  side?: "left" | "right" | "center";
  wide?: boolean;
}) {
  return (
    <div className={`chapter-panel ${side !== "left" ? side : ""} ${wide ? "wide" : ""}`}>
      <p className="chapter-kicker">{kicker}</p>
      {title ? <h2>{title}</h2> : null}
      {children}
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return <div className="stat-row">{children}</div>;
}

export function Stat({ value, label, accent = false }: { value: string; label: string; accent?: boolean }) {
  return (
    <div className={`stat-chip ${accent ? "accent" : ""}`}>
      <b>{value}</b>
      {label}
    </div>
  );
}

export function Badge({ children, live = false, ghost = false }: { children: ReactNode; live?: boolean; ghost?: boolean }) {
  return <span className={`badge ${live ? "live" : ""} ${ghost ? "ghost" : ""}`}>{children}</span>;
}
