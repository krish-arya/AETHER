import type { ReactNode } from "react";

type PanelProps = {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function Panel({ title, aside, children, className = "" }: PanelProps) {
  return (
    <section className={`flex flex-col rounded-md border border-line bg-panel ${className}`}>
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <h2 className="num text-[11px] font-medium tracking-[0.14em] text-muted uppercase">
          {title}
        </h2>
        {aside ? <div className="text-xs text-muted">{aside}</div> : null}
      </header>
      <div className="flex-1 p-4">{children}</div>
    </section>
  );
}

export function PanelSkeleton({ title, className = "" }: { title: string; className?: string }) {
  return (
    <Panel title={title} className={className}>
      <p className="text-sm text-faint">Loading…</p>
    </Panel>
  );
}

/** A labelled value in a definition grid. */
export function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] tracking-wide text-faint uppercase">{label}</dt>
      <dd className="num mt-0.5 truncate text-sm text-text">{children}</dd>
    </div>
  );
}
