import type { ReactNode } from "react";

import type { ModuleStatus } from "@/lib/types";

const TONES = {
  ok: "border-ok/40 text-ok",
  down: "border-down/40 text-down",
  warn: "border-warn/40 text-warn",
  muted: "border-line text-muted",
  synthetic: "border-synthetic/50 text-synthetic",
} as const;

export type Tone = keyof typeof TONES;

export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span
      className={`num inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[10px] font-medium tracking-wider whitespace-nowrap uppercase ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

export function Dot({ tone }: { tone: Tone }) {
  const color = { ok: "bg-ok", down: "bg-down", warn: "bg-warn", muted: "bg-faint", synthetic: "bg-synthetic" }[tone];
  return <span aria-hidden className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} />;
}

/** Mock development data must be unmistakable wherever it appears (PRD rule 2). */
export function SyntheticBadge() {
  return (
    <span title="Generated mock data for development. Not real market data.">
      <Badge tone="synthetic">Synthetic · not market data</Badge>
    </span>
  );
}

const MODULE_TONE: Record<ModuleStatus, Tone> = {
  available: "ok",
  in_progress: "warn",
  planned: "muted",
};

const MODULE_LABEL: Record<ModuleStatus, string> = {
  available: "Available",
  in_progress: "In progress",
  planned: "Not built",
};

export function ModuleStatusBadge({ status }: { status: ModuleStatus }) {
  return (
    <Badge tone={MODULE_TONE[status]}>
      <Dot tone={MODULE_TONE[status]} />
      {MODULE_LABEL[status]}
    </Badge>
  );
}
