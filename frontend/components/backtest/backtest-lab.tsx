"use client";

import { useState, type FormEvent } from "react";

import { Badge, Dot, SyntheticBadge, type Tone } from "@/components/ui/badges";
import { Panel } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useBacktestMetrics, useBacktestRuns, useSubmitBacktest } from "@/hooks/use-backtest";
import { MISSING, formatInteger, formatNumber, formatPercent } from "@/lib/format";
import type { BacktestRequest, BacktestRun, BacktestStatus, MetricSet } from "@/lib/types";

const STATUS_TONE: Record<BacktestStatus, Tone> = {
  queued: "muted",
  running: "warn",
  succeeded: "ok",
  failed: "down",
};

const REBALANCE = ["daily", "weekly", "monthly", "quarterly"] as const;

const signed = (v: number | null | undefined) => formatPercent(v);
const unsigned = (v: number | null | undefined) => formatPercent(v).replace("+", "");
const ratio = (v: number | null | undefined) => formatNumber(v, 2);

const COLUMNS: { key: string; label: string; format: (v: number | null | undefined) => string }[] = [
  { key: "cagr", label: "CAGR", format: signed },
  { key: "volatility", label: "Volatility", format: unsigned },
  { key: "sharpe", label: "Sharpe", format: ratio },
  { key: "sortino", label: "Sortino", format: ratio },
  { key: "max_drawdown", label: "Max DD", format: unsigned },
  { key: "calmar", label: "Calmar", format: ratio },
  { key: "turnover", label: "Turnover /yr", format: ratio },
  { key: "alpha", label: "Alpha", format: signed },
  { key: "beta", label: "Beta", format: ratio },
];

function StatusBadge({ status }: { status: BacktestStatus }) {
  return (
    <Badge tone={STATUS_TONE[status]}>
      <Dot tone={STATUS_TONE[status]} />
      {status}
    </Badge>
  );
}

function NewRunForm() {
  const submit = useSubmitBacktest();
  const [name, setName] = useState("baselines");
  const [capital, setCapital] = useState("1000000");
  const [rebalance, setRebalance] = useState<(typeof REBALANCE)[number]>("monthly");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const request: BacktestRequest = { name, rebalance, initial_capital: Number(capital) };
    submit.mutate(request);
  }

  const field =
    "num mt-1 w-full rounded border border-line bg-raised px-2 py-1.5 text-sm text-text";
  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <label className="block text-[11px] tracking-wide text-faint uppercase">
        Run name
        <input
          className={field}
          value={name}
          onChange={(e) => setName(e.target.value)}
          pattern="[a-z0-9][a-z0-9\-]{1,48}"
          title="Lower-case letters, digits and hyphens"
          required
        />
      </label>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
        <label className="block text-[11px] tracking-wide text-faint uppercase">
          Initial capital (₹)
          <input
            className={field}
            type="number"
            min={1000}
            step={1000}
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            required
          />
        </label>
        <label className="block text-[11px] tracking-wide text-faint uppercase">
          Rebalance
          <select
            className={field}
            value={rebalance}
            onChange={(e) => setRebalance(e.target.value as (typeof REBALANCE)[number])}
          >
            {REBALANCE.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
      <button
        type="submit"
        disabled={submit.isPending}
        className="w-full cursor-pointer rounded border border-accent/60 px-3 py-2 text-sm text-accent transition-colors hover:bg-accent/10 disabled:cursor-wait disabled:opacity-50"
      >
        {submit.isPending ? "Submitting…" : "Run baseline backtest"}
      </button>
      {submit.error ? <QueryError error={submit.error} /> : null}
      <p className="text-xs leading-5 text-faint">
        Runs the six rule-based baselines on the most recently loaded dataset with the
        server&apos;s default cost schedule and walk-forward windows. The job executes in a
        background worker.
      </p>
    </form>
  );
}

function RunList({
  runs,
  selected,
  onSelect,
}: {
  runs: BacktestRun[];
  selected: string | null;
  onSelect: (runId: string) => void;
}) {
  if (runs.length === 0) {
    return <p className="text-sm text-muted">No backtests have been run yet.</p>;
  }
  return (
    <ul className="-mx-4 -my-4 divide-y divide-line/60">
      {runs.map((run) => {
        const isSelected = run.run_id === selected;
        return (
          <li key={run.run_id}>
            <button
              type="button"
              onClick={() => onSelect(run.run_id)}
              aria-pressed={isSelected}
              className={`flex w-full cursor-pointer items-center justify-between gap-3 px-4 py-2.5 text-left ${
                isSelected ? "bg-raised" : "hover:bg-raised/50"
              }`}
            >
              <span className="min-w-0">
                <span className={`num block truncate text-sm ${isSelected ? "text-accent" : "text-text"}`}>
                  {run.name}
                </span>
                <span className="num block truncate text-[11px] text-faint">
                  {run.created_at.slice(0, 19).replace("T", " ")} UTC · {run.run_id.slice(0, 8)}
                </span>
              </span>
              <StatusBadge status={run.status} />
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function MetricsTable({
  title,
  note,
  rows,
  benchmark,
}: {
  title: string;
  note: string;
  rows: Record<string, MetricSet>;
  benchmark: string;
}) {
  const entries = Object.entries(rows);
  if (entries.length === 0) return null;
  return (
    <div>
      <h3 className="text-[11px] tracking-[0.14em] text-faint uppercase">{title}</h3>
      <p className="mt-1 text-xs text-muted">{note}</p>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-right text-sm">
          <thead>
            <tr className="border-b border-line text-[11px] tracking-wide text-faint uppercase">
              <th className="py-2 pr-4 text-left font-normal">Strategy</th>
              {COLUMNS.map((column) => (
                <th key={column.key} className="px-2 py-2 font-normal whitespace-nowrap">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map(([strategy, metrics]) => (
              <tr key={strategy} className="border-b border-line/60 last:border-0">
                <td className="num py-2 pr-4 text-left whitespace-nowrap text-text">
                  {strategy}
                  {strategy === benchmark ? (
                    <span className="ml-2 text-[10px] tracking-wider text-accent uppercase">
                      benchmark
                    </span>
                  ) : null}
                </td>
                {COLUMNS.map((column) => (
                  <td key={column.key} className="num px-2 py-2 text-text">
                    {column.key in metrics ? column.format(metrics[column.key]) : MISSING}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RunResults({ run }: { run: BacktestRun }) {
  const ready = run.status === "succeeded";
  const metrics = useBacktestMetrics(ready ? run.run_id : null);

  if (run.status === "queued" || run.status === "running") {
    return (
      <p className="text-sm text-muted">
        This backtest is {run.status}. Results appear here when the worker finishes — this page
        refreshes on its own.
      </p>
    );
  }
  if (run.status === "failed") {
    return (
      <div role="alert" className="text-sm">
        <p className="text-down">The backtest failed. No results were produced.</p>
        <p className="num mt-2 rounded border border-line bg-raised p-3 text-xs break-words text-muted">
          {run.error ?? "No error message was recorded."}
        </p>
      </div>
    );
  }
  if (metrics.isPending) return <p className="text-sm text-faint">Loading results…</p>;
  if (metrics.error) return <QueryError error={metrics.error} />;

  const data = metrics.data;
  const costs = Object.entries(data.full_period);
  return (
    <div className="space-y-6">
      <ul className="space-y-1.5">
        {data.caveats.map((caveat) => (
          <li
            key={caveat}
            className="rounded border border-warn/30 bg-warn/5 px-3 py-2 text-xs leading-5 text-warn"
          >
            {caveat}
          </li>
        ))}
      </ul>

      <MetricsTable
        title="Full period"
        note="Every strategy over the whole dataset, same costs and execution rules."
        rows={data.full_period}
        benchmark={data.benchmark}
      />
      <MetricsTable
        title="Walk-forward · out of sample"
        note="Stitched test windows only; each window follows its own training and validation windows."
        rows={data.walk_forward}
        benchmark={data.benchmark}
      />

      <div>
        <h3 className="text-[11px] tracking-[0.14em] text-faint uppercase">
          Trading costs · full period
        </h3>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full max-w-2xl text-right text-sm">
            <thead>
              <tr className="border-b border-line text-[11px] tracking-wide text-faint uppercase">
                <th className="py-2 pr-4 text-left font-normal">Strategy</th>
                <th className="px-2 py-2 font-normal whitespace-nowrap">Fees &amp; taxes (₹)</th>
                <th className="px-2 py-2 font-normal whitespace-nowrap">Slippage (₹)</th>
                <th className="px-2 py-2 font-normal">Trades</th>
              </tr>
            </thead>
            <tbody>
              {costs.map(([strategy, m]) => (
                <tr key={strategy} className="border-b border-line/60 last:border-0">
                  <td className="num py-2 pr-4 text-left whitespace-nowrap text-text">{strategy}</td>
                  <td className="num px-2 py-2 text-text">{formatInteger(m.total_costs)}</td>
                  <td className="num px-2 py-2 text-text">{formatInteger(m.slippage_cost)}</td>
                  <td className="num px-2 py-2 text-text">{formatInteger(m.trades)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs leading-5 text-faint">
        Alpha and beta are measured against the benchmark strategy. Max DD is the largest
        peak-to-trough loss. {MISSING} means the metric is undefined or not applicable (the
        benchmark has no alpha against itself).
      </p>
    </div>
  );
}

export function BacktestLab() {
  const { data, error, isPending } = useBacktestRuns();
  const [picked, setPicked] = useState<string | null>(null);

  const runs = data ?? [];
  const selected = runs.find((run) => run.run_id === picked) ?? runs[0] ?? null;

  return (
    <div className="grid gap-4 xl:grid-cols-4">
      <div className="space-y-4">
        <Panel title="New run">
          <NewRunForm />
        </Panel>
        <Panel
          title="Runs"
          aside={data ? <span className="num">{data.length}</span> : null}
        >
          {isPending ? (
            <p className="text-sm text-faint">Loading…</p>
          ) : error ? (
            <QueryError error={error} />
          ) : (
            <RunList runs={runs} selected={selected?.run_id ?? null} onSelect={setPicked} />
          )}
        </Panel>
      </div>

      <Panel
        title={selected ? `Results · ${selected.name}` : "Results"}
        className="min-w-0 xl:col-span-3"
        aside={
          selected ? (
            <span className="flex items-center gap-2">
              {selected.is_synthetic ? <SyntheticBadge /> : null}
              <StatusBadge status={selected.status} />
            </span>
          ) : null
        }
      >
        {selected ? (
          <RunResults run={selected} />
        ) : (
          <p className="text-sm text-muted">Run a backtest to see results here.</p>
        )}
      </Panel>
    </div>
  );
}
