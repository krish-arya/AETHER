"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badges";
import { DataTable, QueryState, buttonClass, fieldClass, headingClass, labelClass } from "@/components/ui/data-table";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useGanEvaluation, useGenerateScenarios, useModels } from "@/hooks/use-platform";
import { formatNumber, formatPercent } from "@/lib/format";
import type { GanEvaluation, ModelCard } from "@/lib/platform-types";

const REGIMES = ["market_stress", "bear", "high_volatility", "sideways", "low_volatility", "bull"];
const unsigned = (value: number | null | undefined) => formatPercent(value).replace("+", "");

function StatusBadge({ card }: { card: ModelCard }) {
  if (card.status === "validated") return <Badge tone="ok">Passed quality gates</Badge>;
  if (card.status === "rejected") return <Badge tone="down">Failed quality gates</Badge>;
  return <Badge tone="muted">Not evaluated</Badge>;
}

function Evaluation({ report }: { report: GanEvaluation }) {
  const synthetic = report.synthetic_vs_heldout_real;
  const baseline = report.real_baseline_train_vs_heldout;
  const rows = [
    ["Return distribution (KS)", synthetic.return_distribution.ks, baseline.return_distribution.ks],
    ["Volatility distribution (KS)", synthetic.volatility_distribution.ks, baseline.volatility_distribution.ks],
    ["Return autocorrelation error", synthetic.autocorrelation.returns_error, baseline.autocorrelation.returns_error],
    ["|Return| autocorrelation error", synthetic.autocorrelation.abs_returns_error, baseline.autocorrelation.abs_returns_error],
    ["Cross-asset correlation error", synthetic.cross_correlation.mean_abs_error, baseline.cross_correlation.mean_abs_error],
    ["Max-drawdown distribution (KS)", synthetic.drawdown_distribution.ks, baseline.drawdown_distribution.ks],
  ] as const;
  const tails = synthetic.tail_risk;
  return (
    <div className="space-y-6">
      <div>
        <h3 className={headingClass}>Quality gates</h3>
        <div className="mt-2">
          <DataTable
            rows={Object.entries(report.gates)}
            rowKey={([name]) => name}
            columns={[
              { label: "Check", render: ([name]) => name.replaceAll("_", " ") },
              { label: "Value", align: "right", render: ([, gate]) => formatNumber(gate.value, 3) },
              {
                label: "Required",
                align: "right",
                render: ([, gate]) => `${gate.rule} ${formatNumber(gate.threshold, 2)}`,
              },
              {
                label: "Result",
                render: ([, gate]) =>
                  gate.passed ? <Badge tone="ok">pass</Badge> : <Badge tone="down">fail</Badge>,
              },
            ]}
          />
        </div>
      </div>

      <div>
        <h3 className={headingClass}>Synthetic vs unseen real windows</h3>
        <p className="mt-1 text-xs leading-5 text-muted">
          Distances to a real period the generator never saw (lower is closer). The baseline column
          is the same distance between two <em>real</em> periods — markets change, so read the
          synthetic score against it, not against zero.
        </p>
        <div className="mt-2">
          <DataTable
            rows={[...rows]}
            rowKey={([name]) => name}
            columns={[
              { label: "Measure", render: ([name]) => name },
              { label: "Synthetic", align: "right", render: ([, value]) => formatNumber(value, 3) },
              { label: "Real baseline", align: "right", render: ([, , value]) => formatNumber(value, 3) },
            ]}
          />
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Stat label="Daily std, real">{unsigned(synthetic.return_distribution.real_std)}</Stat>
        <Stat label="Daily std, synthetic">{unsigned(synthetic.return_distribution.other_std)}</Stat>
        <Stat label="5% CVaR, real">{formatPercent(tails.real.cvar05)}</Stat>
        <Stat label="5% CVaR, synthetic">{formatPercent(tails.other.cvar05)}</Stat>
        <Stat label="Excess kurtosis, real">{formatNumber(tails.real.excess_kurtosis, 1)}</Stat>
        <Stat label="Excess kurtosis, synthetic">{formatNumber(tails.other.excess_kurtosis, 1)}</Stat>
        <Stat label="Real-vs-synthetic classifier">
          {unsigned(0.5 + report.discriminative.score)} accuracy
        </Stat>
        <Stat label="Near-copies of training data">{unsigned(report.memorization.copied_fraction)}</Stat>
      </dl>
      <p className="text-xs leading-5 text-faint">
        A classifier accuracy near 50% means synthetic windows cannot be told from real ones; near
        100% means they are easy to tell apart. Memorisation ratio{" "}
        {formatNumber(report.memorization.ratio, 2)} (1.0 = synthetic windows are as far from the
        training data as unseen real windows are).
      </p>
    </div>
  );
}

function Generate() {
  const generate = useGenerateScenarios();
  const [regime, setRegime] = useState(REGIMES[0]);
  const set = generate.data;
  return (
    <div>
      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          generate.mutate({ regime, scenarios: 200 });
        }}
      >
        <label className={labelClass}>
          Regime to condition on
          <select className={fieldClass} value={regime} onChange={(e) => setRegime(e.target.value)}>
            {REGIMES.map((r) => (
              <option key={r} value={r}>
                {r.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" className={buttonClass} disabled={generate.isPending}>
          {generate.isPending ? "Generating…" : "Generate 200 scenarios"}
        </button>
      </form>
      {generate.error ? (
        <div className="mt-3">
          <QueryError error={generate.error} />
        </div>
      ) : null}
      {set ? (
        <div className="mt-5 space-y-4">
          <p className="rounded border border-synthetic/40 px-3 py-2 text-xs leading-5 text-synthetic">
            {set.label} Generator status: {set.generator_status}. {set.scenarios} scenarios of{" "}
            {set.horizon_sessions} sessions, conditioned on {set.regime.replace("_", " ")}.
          </p>
          <dl className="grid grid-cols-3 gap-4">
            <Stat label="Equal-weight mean">{formatPercent(set.equal_weight_total_return.mean)}</Stat>
            <Stat label="5th percentile">{formatPercent(set.equal_weight_total_return.p05)}</Stat>
            <Stat label="Worst scenario">{formatPercent(set.equal_weight_total_return.worst)}</Stat>
          </dl>
          <DataTable
            rows={set.assets}
            rowKey={(row) => row.symbol}
            columns={[
              { label: "Asset", render: (row) => <span className="num text-text">{row.symbol}</span> },
              { label: "Mean return", align: "right", render: (row) => formatPercent(row.mean_return) },
              { label: "5th percentile", align: "right", render: (row) => formatPercent(row.p05_return) },
              { label: "Volatility (ann.)", align: "right", render: (row) => unsigned(row.annualised_volatility) },
            ]}
          />
        </div>
      ) : null}
    </div>
  );
}

export function GanExplorer() {
  const models = useModels("gan");
  const [picked, setPicked] = useState<string | null>(null);
  const cards = models.data ?? [];
  const selected = cards.find((card) => card.model_id === picked) ?? cards[0] ?? null;
  const evaluation = useGanEvaluation(selected?.model_id ?? null);

  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <Panel title="Scenario generators" aside={<span className="num">{cards.length}</span>}>
        <QueryState isPending={models.isPending} error={models.error}>
          {cards.length === 0 ? (
            <p className="text-sm leading-6 text-muted">
              No generator has been trained yet. Run{" "}
              <span className="num">scripts/run_experiments.py</span> or POST /api/v1/gan/train.
            </p>
          ) : (
            <ul className="-mx-4 -my-4 divide-y divide-line/60">
              {cards.map((card) => (
                <li key={card.model_id}>
                  <button
                    type="button"
                    onClick={() => setPicked(card.model_id)}
                    className={`w-full cursor-pointer px-4 py-2.5 text-left ${card.model_id === selected?.model_id ? "bg-raised" : "hover:bg-raised/50"}`}
                  >
                    <span className="num block truncate text-sm text-text">{card.model_id}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-2">
                      <StatusBadge card={card} />
                      <span className="num text-[11px] text-faint">
                        trained to{" "}
                        {typeof card.training_window?.end === "string" ? card.training_window.end : "?"}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </QueryState>
      </Panel>

      <div className="space-y-4 xl:col-span-2">
        <Panel title="Statistical evaluation">
          {selected ? (
            <QueryState isPending={evaluation.isPending} error={evaluation.error}>
              {evaluation.data ? <Evaluation report={evaluation.data} /> : null}
            </QueryState>
          ) : (
            <p className="text-sm text-muted">Select a generator to see its evaluation.</p>
          )}
        </Panel>
        <Panel title="Generate scenarios" aside={<Badge tone="synthetic">Synthetic output</Badge>}>
          <Generate />
        </Panel>
      </div>
    </div>
  );
}
