"use client";

import { Badge, SyntheticBadge } from "@/components/ui/badges";
import { Caveats, DataTable, QueryState, buttonClass, headingClass } from "@/components/ui/data-table";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useAllocate, useModels } from "@/hooks/use-platform";
import { MISSING, formatNumber, formatPercent } from "@/lib/format";
import type { AllocationRecord } from "@/lib/platform-types";

const unsigned = (value: number | null | undefined) => formatPercent(value).replace("+", "");

function Explanation({ record }: { record: AllocationRecord }) {
  const risk = record.evidence.risk_engine;
  const sources = Object.entries(record.sensitivity.by_information_source);
  return (
    <div className="space-y-6">
      <Caveats
        items={[
          "Research output for a simulated portfolio. Not investment advice; nothing is executed.",
          ...(record.is_synthetic ? ["Computed on SYNTHETIC development data."] : []),
        ]}
      />

      <div>
        <h3 className={headingClass}>Decision summary · as of {record.as_of}</h3>
        <ul className="mt-2 space-y-1.5 text-sm leading-6 text-text">
          {record.summary.narrative.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Stat label="Invested">{unsigned(record.summary.invested)}</Stat>
          <Stat label="Cash">{unsigned(record.summary.cash)}</Stat>
          <Stat label="Positions">{record.summary.positions}</Stat>
          <Stat label="Turnover needed">{unsigned(record.summary.turnover)}</Stat>
        </dl>
      </div>

      <div>
        <h3 className={headingClass}>Target allocation and evidence</h3>
        <div className="mt-2">
          <DataTable
            rows={record.evidence.positions}
            rowKey={(row) => row.symbol}
            columns={[
              { label: "Asset", render: (row) => <span className="num text-text">{row.symbol}</span> },
              { label: "Sector", render: (row) => row.sector ?? MISSING },
              { label: "Target", align: "right", render: (row) => unsigned(row.weight) },
              { label: "Current", align: "right", render: (row) => unsigned(row.previous_weight) },
              { label: "Change", align: "right", render: (row) => formatPercent(row.change) },
              {
                label: "Model, before risk",
                align: "right",
                render: (row) => unsigned(row.policy_weight_before_risk),
              },
              {
                label: "Most influential inputs (value)",
                render: (row) =>
                  row.most_influential_features
                    .map((f) => `${f.feature} ${formatNumber(f.value, 3)}`)
                    .join(" · ") || MISSING,
              },
            ]}
          />
        </div>
        <p className="mt-2 text-xs text-faint">
          Showing the largest positions; the full allocation holds {record.summary.positions}.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className={headingClass}>Risk engine</h3>
          <p className="mt-2 text-sm leading-6 text-muted">
            {risk.rejected
              ? `Rejected: ${risk.reason}. Current holdings are kept.`
              : risk.applied.length
                ? `Modified the model's proposal to satisfy: ${[...new Set(risk.applied)].join(", ").replaceAll("_", " ")}.`
                : "The model's proposal satisfied every limit; nothing was changed."}
          </p>
          {risk.raw_violations.length ? (
            <ul className="mt-2 space-y-1 text-xs text-muted">
              {risk.raw_violations.slice(0, 6).map((v) => (
                <li key={`${v.limit}-${v.subject}`} className="num">
                  {v.limit.replaceAll("_", " ")}
                  {v.subject ? ` (${v.subject})` : ""}: {formatNumber(v.value, 3)} vs limit{" "}
                  {formatNumber(v.threshold, 3)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <div>
          <h3 className={headingClass}>Sensitivity to each information source</h3>
          <ul className="mt-2 space-y-1">
            {sources.map(([source, value]) => (
              <li key={source} className="flex justify-between gap-4 text-sm">
                <span className="text-muted">{source}</span>
                <span className="num text-text">{unsigned(value)} of the portfolio</span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs leading-5 text-faint">
            {record.sensitivity.definition} {record.sensitivity.note}
          </p>
        </div>
      </div>

      <div>
        <h3 className={headingClass}>Model</h3>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Stat label="Model id">{record.model_id}</Stat>
          <Stat label="Trained">{record.model.trained_at.slice(0, 10)}</Stat>
          <Stat label="Validation Sharpe">{formatNumber(record.model.validation_sharpe, 2)}</Stat>
          <Stat label="Executable from">{record.executable_from ?? "next session (no data yet)"}</Stat>
        </dl>
        {record.model.notes ? (
          <p className="mt-2 text-xs leading-5 text-faint">{record.model.notes}</p>
        ) : null}
      </div>
    </div>
  );
}

export function Optimizer() {
  const models = useModels("rl");
  const allocate = useAllocate();
  const cards = models.data ?? [];
  const record = allocate.data;

  return (
    <div className="grid gap-4 xl:grid-cols-4">
      <Panel title="Allocation" className="min-w-0 xl:col-span-3" aside={record?.is_synthetic ? <SyntheticBadge /> : null}>
        <div className="mb-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            className={buttonClass}
            disabled={allocate.isPending || cards.length === 0}
            onClick={() => allocate.mutate({})}
          >
            {allocate.isPending ? "Computing…" : "Compute target allocation"}
          </button>
          <span className="text-xs text-faint">
            Uses the newest deployment model, the paper portfolio as the starting point, and the
            latest session in the data.
          </span>
        </div>
        {allocate.error ? <QueryError error={allocate.error} /> : null}
        {record ? (
          <Explanation record={record} />
        ) : (
          <p className="text-sm leading-6 text-muted">
            {cards.length === 0
              ? "No RL model is registered yet. Train one (POST /api/v1/rl/train, or scripts/run_experiments.py) and it will appear here."
              : "No allocation has been computed in this session."}
          </p>
        )}
      </Panel>

      <Panel title="RL models" aside={<span className="num">{cards.length}</span>}>
        <QueryState isPending={models.isPending} error={models.error}>
          <ul className="space-y-3">
            {cards.slice(0, 12).map((card) => (
              <li key={card.model_id} className="text-xs">
                <span className="num block truncate text-text">{card.model_id}</span>
                <span className="mt-1 flex flex-wrap items-center gap-2 text-faint">
                  <Badge tone={card.lineage.experiment === "deployment" ? "ok" : "muted"}>
                    {card.lineage.experiment ?? "model"}
                  </Badge>
                  <span className="num">
                    val. Sharpe {formatNumber(card.metrics.validation_sharpe ?? null, 2)} · seed{" "}
                    {card.seed ?? MISSING}
                  </span>
                </span>
              </li>
            ))}
          </ul>
          {cards.length > 12 ? (
            <p className="mt-3 text-xs text-faint">+ {cards.length - 12} more in the registry.</p>
          ) : null}
        </QueryState>
      </Panel>
    </div>
  );
}
