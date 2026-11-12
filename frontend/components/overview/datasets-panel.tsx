"use client";

import { Badge, SyntheticBadge } from "@/components/ui/badges";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useDatasets } from "@/hooks/use-market";
import { formatInteger, formatPercent } from "@/lib/format";
import type { DatasetVersion } from "@/lib/types";

// Checks that are informational bookkeeping rather than data defects.
const HIDDEN_CHECKS = new Set(["feature_undefined"]);

function QualityChecks({ dataset }: { dataset: DatasetVersion }) {
  const checks = Object.entries(dataset.quality_summary.by_check).filter(
    ([check]) => !HIDDEN_CHECKS.has(check),
  );
  if (checks.length === 0) {
    return <p className="text-sm text-muted">No data-quality issues were recorded.</p>;
  }
  return (
    <ul className="space-y-1">
      {checks.map(([check, count]) => (
        <li key={check} className="flex justify-between gap-4 text-sm">
          <span className="num text-muted">{check}</span>
          <span className="num text-text">{formatInteger(count)}</span>
        </li>
      ))}
    </ul>
  );
}

function DatasetCard({ dataset }: { dataset: DatasetVersion }) {
  const quality = dataset.quality_summary;
  return (
    <article>
      <div className="flex flex-wrap items-center gap-2">
        <span className="num text-sm break-all text-text">{dataset.dataset_id}</span>
        {dataset.is_synthetic ? <SyntheticBadge /> : <Badge tone="ok">Real market data</Badge>}
      </div>
      <p className="mt-2 text-xs leading-5 text-muted">{dataset.license_note}</p>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Stat label="Provider">{dataset.provider}</Stat>
        <Stat label="Securities">{formatInteger(dataset.symbol_count)}</Stat>
        <Stat label="Daily bars">{formatInteger(dataset.bar_count)}</Stat>
        <Stat label="Feature set">{dataset.feature_set_version}</Stat>
        <Stat label="First bar">{dataset.start_date}</Stat>
        <Stat label="Last bar">{dataset.end_date}</Stat>
        <Stat label="As of">{dataset.as_of}</Stat>
        <Stat label="Rows quarantined">
          {formatInteger(quality.rows_quarantined)} (
          {formatPercent(quality.quarantine_fraction).replace("+", "")})
        </Stat>
      </dl>

      <h3 className="mt-5 mb-2 text-[11px] tracking-[0.14em] text-faint uppercase">
        Data-quality findings
      </h3>
      <QualityChecks dataset={dataset} />
    </article>
  );
}

export function DatasetsPanel({ className = "" }: { className?: string }) {
  const { data, error, isPending } = useDatasets();

  return (
    <Panel
      title="Loaded datasets"
      className={className}
      aside={data ? <span className="num">{data.length} loaded</span> : null}
    >
      {isPending ? (
        <p className="text-sm text-faint">Loading…</p>
      ) : error ? (
        <QueryError error={error} />
      ) : data.length === 0 ? (
        <div className="text-sm text-muted">
          <p>No dataset has been loaded into the database yet.</p>
          <pre className="num mt-3 overflow-x-auto rounded border border-line bg-raised p-3 text-xs leading-5">
            {"uv run python scripts/build_dataset.py\nuv run python scripts/load_dataset.py --latest"}
          </pre>
        </div>
      ) : (
        <div className="space-y-6 divide-y divide-line [&>article+article]:pt-6">
          {data.map((dataset) => (
            <DatasetCard key={dataset.dataset_id} dataset={dataset} />
          ))}
        </div>
      )}
    </Panel>
  );
}
