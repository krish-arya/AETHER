"use client";

import { Badge, type Tone } from "@/components/ui/badges";
import { Caveats, DataTable, QueryState, buttonClass, headingClass } from "@/components/ui/data-table";
import { Panel } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useExperiments, useJobs, useModels, useSubmitJob } from "@/hooks/use-platform";
import { MISSING, formatNumber, formatPercent } from "@/lib/format";
import type { ExperimentEntry, ExperimentReport } from "@/lib/platform-types";

const unsigned = (value: number | null | undefined) => formatPercent(value).replace("+", "");
const JOB_TONE: Record<string, Tone> = { queued: "muted", running: "warn", succeeded: "ok", failed: "down" };

function metric(entry: ExperimentEntry, name: string, percent: boolean): string {
  const mean = entry.mean?.[name];
  const std = entry.std?.[name];
  if (mean == null) return MISSING;
  const fmt = (v: number) => (percent ? unsigned(v) : formatNumber(v, 2));
  const multi = (entry.seeds?.length ?? 1) > 1 && std != null;
  return multi ? `${fmt(mean)} ± ${fmt(std)}` : fmt(mean);
}

function Report({ report }: { report: ExperimentReport }) {
  const first = report.folds[0];
  const last = report.folds[report.folds.length - 1];
  return (
    <div className="space-y-5">
      <Caveats items={report.caveats} />
      <p className="num text-xs text-faint">
        {report.run_id} · {report.dataset.dataset_id} · out-of-sample {first?.test_start} →{" "}
        {last?.test_end} ({report.folds.length} folds) · seeds {report.seeds.join(", ")} ·{" "}
        {formatNumber(report.duration_seconds / 60, 0)} min
      </p>
      <DataTable
        rows={Object.entries(report.experiments)}
        rowKey={([key]) => key}
        columns={[
          { label: "Id", render: ([key]) => <span className="num text-text">{key}</span> },
          {
            label: "Configuration",
            render: ([, entry]) =>
              entry.skipped ? `${entry.label} — skipped: ${entry.skipped}` : entry.label,
          },
          { label: "CAGR", align: "right", render: ([, e]) => (e.skipped ? MISSING : metric(e, "cagr", true)) },
          { label: "Sharpe", align: "right", render: ([, e]) => (e.skipped ? MISSING : metric(e, "sharpe", false)) },
          {
            label: "Sharpe range",
            align: "right",
            render: ([, e]) =>
              e.min?.sharpe != null && (e.seeds?.length ?? 1) > 1
                ? `${formatNumber(e.min.sharpe, 2)} … ${formatNumber(e.max?.sharpe ?? null, 2)}`
                : MISSING,
          },
          { label: "Max DD", align: "right", render: ([, e]) => (e.skipped ? MISSING : metric(e, "max_drawdown", true)) },
          {
            label: "Sharpe vs E0 (95% CI)",
            align: "right",
            render: ([, e]) => {
              const test = e.sharpe_difference_vs_E0;
              if (!test) return MISSING;
              const ci = `${test.estimate >= 0 ? "+" : ""}${formatNumber(test.estimate, 2)} [${formatNumber(test.ci_low, 2)}, ${formatNumber(test.ci_high, 2)}]`;
              return test.significant ? ci : `${ci} n.s.`;
            },
          },
        ]}
      />
      <p className="text-xs leading-5 text-faint">
        ± and range are across training seeds — every seed is reported, not the best one. n.s. =
        the difference from the equal-weight benchmark is not statistically distinguishable from
        zero. All configurations trade on the same dates, pay the same costs and pass through the
        same risk limits.
      </p>
      {report.generator.length ? (
        <div>
          <h3 className={headingClass}>Scenario generator per fold</h3>
          <ul className="mt-2 space-y-1 text-xs text-muted">
            {report.generator.map((row) => (
              <li key={row.fold} className="num">
                fold {row.fold}: {row.passes ? "passed" : "FAILED"} quality gates —{" "}
                {Object.entries(row.gates)
                  .map(([name, gate]) => `${name.replaceAll("_", " ")} ${formatNumber(gate.value, 2)}${gate.passed ? "" : " ✗"}`)
                  .join(" · ")}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export function ExperimentsView() {
  const experiments = useExperiments();
  const rl = useModels("rl");
  const gan = useModels("gan");
  const jobs = useJobs();
  const train = useSubmitJob("/rl/train");
  const cards = [...(rl.data ?? []), ...(gan.data ?? [])].sort((a, b) =>
    b.trained_at.localeCompare(a.trained_at),
  );
  const latest = experiments.data?.[0];

  return (
    <div className="space-y-4">
      <Panel
        title="Research experiments · latest run"
        aside={experiments.data ? <span className="num">{experiments.data.length} runs</span> : null}
      >
        <QueryState isPending={experiments.isPending} error={experiments.error}>
          {latest ? (
            <Report report={latest} />
          ) : (
            <p className="text-sm leading-6 text-muted">
              No experiment has been run yet:{" "}
              <span className="num">uv run python scripts/run_experiments.py --latest</span>
            </p>
          )}
        </QueryState>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-3">
        <Panel
          title="Model registry"
          className="min-w-0 xl:col-span-2"
          aside={<span className="num">{cards.length} models</span>}
        >
          <QueryState isPending={rl.isPending || gan.isPending} error={rl.error ?? gan.error}>
            <DataTable
              rows={cards.slice(0, 40)}
              rowKey={(card) => card.model_id}
              empty="No model has been trained yet."
              columns={[
                { label: "Model", render: (c) => <span className="num text-text">{c.model_id}</span> },
                { label: "Kind", render: (c) => c.kind },
                { label: "Experiment", render: (c) => c.lineage.experiment ?? MISSING },
                {
                  label: "Status",
                  render: (c) => (
                    <Badge tone={c.status === "validated" ? "ok" : c.status === "rejected" ? "down" : "muted"}>
                      {c.status}
                    </Badge>
                  ),
                },
                { label: "Seed", align: "right", render: (c) => c.seed ?? MISSING },
                {
                  label: "Val. Sharpe",
                  align: "right",
                  render: (c) => formatNumber(c.metrics.validation_sharpe ?? null, 2),
                },
                { label: "Trained", render: (c) => <span className="num">{c.trained_at.slice(0, 16).replace("T", " ")}</span> },
              ]}
            />
          </QueryState>
        </Panel>

        <Panel title="Model jobs">
          <button
            type="button"
            className={buttonClass}
            disabled={train.isPending}
            onClick={() => train.mutate({})}
          >
            {train.isPending ? "Submitting…" : "Train deployment model"}
          </button>
          {train.error ? (
            <div className="mt-3">
              <QueryError error={train.error} />
            </div>
          ) : null}
          <p className="mt-2 mb-4 text-xs leading-5 text-faint">
            Trains the allocation model on the most recent data in a background worker (several
            minutes on CPU).
          </p>
          <QueryState isPending={jobs.isPending} error={jobs.error}>
            <ul className="space-y-2">
              {(jobs.data ?? []).map((job) => (
                <li key={job.job_id} className="text-xs">
                  <span className="flex items-center justify-between gap-2">
                    <span className="num text-text">{job.kind.replace("_", " ")}</span>
                    <Badge tone={JOB_TONE[job.status]}>{job.status}</Badge>
                  </span>
                  <span className="num block truncate text-faint">
                    {job.created_at.slice(0, 16).replace("T", " ")} ·{" "}
                    {job.error ?? (job.result ? JSON.stringify(job.result).slice(0, 80) : "…")}
                  </span>
                </li>
              ))}
              {jobs.data?.length === 0 ? <li className="text-sm text-muted">No jobs yet.</li> : null}
            </ul>
          </QueryState>
        </Panel>
      </div>
    </div>
  );
}
