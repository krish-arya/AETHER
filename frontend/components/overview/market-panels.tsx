"use client";

import { Badge, SyntheticBadge, type Tone } from "@/components/ui/badges";
import { DataTable, QueryState } from "@/components/ui/data-table";
import { Panel, Stat } from "@/components/ui/panel";
import { useAlerts, usePaperPositions, useRegime, useRiskMetrics } from "@/hooks/use-platform";
import { MISSING, formatInteger, formatNumber, formatPercent } from "@/lib/format";
import type { RegimePoint } from "@/lib/platform-types";

const unsigned = (value: number | null | undefined) => formatPercent(value).replace("+", "");

const REGIME_TONE: Record<string, Tone> = {
  bull: "ok",
  low_volatility: "ok",
  sideways: "muted",
  high_volatility: "warn",
  bear: "down",
  market_stress: "down",
};

export function regimeLabel(regime: string | null): string {
  return regime ? regime.replace("_", " ") : "undefined";
}

function changes(history: RegimePoint[]): { date: string; from: string; to: string }[] {
  const out = [];
  for (let i = 1; i < history.length; i++) {
    if (history[i].regime !== history[i - 1].regime) {
      out.push({
        date: history[i].timestamp,
        from: regimeLabel(history[i - 1].regime),
        to: regimeLabel(history[i].regime),
      });
    }
  }
  return out.reverse().slice(0, 6);
}

export function RegimePanel({ className = "" }: { className?: string }) {
  const { data, error, isPending } = useRegime();
  const current = data?.current;
  return (
    <Panel
      title="Market regime"
      className={className}
      aside={data?.is_synthetic ? <SyntheticBadge /> : null}
    >
      <QueryState isPending={isPending} error={error}>
        {current ? (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <Badge tone={REGIME_TONE[current.regime ?? ""] ?? "muted"}>
                {regimeLabel(current.regime)}
              </Badge>
              <span className="num text-xs text-faint">as of {current.timestamp}</span>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
              <Stat label="Volatility percentile">{unsigned(current.volatility_level)}</Stat>
              <Stat label="Realised vol (ann.)">{unsigned(current.realized_vol)}</Stat>
              <Stat label="Trend strength">{formatNumber(current.trend_strength, 2)}</Stat>
              <Stat label="63-session return">{formatPercent(current.trend_return)}</Stat>
              <Stat label="Drawdown from peak">{formatPercent(current.drawdown)}</Stat>
              <Stat label="Rule margin">{formatNumber(current.confidence, 2)}</Stat>
            </dl>
            <h3 className="mt-5 mb-2 text-[11px] tracking-[0.14em] text-faint uppercase">
              Recent regime changes
            </h3>
            <DataTable
              rows={changes(data.history)}
              rowKey={(row) => row.date}
              empty="No regime change in the last 250 sessions."
              columns={[
                { label: "Date", render: (row) => <span className="num whitespace-nowrap">{row.date}</span> },
                { label: "From", render: (row) => row.from },
                { label: "To", render: (row) => <span className="text-text">{row.to}</span> },
              ]}
            />
            <p className="mt-4 text-xs leading-5 text-faint">
              Rule-based classification of the equal-weight universe return, using data up to each
              date only. &ldquo;Rule margin&rdquo; is the distance from the nearest rule boundary,
              not a probability.
            </p>
          </>
        ) : null}
      </QueryState>
    </Panel>
  );
}

export function PortfolioPanel() {
  const positions = usePaperPositions();
  const risk = useRiskMetrics(positions.isSuccess);
  const account = positions.data;
  return (
    <Panel
      title="Paper portfolio"
      aside={account?.is_synthetic ? <SyntheticBadge /> : account ? <Badge tone="muted">Simulated</Badge> : null}
    >
      <QueryState isPending={positions.isPending} error={positions.error}>
        {account ? (
          <>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
              <Stat label="Value (₹)">{formatInteger(account.value)}</Stat>
              <Stat label="Daily return">{formatPercent(account.daily_return)}</Stat>
              <Stat label="Cumulative return">{formatPercent(account.cumulative_return)}</Stat>
              <Stat label="Current drawdown">{formatPercent(account.drawdown)}</Stat>
              <Stat label="Cash (₹)">{formatInteger(account.cash)}</Stat>
              <Stat label="Positions">{account.positions}</Stat>
              <Stat label="95% 1-day VaR">{risk.data ? unsigned(risk.data.var) : MISSING}</Stat>
              <Stat label="Expected shortfall">{risk.data ? unsigned(risk.data.cvar) : MISSING}</Stat>
            </dl>
            <p className="mt-4 text-xs leading-5 text-faint">
              {account.name} · {account.strategy.kind}
              {account.strategy.ref ? ` (${account.strategy.ref})` : ""} · since {account.start_date}{" "}
              · as of {account.as_of}. {account.note}
            </p>
          </>
        ) : null}
      </QueryState>
    </Panel>
  );
}

const ALERT_TONE: Record<string, Tone> = { critical: "down", warning: "warn", info: "muted" };

/** Operational alerts: stale data, synthetic data in use, models outside validation. */
export function AlertsPanel({ className = "" }: { className?: string }) {
  const { data, error, isPending } = useAlerts();
  return (
    <Panel
      title="Alerts"
      className={className}
      aside={data ? <span className="num">{data.alerts.length}</span> : null}
    >
      <QueryState isPending={isPending} error={error}>
        {data?.alerts.length ? (
          <ul className="space-y-2">
            {data.alerts.map((alert) => (
              <li key={alert.code} className="flex items-start gap-3 text-sm">
                <span className="mt-0.5 shrink-0">
                  <Badge tone={ALERT_TONE[alert.severity]}>{alert.severity}</Badge>
                </span>
                <span className="leading-6 text-muted">{alert.message}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted">No alerts.</p>
        )}
      </QueryState>
    </Panel>
  );
}
