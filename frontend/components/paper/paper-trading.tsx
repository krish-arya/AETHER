"use client";

import { useState, type FormEvent } from "react";

import { Badge, SyntheticBadge, type Tone } from "@/components/ui/badges";
import { DataTable, QueryState, buttonClass, fieldClass, headingClass, labelClass } from "@/components/ui/data-table";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import {
  useInitializePaper,
  useModels,
  usePaperOrders,
  usePaperPerformance,
  usePaperPositions,
  usePlaceOrder,
} from "@/hooks/use-platform";
import { useStocks } from "@/hooks/use-market";
import { MISSING, formatInteger, formatNumber, formatPercent, formatPrice } from "@/lib/format";

const unsigned = (value: number | null | undefined) => formatPercent(value).replace("+", "");
const STRATEGIES = [
  { value: "baseline:equal_weight", label: "Equal weight (baseline)" },
  { value: "baseline:momentum", label: "Momentum (baseline)" },
  { value: "baseline:moving_average", label: "Moving-average trend (baseline)" },
  { value: "baseline:mean_variance", label: "Mean-variance (baseline)" },
  { value: "manual:", label: "Manual orders" },
];
const ORDER_TONE: Record<string, Tone> = { executed: "ok", pending: "warn", rejected: "down" };

function NewAccount() {
  const initialize = useInitializePaper();
  const models = useModels("rl");
  const deployed = (models.data ?? []).filter((m) => m.lineage.experiment === "deployment");
  const [strategy, setStrategy] = useState(STRATEGIES[0].value);
  const [capital, setCapital] = useState("1000000");

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const [kind, ref] = strategy.split(":");
    initialize.mutate({
      name: "paper",
      initial_capital: Number(capital),
      strategy: ref ? { kind, ref } : { kind },
      rebalance_every: 5,
    });
  }

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <label className={labelClass}>
        Decided by
        <select className={fieldClass} value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          {STRATEGIES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
          {deployed.map((m) => (
            <option key={m.model_id} value={`rl:${m.model_id}`}>
              RL model {m.model_id.slice(-10)}
            </option>
          ))}
        </select>
      </label>
      <label className={labelClass}>
        Virtual capital (₹)
        <input
          className={fieldClass}
          type="number"
          min={1000}
          step={1000}
          value={capital}
          onChange={(e) => setCapital(e.target.value)}
          required
        />
      </label>
      <button type="submit" className={`${buttonClass} w-full`} disabled={initialize.isPending}>
        {initialize.isPending ? "Creating…" : "Start a new paper account"}
      </button>
      {initialize.error ? <QueryError error={initialize.error} /> : null}
      <p className="text-xs leading-5 text-faint">
        Replaces the current account. The account starts about six months back and is replayed to
        the latest session in the data; it advances when newer data is loaded. Virtual money only —
        no order is ever sent to a broker.
      </p>
    </form>
  );
}

function OrderForm({ symbols }: { symbols: string[] }) {
  const place = usePlaceOrder();
  const [symbol, setSymbol] = useState("");
  const [weight, setWeight] = useState("10");
  return (
    <form
      className="flex flex-wrap items-end gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        place.mutate({ symbol, target_weight: Number(weight) / 100 });
      }}
    >
      <label className={labelClass}>
        Symbol
        <input
          className={fieldClass}
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          list="paper-symbols"
          required
        />
        <datalist id="paper-symbols">
          {symbols.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      </label>
      <label className={labelClass}>
        Target weight (%)
        <input
          className={fieldClass}
          type="number"
          min={0}
          max={100}
          step={0.5}
          value={weight}
          onChange={(e) => setWeight(e.target.value)}
          required
        />
      </label>
      <button type="submit" className={buttonClass} disabled={place.isPending}>
        Place simulated order
      </button>
      {place.data ? (
        <span className="text-xs text-muted">
          {place.data.status}: {place.data.detail}
        </span>
      ) : null}
      {place.error ? <QueryError error={place.error} /> : null}
    </form>
  );
}

export function PaperTrading() {
  const positions = usePaperPositions();
  const active = positions.isSuccess;
  const performance = usePaperPerformance(active);
  const orders = usePaperOrders(active);
  const stocks = useStocks();
  const account = positions.data;
  const metrics = performance.data?.metrics;

  return (
    <div className="grid gap-4 xl:grid-cols-4">
      <div className="space-y-4">
        <Panel title="Paper account">
          <NewAccount />
        </Panel>
      </div>

      <div className="min-w-0 space-y-4 xl:col-span-3">
        <Panel
          title="Simulated portfolio"
          aside={
            account ? (
              <span className="flex items-center gap-2">
                {account.is_synthetic ? <SyntheticBadge /> : null}
                <Badge tone="muted">Simulated · no broker</Badge>
              </span>
            ) : null
          }
        >
          <QueryState isPending={positions.isPending} error={positions.error}>
            {account ? (
              <>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
                  <Stat label="Value (₹)">{formatInteger(account.value)}</Stat>
                  <Stat label="Virtual cash (₹)">{formatInteger(account.cash)}</Stat>
                  <Stat label="Daily return">{formatPercent(account.daily_return)}</Stat>
                  <Stat label="Since start">{formatPercent(account.cumulative_return)}</Stat>
                  <Stat label="Sharpe">{formatNumber(metrics?.sharpe ?? null, 2)}</Stat>
                  <Stat label="Max drawdown">{formatPercent(metrics?.max_drawdown ?? null)}</Stat>
                  <Stat label="Fees & taxes (₹)">{formatInteger(metrics?.total_costs ?? null)}</Stat>
                  <Stat label="Slippage (₹)">{formatInteger(performance.data?.slippage_cost ?? null)}</Stat>
                </dl>
                <p className="mt-3 text-xs text-faint">
                  {account.strategy.kind}
                  {account.strategy.ref ? ` · ${account.strategy.ref}` : ""} · {account.start_date} →{" "}
                  {account.as_of} · started with ₹{formatInteger(account.initial_capital)}
                </p>
                <h3 className={`${headingClass} mt-5 mb-2`}>Positions</h3>
                <DataTable
                  rows={account.holdings}
                  rowKey={(row) => row.symbol}
                  empty="No positions: the account is entirely in cash."
                  columns={[
                    { label: "Asset", render: (row) => <span className="num text-text">{row.symbol}</span> },
                    { label: "Sector", render: (row) => row.sector ?? MISSING },
                    { label: "Weight", align: "right", render: (row) => unsigned(row.weight) },
                    { label: "Value (₹)", align: "right", render: (row) => formatInteger(row.value) },
                    { label: "Last close", align: "right", render: (row) => formatPrice(row.last_close) },
                  ]}
                />
                {account.strategy.kind === "manual" ? (
                  <div className="mt-5">
                    <h3 className={`${headingClass} mb-2`}>New order</h3>
                    <OrderForm symbols={(stocks.data ?? []).map((stock) => stock.symbol)} />
                  </div>
                ) : null}
              </>
            ) : null}
          </QueryState>
        </Panel>

        {active ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Orders and executed trades">
              <QueryState isPending={orders.isPending} error={orders.error}>
                {orders.data?.manual_orders.length ? (
                  <ul className="mb-4 space-y-1.5">
                    {orders.data.manual_orders.map((order) => (
                      <li key={order.order_id} className="flex flex-wrap items-center gap-2 text-xs">
                        <Badge tone={ORDER_TONE[order.status] ?? "muted"}>{order.status}</Badge>
                        <span className="num text-text">
                          {order.symbol} → {unsigned(order.target_weight)} · decided {order.decision_date}
                        </span>
                        <span className="text-faint">{order.detail}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
                <DataTable
                  rows={[...(orders.data?.executed_trades ?? [])].reverse().slice(0, 12)}
                  rowKey={(row, index) => `${row.trade_date}-${row.symbol}-${index}`}
                  empty="No trades yet."
                  columns={[
                    { label: "Date", render: (row) => <span className="num whitespace-nowrap">{row.trade_date}</span> },
                    { label: "Asset", render: (row) => <span className="num text-text">{row.symbol}</span> },
                    { label: "Side", render: (row) => row.side },
                    { label: "Value (₹)", align: "right", render: (row) => formatInteger(row.value) },
                    { label: "Costs (₹)", align: "right", render: (row) => formatInteger(row.total_cost) },
                  ]}
                />
              </QueryState>
            </Panel>
            <Panel title="Rejected or limited · daily P&L">
              <QueryState isPending={orders.isPending} error={orders.error}>
                <DataTable
                  rows={[...(orders.data?.rejected_or_limited ?? [])].reverse().slice(0, 6)}
                  rowKey={(row, index) => `${row.trade_date}-${row.event}-${index}`}
                  empty="No order was rejected or limited."
                  columns={[
                    { label: "Date", render: (row) => <span className="num whitespace-nowrap">{row.trade_date}</span> },
                    { label: "Asset", render: (row) => row.symbol ?? MISSING },
                    { label: "What happened", render: (row) => `${row.event.replaceAll("_", " ")} — ${row.detail}` },
                  ]}
                />
              </QueryState>
              <h3 className={`${headingClass} mt-5 mb-2`}>Last sessions</h3>
              <DataTable
                rows={[...(performance.data?.daily_pnl ?? [])].reverse().slice(0, 8)}
                rowKey={(row) => row.date}
                empty="No history yet."
                columns={[
                  { label: "Date", render: (row) => <span className="num whitespace-nowrap">{row.date}</span> },
                  { label: "P&L (₹)", align: "right", render: (row) => formatInteger(row.pnl) },
                  { label: "Return", align: "right", render: (row) => formatPercent(row.return) },
                ]}
              />
            </Panel>
          </div>
        ) : null}
      </div>
    </div>
  );
}
