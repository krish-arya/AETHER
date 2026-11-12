"use client";

import { useState } from "react";

import { Badge, SyntheticBadge } from "@/components/ui/badges";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useLatestIndicators, useStockDetail, useStocks } from "@/hooks/use-market";
import { MISSING, formatCrore, formatInteger, formatNumber, formatPercent, formatPrice } from "@/lib/format";
import type { Security } from "@/lib/types";

type FeatureSpec = { key: string; label: string; format: (v: number | null) => string };

const pct = (v: number | null) => formatPercent(v);
// Magnitudes (volatility, ranges) are never negative, so they carry no sign.
const magnitude = (v: number | null) => formatPercent(v).replace("+", "");
const level = (v: number | null) => formatNumber(v, 1);

function listingWindow(stock: Security): string {
  if (stock.listing_date && stock.delisting_date) {
    return `${stock.listing_date} – ${stock.delisting_date}`;
  }
  if (stock.listing_date) return `from ${stock.listing_date}`;
  if (stock.delisting_date) return `until ${stock.delisting_date}`;
  return "full history";
}
const ratio = (v: number | null) => formatNumber(v, 2);

// Default feature-set column names (configs/data.yaml). A column missing from the
// loaded feature set is simply not shown.
const FEATURES: { group: string; items: FeatureSpec[] }[] = [
  {
    group: "Returns",
    items: [
      { key: "ret_1d", label: "1-day return", format: pct },
      { key: "cum_ret_21d", label: "21-day return", format: pct },
      { key: "cum_ret_63d", label: "63-day return", format: pct },
    ],
  },
  {
    group: "Trend",
    items: [
      { key: "sma_50_ratio", label: "Close vs SMA 50", format: pct },
      { key: "macd_hist_pct", label: "MACD histogram / close", format: pct },
      { key: "adx_14", label: "ADX 14", format: level },
    ],
  },
  {
    group: "Momentum",
    items: [
      { key: "rsi_14", label: "RSI 14", format: level },
      { key: "stoch_k_14", label: "Stochastic %K 14", format: level },
      { key: "williams_r_14", label: "Williams %R 14", format: level },
    ],
  },
  {
    group: "Volatility",
    items: [
      { key: "hist_vol_20d", label: "Hist. volatility 20d (ann.)", format: magnitude },
      { key: "atr_pct_14", label: "ATR 14 / close", format: magnitude },
      { key: "bb_pctb_20", label: "Bollinger %B 20", format: ratio },
    ],
  },
  {
    group: "Volume & market",
    items: [
      { key: "mfi_14", label: "Money Flow Index 14", format: level },
      { key: "obv_norm_20", label: "Net signed volume 20d", format: ratio },
      { key: "beta_60d", label: "Beta 60d (vs universe)", format: ratio },
    ],
  },
];

function StockTable({
  stocks,
  selected,
  onSelect,
}: {
  stocks: Security[];
  selected: string | null;
  onSelect: (symbol: string) => void;
}) {
  return (
    <div className="-mx-4 -my-4 overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-line text-[11px] tracking-wide text-faint uppercase">
            <th className="px-4 py-2 font-normal">Symbol</th>
            <th className="px-4 py-2 font-normal">Name</th>
            <th className="px-4 py-2 font-normal">Sector</th>
            <th className="px-4 py-2 font-normal whitespace-nowrap">Listed</th>
          </tr>
        </thead>
        <tbody>
          {stocks.map((stock) => {
            const isSelected = stock.symbol === selected;
            return (
              <tr
                key={stock.symbol}
                className={`border-b border-line/60 last:border-0 ${isSelected ? "bg-raised" : "hover:bg-raised/50"}`}
              >
                <td className="px-4 py-2">
                  <button
                    type="button"
                    onClick={() => onSelect(stock.symbol)}
                    aria-pressed={isSelected}
                    className={`num cursor-pointer text-left ${isSelected ? "text-accent" : "text-text hover:text-accent"}`}
                  >
                    {stock.symbol}
                  </button>
                </td>
                <td className="px-4 py-2 text-muted">{stock.name}</td>
                <td className="px-4 py-2 text-muted">{stock.sector ?? MISSING}</td>
                <td className="num px-4 py-2 text-xs whitespace-nowrap text-muted">
                  {listingWindow(stock)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function StockDetail({ symbol }: { symbol: string }) {
  const detail = useStockDetail(symbol);
  const indicators = useLatestIndicators(symbol);

  if (detail.isPending) return <p className="text-sm text-faint">Loading {symbol}…</p>;
  if (detail.error) return <QueryError error={detail.error} />;

  const stock = detail.data;
  const bar = stock.latest_bar;
  const latest = indicators.data?.rows.at(-1);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="num text-lg text-text">{stock.symbol}</span>
        <span className="text-sm text-muted">{stock.name}</span>
        {stock.is_synthetic ? <SyntheticBadge /> : null}
        {stock.delisting_date ? <Badge tone="warn">Delisted {stock.delisting_date}</Badge> : null}
      </div>

      {bar ? (
        <>
          <p className="mt-4 text-[11px] tracking-[0.14em] text-faint uppercase">
            Latest bar · {bar.trade_date}
          </p>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Stat label="Open">{formatPrice(bar.open)}</Stat>
            <Stat label="High">{formatPrice(bar.high)}</Stat>
            <Stat label="Low">{formatPrice(bar.low)}</Stat>
            <Stat label="Close">{formatPrice(bar.close)}</Stat>
            <Stat label="Volume">{formatInteger(bar.volume)}</Stat>
            <Stat label="Delivery volume">{formatInteger(bar.delivery_volume)}</Stat>
            <Stat label="Market cap">{formatCrore(bar.market_cap)}</Stat>
            <Stat label="Bars in history">{formatInteger(stock.bar_count)}</Stat>
          </dl>
        </>
      ) : (
        <p className="mt-4 text-sm text-muted">No bars loaded for this security.</p>
      )}

      <p className="mt-6 text-[11px] tracking-[0.14em] text-faint uppercase">
        Technical features{latest ? ` · ${latest.trade_date}` : ""}
        {indicators.data ? ` · set ${indicators.data.feature_set_version}` : ""}
      </p>
      {indicators.isPending ? (
        <p className="mt-2 text-sm text-faint">Loading features…</p>
      ) : indicators.error || !latest ? (
        <p className="mt-2 text-sm text-muted">No features loaded for this security.</p>
      ) : (
        <div className="mt-2 grid gap-x-8 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
          {FEATURES.map(({ group, items }) => {
            const present = items.filter((item) => item.key in latest.features);
            if (present.length === 0) return null;
            return (
              <div key={group}>
                <p className="mb-1 text-xs text-muted">{group}</p>
                <ul className="space-y-1">
                  {present.map((item) => (
                    <li key={item.key} className="flex justify-between gap-4 text-sm">
                      <span className="text-muted">{item.label}</span>
                      <span className="num text-text">{item.format(latest.features[item.key])}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      )}
      <p className="mt-4 text-xs leading-5 text-faint">
        {MISSING} means the value is undefined at that date (warm-up window or mathematically
        undefined). Features use data up to and including the bar date only.
      </p>
    </div>
  );
}

export function MarketExplorer() {
  const { data, error, isPending } = useStocks();
  const [picked, setPicked] = useState<string | null>(null);

  if (isPending) {
    return (
      <Panel title="Universe">
        <p className="text-sm text-faint">Loading…</p>
      </Panel>
    );
  }
  if (error) {
    return (
      <Panel title="Universe">
        <QueryError error={error} />
      </Panel>
    );
  }
  if (data.length === 0) {
    return (
      <Panel title="Universe">
        <p className="text-sm text-muted">
          No securities are loaded. Build and load a dataset (see the Overview page).
        </p>
      </Panel>
    );
  }

  const selected = picked ?? data[0].symbol;
  const anySynthetic = data.some((stock) => stock.is_synthetic);

  return (
    <div className="grid gap-4 xl:grid-cols-5">
      <Panel
        title="Universe"
        className="xl:col-span-2"
        aside={
          <span className="flex items-center gap-2">
            <span className="num">{data.length} securities</span>
            {anySynthetic ? <SyntheticBadge /> : null}
          </span>
        }
      >
        <StockTable stocks={data} selected={selected} onSelect={setPicked} />
      </Panel>
      <Panel title="Security" className="xl:col-span-3">
        <StockDetail symbol={selected} />
      </Panel>
    </div>
  );
}
