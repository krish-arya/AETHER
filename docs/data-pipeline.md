# Data and feature pipeline (Phase 1)

Module documentation for `ml/data` and `ml/features`, in the format required by PRD
section 24 (Rule 6).

## Purpose

Turn raw daily market data from any provider into a **validated, corporate-action
adjusted, point-in-time, versioned** dataset with a technical feature table. Everything
downstream (backtests, GAN, RL, fusion) reads these snapshots, so the correctness of
this layer bounds the correctness of every later result.

```text
MarketDataProvider ─► clean_bars ─► validate_actions ─► apply_adjustments
                                                             │
        snapshot ◄─ quality gates ◄─ compute_features ◄─ statistical checks
```

## Input schema

Supplied by a `MarketDataProvider` ([ml/data/ingestion/base.py](../ml/data/ingestion/base.py)).
Dates are exchange-local (IST) trading dates.

**Daily bars** — raw, as traded, never pre-adjusted

| Column | Required | Meaning |
|---|---|---|
| `symbol` | yes | NSE symbol |
| `trade_date` | yes | Trading date |
| `open`, `high`, `low`, `close` | yes | Prices in rupees |
| `volume` | yes | Shares traded |
| `turnover` | no | Traded value |
| `delivery_volume` | no | Delivered quantity |
| `market_cap` | no | Point-in-time market capitalisation |

**Corporate actions** — `symbol, ex_date, action_type, value, announced_date`

| `action_type` | `value` |
|---|---|
| `split`, `bonus` | Shares held after per share held before (1:1 bonus = `2.0`) |
| `dividend` | Cash per share in rupees, on the pre-action share basis |

**Securities** — `symbol, name, sector, industry, isin, listing_date, delisting_date`.
`delisting_date` is the first date the security no longer trades; an empty
`listing_date` means "listed before the data begins".

Providers available:

| Kind | Class | Use |
|---|---|---|
| `synthetic_dev` | `SyntheticDevProvider` | Labelled mock data for development and tests |
| `csv_directory` | `CsvDirectoryProvider` | CSV exports from a licensed source you supply |

## Output schema

One directory per dataset under `artifacts/datasets/<dataset_id>/`:

| File | Contents |
|---|---|
| `manifest.json` | Provenance, full config, per-file SHA-256, environment, quality summary |
| `quality_report.json` | Every issue found, by check and symbol, with sample dates |
| `securities.csv` | Securities master for the universe |
| `bars.csv` | Raw OHLCV + `adj_factor`, `split_factor`, `adj_open/high/low/close`, `adj_volume` |
| `corporate_actions.csv` | Validated actions that were applied |
| `features.csv` | `symbol, trade_date` + one column per feature |
| `quarantine.csv` | Rejected rows with a `reason` |

`dataset_id = <name>-<first 12 hex of content hash>`. The content hash covers the four
data files, so the id changes whenever data, cleaning rules or features change, and is
identical when nothing does.

### Features

All features are scale-free (see [Point-in-time guarantees](#point-in-time-guarantees)).
Names include the window, so they follow the configuration.

| Family | Columns (default config) |
|---|---|
| Returns | `ret_1d`, `log_ret_1d`, `cum_ret_{5,21,63}d`, `beta_60d`, `corr_mkt_60d` |
| Trend | `sma_{20,50}_ratio`, `ema_{12,26}_ratio`, `macd_pct`, `macd_signal_pct`, `macd_hist_pct`, `adx_14`, `plus_di_14`, `minus_di_14` |
| Momentum | `rsi_14`, `stoch_k_14`, `stoch_d_14`, `roc_10`, `williams_r_14` |
| Volatility | `atr_pct_14`, `bb_pctb_20`, `bb_width_20`, `hist_vol_{20,60}d`, `ret_std_20d` |
| Volume | `obv_norm_20`, `vwap_ratio_20`, `volume_change_1d`, `mfi_14` |

The classic level forms (SMA, EMA, MACD, ATR, Bollinger bands, OBV, VWAP) are
available as functions in [ml/features/indicators.py](../ml/features/indicators.py).

## Configuration

[configs/data.yaml](../configs/data.yaml), validated by
[ml/data/config.py](../ml/data/config.py). Unknown keys are rejected.

| Section | Controls |
|---|---|
| `dataset` | Name, date range, `as_of` knowledge cut-off |
| `provider` | Which provider and its settings (seed, CSV path, provenance) |
| `universe` | Optional symbol subset; historical-ticker → current-ticker map |
| `validation` | Outlier z-score, unexplained-gap threshold, calendar inference, quarantine limit |
| `features` | Every indicator window |
| `output` | Where snapshots are written |

## Running it

There is nothing to train in this phase.

```bash
uv run python scripts/build_dataset.py --config configs/data.yaml   # build a snapshot
uv run python scripts/load_dataset.py --latest                      # load it into the database
```

`build_dataset.py` exits with code 2 and writes nothing when a quality gate fails.

## Data-quality rules

Policy: **detect, quarantine, report — never repair.** No value is interpolated,
forward-filled or clipped.

Hard checks (row moved to `quarantine.csv`):

| Check | Trigger |
|---|---|
| `missing_required_value` | Symbol, date, OHLC or volume missing or non-numeric |
| `future_dated` | `trade_date` after `as_of` |
| `non_session_date` | Not an exchange session |
| `unknown_symbol` | Not in the securities master |
| `outside_listing_window` | Before listing or on/after delisting |
| `invalid_ohlcv` | Non-positive price, negative/fractional volume, inconsistent high/low |
| `duplicate_exact` | Identical copy of an earlier row (first copy kept) |
| `duplicate_conflict` | Same key, different values (all copies quarantined) |

Flags (row kept, issue reported): `zero_volume`, `missing_session`,
`invalid_optional_value`, `return_outlier`, `unexplained_price_gap`,
`survivorship_unverified`, `action_announced_after_ex_date`.

Quality gates (build fails): quarantined share above
`validation.max_quarantine_fraction`; any invalid or conflicting corporate action; a
dividend larger than the price; any infinite or out-of-range feature value.

## Corporate-action adjustment

Proportional back-adjustment. For an action effective at the first bar on or after its
ex-date, every earlier bar is scaled by:

- split / bonus with ratio `r`: price × `1/r`, volume × `r`
- cash dividend `D`: price × `(1 − D / C_prev)`, where `C_prev` is the last raw close
  before the ex-date

`adj_close` is therefore a total-return series. Raw prices are stored unchanged
alongside the factors, so any adjusted value can be re-derived and audited.

## Point-in-time guarantees

- **Indicators are causal.** A value at date T uses bars dated ≤ T only.
- **No leakage through adjustment.** Back-adjusted price *levels* shift when a later
  split or dividend occurs, so a level-based feature would reveal future corporate
  actions. The feature table therefore contains only ratios, percentages and bounded
  oscillators, which are invariant to that rescaling.
- **Snapshot cut-off.** Bars and corporate actions dated after `as_of` are excluded.
- **Point-in-time universe.** `ml.data.universe.universe_as_of(securities, date)`
  returns the securities listed on that date; delisted names stay in the dataset.
- **Decision timing.** A feature row for T is known after the close of T; a decision
  based on it executes no earlier than `TradingCalendar.next_session(T)`.

## Evaluation

```bash
uv run pytest tests/ml
```

| Test file | Verifies |
|---|---|
| `test_no_lookahead.py` | Truncation invariance of every indicator; corrupting the future leaves past features bit-identical; rebuilding as of earlier dates (different adjustment factors) reproduces the features |
| `test_corporate_actions.py` | Factors against hand-computed values; total return preserved across dividends; a missing split is caught |
| `test_cleaning.py` | Each defect class is quarantined or flagged and never filled |
| `test_indicators.py` | Each indicator against hand-computed values or an independent reference |
| `test_pipeline.py` | Reproducible ids, tamper detection, quality gates, as-of cut-off, CSV/synthetic equivalence |
| `test_calendar_universe.py` | Calendar inference, next-session logic, point-in-time universe |

To review a real dataset, read `quality_report.json`: every `error` and `warning` entry
names the check, the symbol, the count and sample dates.

## Known limitations

- **No real data source is connected.** The pipeline has been exercised on synthetic
  development data and CSV fixtures only; it has not yet met real NSE data and its
  quirks (symbol changes, suspensions, circuit limits, face-value changes).
- **Corporate actions modelled:** splits, bonus issues, cash dividends. Rights issues,
  demergers, mergers and capital reductions are not; they surface as
  `unexplained_price_gap` warnings rather than being adjusted.
- A dividend and a split sharing an ex-date assume the dividend is quoted pre-split.
- **Index constituent history** (NIFTY 50/100 membership by date) is not modelled, so
  the point-in-time universe reflects listing windows only.
- **Trading calendar:** NSE holidays are not built in. Sessions come from the provider
  or are inferred from the data; special sessions (e.g. Muhurat trading) are handled
  only insofar as the data contains them.
- Indicators run over each symbol's consecutive available bars; a missing session makes
  the adjacent "1-day" return span more than one day (the gap itself is reported).
- The market proxy for beta/correlation is the equal-weight universe mean, which
  includes the asset itself and is biased towards 1 for small universes. A benchmark
  index series should replace it once index data is licensed.
- **Not yet ingested:** fundamentals, news, macroeconomic series (PRD 5.2–5.4).
- Snapshots are CSV (pyarrow is blocked on the primary development machine); fine at
  tens of symbols, slower than a columnar format at several hundred.
- Dataset versioning is a manifest scheme, not DVC; there is no remote storage yet.
