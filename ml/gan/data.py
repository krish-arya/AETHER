"""GAN preprocessing: market history -> standardised training windows, and back.

Variables modelled per asset and session (all stationary, all recoverable to OHLCV):

    gap      log(open_t / close_{t-1})          overnight move
    intra    log(close_t / open_t)              open-to-close move
    wick     log(high_t / low_t) - |intra_t|    range beyond the body (>= 0)
    volume   log(traded_value_t / median traded value of the previous
             ``volume_window`` sessions)        activity relative to its own recent level

A window is ``sequence_length`` consecutive sessions for all modelled assets, i.e. an
array [L, 4N]. Only assets present on at least 95% of the period's sessions are
modelled; the others are reported, never filled. Windows are standardised with the
mean/std of the training period and clipped to ``+/- clip_sigma``.

Each window carries a condition: the most frequent market regime inside it
(``ml.features.regime``), or ``undefined`` during the regime warm-up.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.backtesting.data import MarketData
from ml.features.regime import REGIMES

VARIABLES = ("gap", "intra", "wick", "volume")
MIN_COVERAGE = 0.95  # share of sessions an asset must have to be modelled
CONDITIONS = (*REGIMES, "undefined")


@dataclass
class WindowSet:
    """Training material for the generator."""

    windows: np.ndarray  # [n, L, 4N] standardised
    conditions: np.ndarray  # [n] index into CONDITIONS
    end_dates: pd.DatetimeIndex  # last session of each window
    symbols: list[str]
    excluded: list[str]  # assets without a complete history in the period
    mean: np.ndarray  # [4N]
    std: np.ndarray  # [4N]
    clip_sigma: float
    clipped_fraction: float
    base_value: np.ndarray  # [N] typical traded value, for turning ratios into rupees
    last_close: np.ndarray  # [N] last real close of the period (a price anchor)

    @property
    def dim(self) -> int:
        return self.windows.shape[2]

    def inverse(self, standardised: np.ndarray) -> np.ndarray:
        """Standardised windows -> raw variables (gap, intra, wick, volume)."""
        raw = standardised * self.std + self.mean
        n = len(self.symbols)
        raw[..., 2 * n : 3 * n] = np.maximum(raw[..., 2 * n : 3 * n], 0.0)  # wick >= 0
        return raw

    def split(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Raw array [..., 4N] -> {variable: array [..., N]}."""
        n = len(self.symbols)
        return {name: raw[..., i * n : (i + 1) * n] for i, name in enumerate(VARIABLES)}


def raw_variables(market: MarketData, volume_window: int) -> tuple[pd.DataFrame, list[str]]:
    """Per-session variables for every asset (wide, columns = (variable, symbol))."""
    if market.high is None or market.low is None:
        raise ValueError("the market view has no high/low prices")
    previous_close = market.close.shift(1)
    intra = np.log(market.close / market.open)
    frames = {
        "gap": np.log(market.open / previous_close),
        "intra": intra,
        "wick": (np.log(market.high / market.low) - intra.abs()).clip(lower=0.0),
        "volume": np.log(
            market.traded_value
            / market.traded_value.shift(1)
            .rolling(volume_window, min_periods=volume_window)
            .median()
        ),
    }
    table = pd.concat(frames, axis=1)
    return table.replace([np.inf, -np.inf], np.nan), list(market.symbols)


def window_conditions(market: MarketData, end_rows: np.ndarray, length: int) -> np.ndarray:
    """Most frequent regime inside each window (``undefined`` if there is none)."""
    undefined = CONDITIONS.index("undefined")
    if market.regimes is None:
        return np.full(len(end_rows), undefined)
    labels = market.regimes.reindex(market.sessions)["regime"]
    codes = labels.map({name: i for i, name in enumerate(CONDITIONS)}).fillna(undefined)
    codes = codes.to_numpy(dtype=np.int64)
    out = np.empty(len(end_rows), dtype=np.int64)
    for k, end in enumerate(end_rows):
        inside = codes[end - length + 1 : end + 1]
        counts = np.bincount(inside, minlength=len(CONDITIONS))
        counts[undefined] = 0 if counts[:undefined].sum() else counts[undefined]
        out[k] = int(counts.argmax())
    return out


def build_windows(
    market: MarketData,
    start,
    end,
    *,
    sequence_length: int,
    volume_window: int,
    clip_sigma: float,
    scaler: WindowSet | None = None,
) -> WindowSet:
    """Windows whose sessions all fall in ``start..end``.

    Pass ``scaler`` (the training WindowSet) to standardise another period with the
    training statistics and the same asset list -- never re-fit on held-out data.
    """
    table, symbols = raw_variables(market, volume_window)
    lo = market.sessions.searchsorted(pd.Timestamp(start), side="left")
    hi = market.sessions.searchsorted(pd.Timestamp(end), side="right")
    period = table.iloc[lo:hi]

    if scaler is None:
        valid = period.notna().sum()
        # An asset is modelled only if every variable is present on nearly every
        # session of the period (the volume baseline needs ``volume_window`` sessions
        # of warm-up). Sessions where a modelled asset has a gap produce no window.
        required = MIN_COVERAGE * (len(period) - volume_window - 1)
        complete = [s for s in symbols if all(valid[(v, s)] >= required for v in VARIABLES)]
    else:
        complete = scaler.symbols
    excluded = [s for s in symbols if s not in complete]
    if not complete:
        raise ValueError("no asset has a complete history in the requested period")

    columns = [(variable, symbol) for variable in VARIABLES for symbol in complete]
    values = period[columns].to_numpy(dtype="float64")
    row_ok = np.isfinite(values).all(axis=1)
    ends = [
        i
        for i in range(sequence_length - 1, len(values))
        if row_ok[i - sequence_length + 1 : i + 1].all()
    ]
    if len(ends) < 2:
        raise ValueError("the period is too short to form training windows")
    stacked = np.stack([values[i - sequence_length + 1 : i + 1] for i in ends])

    if scaler is None:
        flat = values[row_ok]
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
    else:
        mean, std = scaler.mean, scaler.std
    standardised = (stacked - mean) / std
    clipped = float((np.abs(standardised) > clip_sigma).mean())
    standardised = np.clip(standardised, -clip_sigma, clip_sigma)

    end_rows = np.asarray(ends) + lo
    position = [market.symbols.index(s) for s in complete]
    traded = market.traded_value.iloc[lo:hi].to_numpy()[:, position]
    close = market.close.iloc[lo:hi].to_numpy()[:, position]
    return WindowSet(
        windows=standardised.astype(np.float32),
        conditions=window_conditions(market, end_rows, sequence_length),
        end_dates=market.sessions[end_rows],
        symbols=complete,
        excluded=excluded,
        mean=mean,
        std=std,
        clip_sigma=clip_sigma,
        clipped_fraction=clipped,
        base_value=np.nanmedian(traded, axis=0),
        last_close=close[np.isfinite(close).all(axis=1)][-1],
    )
