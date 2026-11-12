"""Read-only market view used by the backtester.

Built from a versioned dataset snapshot (``ml.data.pipeline``). Prices are the
total-return adjusted series, so:

* portfolio returns include dividends (assumed reinvested on the ex-date, untaxed);
* share counts are in *adjusted* units. ``adjusted shares x adjusted price`` equals the
  real rupee value at the time, so notionals -- and therefore transaction costs and
  liquidity limits -- are exact even though the share counts are not literal.

Wide frames are indexed by session with one column per symbol; NaN means the symbol
has no bar that day (not yet listed, delisted, suspended or simply missing).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.data.schemas import to_date
from ml.data.versioning import read_frame, verify_dataset


@dataclass(frozen=True)
class MarketData:
    sessions: pd.DatetimeIndex
    symbols: list[str]
    open: pd.DataFrame  # adjusted open
    close: pd.DataFrame  # adjusted close
    traded_value: pd.DataFrame  # rupees traded that day (liquidity)
    securities: pd.DataFrame  # indexed by symbol; listing_date, delisting_date
    features: pd.DataFrame | None = None  # indexed by (trade_date, symbol)
    regimes: pd.DataFrame | None = None  # indexed by trade_date
    high: pd.DataFrame | None = None  # adjusted high (when the bars carry it)
    low: pd.DataFrame | None = None  # adjusted low
    dataset_id: str | None = None
    is_synthetic: bool | None = None
    provider: str | None = None
    content_hash: str | None = None

    # ---------------------------------------------------------------- constructors

    @classmethod
    def from_frames(
        cls,
        bars: pd.DataFrame,
        securities: pd.DataFrame,
        features: pd.DataFrame | None = None,
        regimes: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
        **provenance,
    ) -> "MarketData":
        """``symbols`` fixes the column set and order (symbols without bars become
        all-NaN columns); by default the symbols present in ``bars`` are used."""
        required = {"symbol", "trade_date", "adj_open", "adj_close", "close", "volume"}
        missing = sorted(required - set(bars.columns))
        if missing:
            raise ValueError(f"bars are missing columns: {missing}")
        if bars.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("bars contain duplicate (symbol, trade_date) rows")

        bars = bars.assign(trade_date=to_date(bars["trade_date"]))
        value = bars["volume"] * bars["close"]
        if "turnover" in bars.columns:
            value = bars["turnover"].where(bars["turnover"].notna(), value)
        bars = bars.assign(_traded_value=value)

        def wide(column: str) -> pd.DataFrame:
            frame = bars.pivot(index="trade_date", columns="symbol", values=column)
            frame = frame.sort_index().sort_index(axis=1).astype("float64")
            return frame if symbols is None else frame.reindex(columns=symbols)

        open_, close, traded = wide("adj_open"), wide("adj_close"), wide("_traded_value")
        master = securities.set_index("symbol").reindex(close.columns)
        for column in ("listing_date", "delisting_date"):
            master[column] = to_date(master[column]) if column in master else pd.NaT

        indexed_features = None
        if features is not None:
            indexed_features = (
                features.assign(trade_date=to_date(features["trade_date"]))
                .set_index(["trade_date", "symbol"])
                .sort_index()
            )
        return cls(
            sessions=pd.DatetimeIndex(close.index),
            symbols=list(close.columns),
            open=open_,
            close=close,
            traded_value=traded,
            securities=master,
            high=wide("adj_high") if "adj_high" in bars.columns else None,
            low=wide("adj_low") if "adj_low" in bars.columns else None,
            features=indexed_features,
            regimes=None
            if regimes is None
            else regimes.assign(trade_date=to_date(regimes["trade_date"])).set_index("trade_date"),
            **provenance,
        )

    @classmethod
    def from_snapshot(cls, dataset_dir: str | Path) -> "MarketData":
        """Load a dataset snapshot after verifying it against its manifest."""
        dataset_dir = Path(dataset_dir)
        manifest = verify_dataset(dataset_dir)
        return cls.from_frames(
            read_frame(dataset_dir / "bars.csv", ("trade_date",)),
            read_frame(dataset_dir / "securities.csv", ("listing_date", "delisting_date")),
            read_frame(dataset_dir / "features.csv", ("trade_date",)),
            read_frame(dataset_dir / "regimes.csv", ("trade_date",))
            if (dataset_dir / "regimes.csv").is_file()
            else None,
            dataset_id=manifest["dataset_id"],
            is_synthetic=manifest["provider"]["is_synthetic"],
            provider=manifest["provider"]["name"],
            content_hash=manifest["content_hash"],
        )

    # -------------------------------------------------------------------- queries

    def sessions_between(self, start, end) -> pd.DatetimeIndex:
        lo = self.sessions.searchsorted(to_date(start), side="left")
        hi = self.sessions.searchsorted(to_date(end), side="right")
        return self.sessions[lo:hi]

    def listed_on(self, day: pd.Timestamp) -> pd.Series:
        """Boolean per symbol: inside its listing window on ``day``."""
        listing = self.securities["listing_date"]
        delisting = self.securities["delisting_date"]
        return (listing.isna() | (listing <= day)) & (delisting.isna() | (day < delisting))

    def regime_on(self, day: pd.Timestamp) -> str | None:
        """Market regime label known at the close of ``day`` (None if undefined)."""
        if self.regimes is None or day not in self.regimes.index:
            return None
        label = self.regimes.at[day, "regime"]
        return None if pd.isna(label) else str(label)

    def features_on(self, day: pd.Timestamp) -> pd.DataFrame | None:
        """Feature cross-section for exactly ``day`` (indexed by symbol), if available."""
        if self.features is None:
            return None
        try:
            return self.features.xs(day, level="trade_date")
        except KeyError:
            return self.features.iloc[0:0].droplevel("trade_date")
