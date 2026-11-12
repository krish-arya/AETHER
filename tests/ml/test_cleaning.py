"""Data cleaning: every defect is quarantined or flagged, none is silently repaired."""

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocessing.clean import clean_bars
from ml.data.validation.report import Severity

from .conftest import END


def _clean(raw, securities, calendar, **kwargs):
    return clean_bars(raw, securities=securities, calendar=calendar, as_of=END, **kwargs)


def _row(raw, symbol="SYNTH01", position=50):
    return raw.index[raw["symbol"] == symbol][position]


def _reasons(result) -> dict[str, int]:
    return result.quarantine["reason"].value_counts().to_dict()


def test_row_accounting_always_balances(raw_bars, securities, calendar):
    raw_bars.loc[_row(raw_bars), "close"] = np.nan
    result = _clean(raw_bars, securities, calendar)
    report = result.report
    assert report.rows_in == len(raw_bars)
    assert report.rows_in == report.rows_out + report.rows_quarantined
    assert report.rows_out == len(result.bars)


def test_missing_value_is_quarantined_and_never_filled(raw_bars, securities, calendar):
    index = _row(raw_bars)
    key = (raw_bars.loc[index, "symbol"], raw_bars.loc[index, "trade_date"])
    raw_bars.loc[index, "close"] = np.nan
    result = _clean(raw_bars, securities, calendar)

    assert _reasons(result) == {"missing_required_value": 1}
    remaining = result.bars.set_index(["symbol", "trade_date"])
    assert key not in remaining.index  # the bar is absent, not interpolated
    assert not result.bars[["open", "high", "low", "close", "volume"]].isna().any().any()
    # ...and its absence is reported as a missing session for that symbol.
    assert result.report.count("missing_session") == 1


def test_non_numeric_value_is_treated_as_missing(raw_bars, securities, calendar):
    raw_bars["close"] = raw_bars["close"].astype("object")
    raw_bars.loc[_row(raw_bars), "close"] = "n/a"
    assert _reasons(_clean(raw_bars, securities, calendar)) == {"missing_required_value": 1}


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("high", 0.01),  # high below low / open / close
        ("low", 1e9),  # low above everything
        ("close", -5.0),  # negative price
        ("open", 0.0),  # zero price
        ("volume", -1),  # negative volume
        ("volume", 10.5),  # fractional shares
    ],
)
def test_invalid_ohlcv_is_quarantined(raw_bars, securities, calendar, column, value):
    raw_bars[column] = raw_bars[column].astype("float64")
    raw_bars.loc[_row(raw_bars), column] = value
    result = _clean(raw_bars, securities, calendar)
    assert _reasons(result) == {"invalid_ohlcv": 1}
    assert result.report.issues[0].severity is Severity.ERROR


def test_infinite_price_is_quarantined(raw_bars, securities, calendar):
    raw_bars.loc[_row(raw_bars), "high"] = np.inf
    assert _reasons(_clean(raw_bars, securities, calendar)) == {"missing_required_value": 1}


def test_exact_duplicate_keeps_one_copy(raw_bars, securities, calendar):
    duplicated = pd.concat([raw_bars, raw_bars.iloc[[10, 10, 20]]], ignore_index=True)
    result = _clean(duplicated, securities, calendar)
    assert _reasons(result) == {"duplicate_exact": 3}
    assert len(result.bars) == len(raw_bars)
    assert not result.bars.duplicated(["symbol", "trade_date"]).any()


def test_conflicting_duplicates_are_all_quarantined(raw_bars, securities, calendar):
    conflicting = raw_bars.iloc[[10]].copy()
    conflicting["close"] = conflicting["close"] * 1.001
    conflicting["high"] = conflicting[["high", "close"]].max(axis=1)
    result = _clean(pd.concat([raw_bars, conflicting], ignore_index=True), securities, calendar)
    # We cannot know which version is right, so neither is kept.
    assert _reasons(result) == {"duplicate_conflict": 2}
    assert len(result.bars) == len(raw_bars) - 1


def test_future_dated_rows_are_rejected(raw_bars, securities, calendar):
    result = clean_bars(raw_bars, securities=securities, calendar=calendar, as_of="2021-06-30")
    assert set(_reasons(result)) == {"future_dated"}
    assert result.bars["trade_date"].max() <= pd.Timestamp("2021-06-30")


def test_non_session_date_is_rejected(raw_bars, securities, calendar):
    saturday = raw_bars.iloc[[_row(raw_bars)]].copy()
    saturday["trade_date"] = pd.Timestamp("2020-06-06")
    assert not calendar.is_session("2020-06-06")
    result = _clean(pd.concat([raw_bars, saturday], ignore_index=True), securities, calendar)
    assert _reasons(result) == {"non_session_date": 1}


def test_bars_outside_listing_window_are_rejected(raw_bars, securities, calendar):
    late = securities[securities["listing_date"].notna()].iloc[0]
    gone = securities[securities["delisting_date"].notna()].iloc[0]
    donor = raw_bars[raw_bars["symbol"] == "SYNTH01"]

    before_listing = donor[donor["trade_date"] < late["listing_date"]].iloc[[3]].copy()
    before_listing["symbol"] = late["symbol"]
    after_delisting = donor[donor["trade_date"] >= gone["delisting_date"]].iloc[[3]].copy()
    after_delisting["symbol"] = gone["symbol"]

    result = _clean(
        pd.concat([raw_bars, before_listing, after_delisting], ignore_index=True),
        securities,
        calendar,
    )
    assert _reasons(result) == {"outside_listing_window": 2}


def test_unknown_symbol_is_rejected(raw_bars, securities, calendar):
    stranger = raw_bars.iloc[[5]].copy()
    stranger["symbol"] = "NOTLISTED"
    result = _clean(pd.concat([raw_bars, stranger], ignore_index=True), securities, calendar)
    assert _reasons(result) == {"unknown_symbol": 1}


def test_missing_session_is_reported_and_left_absent(raw_bars, securities, calendar):
    index = _row(raw_bars, position=100)
    missing_date = raw_bars.loc[index, "trade_date"]
    result = _clean(raw_bars.drop(index), securities, calendar)

    assert result.quarantine.empty
    issues = [i for i in result.report.issues if i.check == "missing_session"]
    assert len(issues) == 1
    assert issues[0].symbol == "SYNTH01" and issues[0].count == 1
    assert issues[0].severity is Severity.WARNING
    assert issues[0].sample_dates == [missing_date.strftime("%Y-%m-%d")]
    symbol_dates = result.bars.loc[result.bars["symbol"] == "SYNTH01", "trade_date"]
    assert missing_date not in set(symbol_dates)


def test_placeholder_bar_is_rejected_but_a_real_zero_volume_bar_is_not(
    raw_bars, securities, calendar
):
    """Zero volume with O=H=L=C equal to the previous close is a filler row."""
    symbol_rows = raw_bars.index[raw_bars["symbol"] == "SYNTH01"]
    filler, previous = symbol_rows[60], symbol_rows[59]
    for column in ("open", "high", "low", "close"):
        raw_bars.loc[filler, column] = raw_bars.loc[previous, "close"]
    raw_bars.loc[filler, "volume"] = 0
    # Same shape but the price is not the previous close: a real (if odd) bar.
    odd = symbol_rows[120]
    for column in ("open", "high", "low"):
        raw_bars.loc[odd, column] = raw_bars.loc[odd, "close"]
    raw_bars.loc[odd, "volume"] = 0

    result = _clean(raw_bars, securities, calendar)
    assert _reasons(result) == {"stale_placeholder_bar": 1}
    assert result.quarantine["trade_date"].iloc[0] == raw_bars.loc[filler, "trade_date"]
    assert result.report.count("zero_volume") == 1  # the odd bar is kept and flagged
    assert result.report.count("missing_session") == 1  # the filler's date is now a gap


def test_zero_volume_is_flagged_but_kept(raw_bars, securities, calendar):
    raw_bars.loc[_row(raw_bars), "volume"] = 0
    result = _clean(raw_bars, securities, calendar)
    assert result.quarantine.empty
    assert result.report.count("zero_volume") == 1
    assert len(result.bars) == len(raw_bars)


def test_invalid_optional_value_is_marked_missing_and_bar_kept(raw_bars, securities, calendar):
    index = _row(raw_bars)
    raw_bars.loc[index, "turnover"] = -10.0
    result = _clean(raw_bars, securities, calendar)
    assert result.quarantine.empty
    assert result.report.count("invalid_optional_value") == 1
    assert result.bars["turnover"].isna().sum() == 1


def test_symbols_are_normalised_and_renamed(raw_bars, securities, calendar):
    raw_bars["symbol"] = raw_bars["symbol"].replace({"SYNTH01": "  oldname ", "SYNTH02": "synth02"})
    result = _clean(raw_bars, securities, calendar, symbol_map={"OLDNAME": "SYNTH01"})
    assert result.quarantine.empty
    assert set(result.bars["symbol"]) == set(securities["symbol"])
    assert result.report.count("symbol_remapped") == (raw_bars["symbol"] == "  oldname ").sum()


def test_output_is_sorted_and_typed(raw_bars, securities, calendar):
    shuffled = raw_bars.sample(frac=1.0, random_state=0)
    result = _clean(shuffled, securities, calendar)
    expected = result.bars.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.bars, expected)
    assert result.bars["volume"].dtype == "int64"
    assert result.bars["trade_date"].dtype == "datetime64[ns]"


def test_missing_required_column_raises(raw_bars, securities, calendar):
    with pytest.raises(ValueError, match="missing required columns"):
        _clean(raw_bars.drop(columns=["volume"]), securities, calendar)
