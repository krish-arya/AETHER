"""Corporate-action adjustment, verified against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocessing.clean import clean_bars
from ml.data.preprocessing.corporate_actions import (
    apply_adjustments,
    compute_adjustment_factors,
    validate_actions,
)
from ml.data.schemas import empty_actions
from ml.data.validation.checks import check_unexplained_gaps
from ml.data.validation.report import DataQualityReport

from .conftest import END

DATES = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])


def _actions(*rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["symbol", "ex_date", "action_type", "value"])
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    frame["announced_date"] = pd.NaT
    return frame


def _bars(close, volume=None, symbol="AAA") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": DATES[: len(close)],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume if volume is not None else [1000] * len(close),
        }
    )


def test_split_factors_hand_computed():
    # 2-for-1 split effective 2024-01-03: price halves, share count doubles.
    close = np.array([100.0, 102.0, 51.0, 52.0])
    split, adj, skipped = compute_adjustment_factors(
        DATES, close, _actions(("AAA", "2024-01-03", "split", 2.0))
    )
    assert skipped == []
    np.testing.assert_allclose(split, [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(adj, [0.5, 0.5, 1.0, 1.0])


def test_dividend_factor_hand_computed():
    # Rs 2 dividend, previous close 100 -> earlier prices scaled by 1 - 2/100 = 0.98.
    close = np.array([100.0, 100.0, 98.0, 99.0])
    split, adj, _ = compute_adjustment_factors(
        DATES, close, _actions(("AAA", "2024-01-03", "dividend", 2.0))
    )
    np.testing.assert_allclose(split, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(adj, [0.98, 0.98, 1.0, 1.0])


def test_split_and_dividend_compound():
    close = np.array([200.0, 100.0, 98.0, 99.0])
    actions = _actions(
        ("AAA", "2024-01-02", "bonus", 2.0),  # 1:1 bonus
        ("AAA", "2024-01-03", "dividend", 2.0),  # 2 / 100 on the post-bonus price
    )
    split, adj, _ = compute_adjustment_factors(DATES, close, actions)
    np.testing.assert_allclose(split, [0.5, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(adj, [0.5 * 0.98, 0.98, 1.0, 1.0])


def test_adjusted_series_and_volume():
    bars = _bars([100.0, 102.0, 51.0, 52.0], volume=[1000, 1100, 2400, 2000])
    report = DataQualityReport()
    out = apply_adjustments(bars, _actions(("AAA", "2024-01-03", "split", 2.0)), report)

    np.testing.assert_allclose(out["adj_close"], [50.0, 51.0, 51.0, 52.0])
    np.testing.assert_allclose(out["adj_volume"], [2000, 2200, 2400, 2000])
    np.testing.assert_allclose(out["close"], [100.0, 102.0, 51.0, 52.0])  # raw untouched
    # The -50% raw "crash" disappears from adjusted returns.
    np.testing.assert_allclose(out["adj_close"].pct_change().iloc[1:], [0.02, 0.0, 1 / 51])
    assert report.issues == []


def test_dividend_total_return_is_preserved():
    # Price falls by exactly the dividend: total return on the ex-date is zero.
    bars = _bars([100.0, 100.0, 98.0, 98.0])
    out = apply_adjustments(
        bars, _actions(("AAA", "2024-01-03", "dividend", 2.0)), DataQualityReport()
    )
    np.testing.assert_allclose(out["adj_close"].pct_change().iloc[1:], [0.0, 0.0, 0.0], atol=1e-12)


def test_ex_date_on_non_trading_day_applies_at_next_bar():
    dates = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    close = np.array([100.0, 100.0, 50.0, 50.0])
    split, _, skipped = compute_adjustment_factors(
        dates,
        close,
        _actions(("AAA", "2024-01-06", "split", 2.0)),  # a Saturday
    )
    assert skipped == []
    np.testing.assert_allclose(split, [0.5, 0.5, 1.0, 1.0])


def test_actions_outside_the_bar_range_are_reported_not_applied():
    close = np.array([100.0, 100.0, 100.0, 100.0])
    actions = _actions(
        ("AAA", "2023-12-01", "split", 2.0),
        ("AAA", "2024-01-01", "dividend", 1.0),  # ex-date is the first bar: no prior close
        ("AAA", "2024-02-01", "split", 2.0),
    )
    split, adj, skipped = compute_adjustment_factors(DATES, close, actions)
    np.testing.assert_allclose(split, 1.0)
    np.testing.assert_allclose(adj, 1.0)
    assert [reason for reason, _ in skipped] == [
        "action_before_first_bar",
        "action_before_first_bar",
        "action_after_last_bar",
    ]


def test_dividend_larger_than_price_is_a_blocking_error():
    report = DataQualityReport()
    out = apply_adjustments(
        _bars([10.0, 10.0, 10.0, 10.0]), _actions(("AAA", "2024-01-03", "dividend", 15.0)), report
    )
    np.testing.assert_allclose(out["adj_factor"], 1.0)
    assert report.count("dividend_exceeds_price") == 1


def test_no_actions_means_unit_factors():
    out = apply_adjustments(_bars([1.0, 2.0, 3.0]), empty_actions(), DataQualityReport())
    assert (out["adj_factor"] == 1.0).all() and (out["split_factor"] == 1.0).all()
    np.testing.assert_allclose(out["adj_close"], out["close"])


# ------------------------------------------------------------------- validation


def test_validate_actions_drops_and_reports_bad_records():
    actions = _actions(
        ("AAA", "2024-01-03", "split", 2.0),  # ok
        ("AAA", "2024-01-03", "split", 2.0),  # identical duplicate
        ("AAA", "2024-01-04", "merger", 1.5),  # unknown type
        ("AAA", "2024-01-04", "dividend", -1.0),  # non-positive
        ("AAA", "2024-01-04", "bonus", 1.0),  # ratio of 1 is not an action
        ("ZZZ", "2024-01-03", "split", 2.0),  # symbol not in the dataset
        ("AAA", "2025-06-01", "dividend", 1.0),  # after as-of
    )
    report = DataQualityReport()
    valid = validate_actions(actions, known_symbols={"AAA"}, as_of="2024-12-31", report=report)

    assert len(valid) == 1
    assert report.count("invalid_corporate_action") == 3
    assert report.count("duplicate_action") == 1
    assert report.count("action_for_unknown_symbol") == 1
    assert report.count("action_after_as_of") == 1


def test_conflicting_duplicate_actions_are_all_dropped():
    actions = _actions(("AAA", "2024-01-03", "split", 2.0), ("AAA", "2024-01-03", "split", 5.0))
    report = DataQualityReport()
    valid = validate_actions(actions, known_symbols={"AAA"}, as_of="2024-12-31", report=report)
    assert valid.empty
    assert report.count("duplicate_action_conflict") == 2


def test_announcement_after_ex_date_is_flagged():
    actions = _actions(("AAA", "2024-01-03", "dividend", 1.0))
    actions["announced_date"] = pd.Timestamp("2024-01-10")
    report = DataQualityReport()
    valid = validate_actions(actions, known_symbols={"AAA"}, as_of="2024-12-31", report=report)
    assert len(valid) == 1  # kept, but the point-in-time timestamp is suspect
    assert report.count("action_announced_after_ex_date") == 1


# ----------------------------------------------- end to end on the synthetic data


@pytest.fixture
def adjusted(raw_bars, raw_actions, securities, calendar):
    cleaned = clean_bars(raw_bars, securities=securities, calendar=calendar, as_of=END)
    report = cleaned.report
    actions = validate_actions(
        raw_actions, known_symbols=set(cleaned.bars["symbol"]), as_of=END, report=report
    )
    return apply_adjustments(cleaned.bars, actions, report), actions, report


def test_synthetic_split_is_removed_from_adjusted_returns(adjusted):
    bars, actions, _ = adjusted
    splits = actions[actions["action_type"].isin(["split", "bonus"])]
    assert len(splits) >= 2
    for action in splits.itertuples():
        series = bars[bars["symbol"] == action.symbol].set_index("trade_date")
        raw_return = series["close"].pct_change().loc[action.ex_date]
        adjusted_return = series["adj_close"].pct_change().loc[action.ex_date]
        assert raw_return < -0.35  # the raw price really does drop by ~1 - 1/ratio
        assert abs(adjusted_return) < 0.2  # ...and the adjusted series does not


def test_synthetic_dividends_raise_adjusted_return_by_the_yield(adjusted):
    bars, actions, _ = adjusted
    dividends = actions[actions["action_type"] == "dividend"]
    assert len(dividends) >= 4
    for action in dividends.itertuples():
        series = bars[bars["symbol"] == action.symbol].set_index("trade_date")
        position = series.index.get_loc(action.ex_date)
        previous_close = series["close"].iloc[position - 1]
        raw_return = series["close"].iloc[position] / previous_close - 1
        adjusted_return = series["adj_close"].pct_change().iloc[position]
        expected = (1 + raw_return) / (1 - action.value / previous_close) - 1
        assert adjusted_return == pytest.approx(expected, abs=1e-12)
        assert adjusted_return > raw_return


def test_latest_bar_is_never_adjusted(adjusted):
    bars, _, _ = adjusted
    last = bars.groupby("symbol").tail(1)
    assert (last["adj_factor"] == 1.0).all()
    np.testing.assert_allclose(last["adj_close"], last["close"])


def test_adjusted_ohlc_relationships_stay_valid(adjusted):
    bars, _, _ = adjusted
    assert (bars["adj_high"] >= bars[["adj_open", "adj_close"]].max(axis=1) - 1e-9).all()
    assert (bars["adj_low"] <= bars[["adj_open", "adj_close"]].min(axis=1) + 1e-9).all()
    assert (bars["adj_factor"] > 0).all() and (bars["adj_factor"] <= 1.0 + 1e-12).all()


def test_missing_split_is_caught_as_unexplained_gap(adjusted):
    bars, actions, _ = adjusted
    report = DataQualityReport()
    check_unexplained_gaps(bars, actions, 0.35, report)
    assert report.count("unexplained_price_gap") == 0  # every big move is explained

    without_splits = actions[actions["action_type"] == "dividend"]
    report = DataQualityReport()
    flagged = check_unexplained_gaps(bars, without_splits, 0.35, report)
    lost = actions[actions["action_type"] != "dividend"]
    assert report.count("unexplained_price_gap") == len(lost)
    assert set(flagged["trade_date"]) == set(lost["ex_date"])
