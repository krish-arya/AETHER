"""News feature aggregation: hand-computed weights and strict point-in-time behaviour."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ml.llm.config import AggregationConfig
from ml.llm.schema import FinancialEvent
from ml.llm.sentiment.aggregation import (
    COLUMNS,
    FEATURE_COLUMNS,
    aggregate_news_features,
    decision_cutoff,
)

CONFIG = AggregationConfig(
    default_half_life_days=5, half_life_days={"earnings": 5, "litigation": 20}, max_age_days=60
)
DAY = "2026-11-10"
CLOSE = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)  # 15:30 IST


def _event(event_id: str, published: datetime, **overrides) -> FinancialEvent:
    fields = {
        "event_id": event_id, "company": "Zentara Infra", "symbol": "ZENTARA",
        "event_type": "earnings", "sentiment": "positive", "sentiment_score": 0.6,
        "impact_direction": "potentially_positive", "impact_horizon": "short_term",
        "severity": 0.5, "confidence": 0.8, "key_facts": [], "risks": [],
        "source_ids": [f"src_{event_id}"], "publication_timestamp": published,
    }  # fmt: skip
    return FinancialEvent(**{**fields, **overrides})


def _row(frame: pd.DataFrame, symbol: str = "ZENTARA", day: str = DAY) -> pd.Series:
    match = frame[(frame["symbol"] == symbol) & (frame["trade_date"] == pd.Timestamp(day))]
    assert len(match) == 1
    return match.iloc[0]


def test_decision_cutoff_is_the_nse_close_in_utc():
    assert decision_cutoff(DAY, CONFIG) == CLOSE
    assert decision_cutoff(pd.Timestamp(DAY), AggregationConfig(market_close="09:15")) == datetime(
        2026, 11, 10, 3, 45, tzinfo=UTC
    )


def test_single_event_features_by_hand():
    # Exactly one half-life (5 days) old: d = 0.5, w = d * r * c = 0.5 * 1 * 0.8 = 0.4.
    event = _event("e1", CLOSE - timedelta(days=5))
    row = _row(aggregate_news_features([event], ["ZENTARA"], [DAY], CONFIG))
    assert row["news_available"] == 1 and row["news_event_count"] == 1
    assert row["news_sentiment"] == pytest.approx(0.6)
    assert row["news_impact"] == pytest.approx(1.0)
    assert row["news_severity_max"] == pytest.approx(0.25)  # d * v
    assert row["news_severity_weighted"] == pytest.approx(0.5)
    assert row["news_event_score"] == pytest.approx(0.4 * 0.5 * 0.6)  # w * v * s
    assert row["news_confidence"] == pytest.approx(0.8)
    assert row["news_source_reliability"] == pytest.approx(1.0)
    assert row["news_freshness"] == pytest.approx(0.5)
    assert row["news_days_since_last"] == pytest.approx(5.0)
    assert row["news_event_frequency"] == pytest.approx(1 / 7)


def test_event_type_specific_half_lives_change_the_blend():
    earnings = _event("e1", CLOSE - timedelta(days=5))  # d = 0.5 ** (5/5) = 0.5
    lawsuit = _event(
        "e2", CLOSE - timedelta(days=20), event_type="litigation", sentiment="negative",
        sentiment_score=-0.8, impact_direction="potentially_negative", severity=1.0,
        confidence=0.5,
    )  # fmt: skip
    # Litigation half-life 20 days: d = 0.5, w = 0.25. Earnings: w = 0.4.
    row = _row(aggregate_news_features([earnings, lawsuit], ["ZENTARA"], [DAY], CONFIG))
    assert row["news_sentiment"] == pytest.approx((0.4 * 0.6 + 0.25 * -0.8) / 0.65)
    assert row["news_impact"] == pytest.approx((0.4 - 0.25) / 0.65)
    assert row["news_severity_max"] == pytest.approx(0.5)  # 0.5 * 1.0 from the lawsuit
    assert row["news_event_score"] == pytest.approx(0.4 * 0.5 * 0.6 + 0.25 * 1.0 * -0.8)
    assert row["news_event_count"] == 2
    assert row["news_event_frequency"] == pytest.approx(1 / 7)  # only one in the last 7 days
    assert row["news_days_since_last"] == pytest.approx(5.0)

    # With one half-life for everything the 20-day-old lawsuit has almost faded:
    # d = 0.5 ** (20/5) = 0.0625, w = 0.03125.
    flat = AggregationConfig(default_half_life_days=5, max_age_days=60)
    row = _row(aggregate_news_features([earnings, lawsuit], ["ZENTARA"], [DAY], flat))
    assert row["news_sentiment"] == pytest.approx((0.4 * 0.6 + 0.03125 * -0.8) / 0.43125)
    assert row["news_sentiment"] > 0.49  # versus 0.06 when litigation is long-lived


def test_source_reliability_weights_publishers():
    config = AggregationConfig(
        default_half_life_days=5, source_reliability={"Rumour Mill": 0.25}, max_age_days=60
    )
    good = _event("e1", CLOSE, sentiment_score=0.5, confidence=1.0)
    weak = _event("e2", CLOSE, sentiment="negative", sentiment_score=-0.5, confidence=1.0,
                  impact_direction="potentially_negative")  # fmt: skip
    publishers = {"e1": "Fictional Wire", "e2": "rumour mill"}  # case-insensitive
    row = _row(aggregate_news_features([good, weak], ["ZENTARA"], [DAY], config, publishers))
    assert row["news_sentiment"] == pytest.approx((1 * 0.5 + 0.25 * -0.5) / 1.25)
    assert row["news_source_reliability"] == pytest.approx((1 + 0.25) / 2)
    neutral = _row(aggregate_news_features([good, weak], ["ZENTARA"], [DAY], config))
    assert neutral["news_sentiment"] == pytest.approx(0.0)


def test_news_after_the_close_counts_from_the_next_session():
    at_close = _event("e1", CLOSE)
    after_close = _event("e2", CLOSE + timedelta(seconds=1))
    days = ["2026-11-10", "2026-11-11"]
    frame = aggregate_news_features([at_close, after_close], ["ZENTARA"], days, CONFIG)
    assert _row(frame, day="2026-11-10")["news_event_count"] == 1
    assert _row(frame, day="2026-11-10")["news_days_since_last"] == 0
    assert _row(frame, day="2026-11-11")["news_event_count"] == 2

    only_late = aggregate_news_features([after_close], ["ZENTARA"], days, CONFIG)
    assert _row(only_late, day="2026-11-10")["news_available"] == 0


def test_no_news_rows_are_zero_and_flagged_never_nan():
    event = _event("e1", CLOSE - timedelta(days=1))
    days = pd.bdate_range("2026-11-02", "2026-11-12")
    frame = aggregate_news_features([event], ["ZENTARA", "QUILLON"], days, CONFIG)
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 2 * len(days)
    assert not frame.isna().any().any()
    quiet = frame[frame["symbol"] == "QUILLON"]
    assert (quiet[FEATURE_COLUMNS] == 0).all().all()
    before = frame[(frame["symbol"] == "ZENTARA") & (frame["trade_date"] < "2026-11-09")]
    assert (before[FEATURE_COLUMNS] == 0).all().all()
    assert frame["trade_date"].dtype == "datetime64[ns]"
    assert frame["news_available"].dtype == "int64"

    empty = aggregate_news_features([], ["ZENTARA"], days, CONFIG)
    assert len(empty) == len(days) and (empty[FEATURE_COLUMNS] == 0).all().all()


def test_events_expire_and_unknown_events_carry_no_signal():
    config = AggregationConfig(max_age_days=10)
    old = _event("e1", CLOSE - timedelta(days=10, seconds=1))
    unknown = _event("e2", CLOSE, event_type="unknown", sentiment="uncertain", sentiment_score=0)
    frame = aggregate_news_features([old, unknown], ["ZENTARA"], [DAY], config)
    assert _row(frame)["news_available"] == 0


def test_future_news_cannot_change_past_features():
    """Truncation test: features up to T are identical with and without later events."""
    base = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)
    events = [
        _event(
            f"e{i}", base + timedelta(hours=31 * i), sentiment_score=((i % 5) - 2) / 4,
            sentiment="mixed", severity=(i % 4) / 4, confidence=0.5 + (i % 3) / 6,
            event_type=["earnings", "litigation", "dividend"][i % 3],
        )
        for i in range(30)
    ]  # fmt: skip
    days = pd.bdate_range("2026-10-01", "2026-11-12")
    full = aggregate_news_features(events, ["ZENTARA"], days, CONFIG)

    cut = pd.Timestamp("2026-10-20")
    cutoff = decision_cutoff(cut, CONFIG)
    known = [e for e in events if e.publication_timestamp <= cutoff]
    assert 0 < len(known) < len(events)
    truncated = aggregate_news_features(known, ["ZENTARA"], days, CONFIG)

    past = full["trade_date"] <= cut
    pd.testing.assert_frame_equal(full[past], truncated[past], check_exact=True)
    assert not full[~past].equals(truncated[~past])  # the later events do matter later

    # Scrambling the content of later events leaves the past untouched as well.
    scrambled = [
        e if e.publication_timestamp <= cutoff
        else e.model_copy(update={"sentiment_score": -e.sentiment_score, "severity": 1.0})
        for e in events
    ]  # fmt: skip
    again = aggregate_news_features(scrambled, ["ZENTARA"], days, CONFIG)
    pd.testing.assert_frame_equal(full[past], again[past], check_exact=True)


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        AggregationConfig(half_life_days={"earnings": 0})
    with pytest.raises(ValueError):
        AggregationConfig(half_life_days={"not_an_event": 3})
    with pytest.raises(ValueError):
        AggregationConfig(source_reliability={"x": -1})
    with pytest.raises(ValueError):
        AggregationConfig(unknown_key=1)
