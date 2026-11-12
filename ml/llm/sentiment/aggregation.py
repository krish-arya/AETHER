"""Point-in-time aggregation of financial events into per-stock daily features (PRD 8.5).

For a decision date ``T`` the features of a symbol use **only** events published at or
before the market close of ``T`` (15:30 Asia/Kolkata by default) and at most
``max_age_days`` old. News published after the close of ``T`` first appears on the next
date. Adding later events can therefore never change an earlier row.

Per event ``i`` (age in days measured from the close of ``T``):

    d_i = 0.5 ** (age_i / half_life[event_type_i])    recency; event-type specific
    r_i = reliability weight of the event's publisher  (default: neutral 1.0)
    c_i = confidence, s_i = sentiment_score, v_i = severity
    m_i = +1 / -1 / 0 for impact potentially_positive / potentially_negative / other
    w_i = d_i * r_i * c_i

Columns (one row per symbol and date; all 0 with ``news_available = 0`` when there is
no usable event -- never NaN, never invented):

    news_available           1 if at least one event is in the window, else 0
    news_event_count         events in the window
    news_event_frequency     events in the last ``frequency_window_days`` per day
    news_sentiment           sum(w s) / sum(w)        confidence-weighted sentiment
    news_impact              sum(w m) / sum(w)        weighted impact direction
    news_severity_max        max(d v)                 recency-decayed peak severity
    news_severity_weighted   sum(w v) / sum(w)
    news_event_score         sum(w v s)               signed strength; decays to 0
    news_confidence          sum(d r c) / sum(d r)
    news_source_reliability  sum(d c r) / sum(d c)
    news_freshness           max(d)                   1 = an event at the close
    news_days_since_last     age of the newest event in days (0 when none: check
                             ``news_available``)

``news_sentiment`` is an average: it says how good the news is, not how much or how
fresh. ``news_event_score`` is a decayed sum and carries strength.

Events of type ``unknown`` (insufficient information) are ignored.

Limitations: half-lives and reliability weights are configured assumptions, not
estimated from data. Without a historical news archive these features only exist for
the dates news was actually collected for.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ml.llm.config import AggregationConfig
from ml.llm.schema import EventType, FinancialEvent, ImpactDirection

FEATURE_COLUMNS = [
    "news_available",
    "news_event_count",
    "news_event_frequency",
    "news_sentiment",
    "news_impact",
    "news_severity_max",
    "news_severity_weighted",
    "news_event_score",
    "news_confidence",
    "news_source_reliability",
    "news_freshness",
    "news_days_since_last",
]
COLUMNS = ["symbol", "trade_date", *FEATURE_COLUMNS]
_IMPACT_SIGN = {
    ImpactDirection.POTENTIALLY_POSITIVE: 1.0,
    ImpactDirection.POTENTIALLY_NEGATIVE: -1.0,
}
SECONDS_PER_DAY = 86_400.0


def decision_cutoff(day: date | pd.Timestamp | str, config: AggregationConfig) -> datetime:
    """The instant (UTC) at which the decision for trading date ``day`` is taken."""
    day = pd.Timestamp(day).date()
    hour, minute = (int(part) for part in config.market_close.split(":"))
    local = datetime.combine(day, time(hour, minute), ZoneInfo(config.market_timezone))
    return local.astimezone(UTC)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def aggregate_news_features(
    events: Iterable[FinancialEvent],
    symbols: Sequence[str],
    dates: Iterable[date | pd.Timestamp | str],
    config: AggregationConfig | None = None,
    publishers: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """News features for every (symbol, date).

    Inputs:  validated events; the symbols and decision dates wanted; the aggregation
             config; optionally ``event_id -> publisher`` for source reliability.
    Output:  DataFrame with columns ``COLUMNS``, one row per symbol and date, sorted by
             symbol then date. ``trade_date`` is a normalised ``datetime64[ns]``.
    """
    config = config or AggregationConfig()
    publishers = publishers or {}
    days = sorted({pd.Timestamp(d).normalize() for d in dates})
    cutoffs = np.array([decision_cutoff(day, config).timestamp() for day in days])

    by_symbol: dict[str, list[FinancialEvent]] = {}
    for event in events:
        if event.event_type is not EventType.UNKNOWN:
            by_symbol.setdefault(event.symbol, []).append(event)

    rows: list[list] = []
    for symbol in symbols:
        chosen = by_symbol.get(symbol, [])
        stamp = np.array([e.publication_timestamp.timestamp() for e in chosen])
        score = np.array([e.sentiment_score for e in chosen])
        conf = np.array([e.confidence for e in chosen])
        severity = np.array([e.severity for e in chosen])
        impact = np.array([_IMPACT_SIGN.get(e.impact_direction, 0.0) for e in chosen])
        half_life = np.array([config.half_life(e.event_type) for e in chosen])
        reliability = np.array([config.reliability(publishers.get(e.event_id)) for e in chosen])
        for day, cutoff in zip(days, cutoffs, strict=True):
            age = (cutoff - stamp) / SECONDS_PER_DAY
            use = (age >= 0) & (age <= config.max_age_days)
            if not use.any():
                rows.append([symbol, day, *([0.0] * len(FEATURE_COLUMNS))])
                continue
            a, s, c, v, m, r = (
                age[use],
                score[use],
                conf[use],
                severity[use],
                impact[use],
                reliability[use],
            )  # noqa: E501
            d = 0.5 ** (a / half_life[use])
            w = d * r * c
            rows.append(
                [
                    symbol,
                    day,
                    1.0,
                    float(use.sum()),
                    float((a <= config.frequency_window_days).sum()) / config.frequency_window_days,
                    _ratio((w * s).sum(), w.sum()),
                    _ratio((w * m).sum(), w.sum()),
                    float((d * v).max()),
                    _ratio((w * v).sum(), w.sum()),
                    float((w * v * s).sum()),
                    _ratio((d * r * c).sum(), (d * r).sum()),
                    _ratio((d * c * r).sum(), (d * c).sum()),
                    float(d.max()),
                    float(a.min()),
                ]
            )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column in ("news_available", "news_event_count"):
        frame[column] = frame[column].astype("int64")
    for column in FEATURE_COLUMNS[2:]:
        frame[column] = frame[column].astype("float64")
    frame["symbol"] = frame["symbol"].astype("object")
    return frame


def news_feature_table(
    symbols: Sequence[str],
    start: date | pd.Timestamp | str,
    end: date | pd.Timestamp | str,
    *,
    news_root: str = "artifacts/news",
    config: AggregationConfig | None = None,
    sessions: Iterable[date | pd.Timestamp | str] | None = None,
) -> pd.DataFrame:
    """News features for ``symbols`` over ``[start, end]`` from every news run on disk.

    ``sessions`` are the decision dates (pass the trading calendar); when omitted,
    Monday-Friday dates are used, which includes exchange holidays. Events are read
    from all verified run directories under ``news_root``; an event analysed in several
    runs is taken from the most recent one. With no runs on disk, or outside the dates
    news was collected for, every row is zero with ``news_available = 0``.
    """
    from ml.llm.runs import load_all_events

    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if sessions is None:
        days = list(pd.bdate_range(start, end))
    else:
        days = [d for d in (pd.Timestamp(s).normalize() for s in sessions) if start <= d <= end]
    analyzed = load_all_events(news_root)
    return aggregate_news_features(
        [a.event for a in analyzed],
        symbols,
        days,
        config,
        publishers={a.event.event_id: a.source for a in analyzed},
    )
