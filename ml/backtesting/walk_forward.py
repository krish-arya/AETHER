"""Rebalancing schedules and chronological walk-forward splits (PRD 11.2, 11.4).

Walk-forward
    Windows roll forward in time and never overlap backwards::

        fold 0   train 2015-2019 | validation 2020 | test 2021
        fold 1   train 2016-2020 | validation 2021 | test 2022
        ...

    * Models are fitted on ``train`` only.
    * Hyper-parameters are chosen on ``validation`` only.
    * ``test`` is touched once, for reporting. It must never drive a tuning decision.
    * ``embargo_sessions`` drops that many sessions from the start of the validation
      and test windows, so information that straddles a boundary (overlapping labels,
      multi-day features) cannot leak across it.

    A fold is only produced when its whole test window lies inside the data.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from ml.data.schemas import to_date

FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")


def rebalance_dates(sessions: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Decision dates: the first session, then the last session of each period.

    The first session is always included so capital is deployed at the start of the
    window instead of idling until the first period end.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
    if len(sessions) == 0:
        return sessions
    if frequency == "daily":
        return sessions
    period = {"weekly": "W", "monthly": "M", "quarterly": "Q"}[frequency]
    period_ends = pd.Series(sessions, index=sessions).groupby(sessions.to_period(period)).max()
    return pd.DatetimeIndex(sorted({sessions[0], *period_ends})).unique()


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            key: value.strftime("%Y-%m-%d") if isinstance(value, pd.Timestamp) else value
            for key, value in asdict(self).items()
        }


def walk_forward_folds(
    sessions: pd.DatetimeIndex,
    *,
    train_years: int,
    validation_years: int,
    test_years: int,
    step_years: int = 1,
    embargo_sessions: int = 0,
    start=None,
) -> list[Fold]:
    """Chronological folds over ``sessions``; every bound is an actual session date."""
    if min(train_years, validation_years, test_years, step_years) < 1 or embargo_sessions < 0:
        raise ValueError("window lengths must be >= 1 year and embargo_sessions >= 0")
    if len(sessions) == 0:
        return []

    origin = to_date(start) if start is not None else sessions[0]
    last = sessions[-1]

    def plus_years(day: pd.Timestamp, years: int) -> pd.Timestamp:
        return day + pd.DateOffset(years=years)

    def window(begin: pd.Timestamp, end: pd.Timestamp, embargo: int) -> pd.DatetimeIndex:
        inside = sessions[(sessions >= begin) & (sessions < end)]
        return inside[embargo:]

    folds: list[Fold] = []
    while True:
        train_begin = plus_years(origin, len(folds) * step_years)
        validation_begin = plus_years(train_begin, train_years)
        test_begin = plus_years(validation_begin, validation_years)
        test_finish = plus_years(test_begin, test_years)
        if test_finish - pd.Timedelta(days=1) > last + pd.Timedelta(days=7):
            break  # the test window would run past the data
        train = window(train_begin, validation_begin, 0)
        validation = window(validation_begin, test_begin, embargo_sessions)
        test = window(test_begin, test_finish, embargo_sessions)
        if min(len(train), len(validation), len(test)) < 2:
            break
        folds.append(
            Fold(len(folds), train[0], train[-1], validation[0], validation[-1], test[0], test[-1])
        )
    return folds
