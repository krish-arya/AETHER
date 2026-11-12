import pandas as pd

from ml.data.universe import check_survivorship, membership_mask, universe_as_of
from ml.data.validation.calendar import TradingCalendar
from ml.data.validation.report import DataQualityReport

from .conftest import END, START

# ---------------------------------------------------------------------- calendar


def test_weekday_calendar_excludes_weekends_and_holidays():
    calendar = TradingCalendar.from_weekdays("2024-01-01", "2024-01-14", holidays=["2024-01-03"])
    assert [d.strftime("%m-%d") for d in calendar.sessions] == [
        "01-01", "01-02", "01-04", "01-05", "01-08", "01-09", "01-10", "01-11", "01-12",
    ]  # fmt: skip
    assert not calendar.is_session("2024-01-03")  # holiday
    assert not calendar.is_session("2024-01-06")  # Saturday


def test_next_and_previous_session_skip_closed_days():
    calendar = TradingCalendar.from_weekdays("2024-01-01", "2024-01-14", holidays=["2024-01-08"])
    friday = pd.Timestamp("2024-01-05")
    # A decision made after Friday's close executes on Tuesday (Monday is a holiday).
    assert calendar.next_session(friday) == pd.Timestamp("2024-01-09")
    assert calendar.previous_session(friday) == pd.Timestamp("2024-01-04")
    assert calendar.next_session("2024-01-06") == pd.Timestamp("2024-01-09")
    assert calendar.next_session("2024-01-12") is None
    assert calendar.previous_session("2024-01-01") is None


def test_sessions_between_is_inclusive():
    calendar = TradingCalendar.from_weekdays("2024-01-01", "2024-01-14")
    assert len(calendar.sessions_between("2024-01-01", "2024-01-05")) == 5
    assert len(calendar.sessions_between("2024-01-06", "2024-01-07")) == 0


def test_calendar_inferred_from_bars_matches_the_true_calendar(raw_bars, calendar):
    # Includes a late listing and a delisting; neither may distort the inference.
    inferred = TradingCalendar.infer_from_bars(raw_bars, min_symbol_fraction=0.5)
    assert inferred.source == "inferred"
    pd.testing.assert_index_equal(inferred.sessions, calendar.sessions, check_names=False)


def test_inferred_calendar_ignores_stray_rows(raw_bars, calendar):
    stray = raw_bars.iloc[[0]].copy()
    stray["trade_date"] = pd.Timestamp("2020-06-06")  # Saturday, one symbol only
    inferred = TradingCalendar.infer_from_bars(pd.concat([raw_bars, stray]))
    assert not inferred.is_session("2020-06-06")
    assert len(inferred) == len(calendar)


def test_inferred_calendar_ignores_zero_volume_placeholder_days(raw_bars, calendar):
    """A holiday on which the source emits zero-volume filler rows is not a session."""
    holiday = pd.Timestamp("2020-06-07")  # a Sunday: certainly not in the true calendar
    filler = raw_bars[raw_bars["trade_date"] == raw_bars["trade_date"].iloc[100]].copy()
    filler["trade_date"] = holiday
    filler["volume"] = 0
    inferred = TradingCalendar.infer_from_bars(pd.concat([raw_bars, filler]))
    assert not inferred.is_session(holiday)
    pd.testing.assert_index_equal(inferred.sessions, calendar.sessions, check_names=False)


# ---------------------------------------------------------------------- universe


def test_universe_is_reconstructed_point_in_time(securities):
    late = securities[securities["listing_date"].notna()].iloc[0]
    gone = securities[securities["delisting_date"].notna()].iloc[0]
    everyone = set(securities["symbol"])

    at_start = set(universe_as_of(securities, START))
    assert at_start == everyone - {late["symbol"]}  # not yet listed

    day_before_listing = late["listing_date"] - pd.Timedelta(days=1)
    assert late["symbol"] not in universe_as_of(securities, day_before_listing)
    assert late["symbol"] in universe_as_of(securities, late["listing_date"])

    last_day = gone["delisting_date"] - pd.Timedelta(days=1)
    assert gone["symbol"] in universe_as_of(securities, last_day)
    assert gone["symbol"] not in universe_as_of(securities, gone["delisting_date"])

    assert set(universe_as_of(securities, END)) == everyone - {gone["symbol"]}


def test_every_bar_falls_inside_its_listing_window(raw_bars, securities):
    assert membership_mask(raw_bars, securities).all()


def test_survivorship_warning_when_no_delisted_securities(securities):
    report = DataQualityReport()
    check_survivorship(securities, report)
    assert not report.has("survivorship_unverified")

    survivors_only = securities.assign(delisting_date=pd.NaT)
    check_survivorship(survivors_only, report)
    assert report.has("survivorship_unverified")
