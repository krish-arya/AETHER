import pandas as pd
import pytest

from ml.data.ingestion.base import MarketDataProvider
from ml.data.ingestion.csv_provider import CsvDirectoryProvider
from ml.data.ingestion.synthetic import SyntheticDevProvider
from ml.data.preprocessing.clean import clean_bars
from ml.data.schemas import BAR_COLUMNS, ProviderError

from .conftest import END, START


def test_synthetic_provider_is_unmistakably_labelled(provider, securities):
    assert isinstance(provider, MarketDataProvider)
    assert provider.info.is_synthetic is True
    assert "not real market data" in provider.info.license_note
    assert securities["symbol"].str.startswith("SYNTH").all()
    assert securities["name"].str.startswith("Synthetic").all()
    assert securities["sector"].str.startswith("Synthetic").all()


def test_synthetic_provider_is_deterministic(provider, symbols):
    twin = SyntheticDevProvider(seed=11, n_symbols=4, history_start=START, history_end=END)
    pd.testing.assert_frame_equal(
        provider.get_daily_bars(symbols, START, END), twin.get_daily_bars(symbols, START, END)
    )
    pd.testing.assert_frame_equal(
        provider.get_corporate_actions(symbols, START, END),
        twin.get_corporate_actions(symbols, START, END),
    )


def test_different_seed_gives_different_data(provider, symbols):
    other = SyntheticDevProvider(seed=12, n_symbols=4, history_start=START, history_end=END)
    a = provider.get_daily_bars(symbols, START, END)
    b = other.get_daily_bars(symbols, START, END)
    assert not a["close"].reset_index(drop=True).equals(b["close"].reset_index(drop=True))


def test_slices_do_not_depend_on_what_was_requested(provider, symbols):
    full = provider.get_daily_bars(symbols, START, END)
    part = provider.get_daily_bars(["SYNTH02"], "2020-06-01", "2020-12-31")
    expected = full[
        (full["symbol"] == "SYNTH02")
        & (full["trade_date"] >= "2020-06-01")
        & (full["trade_date"] <= "2020-12-31")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(part, expected)


def test_synthetic_bars_pass_every_hard_check(raw_bars, securities, calendar):
    result = clean_bars(raw_bars, securities=securities, calendar=calendar, as_of=END)
    assert result.quarantine.empty
    assert len(result.bars) == len(raw_bars)
    assert list(raw_bars.columns) == BAR_COLUMNS


def test_synthetic_universe_contains_late_listing_and_delisting(securities, raw_bars):
    late = securities[securities["listing_date"].notna()]
    gone = securities[securities["delisting_date"].notna()]
    assert len(late) == 1 and len(gone) == 1
    late_symbol, gone_symbol = late["symbol"].iloc[0], gone["symbol"].iloc[0]
    first_bar = raw_bars.loc[raw_bars["symbol"] == late_symbol, "trade_date"].min()
    last_bar = raw_bars.loc[raw_bars["symbol"] == gone_symbol, "trade_date"].max()
    assert first_bar == late["listing_date"].iloc[0]
    assert last_bar < gone["delisting_date"].iloc[0]


def test_synthetic_calendar_has_weekday_holidays(provider):
    sessions = provider.get_trading_sessions(START, END)
    weekdays = pd.bdate_range(START, END)
    assert (sessions.dayofweek < 5).all()
    assert 0 < len(weekdays) - len(sessions) <= 24  # up to 12 holidays per year


def test_unknown_symbol_is_an_error_not_an_empty_frame(provider):
    with pytest.raises(ProviderError):
        provider.get_daily_bars(["RELIANCE"], START, END)


# ------------------------------------------------------------------ CSV provider


def _export(provider, symbols, directory):
    provider.get_daily_bars(symbols, START, END).to_csv(directory / "bars.csv", index=False)
    provider.get_corporate_actions(symbols, START, END).to_csv(
        directory / "corporate_actions.csv", index=False
    )
    provider.list_securities().to_csv(directory / "securities.csv", index=False)


def test_csv_provider_round_trips_data(provider, symbols, tmp_path):
    _export(provider, symbols, tmp_path)
    csv = CsvDirectoryProvider(tmp_path, source_name="unit-test-fixture", license_note="test only")
    assert csv.info.is_synthetic is False
    assert csv.info.name == "unit-test-fixture"

    pd.testing.assert_frame_equal(
        csv.get_daily_bars(symbols, START, END),
        provider.get_daily_bars(symbols, START, END),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        csv.get_corporate_actions(symbols, START, END),
        provider.get_corporate_actions(symbols, START, END),
        check_dtype=False,
    )
    assert list(csv.list_securities()["symbol"]) == symbols
    assert csv.get_trading_sessions(START, END) is None  # no sessions.csv supplied


def test_csv_provider_requires_declared_provenance(tmp_path):
    (tmp_path / "bars.csv").write_text("symbol,trade_date,open,high,low,close,volume\n")
    with pytest.raises(ValueError, match="source_name and license_note"):
        CsvDirectoryProvider(tmp_path, source_name="", license_note="")


def test_csv_provider_rejects_missing_file_and_columns(tmp_path):
    with pytest.raises(ProviderError, match="bars.csv not found"):
        CsvDirectoryProvider(tmp_path, source_name="x", license_note="y")
    (tmp_path / "bars.csv").write_text("symbol,trade_date,close\nA,2020-01-01,1\n")
    csv = CsvDirectoryProvider(tmp_path, source_name="x", license_note="y")
    with pytest.raises(ProviderError, match="missing required columns"):
        csv.get_daily_bars(["A"], START, END)


def test_csv_provider_does_not_repair_bad_rows(tmp_path):
    (tmp_path / "bars.csv").write_text(
        "symbol,trade_date,open,high,low,close,volume\n"
        "A,2020-01-02,10,11,9,10.5,100\n"
        "A,not-a-date,10,11,9,10.5,100\n"
        "A,2020-01-03,,11,9,10.5,100\n"
    )
    csv = CsvDirectoryProvider(tmp_path, source_name="x", license_note="y")
    bars = csv.get_daily_bars(["A"], START, END)
    assert len(bars) == 3  # defects are passed through for validation to report
    assert bars["trade_date"].isna().sum() == 1
    assert bars["open"].isna().sum() == 1
