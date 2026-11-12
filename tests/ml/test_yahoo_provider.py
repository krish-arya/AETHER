"""Yahoo chart provider, tested offline against fabricated responses in Yahoo's JSON
shape (no real market data is stored in the repository)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from ml.data.config import DataConfig
from ml.data.ingestion.yahoo import YahooChartProvider, load_universe
from ml.data.pipeline import build_dataset
from ml.data.schemas import ProviderError

REPO_ROOT = Path(__file__).resolve().parents[2]
IST = 19800
START, END = "2024-01-01", "2024-12-31"


def _epoch(day: str) -> int:
    """09:15 IST on ``day`` as epoch seconds (how Yahoo stamps NSE daily bars)."""
    return int(pd.Timestamp(day).timestamp()) - IST + 9 * 3600 + 15 * 60


def _payload(symbol: str, days: list[str], *, base: float = 100.0, events=None, nulls=()) -> dict:
    closes = [None if i in nulls else round(base + i, 2) for i in range(len(days))]
    quote = {
        "open": [None if c is None else c - 0.5 for c in closes],
        "high": [None if c is None else c + 1.0 for c in closes],
        "low": [None if c is None else c - 1.0 for c in closes],
        "close": closes,
        "volume": [None if c is None else 1000 + i for i, c in enumerate(closes)],
    }
    result = {
        "meta": {"symbol": f"{symbol}.NS", "gmtoffset": IST},
        "timestamp": [_epoch(day) for day in days],
        "indicators": {"quote": [quote]},
    }
    if events:
        result["events"] = events
    return {"chart": {"result": [result], "error": None}}


def _universe(*symbols: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "name": list(symbols),
            "sector": "Test",
            "industry": None,
            "isin": None,
            "listing_date": pd.NaT,
            "delisting_date": pd.NaT,
        }
    )


class FakeYahoo:
    """httpx transport that serves canned payloads and records every request."""

    def __init__(self, payloads: dict[str, dict], failures: list[int] | None = None) -> None:
        self.payloads, self.failures, self.requests = payloads, list(failures or []), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            return httpx.Response(self.failures.pop(0))
        ticker = request.url.path.rsplit("/", 1)[-1]
        if ticker not in self.payloads:
            missing = {"chart": {"result": None, "error": {"code": "Not Found"}}}
            return httpx.Response(404, json=missing)
        return httpx.Response(200, json=self.payloads[ticker])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _provider(fake: FakeYahoo, cache: Path, *symbols: str, **kwargs) -> YahooChartProvider:
    return YahooChartProvider(
        _universe(*symbols), cache, client=fake.client(), pause_seconds=0.0, **kwargs
    )


DAYS = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]
EVENTS = {
    "dividends": {"1": {"amount": 2.5, "date": _epoch("2024-03-06")}},
    "splits": {"2": {"date": _epoch("2024-03-05"), "numerator": 2.0, "denominator": 1.0}},
}


def test_bars_are_parsed_with_exchange_local_dates(tmp_path):
    fake = FakeYahoo({"AAA.NS": _payload("AAA", DAYS)})
    bars = _provider(fake, tmp_path, "AAA").get_daily_bars(["AAA"], START, END)

    assert list(bars["trade_date"]) == [pd.Timestamp(day) for day in DAYS]
    first = bars.loc[0, ["open", "high", "low", "close", "volume"]].tolist()
    assert first == [99.5, 101, 99, 100, 1000]
    assert bars[["turnover", "delivery_volume", "market_cap"]].isna().all().all()
    request = fake.requests[0]
    assert request.url.params["interval"] == "1d"
    assert request.url.params["events"] == "div,splits"


def test_provenance_is_declared_honestly(tmp_path):
    info = _provider(FakeYahoo({}), tmp_path, "AAA").info
    assert info.name == "yahoo_chart"
    assert info.is_synthetic is False
    assert info.price_basis == "split_adjusted"
    assert "survivorship" in info.license_note
    assert "not for redistribution" in info.license_note


def test_only_dividends_are_actions_because_prices_are_already_split_adjusted(tmp_path):
    fake = FakeYahoo({"AAA.NS": _payload("AAA", DAYS, events=EVENTS)})
    provider = _provider(fake, tmp_path, "AAA")

    actions = provider.get_corporate_actions(["AAA"], START, END)
    assert actions[["symbol", "action_type", "value"]].values.tolist() == [["AAA", "dividend", 2.5]]
    assert actions["ex_date"].iloc[0] == pd.Timestamp("2024-03-06")

    splits = provider.get_splits(["AAA"], START, END)
    assert splits[["action_type", "value"]].values.tolist() == [["split", 2.0]]


def test_rows_without_any_price_are_dropped_but_partial_rows_are_passed_through(tmp_path):
    payload = _payload("AAA", DAYS, nulls=(1,))
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"][2] = None  # partial row
    provider = _provider(FakeYahoo({"AAA.NS": payload}), tmp_path, "AAA")
    bars = provider.get_daily_bars(["AAA"], START, END)
    assert len(bars) == 3  # the all-null row is "no bar", not a defect
    assert bars["open"].isna().sum() == 1  # the partial row reaches validation


def test_responses_are_cached_for_reproducible_rebuilds(tmp_path):
    fake = FakeYahoo({"AAA.NS": _payload("AAA", DAYS)})
    provider = _provider(fake, tmp_path, "AAA")
    first = provider.get_daily_bars(["AAA"], START, END)
    provider.get_corporate_actions(["AAA"], START, END)
    again = _provider(fake, tmp_path, "AAA").get_daily_bars(["AAA"], START, END)

    assert len(fake.requests) == 1  # later calls and later providers read the cache
    pd.testing.assert_frame_equal(first, again)
    assert [p.name for p in tmp_path.iterdir()] == ["AAA_20240101_20241231.json"]

    _provider(fake, tmp_path, "AAA", refresh=True).get_daily_bars(["AAA"], START, END)
    assert len(fake.requests) == 2


def test_transient_failures_are_retried(tmp_path):
    fake = FakeYahoo({"AAA.NS": _payload("AAA", DAYS)}, failures=[429, 503])
    bars = _provider(fake, tmp_path, "AAA").get_daily_bars(["AAA"], START, END)
    assert len(bars) == 4
    assert len(fake.requests) == 3


def test_persistent_failure_is_an_error_and_nothing_is_cached(tmp_path):
    fake = FakeYahoo({}, failures=[503] * 10)
    with pytest.raises(ProviderError, match="HTTP 503"):
        _provider(fake, tmp_path, "AAA", max_retries=2).get_daily_bars(["AAA"], START, END)
    assert len(fake.requests) == 3
    assert list(tmp_path.iterdir()) == []


def test_unknown_ticker_fails_fast_without_retrying(tmp_path):
    fake = FakeYahoo({})
    with pytest.raises(ProviderError, match="HTTP 404"):
        _provider(fake, tmp_path, "AAA").get_daily_bars(["AAA"], START, END)
    assert len(fake.requests) == 1


def test_error_payload_and_unlisted_symbols_are_rejected(tmp_path):
    broken = {"chart": {"result": None, "error": {"code": "Bad Request"}}}
    provider = _provider(FakeYahoo({"AAA.NS": broken}), tmp_path, "AAA")
    with pytest.raises(ProviderError, match="no data"):
        provider.get_daily_bars(["AAA"], START, END)
    with pytest.raises(ProviderError, match="not in the configured universe"):
        _provider(FakeYahoo({}), tmp_path, "AAA").get_daily_bars(["ZZZ"], START, END)


def test_special_characters_in_symbols_are_url_encoded(tmp_path):
    fake = FakeYahoo({"M&M.NS": _payload("M&M", DAYS)})
    bars = _provider(fake, tmp_path, "M&M").get_daily_bars(["M&M"], START, END)
    assert len(bars) == 4
    assert "M%26M.NS" in str(fake.requests[0].url)
    assert (tmp_path / "M_M_20240101_20241231.json").is_file()


def test_shipped_universe_file_is_valid():
    universe = load_universe(REPO_ROOT / "configs" / "universe_nse_largecap.yaml")
    assert len(universe) == 20
    assert universe["symbol"].is_unique
    assert universe["sector"].notna().all()
    assert universe["name"].notna().all()
    assert universe["delisting_date"].isna().all()  # hence the survivorship warning


def _yahoo_config(**yahoo) -> DataConfig:
    return DataConfig.model_validate(
        {
            "dataset": {
                "name": "yahoo-test",
                "start_date": date(2023, 1, 1),
                "end_date": date(2024, 12, 31),
            },
            "provider": {"kind": "yahoo_chart", "yahoo_chart": yahoo},
        }
    )


def test_pipeline_builds_a_dataset_and_flags_placeholders_and_survivorship(tmp_path):
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-01-02", "2024-12-31")]
    payloads = {}
    for k, symbol in enumerate(("AAA", "BBB", "CCC")):
        payload = _payload(symbol, days, base=100.0 * (k + 1), events=EVENTS if k == 0 else None)
        quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
        # A holiday on which the source emits a filler row: previous close, no volume.
        for field in ("open", "high", "low", "close"):
            quote[field][300] = quote["close"][299]
        quote["volume"][300] = 0
        payloads[f"{symbol}.NS"] = payload

    provider = _provider(FakeYahoo(payloads), tmp_path / "cache", "AAA", "BBB", "CCC")
    result = build_dataset(_yahoo_config(), provider=provider, output_root=tmp_path / "out")

    manifest = result.manifest
    assert manifest["provider"]["price_basis"] == "split_adjusted"
    assert manifest["provider"]["is_synthetic"] is False
    assert manifest["calendar"]["source"] == "inferred"
    assert manifest["quality"]["by_check"]["non_session_date"] == 3  # the filler day
    assert manifest["quality"]["by_check"]["survivorship_unverified"] == 3
    assert pd.Timestamp(days[300]) not in set(result.bars["trade_date"])
    actions = pd.read_csv(result.path / "corporate_actions.csv")
    assert list(actions["action_type"]) == ["dividend"]  # the split was not re-applied
    on_disk = json.loads((result.path / "manifest.json").read_text())
    assert on_disk["counts"]["securities"] == 3


def test_config_hash_ignores_cache_location(tmp_path):
    base = _yahoo_config()
    assert base.config_hash() == _yahoo_config(cache_dir=str(tmp_path), refresh=True).config_hash()
    assert base.config_hash() != _yahoo_config(exchange_suffix=".BO").config_hash()
