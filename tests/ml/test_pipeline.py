"""End-to-end dataset build: reproducibility, provenance, integrity and quality gates."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml.data.config import DataConfig
from ml.data.ingestion.csv_provider import CsvDirectoryProvider
from ml.data.ingestion.synthetic import SyntheticDevProvider
from ml.data.pipeline import DataQualityError, build_dataset
from ml.data.versioning import combine_hashes, read_frame, read_manifest, verify_dataset

from .conftest import END, START, make_config

DATA_FILES = {
    "securities.csv",
    "bars.csv",
    "corporate_actions.csv",
    "features.csv",
    "regimes.csv",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_dataset(make_config(), output_root=tmp_path_factory.mktemp("datasets"))


def test_snapshot_layout_and_manifest(built):
    files = {p.name for p in built.path.iterdir()}
    assert files == DATA_FILES | {"manifest.json", "quality_report.json", "quarantine.csv"}

    manifest = read_manifest(built.path)
    assert manifest["dataset_id"] == built.dataset_id == built.path.name
    assert manifest["provider"]["is_synthetic"] is True
    assert manifest["provider"]["name"] == "synthetic_dev"
    assert set(manifest["files"]) == DATA_FILES
    assert manifest["counts"]["bars"] == len(built.bars) > 1500
    assert manifest["counts"]["quarantined"] == 0
    assert manifest["calendar"]["source"] == "provider"
    assert manifest["config"]["provider"]["synthetic_dev"]["seed"] == 11
    assert manifest["environment"]["pandas"] == pd.__version__
    assert len(manifest["feature_columns"]) == 32


def test_config_serialisation_is_platform_independent():
    cfg = make_config()
    dumped = json.dumps(cfg.model_dump(mode="json"))
    assert "\\\\" not in dumped  # no Windows path separators
    assert cfg.model_dump(mode="json")["provider"]["csv_directory"]["path"] == "data/licensed"
    assert cfg.model_dump(mode="json")["output"]["root"] == "artifacts/datasets"
    # The output location is not part of the dataset's identity.
    elsewhere = make_config()
    elsewhere.output.root = elsewhere.output.root / "somewhere-else"
    assert elsewhere.config_hash() == cfg.config_hash()


def test_manifest_records_platform_and_input_hash(built):
    manifest = built.manifest
    assert manifest["environment"]["platform"]
    derived = ("features.csv", "regimes.csv")
    inputs = {k: v for k, v in manifest["files"].items() if k not in derived}
    assert manifest["data_hash"] == combine_hashes(inputs)
    assert manifest["data_hash"] != manifest["content_hash"]


def test_build_is_reproducible(built, tmp_path):
    again = build_dataset(make_config(), output_root=tmp_path)
    assert again.dataset_id == built.dataset_id
    assert again.manifest["files"] == built.manifest["files"]
    assert again.manifest["content_hash"] == built.manifest["content_hash"]
    assert again.manifest["config_hash"] == built.manifest["config_hash"]


def test_rebuilding_into_the_same_directory_is_idempotent(tmp_path):
    first = build_dataset(make_config(), output_root=tmp_path)
    second = build_dataset(make_config(), output_root=tmp_path)
    assert first.path == second.path
    assert [p.name for p in tmp_path.iterdir()] == [first.dataset_id]  # no staging leftovers


def test_any_change_produces_a_new_dataset_id(built, tmp_path):
    other_seed = DataConfig.model_validate(
        {
            **make_config().model_dump(),
            "provider": {"kind": "synthetic_dev", "synthetic_dev": {"seed": 12, "n_symbols": 4}},
        }
    )
    other_features = make_config()
    other_features.features.rsi_window = 7

    ids = {
        built.dataset_id,
        build_dataset(other_seed, output_root=tmp_path).dataset_id,
        build_dataset(other_features, output_root=tmp_path).dataset_id,
    }
    assert len(ids) == 3
    assert other_features.config_hash() != make_config().config_hash()


def test_integrity_check_detects_tampering(tmp_path):
    result = build_dataset(make_config(), output_root=tmp_path)
    assert verify_dataset(result.path)["dataset_id"] == result.dataset_id

    bars_file = result.path / "bars.csv"
    text = bars_file.read_text()
    bars_file.write_text(text.replace("SYNTH01", "SYNTH99", 1))
    with pytest.raises(ValueError, match="modified after it was built"):
        verify_dataset(result.path)


def test_written_files_round_trip(built):
    bars = read_frame(built.path / "bars.csv", ("trade_date",))
    features = read_frame(built.path / "features.csv", ("trade_date",))
    pd.testing.assert_frame_equal(bars, built.bars.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(features, built.features, check_dtype=False, check_exact=True)

    report = json.loads((built.path / "quality_report.json").read_text())
    assert report["summary"] == built.manifest["quality"]
    assert report["summary"]["rows_in"] == report["summary"]["rows_out"]


def test_point_in_time_as_of_excludes_later_bars_and_actions(tmp_path):
    cfg = make_config(as_of=date(2021, 3, 31))
    cfg.validation.max_quarantine_fraction = 1.0  # later rows are rejected on purpose
    result = build_dataset(cfg, output_root=tmp_path)

    cutoff = pd.Timestamp("2021-03-31")
    assert result.bars["trade_date"].max() <= cutoff
    quarantine = pd.read_csv(result.path / "quarantine.csv")
    assert set(quarantine["reason"]) == {"future_dated"}
    actions = read_frame(result.path / "corporate_actions.csv", ("ex_date",))
    assert actions["ex_date"].max() <= cutoff
    assert result.report.has("action_after_as_of")
    assert result.manifest["date_range"]["as_of"] == "2021-03-31"


class _DirtyProvider(SyntheticDevProvider):
    """Synthetic provider that corrupts a share of its rows."""

    def __init__(self, bad_fraction: float) -> None:
        super().__init__(seed=11, n_symbols=4, history_start=START, history_end=END)
        self.bad_fraction = bad_fraction

    def get_daily_bars(self, symbols, start, end):
        bars = super().get_daily_bars(symbols, start, end).copy()
        bad = bars.sample(frac=self.bad_fraction, random_state=1).index
        bars.loc[bad, "high"] = bars.loc[bad, "low"] * 0.5
        return bars


def test_quality_gate_refuses_to_write_a_bad_dataset(tmp_path):
    with pytest.raises(DataQualityError) as raised:
        build_dataset(make_config(), provider=_DirtyProvider(0.05), output_root=tmp_path)
    assert "quarantined" in str(raised.value)
    assert raised.value.report.count("invalid_ohlcv") > 0
    assert not any(p.name.startswith("test-synthetic") for p in tmp_path.iterdir())


def test_small_defect_rate_passes_gate_and_is_reported(tmp_path):
    result = build_dataset(make_config(), provider=_DirtyProvider(0.002), output_root=tmp_path)
    quarantined = result.manifest["counts"]["quarantined"]
    assert 0 < quarantined < 10
    assert result.manifest["quality"]["by_check"]["invalid_ohlcv"] == quarantined
    assert result.manifest["quality"]["by_check"]["missing_session"] == quarantined
    assert len(pd.read_csv(result.path / "quarantine.csv")) == quarantined


def test_unknown_configured_symbol_is_rejected(tmp_path):
    cfg = make_config()
    cfg.universe.symbols = ["SYNTH01", "RELIANCE"]
    with pytest.raises(ValueError, match="not offered by the provider"):
        build_dataset(cfg, output_root=tmp_path)


def test_universe_can_be_restricted(tmp_path):
    cfg = make_config()
    cfg.universe.symbols = ["synth01", "SYNTH02"]
    result = build_dataset(cfg, output_root=tmp_path)
    assert result.manifest["symbols"] == ["SYNTH01", "SYNTH02"]
    assert set(result.bars["symbol"]) == {"SYNTH01", "SYNTH02"}
    # No delisted name in this universe -> survivorship warning is raised.
    assert result.report.has("survivorship_unverified")


def test_csv_provider_reproduces_the_same_adjusted_data(built, provider, symbols, tmp_path):
    """Same raw data via a different provider (and an inferred calendar) -> same bars."""
    source = tmp_path / "source"
    source.mkdir()
    provider.get_daily_bars(symbols, START, END).to_csv(source / "bars.csv", index=False)
    provider.get_corporate_actions(symbols, START, END).to_csv(
        source / "corporate_actions.csv", index=False
    )
    provider.list_securities().to_csv(source / "securities.csv", index=False)

    cfg = DataConfig.model_validate(
        {
            "dataset": {"name": "test-csv", "start_date": START, "end_date": END},
            "provider": {
                "kind": "csv_directory",
                "csv_directory": {
                    "path": str(source),
                    "source_name": "unit-test-fixture",
                    "license_note": "generated test fixture",
                },
            },
        }
    )
    result = build_dataset(cfg, output_root=tmp_path / "out")

    assert result.manifest["provider"] == {
        "name": "unit-test-fixture",
        "is_synthetic": False,
        "license_note": "generated test fixture",
        "price_basis": "raw",
    }
    assert result.manifest["calendar"]["source"] == "inferred"
    assert result.manifest["counts"]["quarantined"] == 0
    np.testing.assert_allclose(result.bars["adj_close"], built.bars["adj_close"], rtol=1e-12)
    np.testing.assert_allclose(
        result.features["rsi_14"], built.features["rsi_14"], rtol=1e-9, equal_nan=True
    )


def test_csv_provider_instance_is_used_for_csv_kind(tmp_path):
    (tmp_path / "bars.csv").write_text("symbol,trade_date,open,high,low,close,volume\n")
    assert CsvDirectoryProvider(tmp_path, "x", "y").info.is_synthetic is False
