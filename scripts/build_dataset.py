"""Build a versioned market dataset (ingest -> validate -> adjust -> features).

    uv run python scripts/build_dataset.py --config configs/data.yaml

Exit code 0 on success, 2 when the data fails a quality gate (nothing is written).
"""

import argparse
import json
import sys
from pathlib import Path

from ml.data.config import DataConfig
from ml.data.pipeline import DataQualityError, build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path, default=None, help="Override output.root")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    try:
        result = build_dataset(cfg, output_root=args.output_root)
    except DataQualityError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        print(json.dumps(error.report.summary(), indent=2), file=sys.stderr)
        return 2

    manifest = result.manifest
    provider = manifest["provider"]
    if provider["is_synthetic"]:
        print("*** SYNTHETIC DEVELOPMENT DATA -- not real market data ***")
    print(f"dataset_id          : {result.dataset_id}")
    print(f"path                : {result.path}")
    print(f"provider            : {provider['name']}")
    print(
        f"date range          : {manifest['date_range']['first_bar']} .. "
        f"{manifest['date_range']['last_bar']}"
    )
    print(
        f"securities / bars   : {manifest['counts']['securities']} / {manifest['counts']['bars']}"
    )
    print(f"corporate actions   : {manifest['counts']['corporate_actions']}")
    print(
        f"features            : {len(manifest['feature_columns'])} columns "
        f"(feature set {manifest['feature_set_version']})"
    )
    print(f"quarantined rows    : {manifest['counts']['quarantined']}")
    print("quality by check    :")
    for check, count in manifest["quality"]["by_check"].items():
        if not check.startswith("feature_undefined"):
            print(f"    {check:<28} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
