"""Evaluate the configured LLM on the labelled fictional news cases.

    uv run python scripts/evaluate_llm.py
    uv run python scripts/evaluate_llm.py --model qwen2.5:1.5b

Writes artifacts/llm_eval/<run_id>/report.json and prints the metrics. Every company
in the evaluation set is invented; see ml/llm/evaluation/dataset.py.
"""

import argparse
import sys
from pathlib import Path

from ml.data.versioning import write_json
from ml.llm.config import LLMConfig, LLMEnvironment
from ml.llm.evaluation.suite import evaluate
from ml.llm.providers import LLMError, OpenAICompatibleClient


def _progress(index: int, total: int, result: dict) -> None:
    if result["kind"] == "injection":
        verdict = "resisted" if result["injection_resisted"] else "OBEYED INJECTION"
    elif result["kind"] in ("insufficient", "no_entity"):
        verdict = "handled" if result["handled"] else "NOT handled"
    else:
        verdict = (
            f"sentiment {'ok' if result['sentiment_correct'] else 'WRONG'}, "
            f"type {'ok' if result['event_type_correct'] else 'WRONG'}"
        )
    rejected = f"  rejected: {', '.join(result['rejections'])}" if result["rejections"] else ""
    print(f"  [{index:>2}/{total}] {result['case_id']:<24}{verdict}{rejected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/llm.yaml"))
    parser.add_argument("--model", default=None, help="Override the configured model")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/llm_eval"))
    args = parser.parse_args()

    environment = LLMEnvironment()
    config = environment.apply(LLMConfig.from_yaml(args.config)).with_provider(model=args.model)
    client = OpenAICompatibleClient.from_config(config.provider, environment.key)
    print(f"provider : {client.provider}  model: {client.model}  ({config.provider.base_url})")
    try:
        report = evaluate(client, config.extraction, progress=_progress)
    except LLMError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    report["config_hash"] = config.config_hash()
    report["extraction_config"] = config.extraction.model_dump(mode="json")
    stamp = report["evaluated_at"][:19].replace("-", "").replace(":", "")
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in client.model)
    path = args.output_root / f"eval-{stamp}Z-{safe_model}" / "report.json"
    write_json(report, path)

    print(f"\nreport   : {path}")
    print(f"cases    : {report['counts']}")
    for name, value in report["metrics"].items():
        shown = (
            "n/a" if value is None else (f"{value:.2f}s" if "latency" in name else f"{value:.1%}")
        )
        print(f"  {name:<38}{shown:>8}")
    print(f"  injection resisted by: {report['injection_resisted_by']}")
    for reason, count in report["rejections_by_reason"].items():
        print(f"  rejected  {reason:<28}{count:>4}")
    for flag, count in report["flags_by_flag"].items():
        print(f"  flagged   {flag:<28}{count:>4}")
    print("\nSmall hand-labelled fictional set: a smoke test, not a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
