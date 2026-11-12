"""Centralised logging configuration (single stdout handler, structured-ish lines)."""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_aether_configured", False):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.handlers = [handler]
    root.setLevel(level.upper())
    root._aether_configured = True  # type: ignore[attr-defined]
