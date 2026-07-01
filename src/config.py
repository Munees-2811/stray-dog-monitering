"""
Configuration loader.

Reads ``config/config.yaml`` once and exposes it as a plain nested dict via
``load_config()``. Every module reads its knobs from here so there is a single
source of truth for thresholds, model names and hardware settings.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_cache = None


def load_config(path=None, reload=False):
    """Return the parsed configuration dict (cached after the first read)."""
    global _cache
    if _cache is not None and not reload and path is None:
        return _cache

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if path is None:
        _cache = cfg
    return cfg
