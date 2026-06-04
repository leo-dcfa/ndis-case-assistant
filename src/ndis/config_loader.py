"""Cached access to the parsed application config.

:func:`ndis.models.load_config` reads and parses the YAML on every call. The
eval harness touches the config once per scored field, so this wraps it in an
LRU cache. Pass a ``config_dir`` to load an alternate config (tests); the
default is cached.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

from ndis.models import AppConfig, load_config


@lru_cache(maxsize=8)
def _cached(config_dir: pathlib.Path | None) -> AppConfig:
    return load_config(config_dir)


def get_config(config_dir: pathlib.Path | None = None) -> AppConfig:
    return _cached(config_dir)
