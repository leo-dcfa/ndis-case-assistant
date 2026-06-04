"""Stratified test-set assembly (PRD §6.2).

Loads a split (default the frozen test split) and groups examples by stratum so
the harness can report a within-stratum failure rate, not just an aggregate.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Any

from ndis.splits import SPLIT_FILES, read_jsonl

DEFAULT_SPLITS_DIR = pathlib.Path("data/splits")

# Strata expected to exist (used for reporting coverage gaps).
EXPECTED_STRATA = [
    "routine_session",
    "incident",
    "capacity_building",
    "sparse_input",
    "adv_pii_check",
    "adv_missing_field",
]


def load_split(
    split: str = "test", splits_dir: pathlib.Path = DEFAULT_SPLITS_DIR
) -> list[dict[str, Any]]:
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r} (expected one of {list(SPLIT_FILES)})")
    return read_jsonl(splits_dir / SPLIT_FILES[split])


def group_by_stratum(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        out[ex.get("stratum", "unknown")].append(ex)
    return dict(out)


def coverage_report(examples: list[dict[str, Any]]) -> dict[str, int]:
    grouped = group_by_stratum(examples)
    return {s: len(grouped.get(s, [])) for s in EXPECTED_STRATA}
