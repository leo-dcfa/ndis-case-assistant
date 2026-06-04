"""Train / validation / held-out-test splitting (PRD §5.4).

The test split is *frozen*: it is written once and must never be seen during
training or hyperparameter selection. Splitting is stratified so every case
type is represented in every split, and deterministic given a seed so a run is
reproducible.
"""

from __future__ import annotations

import json
import pathlib
import random
from collections import defaultdict
from typing import Any

SPLIT_FILES = {
    "seed": "seed.jsonl",
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def write_jsonl(records: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stratified_split(
    records: list[dict[str, Any]],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val/test, stratified by ``record["stratum"]``.

    Each stratum is shuffled deterministically then sliced, so small strata
    still contribute to every split.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_stratum[rec.get("stratum", "unknown")].append(rec)

    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for stratum, items in sorted(by_stratum.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_frac)) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        test = items[:n_test]
        val = items[n_test : n_test + n_val]
        train = items[n_test + n_val :]
        out["test"].extend(test)
        out["val"].extend(val)
        out["train"].extend(train)
    return out


def write_splits(
    records: list[dict[str, Any]],
    out_dir: pathlib.Path,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Write seed.jsonl plus train/val/test splits. Returns per-split counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / SPLIT_FILES["seed"])
    splits = stratified_split(records, val_frac=val_frac, test_frac=test_frac, seed=seed)
    counts: dict[str, int] = {"seed": len(records)}
    for name, recs in splits.items():
        write_jsonl(recs, out_dir / SPLIT_FILES[name])
        counts[name] = len(recs)
    return counts
