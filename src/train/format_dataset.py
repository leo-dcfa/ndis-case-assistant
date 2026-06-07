"""Convert {input, target} JSONL splits into chat-format SFT examples.

Each record becomes a 3-turn chat (system / user / assistant) using the SAME
prompt contract the served model uses (``ndis.prompts``), so what the model
learns is exactly what it sees at inference. Returns HF ``datasets`` with a
``messages`` column that TRL's SFTTrainer consumes directly.
"""

from __future__ import annotations

import pathlib
from typing import Any

from ndis.prompts import build_chat
from ndis.splits import read_jsonl


def to_chat_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": build_chat(r["input"], r["target"])} for r in records]


def load_chat_dataset(splits_dir: pathlib.Path, split: str) -> Any:
    from datasets import Dataset

    records = read_jsonl(splits_dir / f"{split}.jsonl")
    if not records:
        raise FileNotFoundError(f"no records in {splits_dir / f'{split}.jsonl'}")
    return Dataset.from_list(to_chat_records(records))
