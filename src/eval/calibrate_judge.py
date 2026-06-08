"""Judge calibration (PRD §6.4).

Runs a judge over a set of human-labelled notes and reports judge–human
agreement (accuracy + Cohen's kappa) for the faithfulness and register
dimensions. Do **not** trust the LLM judge in release gating until agreement is
confirmed on a sample.

    python -m eval.calibrate_judge                 # heuristic judge
    python -m eval.calibrate_judge --judge llm     # local open-weight judge
    python -m eval.calibrate_judge --export runs/judge_review.jsonl

The ``--export`` form writes the judge's verdicts alongside the input/note so an
owner can hand-review a sample and extend ``human_labels.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from eval.judge import make_judge

LABELS_PATH = pathlib.Path(__file__).resolve().parent / "calibration" / "human_labels.jsonl"


def _load_labels(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _agreement(human: list[bool], judge: list[bool]) -> dict[str, float]:
    n = len(human)
    accuracy = sum(h == j for h, j in zip(human, judge)) / n if n else 1.0
    try:
        from sklearn.metrics import cohen_kappa_score

        # cohen_kappa is undefined when a rater is constant; guard it.
        if len(set(human)) < 2 or len(set(judge)) < 2:
            kappa = float("nan")
        else:
            kappa = float(cohen_kappa_score(human, judge))
    except Exception:
        kappa = float("nan")
    return {"n": n, "accuracy": accuracy, "cohen_kappa": kappa}


def run(args: argparse.Namespace) -> int:
    path = pathlib.Path(args.labels) if args.labels else LABELS_PATH
    if not path.exists():
        print(f"[calibrate] labels not found: {path}", file=sys.stderr)
        return 2
    rows = _load_labels(path)
    judge_kwargs: dict[str, Any] = {}
    if args.model:
        judge_kwargs["model"] = args.model
    if args.judge_url:
        judge_kwargs["base_url"] = args.judge_url
    judge = make_judge(args.judge, **judge_kwargs)

    h_faith, j_faith, h_reg, j_reg = [], [], [], []
    exported: list[dict[str, Any]] = []
    for row in rows:
        note, inp, human = row["note"], row["input"], row["human"]
        fc = judge.judge_faithfulness(inp, note)
        rc = judge.judge_register(note)
        h_faith.append(bool(human["faithful"]))
        j_faith.append(bool(fc.is_faithful))
        h_reg.append(bool(human["meets_register"]))
        j_reg.append(bool(rc.meets_register))
        exported.append(
            {
                "id": row.get("id"),
                "human": human,
                "judge": {"faithful": fc.is_faithful, "meets_register": rc.meets_register},
                "faithful_agree": bool(human["faithful"]) == fc.is_faithful,
                "register_agree": bool(human["meets_register"]) == rc.meets_register,
            }
        )

    faith = _agreement(h_faith, j_faith)
    reg = _agreement(h_reg, j_reg)

    print(f"Judge: {judge.name}   samples: {len(rows)}\n")
    print(f"{'Dimension':<16}{'Accuracy':>12}{'Cohen kappa':>14}")
    print("-" * 42)
    print(f"{'faithfulness':<16}{faith['accuracy'] * 100:>11.1f}%{faith['cohen_kappa']:>14.3f}")
    print(f"{'register':<16}{reg['accuracy'] * 100:>11.1f}%{reg['cohen_kappa']:>14.3f}")
    print()
    disagreements = [
        e["id"] for e in exported if not (e["faithful_agree"] and e["register_agree"])
    ]
    if disagreements:
        print(f"Disagreements on: {disagreements}")
    verdict = (
        "OK to trust" if faith["accuracy"] >= 0.9 and reg["accuracy"] >= 0.9 else "NOT calibrated"
    )
    print(f"Calibration verdict: {verdict} (require >=90% agreement before gating).")

    if args.export:
        out = pathlib.Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for e in exported:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[calibrate] wrote review file -> {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report judge-vs-human agreement.")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--model", default=None, help="Override judge model (llm only).")
    parser.add_argument(
        "--judge-url", default=None, help="OpenAI-compatible base_url (e.g. LM Studio on the Mac)."
    )
    parser.add_argument("--labels", default=None, help="Path to a human_labels.jsonl.")
    parser.add_argument("--export", default=None, help="Write per-example agreement to this path.")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
