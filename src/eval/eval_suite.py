"""Evaluation harness entry point (PRD §6 / Phase 1 DoD).

Runs a model-under-test over the held-out split and the adversarial failure
suites, scores every example on the automated + judged dimensions, and emits a
scorecard against the §2 bar.

    python -m eval.eval_suite --mode dummy --judge heuristic
    python -m eval.eval_suite --mode openai --judge llm --split test --html runs/sc.html

``--mode dummy`` uses the GoldenModel (harness self-test): it should clear every
gate, proving the pipeline end-to-end. ``--mode naive`` shows the gates failing.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from datetime import datetime
from typing import Any

from eval import failure_suites
from eval.judge import Judge, make_judge
from eval.model_under_test import ModelUnderTest, build_model
from eval.models import ExampleScore
from eval.rubric import score_pii, score_structure
from eval.scorecard import build_scorecard, render_console, write_html, write_json
from eval.strata import coverage_report, load_split
from ndis.splits import SPLIT_FILES


def score_example(example: dict[str, Any], model: ModelUnderTest, judge: Judge) -> ExampleScore:
    input_text = example["input"]
    stratum = example.get("stratum", "unknown")
    t0 = time.perf_counter()
    draft = model.draft_note(input_text, stratum=stratum)
    latency_s = time.perf_counter() - t0

    structure = score_structure(draft, stratum=stratum)
    pii = score_pii(draft, forbidden=example.get("forbidden_pii"))
    faithfulness = judge.judge_faithfulness(input_text, draft)
    register = judge.judge_register(draft)
    billable = judge.judge_billable(input_text, draft)

    return ExampleScore(
        id=example.get("id", "?"),
        stratum=stratum,
        structure=structure,
        pii=pii,
        faithfulness=faithfulness,
        register=register,
        billable=billable,
        latency_s=latency_s,
    )


def _bootstrap_offline(splits_dir: pathlib.Path, n: int = 60) -> None:
    """Generate a small offline dataset so the harness runs with no data present."""
    from ndis.splits import write_splits
    from ndis.synth_generate import STRATUM_PLAN, generate_dataset

    total = sum(STRATUM_PLAN.values())
    scale = n / total
    plan = {s: max(1, round(c * scale)) for s, c in STRATUM_PLAN.items()}
    print(
        f"[eval] no split found — bootstrapping ~{sum(plan.values())} offline records",
        file=sys.stderr,
    )
    records = generate_dataset(plan, offline=True)
    write_splits(records, splits_dir)


def run(args: argparse.Namespace) -> int:
    splits_dir = pathlib.Path(args.splits_dir)
    if not (splits_dir / SPLIT_FILES[args.split]).exists():
        _bootstrap_offline(splits_dir)

    examples = load_split(args.split, splits_dir)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print(f"[eval] split {args.split!r} is empty in {splits_dir}", file=sys.stderr)
        return 2

    coverage = coverage_report(examples)
    missing_strata = [s for s, n in coverage.items() if n == 0]
    if missing_strata:
        print(f"[eval] WARNING: strata with no examples: {missing_strata}", file=sys.stderr)

    # The dummy GoldenModel must know suite ideal answers too, so it can clear
    # the failure gates (a perfect-model self-test).
    golden_examples = examples + (failure_suites.ideal_examples() if args.mode == "dummy" else [])
    model = build_model(args.mode, golden_examples)
    judge = make_judge(args.judge)

    scores = [score_example(ex, model, judge) for ex in examples]
    suites = failure_suites.run_all(model, judge)

    card = build_scorecard(
        model_name=model.name,
        judge_name=judge.name,
        split=args.split,
        scores=scores,
        suites=suites,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    render_console(card)
    if args.out:
        write_json(card, pathlib.Path(args.out))
        print(f"[eval] wrote JSON scorecard -> {args.out}", file=sys.stderr)
    if args.html:
        write_html(card, pathlib.Path(args.html))
        print(f"[eval] wrote HTML scorecard -> {args.html}", file=sys.stderr)

    if args.fail_on_block and not card.release_ok:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NDIS case-note evaluation harness.")
    parser.add_argument(
        "--mode",
        default="dummy",
        choices=["dummy", "naive", "openai"],
        help="Model under test (dummy=golden self-test).",
    )
    parser.add_argument(
        "--judge",
        default="heuristic",
        choices=["heuristic", "llm"],
        help="Judge for fuzzy dimensions.",
    )
    parser.add_argument("--split", default="test", choices=list(SPLIT_FILES))
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of examples (0 = all).")
    parser.add_argument("--out", default=None, help="Write JSON scorecard to this path.")
    parser.add_argument("--html", default=None, help="Write HTML scorecard to this path.")
    parser.add_argument(
        "--fail-on-block", action="store_true", help="Exit non-zero if any hard gate fails."
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
