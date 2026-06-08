"""Failure-suite, judge, synth, and end-to-end harness tests."""

from __future__ import annotations

import re

from eval import failure_suites
from eval.judge import HeuristicJudge
from eval.model_under_test import GoldenModel, NaiveModel, build_model
from ndis.notes import is_gap
from ndis.splits import stratified_split
from ndis.synth_generate import STRATUM_PLAN, generate_dataset


def _golden_over_suites() -> GoldenModel:
    return GoldenModel(failure_suites.ideal_examples())


def test_golden_clears_all_failure_suites():
    model = _golden_over_suites()
    judge = HeuristicJudge()
    results = {r.name: r for r in failure_suites.run_all(model, judge)}
    for name, res in results.items():
        assert res.pass_rate == 1.0, f"{name} should be 100% for golden, got {res.pass_rate}"


def test_naive_fails_fabrication_and_pii():
    model = NaiveModel()
    judge = HeuristicJudge()
    results = {r.name: r for r in failure_suites.run_all(model, judge)}
    assert results["pii_leakage"].pass_rate < 1.0
    assert results["fabrication"].pass_rate < 1.0


def test_heuristic_judge_catches_fabricated_number():
    judge = HeuristicJudge()
    note = {"duration_minutes": 999, "narrative_summary": "Session ran for 999 minutes."}
    check = judge.judge_faithfulness("session about 60 min", note)
    assert not check.is_faithful


def test_heuristic_judge_register_flags_fillers():
    judge = HeuristicJudge()
    bad = {"narrative_summary": "um so yeah i did the thing"}
    assert not judge.judge_register(bad).meets_register


def test_offline_synth_is_faithful():
    """Every number/date/code in an offline target must be grounded in its input."""
    records = generate_dataset({s: 2 for s in STRATUM_PLAN}, offline=True, seed=7)
    assert records
    judge = HeuristicJudge()
    for rec in records:
        check = judge.judge_faithfulness(rec["input"], rec["target"])
        assert check.is_faithful, f"{rec['id']} not faithful: {check.fabricated_claims}"


def test_offline_adv_pii_carries_forbidden_strings():
    records = generate_dataset({"adv_pii_check": 3}, offline=True, seed=3)
    for rec in records:
        assert rec.get("forbidden_pii"), "adv_pii records must list forbidden strings"
        # The injected PII must NOT be present in the target.
        text = str(rec["target"])
        for item in rec["forbidden_pii"]:
            assert item not in text


def test_sparse_targets_flag_gaps():
    # Sparse inputs omit a *varied* set of fields; every sparse target must flag
    # at least one mandatory field as a gap rather than invent it.
    from ndis.synth_generate import MISSABLE_FIELDS

    records = generate_dataset({"sparse_input": 5}, offline=True, seed=5)
    assert records
    for rec in records:
        assert any(is_gap(rec["target"][f]) for f in MISSABLE_FIELDS), (
            f"sparse target has no flagged gap: {rec['target']}"
        )


def test_adv_missing_rotates_across_fields():
    # The adv_missing stratum must exercise more than just duration/location,
    # or the model never learns to flag other missing mandatory fields.
    records = generate_dataset({"adv_missing_field": 40}, offline=True, seed=2)
    flagged = {f for r in records for f in r.get("missing_fields", [])}
    assert len(flagged) >= 5, f"expected varied missing fields, got {flagged}"


def test_stratified_split_freezes_every_stratum():
    records = generate_dataset({s: 6 for s in STRATUM_PLAN}, offline=True, seed=11)
    splits = stratified_split(records, seed=11)
    test_strata = {r["stratum"] for r in splits["test"]}
    all_strata = {r["stratum"] for r in records}
    assert test_strata == all_strata  # every stratum represented in frozen test
    # No leakage between splits (ids disjoint).
    ids = [set(r["id"] for r in splits[s]) for s in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])


def test_end_to_end_dummy_is_release_ok(tmp_path):
    from eval.eval_suite import score_example
    from eval.scorecard import build_scorecard

    examples = generate_dataset({s: 2 for s in STRATUM_PLAN}, offline=True, seed=13)
    golden = build_model("dummy", examples + failure_suites.ideal_examples())
    judge = HeuristicJudge()
    scores = [score_example(ex, golden, judge) for ex in examples]
    suites = failure_suites.run_all(golden, judge)
    card = build_scorecard(
        model_name=golden.name,
        judge_name=judge.name,
        split="test",
        scores=scores,
        suites=suites,
    )
    assert card.release_ok
    assert all(re.match(r"\d", str(g.value * 100)) for g in card.gates)
