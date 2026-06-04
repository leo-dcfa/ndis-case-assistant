"""Scorecard: aggregate per-example + suite results against the PRD §2 bar.

Distinguishes *targets* (structural ≥ 98%, register ≥ 95%) from *hard gates*
(fabrication / PII / missing-element suites must be 100%). ``release_ok`` is
true only when every hard gate passes. Renders to console, JSON, and HTML.
"""

from __future__ import annotations

import html
import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from eval.models import ExampleScore, SuiteResult

# PRD §2 bar.
STRUCTURAL_TARGET = 0.98
REGISTER_TARGET = 0.95
HARD_GATE = 1.0


@dataclass
class Gate:
    name: str
    value: float
    threshold: float
    hard: bool

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "hard_gate": self.hard,
            "passed": self.passed,
        }


@dataclass
class Scorecard:
    model_name: str
    judge_name: str
    split: str
    n_examples: int
    gates: list[Gate] = field(default_factory=list)
    per_stratum: dict[str, dict[str, Any]] = field(default_factory=dict)
    suites: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def release_ok(self) -> bool:
        return all(g.passed for g in self.gates if g.hard)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "judge": self.judge_name,
            "split": self.split,
            "n_examples": self.n_examples,
            "release_ok": self.release_ok,
            "gates": [g.to_dict() for g in self.gates],
            "per_stratum": self.per_stratum,
            "suites": self.suites,
            **self.extra,
        }


def _rate(items: list[bool]) -> float:
    return sum(items) / len(items) if items else 1.0


def build_scorecard(
    *,
    model_name: str,
    judge_name: str,
    split: str,
    scores: list[ExampleScore],
    suites: list[SuiteResult],
    generated_at: str | None = None,
) -> Scorecard:
    structural = _rate([s.structure.passed for s in scores])
    faithful = _rate([s.faithfulness.is_faithful for s in scores])
    pii_clean = _rate([s.pii.is_clean for s in scores])
    register = _rate([s.register.meets_register for s in scores])
    billable = _rate([s.billable.present for s in scores])

    suite_by_name = {s.name: s for s in suites}

    def suite_rate(name: str) -> float:
        return suite_by_name[name].pass_rate if name in suite_by_name else 1.0

    gates = [
        Gate("structural_compliance", structural, STRUCTURAL_TARGET, hard=False),
        Gate("professional_register", register, REGISTER_TARGET, hard=False),
        Gate("strata_faithfulness", faithful, STRUCTURAL_TARGET, hard=False),
        Gate("strata_pii_clean", pii_clean, HARD_GATE, hard=False),
        Gate("strata_billable_evidence", billable, REGISTER_TARGET, hard=False),
        Gate("suite_fabrication", suite_rate("fabrication"), HARD_GATE, hard=True),
        Gate("suite_pii_leakage", suite_rate("pii_leakage"), HARD_GATE, hard=True),
        Gate("suite_missing_element", suite_rate("missing_element"), HARD_GATE, hard=True),
    ]

    # Per-stratum breakdown.
    grouped: dict[str, list[ExampleScore]] = defaultdict(list)
    for s in scores:
        grouped[s.stratum].append(s)
    per_stratum: dict[str, dict[str, Any]] = {}
    for stratum, items in sorted(grouped.items()):
        per_stratum[stratum] = {
            "n": len(items),
            "structural": round(_rate([i.structure.passed for i in items]), 4),
            "faithful": round(_rate([i.faithfulness.is_faithful for i in items]), 4),
            "pii_clean": round(_rate([i.pii.is_clean for i in items]), 4),
            "register": round(_rate([i.register.meets_register for i in items]), 4),
            "billable": round(_rate([i.billable.present for i in items]), 4),
            "overall_pass": round(_rate([i.passed for i in items]), 4),
        }

    extra: dict[str, Any] = {}
    if generated_at:
        extra["generated_at"] = generated_at

    return Scorecard(
        model_name=model_name,
        judge_name=judge_name,
        split=split,
        n_examples=len(scores),
        gates=gates,
        per_stratum=per_stratum,
        suites=[s.to_dict() for s in suites],
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_console(card: Scorecard) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule(f"[bold]NDIS Eval Scorecard — {card.model_name}")
    console.print(
        f"split=[cyan]{card.split}[/]  examples=[cyan]{card.n_examples}[/]  "
        f"judge=[cyan]{card.judge_name}[/]"
    )

    gate_table = Table(title="Gates", show_lines=False)
    gate_table.add_column("Metric")
    gate_table.add_column("Value", justify="right")
    gate_table.add_column("Threshold", justify="right")
    gate_table.add_column("Type")
    gate_table.add_column("Result", justify="center")
    for g in card.gates:
        result = "[green]PASS[/]" if g.passed else "[red]FAIL[/]"
        gate_table.add_row(
            g.name,
            f"{g.value * 100:.1f}%",
            f"{g.threshold * 100:.0f}%",
            "hard" if g.hard else "target",
            result,
        )
    console.print(gate_table)

    strat_table = Table(title="Per-stratum (automated + judged)")
    strat_table.add_column("Stratum")
    strat_table.add_column("N", justify="right")
    for col in ("structural", "faithful", "pii_clean", "register", "billable", "overall_pass"):
        strat_table.add_column(col, justify="right")
    for stratum, row in card.per_stratum.items():
        strat_table.add_row(
            stratum,
            str(row["n"]),
            *[
                f"{row[c] * 100:.0f}%"
                for c in (
                    "structural",
                    "faithful",
                    "pii_clean",
                    "register",
                    "billable",
                    "overall_pass",
                )
            ],
        )
    console.print(strat_table)

    suite_table = Table(title="Failure suites (hard gates)")
    suite_table.add_column("Suite")
    suite_table.add_column("Passed", justify="right")
    suite_table.add_column("Total", justify="right")
    suite_table.add_column("Rate", justify="right")
    suite_table.add_column("Result", justify="center")
    for s in card.suites:
        result = "[green]100%[/]" if s["pass_rate"] >= 1.0 else "[red]FAIL[/]"
        suite_table.add_row(
            s["name"],
            str(s["passed"]),
            str(s["total"]),
            f"{s['pass_rate'] * 100:.1f}%",
            result,
        )
    console.print(suite_table)

    verdict = "[bold green]RELEASE-OK[/]" if card.release_ok else "[bold red]BLOCKED[/]"
    console.print(f"\nRelease decision (hard gates): {verdict}")


def write_json(card: Scorecard, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")


def write_html(card: Scorecard, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = card.to_dict()

    def row(cells: list[str], header: bool = False) -> str:
        tag = "th" if header else "td"
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"

    gate_rows = "".join(
        row(
            [
                g["name"],
                f"{g['value'] * 100:.1f}%",
                f"{g['threshold'] * 100:.0f}%",
                "hard" if g["hard_gate"] else "target",
                "PASS" if g["passed"] else "FAIL",
            ]
        )
        for g in d["gates"]
    )
    strat_rows = "".join(
        row(
            [
                s,
                r["n"],
                f"{r['structural'] * 100:.0f}%",
                f"{r['faithful'] * 100:.0f}%",
                f"{r['pii_clean'] * 100:.0f}%",
                f"{r['register'] * 100:.0f}%",
                f"{r['billable'] * 100:.0f}%",
                f"{r['overall_pass'] * 100:.0f}%",
            ]
        )
        for s, r in d["per_stratum"].items()
    )
    suite_rows = "".join(
        row([s["name"], s["passed"], s["total"], f"{s['pass_rate'] * 100:.1f}%"])
        for s in d["suites"]
    )
    verdict = "RELEASE-OK" if d["release_ok"] else "BLOCKED"
    color = "#1a7f37" if d["release_ok"] else "#cf222e"

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NDIS Eval Scorecard</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; }}
 h1 {{ margin-bottom: 0; }}
 .meta {{ color: #57606a; margin-bottom: 1.5rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
 th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 .verdict {{ font-size: 1.3rem; font-weight: 700; color: {color}; }}
</style></head><body>
<h1>NDIS Eval Scorecard</h1>
<div class="meta">model: <b>{html.escape(d["model"])}</b> &middot; judge: {html.escape(d["judge"])}
 &middot; split: {html.escape(d["split"])} &middot; examples: {d["n_examples"]}</div>
<p class="verdict">Release (hard gates): {verdict}</p>
<h2>Gates</h2><table>{row(["Metric", "Value", "Threshold", "Type", "Result"], header=True)}{gate_rows}</table>
<h2>Per-stratum</h2><table>{row(["Stratum", "N", "Structural", "Faithful", "PII clean", "Register", "Billable", "Overall"], header=True)}{strat_rows}</table>
<h2>Failure suites (hard gates)</h2><table>{row(["Suite", "Passed", "Total", "Rate"], header=True)}{suite_rows}</table>
</body></html>"""
    path.write_text(doc, encoding="utf-8")
