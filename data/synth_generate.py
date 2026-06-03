"""Synthetic seed-data generator for NDIS case notes.

Produces realistic-but-fake input → target-note pairs, stratified across
all required strata (PRD §6.2).  Uses only deterministic Python + random —
no LLM calls needed for the seed set.

Run:
    python -m data.synth_generate --count 20          # default
    python -m data.synth_generate --count 100 --output data/splits/seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import textwrap
from datetime import date, timedelta
from typing import Any

random.seed(42)  # reproducible

# ---------------------------------------------------------------------------
# Constants — fake but realistic data pools
# ---------------------------------------------------------------------------

FAKE_PARTICIPANTS = [
    "P-1001", "P-1002", "P-1003", "P-1004", "P-1005",
    "P-2001", "P-2002", "P-2003", "P-3001", "P-3002",
]

FAKE_STAFF = [
    "Worker-A", "Worker-B", "Worker-C", "Worker-D", "Worker-E",
]

SERVICE_TYPES = [
    "Support Association",
    "Assist Daily Life",
    "Community Participation",
    "Capacity Building",
    "Consumed Materials",
    "Transportation",
    "Group Care Activity",
    "Short Term Accommodation",
    "Assessment and Reporting",
    "Case Management",
    "Other",
]

LOCATIONS = [
    "Participant's home",
    "Community centre",
    "Local park",
    "NDIS provider facility",
    "Local shop / mall",
    "Vocational workshop",
    "Therapy clinic",
    "Group home",
    "Virtual (video call)",
    "Residential aged care facility",
]

GOALS = [
    "Increase independence in daily living",
    "Build social connections and community participation",
    "Develop vocational skills for employment readiness",
    "Improve personal safety and risk management",
    "Manage daily finances independently",
    "Enhance communication skills",
    "Establish a consistent daily routine",
    "Reduce reliance on informal carers",
    "Complete NDIS plan review documentation",
    "Access local health services",
]

NARRATIVE_TEMPLATES = {
    "Routine session": [
        "{staff} supported {pid} with {activity}. {pid} was in a good mood and engaged well. The activity ran as planned for approximately {duration} minutes.",
        "{staff} attended {pid}'s residence at {time_am_pm} to provide {service_type} support. {pid} was ready on time. {narrative_detail}",
    ],
    "Incident note": [
        "During the scheduled session, an unexpected incident occurred involving {incident_desc}. {staff} followed incident response procedures. {pid} was unharmed.",
        "{staff} was providing {service_type} support when {incident_desc}. The situation was managed according to protocol.",
    ],
    "Sparse-input": [
        "{staff} supported {pid}. Session focused on daily living tasks. Progress noted.",
        "{pid} attended a session with {staff}. Support provided as scheduled.",
    ],
}

NARRATIVE_DETAIL_POOL = [
    "Tasks included meal preparation and household organisation.",
    "They went to the local grocery store where {pid} practiced shopping independently.",
    "A walking exercise was conducted around the neighbourhood.",
    "{pid} completed a personal care routine with minimal prompting.",
    "Group activity at the community centre included art and crafts.",
    "Transportation was provided to a medical appointment.",
    "Virtual check-in confirmed {pid}'s wellbeing and upcoming schedule.",
    "Role-play exercises were practiced for employment readiness.",
]

INCIDENT_DESCS = [
    "the participant reported feeling unwell",
    "a family member arrived unexpectedly",
    "the scheduled vehicle was delayed by over 30 minutes",
    "the participant became upset about a change in routine",
    "weather conditions prevented an outdoor activity",
]

OUTCOME_POOL = [
    "Increased confidence in daily tasks",
    "Improved social interaction with peers",
    "Completed planned activity independently",
    "Identified barriers to further progress",
    "Maintained consistent attendance this month",
    "Practiced coping strategies effectively",
    "Progressed towards goal: {goal}",
]


# ---------------------------------------------------------------------------
# Strata definitions
# ---------------------------------------------------------------------------

STRATA = {
    "routine_session": {
        "weight": 0.35,
        "narrative_type": "Routine session",
        "service_types": ["Assist Daily Life", "Support Association", "Community Participation"],
    },
    "incident": {
        "weight": 0.10,
        "narrative_type": "Incident note",
        "service_types": ["Assist Daily Life", "Community Participation", "Group Care Activity"],
        "always_has_risk": True,
    },
    "capacity_building": {
        "weight": 0.15,
        "narrative_type": "Routine session",
        "service_types": ["Capacity Building", "Assessment and Reporting", "Case Management"],
    },
    "consumed_materials": {
        "weight": 0.08,
        "narrative_type": "Routine session",
        "service_types": ["Consumed Materials"],
    },
    "transportation": {
        "weight": 0.07,
        "narrative_type": "Routine session",
        "service_types": ["Transportation"],
    },
    "group_care": {
        "weight": 0.07,
        "narrative_type": "Routine session",
        "service_types": ["Group Care Activity", "Short Term Accommodation"],
    },
    "sparse_input": {
        "weight": 0.12,
        "narrative_type": "Sparse-input",
        "service_types": SERVICE_TYPES,
    },
    "edge_case": {
        "weight": 0.06,
        "narrative_type": "Incident note",
        "service_types": ["Assist Daily Life", "Short Term Accommodation", "Other"],
        "always_has_risk": True,
    },
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _pick_weighted(choices: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in choices)
    r = random.random() * total
    cumulative = 0.0
    for item, weight in choices:
        cumulative += weight
        if r <= cumulative:
            return item
    return choices[-1][0]


def _random_date(start_year: int = 2025) -> date:
    start = date(start_year, 1, 1)
    delta = random.randint(0, 364)
    return start + timedelta(days=delta)


def _random_duration(stype: str) -> int:
    if stype in ("Transportation", "Consumed Materials"):
        return random.choice([15, 30, 45])
    return random.choice([30, 45, 60, 90, 120])


def generate_single_note(stype: str | None = None) -> dict[str, Any]:
    """Generate one synthetic case-note pair (input + target)."""

    # Pick stratum if service type not pre-specified
    if stype is None:
        stratum_name = _pick_weighted(list(STRATA.items()))
    else:
        # Find matching strata
        candidates = [(k, v) for k, v in STRATA.items() if stype in v["service_types"]]
        if not candidates:
            candidates = list(STRATA.items())
        stratum_name = _pick_weighted(candidates)

    s = STRATA[stratum_name]
    narrative_type = s["narrative_type"] if isinstance(s.get("narrative_type"), str) else "Routine session"
    service_pool = s["service_types"]
    svc_type: str = random.choice(service_pool)

    pid = random.choice(FAKE_PARTICIPANTS)
    staff = random.choice(FAKE_STAFF)
    goal = random.choice(GOALS)
    location = random.choice(LOCATIONS)
    svc_date = _random_date()
    duration = _random_duration(svc_type) if stype is None else _random_duration(stype)
    present = random.choice([True, False])

    # Narrative
    templates = NARRATIVE_TEMPLATES[narrative_type]
    template = random.choice(templates)
    detail = random.choice(NARRATIVE_DETAIL_POOL) if narrative_type != "Sparse-input" else ""
    incident_desc = f"{random.choice(INCIDENT_DESCS)}." if narrative_type == "Incident note" else ""

    time_val = f"{random.randint(7,17):02d}:{random.choice(['00', '30'])}"
    am_pm = "AM" if int(time_val[:2]) < 12 else "PM"

    narrative = template.format(
        staff=staff, pid=pid, activity=svc_type.lower(),
        duration=duration, service_type=svc_type,
        narrative_detail=detail, incident_desc=incident_desc,
        goal=goal.split(":")[-1].strip() if ":" in goal else goal,
    )

    # Worker's rough input (the model will transform this)
    worker_input = textwrap.dedent(f"""\
    Worker: {staff}
    Participant: {pid}
    Date: {svc_date.isoformat()}
    Duration: {duration} min
    Service type: {svc_type}
    Location: {location}
    Present: {'Yes' if present else 'No'}
    Goal: {goal}
    Notes: {narrative}
    Outcomes: Increased confidence, completed activity.
    Follow-up: Yes - check next week""")

    # Target structured note
    outcomes = random.sample(OUTCOME_POOL, k=random.randint(1, 3))
    outcomes = [o.format(goal=goal.split(":")[-1].strip() if ":" in goal else goal) for o in outcomes]

    target: dict[str, Any] = {
        "participant_id": pid,
        "date_of_service": svc_date.isoformat(),
        "duration_minutes": duration,
        "service_type": svc_type,
        "goal_linkage": goal,
        "location": location,
        "staff_presented_by": staff,
        "participant_present": present,
        "narrative_summary": narrative.strip(),
        "billable_evidence": f"Direct support provided for {svc_type.lower()}. Participant engaged for full {duration} minutes. Evidence: {outcomes[0] if outcomes else 'Activity completed'}.",
        "outcomes_achieved": outcomes,
        "risk_management": (random.choice(INCIDENT_DESCS) + ".") if s.get("always_has_risk") else None,
        "follow_up_needed": random.choice([True, False]),
        "follow_up_notes": f"Reassess support needs during next {svc_type.lower()} session." if present else "Contact participant to reschedule.",
    }

    return {
        "stratum": stratum_name,
        "worker_input": worker_input.strip(),
        "target_note": target,
    }


def generate_dataset(count: int = 20) -> list[dict[str, Any]]:
    """Generate *count* synthetic records distributed by stratum weights."""
    records: list[dict[str, Any]] = []
    for _ in range(count):
        # Weighted selection
        items = list(STRATA.items())
        stratum_name = _pick_weighted([(k, v["weight"]) for k, v in items])
        s = STRATA[stratum_name]
        record = generate_single_note()
        # Override the recorded stratum for metadata
        record["stratum"] = stratum_name
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NDIS case note data")
    parser.add_argument("--count", type=int, default=20, help="Number of records to generate")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/splits/seed.jsonl)",
    )
    args = parser.parse_args()

    records = generate_dataset(args.count)

    out_dir = pathlib.Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = pathlib.Path(args.output) if args.output else out_dir / "seed.jsonl"
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write stratified splits (train=70%, val=15%, test=15%)
    split_dir = out_dir
    for split_name, ratio in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        split_file = split_dir / f"{split_name}.jsonl"
        # Group by stratum to maintain stratification
        by_stratum: dict[str, list[dict]] = {}
        for r in records:
            by_stratum.setdefault(r["stratum"], []).append(r)

        with open(split_file, "w") as f:
            for sname, srecords in sorted(by_stratum.items()):
                import math
                n_total = len(srecords)
                split_size = round(n_total * ratio)
                random.shuffle(srecords)
                chosen = srecords[:split_size]
                for r in chosen:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} synthetic records → {output_path}")
    stratum_counts = {}
    for r in records:
        stratum_counts[r["stratum"]] = stratum_counts.get(r["stratum"], 0) + 1
    print("Stratum distribution:")
    for k, v in sorted(stratum_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
