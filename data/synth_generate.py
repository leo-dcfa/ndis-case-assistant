"""Synthetic seed data generator for NDIS case notes.

Generates realistic but entirely fake input → target note pairs, stratified
across every required stratum (§6.2 of the PRD).  Uses template-based synthesis
with parameterised values so no real participant data is ever needed.

Usage:
    # Generate a full stratified dataset
    python -m data.synth_generate --count 500 --output data/splits/seed.jsonl

    # Or import and generate programmatically
    from data.synth_generate import generate_stratified_dataset
    dataset = generate_stratified_dataset(count_per_stratum=20)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so imports work with `python -m` invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import CaseNote, ServiceType  # noqa: E402

# ---------------------------------------------------------------------------
# Randomised value pools (all fake)
# ---------------------------------------------------------------------------

PARTICIPANT_NAMES = [
    "P-1001", "P-1002", "P-1003", "P-1004", "P-1005",
    "P-1006", "P-1007", "P-1008", "P-1009", "P-1010",
]

STAFF_NAMES = [
    "Worker-A", "Worker-B", "Worker-C", "Worker-D", "Worker-E",
]

LOCATIONS = [
    "Participant's home (42 Example St, Suburb)",
    "Community centre — Central Park",
    "Local shopping centre — Westfield",
    "Neighbourhood group activity room",
    "Day program venue",
    "Public transport route (bus/tram)",
    "Support worker's office (de-identified)",
    "Gymnasium (community facility)",
]

GOALS = [
    "Increase independence in daily living tasks (Goal 1 — Core Supports)",
    "Develop social skills and community participation (Goal 3 — Capacity Building)",
    "Improve ability to maintain a healthy lifestyle (Goal 2 — Capacity Building)",
    "Enhance communication and conflict resolution (Goal 4 — Capacity Building)",
    "Build confidence using public transport independently (Goal 5 — Core Supports)",
    "Participate in community events without support escalation (Goal 2 — Capacity Building)",
    "Manage personal finances with growing independence (Goal 6 — Capacity Building)",
]

NARRATIVE_TEMPLATES = {
    "routine_session": [
        "Attended participant's residence at {time}. Accompanied the participant to the local supermarket for grocery shopping as part of daily living skills development. Participant was independent in selecting items but required prompting to stay within budget. Used $50 from the participant's petty cash fund.",
        "Support worker arrived at 09:30 to begin the weekly community access session. Supported the participant with meal preparation in their kitchenette. Participant followed a visual recipe card and independently prepared pasta with sauce. Reviewed nutrition labels together to support healthy food choices.",
        "Provided personal care support in the morning routine as scheduled. Assisted with showering and dressing with verbal prompting only — no physical assistance required. The participant expressed satisfaction with how they managed their appearance for the day.",
    ],
    "incident_note": [
        "At approximately 14:00, observed the participant become visibly distressed when a planned community outing was cancelled due to adverse weather. Supported the participant by discussing alternative indoor activities. Participant initially refused to engage but eventually agreed to watch a DVD in their room with a companion phone call from a family member.",
        "Incident reported at 16:30: participant's neighbour notified support team that participant had fallen while attempting to carry laundry upstairs. Emergency services were not required. Attended the residence to assess safety. Assisted participant back to ground-floor bedroom and reviewed home hazards with the NDIS planner on call.",
    ],
    "capacity_building": [
        "Facilitated a group cooking workshop at the community kitchen with 4 other participants. The session focused on basic meal planning and budgeting. Participant contributed actively, preparing a salad and comparing prices between two grocery brands. Documented key learnings in their personal goals journal.",
        "Supported the participant to attend a TAFE orientation session for a short course in customer service. Accompanied to campus, introduced them to the course coordinator, and assisted with completing enrolment paperwork. Participant appeared motivated and expressed interest in completing the application.",
    ],
    "sparse_input": [
        "Met with participant at their home. Did grocery shopping together. Participant seemed okay today.",
        "Morning visit. Helped make breakfast. Watched TV for a bit. Left at noon.",
        "Visited P-1002. They wanted to go to the shops but were too tired. Stayed and chatted instead.",
    ],
    "edge_case": [
        "Participant arrived unexpectedly at the support worker's office at 08:45, having missed their scheduled bus service (arranged via group transport). No pre-booking was in place. Discussed rescheduling options and provided a phone number to call for future unplanned pickups. Documented as a potential transport gap requiring discussion with the coordinator.",
        "The participant's family member (mother) attended the session unannounced to discuss concerns about an upcoming change of support worker. The participant appeared uncomfortable with this conversation. Advised the mother that future discussions about care changes should be scheduled formally through the plan manager, and offered to schedule a three-way call.",
    ],
}

SERVICE_TYPES_FOR_STRATUM = {
    "routine_session": [ServiceType.SUPPORT_ASSOCIATION, ServiceType.ASSIST_DAILY_LIFE],
    "incident_note": [ServiceType.CASE_MANAGEMENT, ServiceType.ASSESSMENT_REPORTING],
    "capacity_building": [ServiceType.CAPACITY_BUILDING, ServiceType.COMMUNITY_PARTICIPATION],
    "sparse_input": [ServiceType.SUPPORT_ASSOCIATION, ServiceType.CONSUMED_MATERIALS],
    "edge_case": [ServiceType.CASE_MANAGEMENT, ServiceType.GROUP_CARE_ACTIVITY],
}

OUTCOME_TEMPLATES = {
    "routine_session": [
        "Participant independently selected three groceries without prompting",
        "Improved ability to use the public card reader at checkout",
        "Completed meal preparation using visual supports with 75% independence",
        "Maintained appropriate spending within allocated budget",
    ],
    "incident_note": [
        "Participant calmed after 20 minutes of de-escalation strategies were applied",
        "Safety of the home environment confirmed during follow-up visit",
        "Plan manager was notified and a meeting scheduled for next week",
        "Incident report filed in accordance with NDIS incident management policy",
    ],
    "capacity_building": [
        "Participant successfully prepared two meals from scratch with minimal prompting",
        "Participant completed course enrolment paperwork independently after scaffolding",
        "Demonstrated improved confidence when interacting with shop staff",
        "Group leader noted positive engagement in peer discussions",
    ],
    "sparse_input": [
        "Groceries purchased within budget",
        "Breakfast consumed without assistance",
        "Participant restful and cooperative during visit",
    ],
    "edge_case": [
        "Alternative phone-based support provided as contingency",
        "Mother was advised on formal escalation pathway for care concerns",
        "Transport coordinator to review group transport schedule",
    ],
}

TIMES = ["08:00", "09:30", "10:00", "10:30", "11:00", "14:00", "15:00", "16:00"]


# ---------------------------------------------------------------------------
# Synthetic data generation functions
# ---------------------------------------------------------------------------

def _random_date() -> str:
    """Return a random date string in YYYY-MM-DD format within 2025."""
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"2025-{month:02d}-{day:02d}"


def _random_duration(stratum: str) -> int:
    durations = {
        "routine_session": [60, 90, 120],
        "incident_note": [30, 45, 60, 90],
        "capacity_building": [90, 120, 180],
        "sparse_input": [30, 45, 60],
        "edge_case": [60, 90],
    }
    return random.choice(durations[stratum])


def generate_single_record(stratum: str, idx: int) -> dict[str, Any]:
    """Generate one synthetic input → target pair for the given stratum."""
    participant = random.choice(PARTICIPANT_NAMES)
    staff = random.choice(STAFF_NAMES)
    location = random.choice(LOCATIONS)
    goal = random.choice(GOALS)
    service_types = SERVICE_TYPES_FOR_STRATUM[stratum]
    service_type = random.choice(service_types)
    narrative_template = random.choice(NARRATIVE_TEMPLATES[stratum])
    time_str = random.choice(TIMES)
    duration = _random_duration(stratum)

    narrative_text = narrative_template.format(time=time_str, participant=participant)

    outcomes = random.sample(
        OUTCOME_TEMPLATES[stratum],
        k=min(random.randint(1, 3), len(OUTCOME_TEMPLATES[stratum])),
    )

    # Build the target CaseNote as a structured dict (the "model output" it should produce)
    target_note = CaseNote(
        participant_id=participant,
        staff_presented_by=staff,
        date_of_service=_random_date(),
        duration_minutes=duration,
        service_type=service_type,
        goal_linkage=goal,
        location=location,
        participant_present=random.choice([True, True, True, False]),  # 75% present
        narrative_summary=narrative_text,
        billable_evidence=f"Provided {service_type.value.lower()} support. {outcomes[0].lower()}. "
                         f"Documented for billing purposes in alignment with NDIS pricing arrangements.",
        outcomes_achieved=outcomes,
        risk_management=random.choice([None, f"Potential concern noted regarding {random.choice(['transport reliability', 'home safety', 'social isolation'])} — flagged to coordinator."]),
        follow_up_needed=random.choice([True, False]),
        follow_up_notes="Discussed with NDIS planner at last review. Follow-up as scheduled in Q{Q}.",
    )

    # The "worker input" is a rough, informal version (simulating dictation / bullet points)
    worker_input = _build_worker_input(stratum, participant, staff, time_str, duration, service_type, narrative_template.format(time=time_str))

    return {
        "id": f"{stratum}_{idx:04d}",
        "stratum": stratum,
        "input_text": worker_input,
        "target_note": target_note.to_jsonl_record(),
    }


def _build_worker_input(
    stratum: str, participant: str, staff: str, time_str: str, duration: int, service_type: ServiceType, narrative_snippet: str
) -> str:
    """Construct a realistic support-worker input (rough notes / dictation)."""
    inputs = {
        "routine_session": f"• {staff} saw {participant} at {time_str}\n• Duration: {duration}min\n• Service type: {service_type.value}\n• Narrative: {narrative_snippet}\n• Outcomes: improved independence with shopping, stayed within budget",
        "incident_note": f"⚠ INCIDENT — {staff} called in at {time_str} about {participant}. {duration}min session. Type: {service_type.value}. {narrative_snippet}",
        "capacity_building": f"• Group activity with {participant}\n• {staff} facilitated\n• Duration: {duration}min\n• Activity: cooking workshop\n• {narrative_snippet}",
        "sparse_input": "",  # sparse input IS the input
        "edge_case": f"* UNUSUAL — {staff}: {participant} showed up unannounced at {time_str}. {narrative_snippet}",
    }

    if stratum == "sparse_input" and not narrative_snippet.startswith("•"):
        # Use raw sparse template directly
        pass  # already set by NARRATIVE_TEMPLATES["sparse_input"]
    elif stratum != "sparse_input":
        return inputs[stratum]

    return narrative_snippet


# ---------------------------------------------------------------------------
# Stratified dataset builder
# ---------------------------------------------------------------------------

STRATUM_WEIGHTS = {
    "routine_session": 0.35,     # majority class
    "incident_note": 0.10,       # less common but critical
    "capacity_building": 0.20,   # core service type
    "sparse_input": 0.20,        # edge case — model must handle minimal input
    "edge_case": 0.15,           # adversarial pressure
}


def generate_stratified_dataset(count_per_stratum: int = 20) -> list[dict[str, Any]]:
    """Generate a stratified synthetic dataset."""
    random.seed(42)  # reproducibility
    records: list[dict[str, Any]] = []

    for stratum, count in STRATUM_WEIGHTS.items():
        for i in range(count_per_stratum):
            record = generate_single_record(stratum, i)
            records.append(record)

    # Shuffle deterministically (but with different seed than generation)
    random.seed(123)
    random.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NDIS case note data")
    parser.add_argument("--count", type=int, default=20, help="Records per stratum (default: 20)")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "splits" / "seed.jsonl"), help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Generating {args.count} records per stratum (5 strata total = {args.count * 5} records)...")
    dataset = generate_stratified_dataset(count_per_stratum=args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write train/val/test split manifests
    _write_split_manifests(dataset, output_path)

    print(f"✓ Wrote {len(dataset)} records to {output_path}")
    print("Split manifest:")
    for split_name in ["train", "val", "test"]:
        p = PROJECT_ROOT / "data" / "splits" / f"{split_name}.jsonl"
        with open(p, encoding="utf-8") as fh:
            cnt = sum(1 for _ in fh)
        print(f"  {split_name}: {cnt} records → {p}")


def _write_split_manifests(dataset: list[dict[str, Any]], seed_path: Path) -> None:
    """Split the dataset into train (70%) / val (15%) / test (15%)."""
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_train = int(len(dataset) * 0.70)
    n_val = int(len(dataset) * 0.15)

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    # Ensure test set has no empty strata
    test_strata = set()
    for i in split_indices["test"]:
        test_strata.add(dataset[i]["stratum"])
    min_per_stratum = max(1, len(split_indices["test"]) // 5)
    for stratum in STRATUM_WEIGHTS:
        if dataset[i]["stratum"] == stratum and len([j for j in split_indices["test"] if dataset[j]["stratum"] == stratum]) < min_per_stratum:
            # Promote from val or train
            src = split_indices.get("val" if split_indices["val"] else "train")
            for j in src:
                if dataset[j]["stratum"] == stratum and dataset[j] not in [split_indices["test"][k] for k in range(len(split_indices["test"]))]:
                    split_indices["test"].append(j)
                    break

    for split_name, indices_list in split_indices.items():
        output_path = PROJECT_ROOT / "data" / "splits" / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for i in indices_list:
                f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
