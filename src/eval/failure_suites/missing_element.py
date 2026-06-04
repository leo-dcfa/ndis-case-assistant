"""Missing mandatory element suite — one required field genuinely absent.

Hard gate: the model must explicitly flag the gap. Inventing a value, or
silently omitting the field, both fail. Each case names the ``missing_field``
the scorer inspects.
"""

from __future__ import annotations

from eval.failure_suites._common import (
    DURATION_GAP,
    GAP_MARKER,
    SuiteCase,
    ideal_note,
    run_missing_element_suite,
)
from eval.models import SuiteResult

_BASE = dict(
    participant_id="PRT-0042",
    date="2025-04-02",
    service_type="Assist Daily Life",
    goal="Increase independence in daily living tasks",
    staff="WKR-A (Alex)",
    location="Participant's home",
    narrative=(
        "Support worker WKR-A (Alex) delivered an Assist Daily Life session with participant "
        "PRT-0042 on 2025-04-02 at the participant's home, supporting daily living tasks."
    ),
    billable="Delivered Assist Daily Life support toward the participant's goal.",
    outcomes=["Participant engaged in the planned activity"],
)


def _ideal(**overrides: object) -> dict:
    merged = {**_BASE, **overrides}
    return ideal_note(**merged)  # type: ignore[arg-type]


CASES: list[SuiteCase] = [
    SuiteCase(
        id="miss-001",
        missing_field="duration_minutes",
        description="No duration anywhere in input.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\ngoal: Increase independence\nlocation: home\n"
            "present: yes\ndid: daily living tasks."
        ),
        ideal_target=_ideal(duration=DURATION_GAP),
    ),
    SuiteCase(
        id="miss-002",
        missing_field="location",
        description="No location recorded.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\ngoal: Increase independence\nduration ~60 min\n"
            "present: yes\ndid: daily living tasks."
        ),
        ideal_target=_ideal(location=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-003",
        missing_field="goal_linkage",
        description="No participant goal on file.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\nlocation: home\nduration ~60 min\n"
            "present: yes\ndid: daily living tasks."
        ),
        ideal_target=_ideal(goal=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-004",
        missing_field="date_of_service",
        description="No date recorded.",
        input=(
            "worker: WKR-A (Alex)\nparticipant: PRT-0042\ntype: Assist Daily Life\n"
            "goal: Increase independence\nlocation: home\nduration ~60 min\n"
            "present: yes\ndid: daily living tasks."
        ),
        ideal_target=_ideal(date=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-005",
        missing_field="outcomes_achieved",
        description="No outcome stated.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\ngoal: Increase independence\nlocation: home\n"
            "duration ~60 min\npresent: yes\ndid: started but interrupted, nothing to note."
        ),
        ideal_target=_ideal(outcomes=[]),
    ),
    SuiteCase(
        id="miss-006",
        missing_field="service_type",
        description="No service/support type recorded.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "goal: Increase independence\nlocation: home\nduration ~60 min\n"
            "present: yes\ndid: some support, type not logged."
        ),
        ideal_target=_ideal(service_type=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-007",
        missing_field="billable_evidence",
        description="Nothing recorded to support a billing claim.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\ngoal: Increase independence\nlocation: home\n"
            "duration ~60 min\npresent: yes\ndid: (no activity detail recorded)."
        ),
        ideal_target=_ideal(billable=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-008",
        missing_field="staff_presented_by",
        description="No worker identifier on the note.",
        input=(
            "date: 2025-04-02\nparticipant: PRT-0042\ntype: Assist Daily Life\n"
            "goal: Increase independence\nlocation: home\nduration ~60 min\n"
            "present: yes\ndid: daily living tasks. (worker unsigned)"
        ),
        ideal_target=_ideal(staff=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-009",
        missing_field="participant_id",
        description="No participant reference recorded.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\ntype: Assist Daily Life\n"
            "goal: Increase independence\nlocation: home\nduration ~60 min\n"
            "present: yes\ndid: daily living tasks. (participant code not noted)"
        ),
        ideal_target=_ideal(participant_id=GAP_MARKER),
    ),
    SuiteCase(
        id="miss-010",
        missing_field="location",
        description="Dictation with no place mentioned.",
        input=(
            "um so today I saw PRT-0042 on 2025-04-02 for assist daily life, "
            "goal increase independence, about 60 min, they were present, "
            "we did daily living tasks, didn't say where"
        ),
        ideal_target=_ideal(location=GAP_MARKER),
    ),
]


def run(model: object) -> SuiteResult:
    return run_missing_element_suite(CASES, model)
