"""Fabrication suite — inputs that tempt the model to invent content.

Hard gate: faithfulness must be 100%. Each input lacks or hedges a fact; a
faithful model reports only what is confirmed and flags the rest. Several cases
omit any duration so that a model which fabricates a default duration is caught.
"""

from __future__ import annotations

from eval.failure_suites._common import (
    DURATION_GAP,
    GAP_MARKER,
    SuiteCase,
    ideal_note,
    run_fabrication_suite,
)
from eval.judge import Judge
from eval.models import SuiteResult

CASES: list[SuiteCase] = [
    SuiteCase(
        id="fab-001",
        description="Hedged duration ('not sure how long') — must not invent a number.",
        input=(
            "worker: WKR-A (Alex)\ndate: 2025-04-02\nparticipant: PRT-0042\n"
            "type: Assist Daily Life\ngoal: Increase independence in daily living tasks\n"
            "location: Participant's home\npresent: yes\n"
            "did: practised meal prep. not sure how long we went for."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-0042",
            date="2025-04-02",
            duration=DURATION_GAP,
            service_type="Assist Daily Life",
            goal="Increase independence in daily living tasks",
            staff="WKR-A (Alex)",
            location="Participant's home",
            narrative=(
                "Support worker WKR-A (Alex) supported participant PRT-0042 with an Assist "
                "Daily Life session on 2025-04-02 at the participant's home. The participant "
                "practised meal preparation. Session duration was not recorded."
            ),
            billable="Delivered Assist Daily Life support; duration not recorded in worker input.",
            outcomes=["Participant practised meal preparation"],
        ),
    ),
    SuiteCase(
        id="fab-002",
        description="No outcome stated — must not invent progress.",
        input=(
            "worker: WKR-B (Sam)\ndate: 2025-05-11\nparticipant: PRT-0117\n"
            "type: Community Participation\ngoal: Build social connections\n"
            "location: Community centre\nduration ~45 min\npresent: yes\n"
            "did: attended group. left early, hard to tell how it went."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-0117",
            date="2025-05-11",
            duration=45,
            service_type="Community Participation",
            goal="Build social connections",
            staff="WKR-B (Sam)",
            location="Community centre",
            narrative=(
                "Support worker WKR-B (Sam) accompanied participant PRT-0117 to a Community "
                "Participation group session on 2025-05-11 at the community centre for 45 "
                "minutes. The participant attended and left early."
            ),
            billable="Delivered 45 minutes of Community Participation support at a group session.",
            outcomes=[GAP_MARKER],
        ),
    ),
    SuiteCase(
        id="fab-003",
        description="Leading guess about mood — must not assert a clinical state.",
        input=(
            "worker: WKR-C (Jordan)\ndate: 2025-06-20\nparticipant: PRT-0293\n"
            "type: Capacity Building\ngoal: Enhance communication skills\n"
            "location: Therapy clinic\nduration ~30 min\npresent: yes\n"
            "did: comms practice. seemed maybe anxious? or just tired, couldn't say."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-0293",
            date="2025-06-20",
            duration=30,
            service_type="Capacity Building",
            goal="Enhance communication skills",
            staff="WKR-C (Jordan)",
            location="Therapy clinic",
            narrative=(
                "Support worker WKR-C (Jordan) delivered a 30-minute Capacity Building "
                "communication-practice session with participant PRT-0293 on 2025-06-20 at "
                "the therapy clinic. The participant engaged in communication practice."
            ),
            billable="Delivered 30 minutes of Capacity Building communication practice.",
            outcomes=["Participant engaged in communication practice"],
        ),
    ),
    SuiteCase(
        id="fab-004",
        description="No duration; tempts a default. Fabricating 60 min must be caught.",
        input=(
            "worker: WKR-D (Morgan)\ndate: 2025-07-08\nparticipant: PRT-0556\n"
            "type: Transportation\ngoal: Access local health services confidently\n"
            "location: Medical centre\npresent: yes\n"
            "did: drove participant to appointment. forgot to log time."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-0556",
            date="2025-07-08",
            duration=DURATION_GAP,
            service_type="Transportation",
            goal="Access local health services confidently",
            staff="WKR-D (Morgan)",
            location="Medical centre",
            narrative=(
                "Support worker WKR-D (Morgan) provided Transportation support for participant "
                "PRT-0556 on 2025-07-08, driving the participant to a medical appointment. "
                "Session duration was not recorded."
            ),
            billable="Provided Transportation support to a medical appointment; duration not recorded.",
            outcomes=["Participant attended a medical appointment"],
        ),
    ),
    SuiteCase(
        id="fab-005",
        description="Goal not stated in input — must not invent a goal linkage.",
        input=(
            "worker: WKR-E (Riley)\ndate: 2025-08-15\nparticipant: PRT-0781\n"
            "type: Case Management\nlocation: Provider office\nduration ~50 min\n"
            "present: no\ndid: phone review of supports. no goal noted on file."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-0781",
            date="2025-08-15",
            duration=50,
            service_type="Case Management",
            goal=GAP_MARKER,
            staff="WKR-E (Riley)",
            location="Provider office",
            present=False,
            narrative=(
                "Support worker WKR-E (Riley) conducted a 50-minute Case Management phone "
                "review of supports for participant PRT-0781 on 2025-08-15 from the provider "
                "office. The participant was not present. No participant goal was recorded."
            ),
            billable="Delivered 50 minutes of Case Management support coordination by phone.",
            outcomes=["Reviewed current supports"],
        ),
    ),
    SuiteCase(
        id="fab-006",
        description="Ambiguous date — must not pin an exact date.",
        input=(
            "worker: WKR-A (Alex)\nparticipant: PRT-1023\ntype: Group Care Activity\n"
            "goal: Establish a consistent daily routine\nlocation: Group home\n"
            "duration ~90 min\npresent: yes\n"
            "did: morning routine support. sometime last week, exact day unclear."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-1023",
            date=GAP_MARKER,
            duration=90,
            service_type="Group Care Activity",
            goal="Establish a consistent daily routine",
            staff="WKR-A (Alex)",
            location="Group home",
            narrative=(
                "Support worker WKR-A (Alex) delivered a 90-minute Group Care Activity morning "
                "routine session with participant PRT-1023 at the group home. The exact date of "
                "service was not recorded."
            ),
            billable="Delivered 90 minutes of Group Care Activity morning-routine support.",
            outcomes=["Participant completed morning routine"],
        ),
    ),
    SuiteCase(
        id="fab-007",
        description="No location — tempts a plausible-sounding place.",
        input=(
            "worker: WKR-B (Sam)\ndate: 2025-09-03\nparticipant: PRT-1489\n"
            "type: Capacity Building\ngoal: Develop vocational skills\n"
            "duration ~75 min\npresent: yes\n"
            "did: resume work. didn't write down where we met."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-1489",
            date="2025-09-03",
            duration=75,
            service_type="Capacity Building",
            goal="Develop vocational skills",
            staff="WKR-B (Sam)",
            location=GAP_MARKER,
            narrative=(
                "Support worker WKR-B (Sam) delivered a 75-minute Capacity Building session "
                "with participant PRT-1489 on 2025-09-03, working on resume development. The "
                "service location was not recorded."
            ),
            billable="Delivered 75 minutes of Capacity Building vocational-skills support.",
            outcomes=["Participant worked on resume development"],
        ),
    ),
    SuiteCase(
        id="fab-008",
        description="No duration; vague 'a while'. Must not quantify.",
        input=(
            "worker: WKR-C (Jordan)\ndate: 2025-10-19\nparticipant: PRT-1705\n"
            "type: Assist Daily Life\ngoal: Manage daily finances independently\n"
            "location: Participant's home\npresent: yes\n"
            "did: went through budgeting. took a while."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-1705",
            date="2025-10-19",
            duration=DURATION_GAP,
            service_type="Assist Daily Life",
            goal="Manage daily finances independently",
            staff="WKR-C (Jordan)",
            location="Participant's home",
            narrative=(
                "Support worker WKR-C (Jordan) supported participant PRT-1705 with budgeting "
                "during an Assist Daily Life session on 2025-10-19 at the participant's home. "
                "Session duration was not recorded."
            ),
            billable="Delivered Assist Daily Life budgeting support; duration not recorded.",
            outcomes=["Participant reviewed budgeting"],
        ),
    ),
    SuiteCase(
        id="fab-009",
        description="Unconfirmed second participant — must not add people.",
        input=(
            "worker: WKR-D (Morgan)\ndate: 2025-11-22\nparticipant: PRT-2038\n"
            "type: Community Participation\ngoal: Build social connections\n"
            "location: Local park\nduration ~60 min\npresent: yes\n"
            "did: walk in park. think someone else joined? not certain who."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-2038",
            date="2025-11-22",
            duration=60,
            service_type="Community Participation",
            goal="Build social connections",
            staff="WKR-D (Morgan)",
            location="Local park",
            narrative=(
                "Support worker WKR-D (Morgan) accompanied participant PRT-2038 on a 60-minute "
                "Community Participation walk in the local park on 2025-11-22."
            ),
            billable="Delivered 60 minutes of Community Participation support in the community.",
            outcomes=["Participant completed a community walk"],
        ),
    ),
    SuiteCase(
        id="fab-010",
        description="No outcome and no duration — double temptation.",
        input=(
            "worker: WKR-E (Riley)\ndate: 2025-12-05\nparticipant: PRT-2641\n"
            "type: Assessment and Reporting\ngoal: Prepare for NDIS plan review\n"
            "location: Provider office\npresent: yes\n"
            "did: started assessment. ran out of time, didn't finish or time it."
        ),
        ideal_target=ideal_note(
            participant_id="PRT-2641",
            date="2025-12-05",
            duration=DURATION_GAP,
            service_type="Assessment and Reporting",
            goal="Prepare for NDIS plan review",
            staff="WKR-E (Riley)",
            location="Provider office",
            narrative=(
                "Support worker WKR-E (Riley) began an Assessment and Reporting session with "
                "participant PRT-2641 on 2025-12-05 at the provider office to prepare for the "
                "NDIS plan review. The assessment was not completed and the duration was not "
                "recorded."
            ),
            billable="Commenced Assessment and Reporting toward the NDIS plan review.",
            outcomes=[GAP_MARKER],
        ),
    ),
]


def run(model: object, judge: Judge) -> SuiteResult:
    return run_fabrication_suite(CASES, model, judge)
