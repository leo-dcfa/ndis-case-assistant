"""Synthetic NDIS case-note data generator (library)."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import date, timedelta
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3.6:27b"
DEFAULT_MODEL = "qwen3:8b"


client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

random.seed(42)


SYSTEM_PROMPT = (
    "You generate SYNTHETIC training data for a model that drafts NDIS "
    "(Australian National Disability Insurance Scheme) case notes. "
    "Every example is COMPLETELY FICTIONAL.\n\n"
    "ABSOLUTE RULES:\n"
    '- All identifiers are fake: participant codes like "PRT-0042", '
    "invented first names only, fictional suburbs/postcodes.\n"
    "- NEVER output a real person's name, address, phone number, or NDIS ID.\n"
    "- Never output a 9-digit number that resembles a real NDIS participant number.\n"
    "- FAITHFULNESS: every fact in the target note must be derived EXACTLY from "
    "the worker input. Restructure and organise — never invent new facts.\n"
    '- Where the input lacks a required element, mark it "[not recorded]" rather '
    "than inventing it.\n"
    "- Output STRICT JSON only. No markdown fences, no commentary, no explanation."
)

STRATUM_PROMPTS = {
    "routine_session": (
        "A regular support session went as planned. The input notes are detailed enough "
        "for a complete structured note."
    ),
    "incident": (
        "An unexpected event occurred during the session. The input may contain some "
        "urgency, fragmented details, or emotional language — reorganise into calm, "
        "professional tone."
    ),
    "capacity_building": (
        "A skill-development or capacity-building activity. Focus on learning outcomes "
        "and progressive milestones."
    ),
    "sparse_input": (
        "The support worker left very brief notes — possibly just bullet fragments. "
        "Reconstruct the structured note from minimal information. Mark missing fields "
        'as "[not recorded]".'
    ),
    "adv_pii_check": (
        "ADVERSARIAL TEST: The input contains sensitive third-party details (a real-ish "
        "phone number, another person's name) that MUST NOT appear in the target note. "
        "The target must redact or anonymise these per privacy policy. Return an extra "
        'key "pii_handling" describing what was redacted.'
    ),
    "adv_missing_field": (
        "ADVERSARIAL TEST: The input is GENUINELY missing one required field (e.g. no "
        "duration recorded). The target must explicitly flag the gap rather than invent "
        'or silently omit it. Return an extra key "missing_fields" with the gap list.'
    ),
}

INPUT_STYLES = {
    "jotted_notes": (
        "The worker's style: lowercase, abbreviations, shorthand, fragments, minimal "
        "punctuation. Example: 'went to mall p with prt bought groceries ok mood fine'"
    ),
    "dictation": (
        "Speech-to-text style: fillers ('um', 'so'), false starts, repetitions, run-on "
        "sentences. Example: 'so today um I went to see P-1004 at their place and we like "
        "we did the cooking session and yeah they were good'"
    ),
}

FAKE_PARTICIPANTS = [
    "PRT-0042",
    "PRT-0117",
    "PRT-0293",
    "PRT-0556",
    "PRT-0781",
    "PRT-1023",
    "PRT-1489",
    "PRT-1705",
    "PRT-2038",
    "PRT-2641",
]

FAKE_STAFF = [
    "WKR-A (Alex)",
    "WKR-B (Sam)",
    "WKR-C (Jordan)",
    "WKR-D (Morgan)",
    "WKR-E (Riley)",
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

FAKE_LOCATIONS = [
    "Participant's home (suburb TBD)",
    "Community centre",
    "Local park / green space",
    "NDIS provider facility",
    "Shopping centre",
    "Vocational workshop",
    "Therapy clinic",
    "Group home",
    "Virtual (video call)",
]

FAKE_GOALS = [
    "Increase independence in daily living tasks",
    "Build social connections and community participation",
    "Develop vocational skills for employment readiness",
    "Improve personal safety and risk management",
    "Manage daily finances independently",
    "Enhance communication and expressive skills",
    "Establish a consistent daily routine",
    "Reduce reliance on informal carers",
    "Prepare for NDIS plan review",
    "Access local health services confidently",
]

OUTCOME_POOL = [
    "Increased confidence in completing daily tasks",
    "Improved social interaction with peers",
    "Completed planned activity independently",
    "Identified new barriers to progress",
    "Maintained consistent attendance for the month",
    "Practiced and reinforced coping strategies",
    "Demonstrated improvement in targeted skill area",
    "No significant change — continued monitoring",
]

FAKE_DATES: list[date] = []
for year in (2025, 2026):
    start = date(year, 1, 1)
    FAKE_DATES.extend(start + timedelta(days=i) for i in range(365))


def _format_worker_input(
    style: str,
    service_type: str,
    goal: str,
    participants_count: int = 1,
    has_risk: bool = False,
    quality: str = "normal",
) -> tuple[str, list[str]]:
    """Build a fake raw worker input string.

    Returns (input_text, missing_fields) where missing_fields is empty
    unless quality == 'sparse'.
    """
    parts: list[str] = []
    missing: list[str] = []

    parts.append(f"worker: {random.choice(FAKE_STAFF)}")
    parts.append(f"date: {random.choice(FAKE_DATES).isoformat()}")
    parts.append(f"participant: {random.choice(FAKE_PARTICIPANTS)}")
    parts.append(f"type: {service_type}")
    parts.append(f"goal: {random.choice(FAKE_GOALS)[:80]}")

    if style == "jotted_notes":
        fragments = [
            f"{service_type.lower()} session",
            f"location: {random.choice(FAKE_LOCATIONS)}",
            f"participant present: {'yes' if random.random() > 0.2 else 'no'}",
        ]
        duration = random.randint(30, 120)
        fragments.append(f"duration ~{duration} min")

        if quality == "sparse":
            missing.append("duration_minutes")
            missing.append("location")
            parts.extend(["brief visit", "prt- attended ok"])
            return "\n".join(parts + fragments[:2]), missing

        if has_risk:
            risk_events = [
                "prt became agitated - de-escalated ok",
                "family member arrived unexpectedly",
                "prt reported feeling unwell - monitored",
                "vehicle delay 45 min - activity shortened",
            ]
            fragments.append(random.choice(risk_events))

        fragments.append(f"outcome: {random.choice(OUTCOME_POOL).lower()}")

        if style == "jotted_notes":
            text = "\n".join(parts + [", ".join(fragments)])
        else:
            filler = random.choice(["um", "so", "right", "yeah", "okay"])
            parts.append(f"{filler} {fragments[0].title().replace(',', '.')}")
            for i, f in enumerate(fragments[1:], 1):
                prefix = random.choice(["also,", "then,", "and ", "also um, "])
                parts.append(f"{prefix}{f}")
            text = "\n".join(parts)

        return text, []

    elif style == "dictation":
        filler = random.choice(["um", "so", "right", "yeah"])
        line = (
            f"{filler} today's session with {random.choice(FAKE_PARTICIPANTS)} "
            f"on {random.choice(FAKE_DATES).isoformat()}"
        )
        parts.append(line)
        parts.append(f"type: {service_type}")
        return "\n".join(parts), missing

    return "\n".join(parts), []


def _build_stratum_seed(stratum: str, service_type: str) -> str:
    """A seed hint that gives the LLM creative direction without being copied verbatim."""
    seeds = {
        "routine_session": f"New routine session for {service_type}",
        "incident": f"Unexpected event during {service_type} support",
        "capacity_building": f"Capacity-building activity — {service_type}",
        "sparse_input": f"Brief, fragmented notes for {service_type}",
        "adv_pii_check": f"Pii-injection scenario in a {service_type} session note",
        "adv_missing_field": f"Incomplete documentation for a {service_type} session",
    }
    seed_variations = [
        "first week with new support worker",
        "participant celebrating a small milestone",
        "session ran longer than scheduled",
        "participant declined to continue partway through",
        "transport cancelled — rescheduled same day",
        "group activity, multiple participants present",
        "introducing a new goal or revising an existing one",
        "participant experiencing anxiety in unfamiliar setting",
    ]
    return f"{seeds[stratum]} ({random.choice(seed_variations)})"


def generate_one(stratum: str) -> dict[str, Any]:
    """Ask the LLM to produce one input → target pair."""

    service_type = random.choice(SERVICE_TYPES)
    style_key = random.choice(list(INPUT_STYLES.keys()))
    quality = "sparse" if stratum == "sparse_input" else "normal"
    has_risk = stratum == "incident"

    worker_text, missing_fields = _format_worker_input(
        style=style_key,
        service_type=service_type,
        goal="",
        has_risk=has_risk,
        quality=quality,
    )

    seed_hint = _build_stratum_seed(stratum, service_type)

    prompt = (
        "Generate ONE synthetic NDIS case-note example.\n\n"
        f"STRATUM: {stratum}\n"
        f"SERVICE TYPE: {service_type}\n"
        f"WORKER INPUT STYLE: {style_key}\n"
        f"{INPUT_STYLES[style_key]}\n\n"
        "Scenario seed (creative direction, do NOT quote): "
        f"{seed_hint}\n"
        f"Described context: {STRATUM_PROMPTS[stratum]}\n\n"
        "--- WORKER INPUT ---\n"
        f"{worker_text}\n"
        "---------------------\n\n"
        "REQUIRED target fields (must appear exactly as keys):\n"
        "  participant_id        — fake code only\n"
        "  date_of_service       — YYYY-MM-DD format\n"
        "  duration_minutes      — integer, or -1 if not recorded\n"
        "  service_type          — one of the NDIS categories\n"
        "  goal_linkage          — link to a specific participant goal\n"
        "  location              — where the service was delivered\n"
        "  staff_presented_by    — fake worker code/name\n"
        "  participant_present   — true / false (or null if unknown)\n"
        "  narrative_summary     — structured third-person narrative\n"
        "  billable_evidence     — what supports the billing claim\n"
        "  outcomes_achieved     — array of strings\n"
        '  risk_management       — string or "[not recorded]"\n'
        "  follow_up_needed      — true / false\n"
        '  follow_up_notes       — string or "[not recorded]"\n\n'
    )

    if stratum == "adv_pii_check":
        prompt += (
            "ADVERSARIAL: The worker input contains a stray phone number and another "
            "person's name. Redact or anonymise these in the target note per privacy "
            'policy. Return extra key "pii_handling" describing what was redacted.\n\n'
        )
    elif stratum == "adv_missing_field":
        prompt += (
            "ADVERSARIAL: The worker input is GENUINELY missing one required field "
            "(e.g. no duration recorded). The target must explicitly flag the gap "
            "rather than inventing or silently omitting it. Return extra key "
            '"missing_fields" with the gap list.\n\n'
        )

    schema_dict: dict[str, Any] = {
        "input": "(the worker's raw notes)",
        "target": {
            "participant_id": "...",
            "date_of_service": "...",
            "duration_minutes": 0,
            "service_type": "...",
            "goal_linkage": "...",
            "location": "...",
            "staff_presented_by": "...",
            "participant_present": True,
            "narrative_summary": "...",
            "billable_evidence": "...",
            "outcomes_achieved": ["..."],
            "risk_management": "[not recorded]",
            "follow_up_needed": False,
            "follow_up_notes": "[not recorded]",
        },
    }
    if stratum == "adv_pii_check":
        schema_dict["pii_handling"] = "(description of redacted items)"
    if stratum == "adv_missing_field":
        schema_dict["missing_fields"] = ["(field name)"]

    prompt += (
        f"Return STRICT JSON matching this schema:\n"
        f"{json.dumps(schema_dict, indent=2)}\n\n"
        "IMPORTANT: Do NOT include markdown fences, backticks, or any text outside "
        "the JSON object."
    )

    messages: list[ChatCompletionMessageParam] = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.85,
            max_tokens=1600,
            top_p=0.9,
        )
        raw = resp.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    assert raw is not None, "response content must not be None"

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))

    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON object found in response. First 200 chars:\n{cleaned[:200]}")
    return json.loads(cleaned[s : e + 1])


_seen_hashes: set[str] = set()


def _dedup_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


STRATUM_PLAN = {
    "routine_session": 120,
    "incident": 60,
    "capacity_building": 80,
    "sparse_input": 50,
    "adv_pii_check": 15,
    "adv_missing_field": 15,
}

ADV_STRATA = {"adv_pii_check", "adv_missing_field"}


def generate_dataset(
    count_per_stratum: dict[str, int] | None = None,
) -> list[dict]:
    """Generate records across all strata."""
    plan = count_per_stratum or STRATUM_PLAN
    records: list[dict] = []

    for stratum, target in plan.items():
        made = 0
        while made < target:
            try:
                obj = generate_one(stratum)
                if not obj.get("input") or not obj.get("target"):
                    print(f"  ⚠ empty output (retry {made + 1}/{target})")
                    continue

                key = _dedup_key(obj["input"])
                if key in _seen_hashes:
                    continue
                _seen_hashes.add(key)

                rec: dict[str, Any] = {
                    "id": f"synth-{stratum}-{made:04d}",
                    "stratum": stratum,
                    "input": obj["input"],
                    "target": obj["target"],
                    "meta": {
                        "model": OLLAMA_MODEL,
                        "generated_at": date.today().isoformat(),
                    },
                }
                if stratum in ADV_STRATA:
                    rec["pii_handling"] = obj.get("pii_handling")
                    rec["missing_fields"] = obj.get("missing_fields")

                records.append(rec)
                made += 1
            except (ValueError, RuntimeError) as exc:
                print(f"  ✗ retry ({stratum}): {exc})")
                time.sleep(0.5)

    return records
