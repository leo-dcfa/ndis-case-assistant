"""Synthetic NDIS case-note data generator (library)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import time
from collections.abc import Callable
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
    "\nJSON TYPES (exact — wrong types are rejected):\n"
    "- participant_present and follow_up_needed are JSON booleans: true / false "
    '(never the strings "yes"/"no").\n'
    "- duration_minutes is a JSON integer (e.g. 60), or -1 if genuinely not recorded.\n"
    '- outcomes_achieved is a JSON array of strings (e.g. ["..."]), never a single string.\n'
    "- Include ALL required keys, every time.\n"
    "\nGAP DISCIPLINE:\n"
    '- Use "[not recorded]" (or -1 for duration) ONLY for an element the worker input '
    "genuinely does not contain. If the input contains it, you MUST fill it — do not "
    'lazily write "[not recorded]".\n'
    "\n- Output STRICT JSON only. No markdown fences, no commentary, no explanation."
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

# Must stay in sync with config/required_fields.yaml service_type options.
SERVICE_TYPES = [
    "Assistance with Daily Life",
    "Assistance with Social, Economic and Community Participation",
    "Development of Daily Living and Life Skills",
    "Group and Centre Based Activities",
    "Transport",
    "Short Term Accommodation and Assistance",
    "Support Coordination",
    "Specialist Support Coordination",
    "Therapy Supports",
    "Consumables",
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


def generate_one(stratum: str, model: str | None = None) -> dict[str, Any]:
    """Ask the local LLM teacher to produce ONE input → target pair.

    Design: the worker INPUT is template-rendered from a controlled set of sampled
    facts (faithful by construction), and the teacher's only job is to write the
    polished TARGET note from that input. This stops the teacher inventing numbers
    the input never contained (the main source of dropped pairs) while still giving
    LLM-quality, varied target prose.
    """
    from ndis.notes import coerce_draft

    model = model or OLLAMA_MODEL
    facts = _sample_facts(stratum, random)
    style_key = random.choice(list(INPUT_STYLES.keys()))
    input_text, extra = _render_input_offline(facts, style_key, stratum)
    seed_hint = _build_stratum_seed(stratum, facts["service_type"])

    target_schema = {
        "participant_id": "...",
        "date_of_service": "YYYY-MM-DD",
        "duration_minutes": 0,
        "service_type": facts["service_type"],
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
    }

    prompt = (
        "You are given a support worker's RAW INPUT. Produce ONLY the structured NDIS "
        "case note (the target) as JSON, derived STRICTLY from that input.\n\n"
        f"Context: {STRATUM_PROMPTS[stratum]}\n"
        f"Tone angle (do NOT add facts): {seed_hint}\n\n"
        "--- RAW WORKER INPUT (the ONLY source of truth) ---\n"
        f"{input_text}\n"
        "---------------------------------------------------\n\n"
        "Rules:\n"
        "- Use ONLY facts present in the raw input. Do NOT introduce any number, date, "
        "name, duration, location, or detail that is not in the input.\n"
        "- Record only OBSERVABLE facts from the input. Do NOT characterise the "
        "participant's mood, engagement, progress, or the significance of events "
        "(e.g. 'engaged well', 'a milestone', 'as usual') unless those words are in the input.\n"
        "- Reword into calm, objective, third-person, past-tense professional prose.\n"
        f"- service_type must be EXACTLY: {facts['service_type']}\n"
    )

    if stratum in ("sparse_input", "adv_missing_field"):
        prompt += (
            "- GAP scenario: the input genuinely omits some required element(s). Flag ONLY "
            'those as "[not recorded]" (or -1 for duration); fill every other field from '
            "the input.\n"
        )
    else:
        prompt += (
            "- The input contains every required element. Fill EVERY field with a real value "
            'from the input — do NOT use "[not recorded]" or -1 anywhere.\n'
        )

    if stratum == "adv_pii_check":
        prompt += (
            "- The input contains third-party personal details (a name and/or phone/email). "
            "These MUST NOT appear in the note — redact/anonymise them.\n"
        )

    prompt += (
        "\nReturn STRICT JSON for the target note with exactly these keys "
        "(no wrapping object, no markdown):\n"
        f"{json.dumps(target_schema, indent=2)}"
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
            model=model,
            messages=messages,
            temperature=0.7,
            # Qwen3 is a thinking model. Disabling thinking (think=False) yields
            # unparseable/empty output here, so we leave it ON and give a budget
            # large enough for the reasoning AND the JSON; coerce_draft() strips the
            # <think> block before parsing. (Measured: 27B ~20s/call, 8B ~5s.)
            max_tokens=6000,
            top_p=0.9,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    target = coerce_draft(raw)
    if not target:
        raise ValueError(f"No JSON object found in response. First 200 chars:\n{raw[:200]!r}")

    obj: dict[str, Any] = {"input": input_text, "target": target}
    if stratum == "adv_pii_check":
        inj = extra.get("pii_injected", {})
        obj["pii_handling"] = (
            f"Redacted third-party details ({inj.get('name')}, {inj.get('phone')})."
        )
        obj["forbidden_pii"] = [s for s in (inj.get("name"), inj.get("phone")) if s]
    if stratum in {"sparse_input", "adv_missing_field"}:
        obj["missing_fields"] = extra.get("missing_fields", [])
    return obj


# ---------------------------------------------------------------------------
# Offline deterministic generator (no LLM).
#
# Produces faithful input -> target pairs from a single sampled set of facts,
# so every value in the target is provably grounded in the input. Used to make
# the whole pipeline (synth -> splits -> eval -> scorecard) runnable with zero
# external dependencies — important for CI and for the Phase 1 DoD, which only
# requires the *eval harness* to run end-to-end on a dummy model. The licensed
# open-weight teacher (Ollama) remains the path for real training data.
# ---------------------------------------------------------------------------

from ndis.notes import DURATION_GAP, GAP_MARKER  # noqa: E402

ACTIVITY_BY_SERVICE = {
    "Assistance with Daily Life": "completed personal-care and household routines",
    "Assistance with Social, Economic and Community Participation": (
        "attended a community group activity"
    ),
    "Development of Daily Living and Life Skills": "practised a targeted independence skill",
    "Group and Centre Based Activities": "took part in a supported group session",
    "Transport": "travelled to a scheduled appointment",
    "Short Term Accommodation and Assistance": "settled into short-term accommodation",
    "Support Coordination": "reviewed supports and coordinated services",
    "Specialist Support Coordination": "coordinated complex supports with providers",
    "Therapy Supports": "completed a structured assessment session",
    "Consumables": "used consumable supports for the activity",
    "Other": "engaged in the planned support activity",
}

# Fake third-party PII injected into adversarial inputs — must NOT survive
# into the target note.
ADV_PII_SNIPPETS = [
    ("call mum Jenny on 0412 345 678", "Jenny", "0412 345 678"),
    ("neighbour Tom Reed phoned 03 9123 4567", "Tom Reed", "03 9123 4567"),
    ("contact sister at sarah.k@example.com", "sarah.k@example.com", ""),
]


def _sample_facts(stratum: str, rng: random.Random) -> dict[str, Any]:
    service = rng.choice(SERVICE_TYPES)
    return {
        "participant_id": rng.choice(FAKE_PARTICIPANTS),
        "date": rng.choice(FAKE_DATES).isoformat(),
        "duration": rng.choice([30, 45, 60, 75, 90, 120]),
        "service_type": service,
        "goal": rng.choice(FAKE_GOALS),
        "location": rng.choice(FAKE_LOCATIONS),
        "staff": rng.choice(FAKE_STAFF),
        "present": rng.random() > 0.15,
        "activity": ACTIVITY_BY_SERVICE.get(service, ACTIVITY_BY_SERVICE["Other"]),
        "outcome": rng.choice(OUTCOME_POOL),
        "risk": rng.choice(
            [
                "participant became briefly agitated and was supported to de-escalate",
                "participant reported feeling unwell and was monitored throughout",
                "transport was delayed, shortening the planned activity",
            ]
        )
        if stratum == "incident"
        else None,
        "follow_up": stratum in {"incident", "capacity_building"},
    }


def _render_input_offline(facts: dict[str, Any], style: str, stratum: str) -> tuple[str, dict]:
    """Render raw worker input plus an ``extra`` dict (pii/missing metadata)."""
    extra: dict[str, Any] = {}
    sparse = stratum in {"sparse_input", "adv_missing_field"}

    lines = [
        f"worker: {facts['staff']}",
        f"date: {facts['date']}",
        f"participant: {facts['participant_id']}",
        f"type: {facts['service_type']}",
        f"goal: {facts['goal']}",
    ]
    if not sparse:
        lines.append(f"location: {facts['location']}")
        lines.append(f"duration ~{facts['duration']} min")
    else:
        # Genuinely omit the duration (and location) from the input.
        extra["missing_fields"] = ["duration_minutes", "location"]
    lines.append(f"present: {'yes' if facts['present'] else 'no'}")
    lines.append(f"did: {facts['activity']}")
    if facts["risk"]:
        lines.append(f"note: {facts['risk']}")
    lines.append(f"outcome: {facts['outcome'].lower()}")

    if stratum == "adv_pii_check":
        snippet, name, phone = rng_choice_pii(facts)
        lines.append(f"aside: {snippet}")
        extra["pii_injected"] = {"name": name, "phone": phone, "snippet": snippet}

    text = "\n".join(lines)
    if style == "dictation":
        filler = "um, so " if stratum != "incident" else "okay so "
        text = filler + text.replace("\n", "; ")
    return text, extra


def rng_choice_pii(facts: dict[str, Any]) -> tuple[str, str, str]:
    # Deterministic pick keyed off the participant id so offline runs are stable.
    idx = sum(ord(c) for c in facts["participant_id"]) % len(ADV_PII_SNIPPETS)
    return ADV_PII_SNIPPETS[idx]


def _render_target_offline(facts: dict[str, Any], stratum: str) -> dict[str, Any]:
    sparse = stratum in {"sparse_input", "adv_missing_field"}
    present_phrase = (
        "The participant was present."
        if facts["present"]
        else ("The participant was not present for the session.")
    )
    narrative = (
        f"Support worker {facts['staff']} delivered a {facts['service_type']} session "
        f"with participant {facts['participant_id']} on {facts['date']}. {present_phrase} "
        f"During the session the participant {facts['activity']}."
    )
    if facts["risk"]:
        narrative += f" Of note, {facts['risk']}."

    duration = DURATION_GAP if sparse else facts["duration"]
    location = GAP_MARKER if sparse else facts["location"]
    billable = (
        f"Session delivered {facts['service_type']} support toward the participant's goal; "
        + (
            "duration not recorded in worker input."
            if sparse
            else f"duration {facts['duration']} minutes."
        )
    )

    target: dict[str, Any] = {
        "participant_id": facts["participant_id"],
        "date_of_service": facts["date"],
        "duration_minutes": duration,
        "service_type": facts["service_type"],
        "goal_linkage": facts["goal"],
        "location": location,
        "staff_presented_by": facts["staff"],
        "participant_present": facts["present"],
        "narrative_summary": narrative,
        "billable_evidence": billable,
        "outcomes_achieved": [facts["outcome"]],
        "risk_management": facts["risk"] if facts["risk"] else GAP_MARKER,
        "follow_up_needed": facts["follow_up"],
        "follow_up_notes": "Continue per support plan." if facts["follow_up"] else GAP_MARKER,
    }
    return target


def generate_one_offline(stratum: str, rng: random.Random) -> dict[str, Any]:
    facts = _sample_facts(stratum, rng)
    style = rng.choice(list(INPUT_STYLES.keys()))
    text, extra = _render_input_offline(facts, style, stratum)
    target = _render_target_offline(facts, stratum)
    obj: dict[str, Any] = {"input": text, "target": target}
    if stratum == "adv_pii_check":
        inj = extra.get("pii_injected", {})
        obj["pii_handling"] = (
            f"Redacted third-party contact details ({inj.get('name')}, "
            f"{inj.get('phone')}) from the note."
        )
        # Exact strings that must NOT survive into the target — used by the
        # PII scorer as a precise cross-check.
        obj["forbidden_pii"] = [s for s in (inj.get("name"), inj.get("phone")) if s]
    if stratum in {"sparse_input", "adv_missing_field"}:
        obj["missing_fields"] = extra.get("missing_fields", [])
    return obj


_seen_hashes: set[str] = set()


def _dedup_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


# Weighted toward the safety-critical behaviours (gap-flagging + PII redaction)
# so training data teaches them, not just routine notes. --count scales these
# proportionally. ~59% varied routine, ~41% hard behaviours.
STRATUM_PLAN = {
    "routine_session": 110,
    "incident": 50,
    "capacity_building": 50,
    "sparse_input": 55,
    "adv_pii_check": 45,
    "adv_missing_field": 45,
}

ADV_STRATA = {"adv_pii_check", "adv_missing_field"}


def generate_dataset(
    count_per_stratum: dict[str, int] | None = None,
    *,
    offline: bool = False,
    seed: int = 42,
    verifier: "Callable[[str, dict, str], bool] | None" = None,
    teacher_model: str | None = None,
    checkpoint_path: "pathlib.Path | None" = None,
) -> list[dict]:
    """Generate records across all strata.

    ``offline=True`` uses the deterministic template generator (no LLM, no
    network); otherwise the licensed open-weight Ollama teacher is used.

    ``verifier(input, target, stratum) -> bool`` is an optional compliance gate:
    any pair it rejects is dropped and regenerated, so only fully-compliant
    exemplars (faithful + schema-valid + PII-clean + structured) become training
    data. Build one with ``eval.verify.build_compliance_verifier`` — it must be
    LOCAL (rubric + local judge), never a frontier API.
    """
    plan = count_per_stratum or STRATUM_PLAN
    records: list[dict] = []
    dropped = 0
    rng = random.Random(seed)

    # Hard safety caps so a flaky teacher or an over-zealous filter can NEVER loop
    # forever (the failure mode that hung a run for 90 min). Whichever trips first
    # ends the stratum; a shortfall is reported, not silently hidden.
    max_consecutive_fail = 60 if offline else 25

    for stratum, target in plan.items():
        made = 0
        attempts = 0
        consecutive_fail = 0
        max_attempts = target * (50 if offline else 20) + 30
        while made < target:
            attempts += 1
            if attempts > max_attempts or consecutive_fail >= max_consecutive_fail:
                print(
                    f"  ⚠ {stratum}: stopped at {made}/{target} after {attempts - 1} attempts "
                    f"({consecutive_fail} consecutive failures) — cap hit"
                )
                break
            try:
                obj = (
                    generate_one_offline(stratum, rng)
                    if offline
                    else generate_one(stratum, model=teacher_model)
                )
                if not obj.get("input") or not obj.get("target"):
                    consecutive_fail += 1
                    continue

                if verifier is not None and not verifier(obj["input"], obj["target"], stratum):
                    dropped += 1
                    consecutive_fail += 1
                    print(f"  ✗ dropped non-compliant pair ({stratum}); {dropped} dropped so far")
                    continue

                key = _dedup_key(obj["input"])
                if key in _seen_hashes:
                    consecutive_fail += 1
                    continue
                _seen_hashes.add(key)

                rec: dict[str, Any] = {
                    "id": f"synth-{stratum}-{made:04d}",
                    "stratum": stratum,
                    "input": obj["input"],
                    "target": obj["target"],
                    "meta": {
                        "model": "offline-template"
                        if offline
                        else (teacher_model or OLLAMA_MODEL),
                        "teacher_license": "n/a" if offline else "open-weight (Apache-2.0)",
                        "generated_at": date.today().isoformat(),
                    },
                }
                if stratum in ADV_STRATA:
                    rec["pii_handling"] = obj.get("pii_handling")
                    rec["missing_fields"] = obj.get("missing_fields")
                    if obj.get("forbidden_pii"):
                        rec["forbidden_pii"] = obj["forbidden_pii"]

                records.append(rec)
                if checkpoint_path is not None:
                    # Append-as-you-go so a long batch survives an interrupt.
                    with checkpoint_path.open("a", encoding="utf-8") as ckpt:
                        ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
                made += 1
                consecutive_fail = 0
            except (ValueError, RuntimeError) as exc:
                consecutive_fail += 1
                print(f"  ✗ retry ({stratum}): {exc}")
                if not offline:
                    time.sleep(0.5)

    if dropped:
        print(f"Compliance verifier dropped {dropped} pair(s) before they entered the dataset.")
    return records


def main(argv: list[str] | None = None) -> int:
    """CLI: generate a stratified synthetic dataset and write train/val/test splits."""
    import argparse

    from ndis.splits import write_splits

    parser = argparse.ArgumentParser(description="Generate synthetic NDIS case-note data.")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Total records to generate (spread across strata by STRATUM_PLAN weights).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic no-LLM generator (default uses the Ollama teacher).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/splits",
        help="Directory to write seed.jsonl + train/val/test splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-model",
        default=OLLAMA_MODEL,
        help=f"Local Ollama teacher model (default {OLLAMA_MODEL}; use qwen3:8b for ~4x speed).",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Keep only fully-compliant pairs (faithful + schema-valid + PII-clean + structured) "
        "via a LOCAL verifier. Recommended for teacher runs.",
    )
    parser.add_argument(
        "--filter-judge",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Faithfulness judge inside --filter. 'heuristic' catches only number/ID fabrication; "
        "'llm' (local 27B) also catches embellished prose. Both run locally.",
    )
    args = parser.parse_args(argv)

    # Scale STRATUM_PLAN down/up to the requested total, keeping proportions and
    # guaranteeing at least one record per stratum.
    total_plan = sum(STRATUM_PLAN.values())
    scale = args.count / total_plan
    plan = {s: max(1, round(n * scale)) for s, n in STRATUM_PLAN.items()}

    verifier: Callable[[str, dict, str], bool] | None = None
    if args.filter:
        # Lazy import keeps ndis independent of the eval package unless filtering.
        from eval.verify import build_compliance_verifier

        verifier = build_compliance_verifier(args.filter_judge)
        print(
            f"Compliance verifier ON: faithful + schema-valid + PII-clean + structured "
            f"(faithfulness judge: {verifier.judge_name})."  # type: ignore[attr-defined]
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "seed.jsonl"
    checkpoint.unlink(missing_ok=True)  # fresh run; append-as-you-go below

    mode = "offline template" if args.offline else f"LLM teacher ({args.teacher_model})"
    print(f"Generating ~{sum(plan.values())} records via {mode} (checkpoint: {checkpoint}) ...")
    records = generate_dataset(
        plan,
        offline=args.offline,
        seed=args.seed,
        verifier=verifier,
        teacher_model=args.teacher_model,
        checkpoint_path=checkpoint,
    )

    counts = write_splits(records, out_dir, seed=args.seed)
    print(f"\nWrote {counts['seed']} records to {out_dir}/")
    for name in ("train", "val", "test"):
        print(f"  {name}: {counts[name]}")

    dist: dict[str, int] = {}
    for rec in records:
        dist[rec["stratum"]] = dist.get(rec["stratum"], 0) + 1
    print("Stratum distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
