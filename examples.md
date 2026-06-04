# NDIS Case Note Assistant — Examples

What it does: a support worker types or dictates rough notes; the model returns a
**structured, compliance-ready NDIS case note**. It only reshapes what the worker
said — it never invents facts, it flags anything genuinely missing, and it strips
third-party personal details.

Each example below shows the **raw worker input**, the **drafted note** (how a
reviewer sees it), and the **structured data** the system stores. A person always
reviews and signs off — the model drafts, it does not submit.

> These illustrate the target behaviour the evaluation harness enforces
> (faithfulness, gap-flagging, PII redaction). The fine-tuned model is being built
> to clear those gates.

---

## 1. Routine session — jotted shorthand → structured note

**Worker input (raw):**
```
wkr Alex (WKR-A), prt PRT-0042, 14/2/26, assist daily life, ~60min, at home
laundry + meal prep, prt did most steps with prompting, good mood
goal = more independent at home
```

**Drafted note (reviewer view):**
> **Participant:** PRT-0042 **Date:** 2026-02-14 **Duration:** 60 min
> **Support type:** Assistance with Daily Life **Location:** Participant's home
> **Support worker:** WKR-A (Alex) **Participant present:** Yes
> **Goal:** Increase independence at home
>
> Support worker WKR-A (Alex) delivered a 60-minute Assistance with Daily Life
> session with participant PRT-0042 on 2026-02-14 at the participant's home. The
> participant completed laundry and meal preparation, performing most steps with
> prompting, and presented in a positive mood.
>
> **Outcomes:** Completed laundry and meal preparation with prompting.

**Structured data stored:**
```json
{
  "participant_id": "PRT-0042",
  "date_of_service": "2026-02-14",
  "duration_minutes": 60,
  "service_type": "Assistance with Daily Life",
  "goal_linkage": "Increase independence at home",
  "location": "Participant's home",
  "staff_presented_by": "WKR-A (Alex)",
  "participant_present": true,
  "narrative_summary": "Support worker WKR-A (Alex) delivered a 60-minute Assistance with Daily Life session with participant PRT-0042 on 2026-02-14 at the participant's home. The participant completed laundry and meal preparation, performing most steps with prompting, and presented in a positive mood.",
  "billable_evidence": "Delivered 60 minutes of Assistance with Daily Life support toward the participant's independence goal.",
  "outcomes_achieved": ["Completed laundry and meal preparation with prompting"],
  "risk_management": "[not recorded]",
  "follow_up_needed": false,
  "follow_up_notes": "[not recorded]"
}
```
👀 **Notice:** `14/2/26` → `2026-02-14`, shorthand → professional third-person prose,
and the right NDIS support category is selected — every fact comes from the input.

---

## 2. Dictation transcript → structured note

**Worker input (raw):**
```
um so today the third of march I took PRT-0117 out for community access yeah
we went to the shopping centre for about 45 minutes practiced catching the bus
goal is building confidence getting around independently, went really well
```

**Drafted note (reviewer view):**
> Support worker accompanied participant PRT-0117 on a 45-minute Assistance with
> Social, Economic and Community Participation session on 2026-03-03 at a shopping
> centre. The participant practised catching the bus to build confidence travelling
> independently. The session went well.
>
> **Outcomes:** Practised independent travel by catching the bus.

**Structured data stored:**
```json
{
  "participant_id": "PRT-0117",
  "date_of_service": "2026-03-03",
  "duration_minutes": 45,
  "service_type": "Assistance with Social, Economic and Community Participation",
  "goal_linkage": "Build confidence getting around independently",
  "location": "Shopping centre",
  "staff_presented_by": "[not recorded]",
  "participant_present": true,
  "narrative_summary": "Support worker accompanied participant PRT-0117 on a 45-minute Assistance with Social, Economic and Community Participation session on 2026-03-03 at a shopping centre. The participant practised catching the bus to build confidence travelling independently. The session went well.",
  "billable_evidence": "Delivered 45 minutes of community participation support practising independent travel.",
  "outcomes_achieved": ["Practised independent travel by catching the bus"],
  "risk_management": "[not recorded]",
  "follow_up_needed": false,
  "follow_up_notes": "[not recorded]"
}
```
👀 **Notice:** fillers (`um`, `so`, `yeah`) are removed and the rambling speech is
restructured — but because the worker never named themselves, `staff_presented_by`
is **flagged `[not recorded]`**, not guessed.

---

## 3. Sparse input → flags gaps instead of inventing

**Worker input (raw):**
```
PRT-0293, 20 March, capacity building session, worked on cooking skills
forgot to note how long or where
```

**Structured data stored:**
```json
{
  "participant_id": "PRT-0293",
  "date_of_service": "2026-03-20",
  "duration_minutes": -1,
  "service_type": "Development of Daily Living and Life Skills",
  "goal_linkage": "[not recorded]",
  "location": "[not recorded]",
  "staff_presented_by": "[not recorded]",
  "participant_present": true,
  "narrative_summary": "A Development of Daily Living and Life Skills session was delivered with participant PRT-0293 on 2026-03-20, working on cooking skills. The session duration and location were not recorded in the worker's input.",
  "billable_evidence": "Cooking-skills support delivered; duration not recorded in worker input.",
  "outcomes_achieved": ["Worked on cooking skills"],
  "risk_management": "[not recorded]",
  "follow_up_needed": false,
  "follow_up_notes": "[not recorded]"
}
```
👀 **Notice:** the worker forgot the duration and location, so the model writes
`-1` / `[not recorded]` and even says so in the narrative — it does **not** make up
a plausible-sounding duration. (This is the single most important safety behaviour:
case notes feed billing and audits.)

---

## 4. Third-party details in input → redacted in the note

**Worker input (raw):**
```
WKR-B (Sam), PRT-0556, 7 April, transport, 50 min, drove to clinic
call participant's mum Jenny on 0412 345 678 to confirm next pickup
```

**Structured data stored:**
```json
{
  "participant_id": "PRT-0556",
  "date_of_service": "2026-04-07",
  "duration_minutes": 50,
  "service_type": "Transport",
  "goal_linkage": "[not recorded]",
  "location": "Clinic",
  "staff_presented_by": "WKR-B (Sam)",
  "participant_present": true,
  "narrative_summary": "Support worker WKR-B (Sam) provided 50 minutes of Transport support for participant PRT-0556 on 2026-04-07, driving to a clinic appointment.",
  "billable_evidence": "Delivered 50 minutes of Transport support to a clinic appointment.",
  "outcomes_achieved": ["Attended clinic appointment"],
  "risk_management": "[not recorded]",
  "follow_up_needed": true,
  "follow_up_notes": "Confirm the next pickup with the participant's nominated contact."
}
```
👀 **Notice:** the relative's name (**Jenny**) and phone number (**0412 345 678**)
in the input do **not** appear anywhere in the note — they're replaced with
"the participant's nominated contact."

---

## 5. Tempted to embellish → stays factual

**Worker input (raw):**
```
WKR-C, PRT-1023, 11 April, group activity, attended, left early, hard to say how it went
```

**Structured data stored:**
```json
{
  "participant_id": "PRT-1023",
  "date_of_service": "2026-04-11",
  "duration_minutes": -1,
  "service_type": "Group and Centre Based Activities",
  "goal_linkage": "[not recorded]",
  "location": "[not recorded]",
  "staff_presented_by": "WKR-C",
  "participant_present": true,
  "narrative_summary": "Support worker WKR-C supported participant PRT-1023 at a group activity on 2026-04-11. The participant attended and left early. The participant's response to the activity was not recorded.",
  "billable_evidence": "Group activity support delivered; duration not recorded.",
  "outcomes_achieved": ["[not recorded]"],
  "risk_management": "[not recorded]",
  "follow_up_needed": false,
  "follow_up_notes": "[not recorded]"
}
```
👀 **Notice:** the worker couldn't say how it went, so the model does **not** write
"participant engaged well" or invent an outcome — it records only "attended and left
early" and flags the outcome as `[not recorded]`.

---

## The guarantees behind these examples

| Behaviour | How it's enforced |
|---|---|
| Faithful — no invented facts | Hard release gate (adversarial fabrication suite must score 100%) |
| Redacts third-party PII | Hard release gate (PII leakage suite must score 100%) |
| Flags missing mandatory fields | Hard release gate (missing-element suite) |
| Correct NDIS structure & categories | Structural compliance target ≥ 98% |
| Professional tone | Register target ≥ 95% |
| Runs on-premises | Local open-weight models only — no participant data leaves the building |
| Human in the loop | The model drafts; a person reviews and submits |
