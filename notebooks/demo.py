"""NDIS Case Note Assistant — interactive demo (marimo).

Run it:
    uvx marimo edit notebooks/demo.py      # edit / present
    uvx marimo run  notebooks/demo.py      # read-only app view

The curated examples render with no model or GPU needed. The optional "Try it
live" cell at the bottom loads the fine-tuned local model on demand.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="NDIS Case Note Assistant")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ---- Theme (original styling; all values defined here) -------------------
    THEME = {
        "bg": "#f5f7fb",
        "card": "#ffffff",
        "ink": "#16233b",
        "muted": "#6b7689",
        "line": "#e6eaf2",
        "navy": "#102a43",
        "accent": "#0e8f88",  # calm teal
        "accent_soft": "#e7f5f4",
        "raw_bg": "#0f1b2d",  # dark panel for the "raw input"
        "raw_ink": "#cdd9e5",
        "gap": "#b4690e",  # amber for [not recorded]
        "ok": "#0e8f88",
        "font": (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
        ),
        "mono": "'SF Mono', 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace",
    }
    return (THEME,)


@app.cell
def _(THEME, mo):
    mo.Html(
        f"""
        <div style="font-family:{THEME["font"]}; background:{THEME["bg"]};
                    border-radius:18px; padding:34px 30px; margin-bottom:6px;
                    border:1px solid {THEME["line"]};">
          <div style="display:inline-block; font-size:12px; font-weight:700;
                      letter-spacing:.12em; text-transform:uppercase;
                      color:{THEME["accent"]}; background:{THEME["accent_soft"]};
                      padding:6px 12px; border-radius:999px;">On-prem · faithful · human-reviewed</div>
          <h1 style="color:{THEME["navy"]}; font-size:30px; margin:16px 0 6px;
                     letter-spacing:-.02em;">NDIS Case Note Assistant</h1>
          <p style="color:{THEME["muted"]}; font-size:16px; max-width:60ch; margin:0;
                    line-height:1.55;">
            A support worker types or dictates rough notes; the model returns a
            structured, compliance-ready case note — reshaping only what was said,
            flagging anything missing, and never inventing facts.
          </p>
        </div>
        """
    )
    return


@app.cell
def _():
    # ---- Curated, faithful examples (no model needed to render) --------------
    EXAMPLES = [
        {
            "tag": "Routine session",
            "title": "Shorthand → structured note",
            "notice": "A messy date and abbreviations become a clean professional note — "
            "every fact traceable to the input.",
            "input": "wkr Alex (WKR-A), prt PRT-0042, 14/2/26, assist daily life, ~60min, at home\n"
            "laundry + meal prep, prt did most steps with prompting, good mood\n"
            "goal = more independent at home",
            "note": {
                "participant_id": "PRT-0042",
                "date_of_service": "2026-02-14",
                "duration_minutes": 60,
                "service_type": "Assistance with Daily Life",
                "goal_linkage": "Increase independence at home",
                "location": "Participant's home",
                "staff_presented_by": "WKR-A (Alex)",
                "participant_present": True,
                "narrative_summary": "Support worker WKR-A (Alex) delivered a 60-minute "
                "Assistance with Daily Life session with participant PRT-0042 on 2026-02-14 at "
                "the participant's home. The participant completed laundry and meal preparation, "
                "performing most steps with prompting, and presented in a positive mood.",
                "billable_evidence": "Delivered 60 minutes of Assistance with Daily Life support "
                "toward the participant's independence goal.",
                "outcomes_achieved": ["Completed laundry and meal preparation with prompting"],
                "risk_management": "[not recorded]",
                "follow_up_needed": False,
                "follow_up_notes": "[not recorded]",
            },
        },
        {
            "tag": "Dictation",
            "title": "Spoken transcript → structured note",
            "notice": "Fillers (“um”, “so”, “yeah”) are stripped — and because the worker never "
            "named themselves, the support-worker field is flagged, not guessed.",
            "input": "um so today the third of march I took PRT-0117 out for community access yeah\n"
            "we went to the shopping centre for about 45 minutes practiced catching the bus\n"
            "goal is building confidence getting around independently, went really well",
            "note": {
                "participant_id": "PRT-0117",
                "date_of_service": "2026-03-03",
                "duration_minutes": 45,
                "service_type": "Assistance with Social, Economic and Community Participation",
                "goal_linkage": "Build confidence getting around independently",
                "location": "Shopping centre",
                "staff_presented_by": "[not recorded]",
                "participant_present": True,
                "narrative_summary": "Support worker accompanied participant PRT-0117 on a "
                "45-minute community participation session on 2026-03-03 at a shopping centre. "
                "The participant practised catching the bus to build confidence travelling "
                "independently.",
                "billable_evidence": "Delivered 45 minutes of community participation support "
                "practising independent travel.",
                "outcomes_achieved": ["Practised independent travel by catching the bus"],
                "risk_management": "[not recorded]",
                "follow_up_needed": False,
                "follow_up_notes": "[not recorded]",
            },
        },
        {
            "tag": "Sparse input",
            "title": "Flags gaps instead of inventing",
            "notice": "The worker forgot the duration and location, so the model writes "
            "“[not recorded]” rather than making up a plausible value — the key safety behaviour.",
            "input": "PRT-0293, 20 March, capacity building session, worked on cooking skills\n"
            "forgot to note how long or where",
            "note": {
                "participant_id": "PRT-0293",
                "date_of_service": "2026-03-20",
                "duration_minutes": -1,
                "service_type": "Development of Daily Living and Life Skills",
                "goal_linkage": "[not recorded]",
                "location": "[not recorded]",
                "staff_presented_by": "[not recorded]",
                "participant_present": True,
                "narrative_summary": "A Development of Daily Living and Life Skills session was "
                "delivered with participant PRT-0293 on 2026-03-20, working on cooking skills. "
                "Session duration and location were not recorded in the worker's input.",
                "billable_evidence": "Cooking-skills support delivered; duration not recorded.",
                "outcomes_achieved": ["Worked on cooking skills"],
                "risk_management": "[not recorded]",
                "follow_up_needed": False,
                "follow_up_notes": "[not recorded]",
            },
        },
        {
            "tag": "Privacy",
            "title": "Third-party details redacted",
            "notice": "A relative's name and phone number in the input do not appear anywhere "
            "in the note.",
            "input": "WKR-B (Sam), PRT-0556, 7 April, transport, 50 min, drove to clinic\n"
            "call participant's mum Jenny on 0412 345 678 to confirm next pickup",
            "note": {
                "participant_id": "PRT-0556",
                "date_of_service": "2026-04-07",
                "duration_minutes": 50,
                "service_type": "Transport",
                "goal_linkage": "[not recorded]",
                "location": "Clinic",
                "staff_presented_by": "WKR-B (Sam)",
                "participant_present": True,
                "narrative_summary": "Support worker WKR-B (Sam) provided 50 minutes of Transport "
                "support for participant PRT-0556 on 2026-04-07, driving to a clinic appointment.",
                "billable_evidence": "Delivered 50 minutes of Transport support to a clinic "
                "appointment.",
                "outcomes_achieved": ["Attended clinic appointment"],
                "risk_management": "[not recorded]",
                "follow_up_needed": True,
                "follow_up_notes": "Confirm the next pickup with the participant's nominated "
                "contact.",
            },
        },
    ]
    return (EXAMPLES,)


@app.cell
def _(THEME):
    # ---- Rendering helpers (inline styles -> always render) ------------------
    LABELS = {
        "participant_id": "Participant",
        "date_of_service": "Date of service",
        "duration_minutes": "Duration (min)",
        "service_type": "Support type",
        "goal_linkage": "Goal",
        "location": "Location",
        "staff_presented_by": "Support worker",
        "participant_present": "Participant present",
        "narrative_summary": "Narrative",
        "billable_evidence": "Billable evidence",
        "outcomes_achieved": "Outcomes",
        "risk_management": "Risk management",
        "follow_up_needed": "Follow-up needed",
        "follow_up_notes": "Follow-up notes",
    }
    GAP_TOKENS = {"[not recorded]", -1, "", None}

    def _is_gap(v):
        if isinstance(v, list):
            return len(v) == 0
        return v in GAP_TOKENS

    def _fmt(v):
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else "[not recorded]"
        if v == -1:
            return "[not recorded]"
        return str(v)

    def render_note(note: dict) -> str:
        rows = []
        for key, label in LABELS.items():
            v = note.get(key, "[not recorded]")
            gap = _is_gap(v)
            val_style = (
                f"color:{THEME['gap']}; font-style:italic;" if gap else f"color:{THEME['ink']};"
            )
            text = "[not recorded]" if gap else _fmt(v)
            rows.append(
                f"""<div style="display:grid; grid-template-columns:150px 1fr; gap:12px;
                        padding:7px 0; border-bottom:1px solid {THEME["line"]};">
                  <div style="color:{THEME["muted"]}; font-size:12px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.04em;">{label}</div>
                  <div style="font-size:14px; line-height:1.5; {val_style}">{text}</div>
                </div>"""
            )
        return "".join(rows)

    def render_card(ex: dict) -> str:
        return f"""
        <div style="font-family:{THEME["font"]}; background:{THEME["card"]};
                    border:1px solid {THEME["line"]}; border-radius:16px;
                    box-shadow:0 1px 3px rgba(16,42,67,.06); padding:22px 24px;
                    margin:18px 0;">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
            <span style="font-size:11px; font-weight:700; letter-spacing:.1em;
                  text-transform:uppercase; color:{THEME["accent"]};
                  background:{THEME["accent_soft"]}; padding:4px 10px;
                  border-radius:999px;">{ex["tag"]}</span>
            <span style="color:{THEME["navy"]}; font-weight:650; font-size:16px;">{ex["title"]}</span>
          </div>
          <div style="display:grid; grid-template-columns:0.9fr 1.1fr; gap:18px;
                      align-items:start;">
            <div>
              <div style="color:{THEME["muted"]}; font-size:11px; font-weight:700;
                    text-transform:uppercase; letter-spacing:.08em;
                    margin-bottom:6px;">Worker input</div>
              <pre style="background:{THEME["raw_bg"]}; color:{THEME["raw_ink"]};
                    font-family:{THEME["mono"]}; font-size:12.5px; line-height:1.55;
                    padding:14px 16px; border-radius:10px; white-space:pre-wrap;
                    margin:0;">{ex["input"]}</pre>
            </div>
            <div>
              <div style="color:{THEME["muted"]}; font-size:11px; font-weight:700;
                    text-transform:uppercase; letter-spacing:.08em;
                    margin-bottom:6px;">Drafted case note</div>
              {render_note(ex["note"])}
            </div>
          </div>
          <div style="margin-top:16px; background:{THEME["accent_soft"]};
                border-left:3px solid {THEME["accent"]}; border-radius:0 8px 8px 0;
                padding:11px 14px; color:{THEME["navy"]}; font-size:13.5px;
                line-height:1.5;">{ex["notice"]}</div>
        </div>
        """

    return (render_card,)


@app.cell
def _(THEME, mo):
    # ---- How it was built ---------------------------------------------------
    mo.Html(
        f"""
        <div style="font-family:{THEME["font"]}; background:{THEME["navy"]};
                    color:#dde7f1; border-radius:16px; padding:24px 26px; margin:22px 0;">
          <div style="font-size:11px; font-weight:700; letter-spacing:.12em;
                text-transform:uppercase; color:#7fd8d2;">Built &amp; trained on-premises</div>
          <h2 style="margin:8px 0 12px; color:#ffffff; font-size:20px;">Nothing leaves the building</h2>
          <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:14px 26px;
                font-size:14px; line-height:1.55;">
            <div>🔒 <b>Trained locally</b> — synthetic data generated and the model
                 fine-tuned (QLoRA) on a <b>single on-prem GPU</b> in minutes.</div>
            <div>🧾 <b>License-clean</b> — no frontier-model outputs in the training
                 lineage; only local open-weight models were used.</div>
            <div>🩺 <b>Privacy by design</b> — runs on-prem / on-device, so participant
                 data never leaves controlled infrastructure.</div>
            <div>🧪 <b>Every training example verified</b> — auto-checked for faithfulness,
                 PII and structure before it was allowed into the dataset.</div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(THEME, mo):
    # ---- How it works (the eval-driven loop) --------------------------------
    steps = [
        (
            "1",
            "Define the standard",
            "A configurable schema of the required NDIS note "
            "elements — the single source of truth for 'compliant'.",
        ),
        (
            "2",
            "Generate &amp; verify data",
            "A local model drafts synthetic examples; each is "
            "auto-checked for faithfulness, PII and structure — only clean ones are kept.",
        ),
        (
            "3",
            "Fine-tune locally",
            "A small model is QLoRA fine-tuned on that data, on one on-prem GPU, in minutes.",
        ),
        (
            "4",
            "Measure against hard gates",
            "Scored on a frozen test set + adversarial suites; "
            "faithfulness and PII must hit 100% before release.",
        ),
        ("5", "Iterate", "Failures point straight back to the data — fix, regenerate, retrain."),
    ]
    cards = "".join(
        f"""<div style="flex:1; min-width:150px; background:{THEME["bg"]};
              border:1px solid {THEME["line"]}; border-radius:12px; padding:14px;">
            <div style="width:26px; height:26px; border-radius:50%;
                  background:{THEME["accent"]}; color:#fff; font-weight:700; font-size:13px;
                  display:flex; align-items:center; justify-content:center;">{n}</div>
            <div style="margin-top:8px; font-weight:650; color:{THEME["navy"]};
                  font-size:14px;">{t}</div>
            <div style="margin-top:4px; color:{THEME["muted"]}; font-size:12.5px;
                  line-height:1.5;">{d}</div></div>"""
        for n, t, d in steps
    )
    mo.Html(
        f"""<div style="font-family:{THEME["font"]}; margin:22px 0;">
          <h2 style="margin:0 0 12px; color:{THEME["navy"]}; font-size:20px;">How it works</h2>
          <div style="display:flex; gap:12px; flex-wrap:wrap;">{cards}</div>
        </div>"""
    )
    return


@app.cell
def _(THEME, mo):
    # ---- Results so far -----------------------------------------------------
    def _row(metric, base, v1, note, warn=False):
        v1col = THEME["gap"] if warn else THEME["ok"]
        return f"""<tr>
          <td style="padding:9px 10px; border-bottom:1px solid {THEME["line"]};
                color:{THEME["ink"]};">{metric}</td>
          <td style="padding:9px 10px; border-bottom:1px solid {THEME["line"]};
                text-align:center; color:{THEME["muted"]};">{base}</td>
          <td style="padding:9px 10px; border-bottom:1px solid {THEME["line"]};
                text-align:center; font-weight:700; color:{v1col};">{v1}</td>
          <td style="padding:9px 10px; border-bottom:1px solid {THEME["line"]};
                color:{THEME["muted"]}; font-size:13px;">{note}</td></tr>"""

    rows = (
        _row("Structural compliance", "0%", "100%", "biggest fix — valid schema every time")
        + _row("Faithfulness (realistic notes)", "—", "~100%", "no invented facts in-distribution")
        + _row("Fabrication suite (hard gate)", "80%", "100%", "resists tempting edge cases")
        + _row("PII redaction (hard gate)", "60%", "50%", "v2 in progress — see below", warn=True)
        + _row(
            "Missing-field flagging (hard gate)",
            "100%",
            "60%",
            "v2 in progress — see below",
            warn=True,
        )
    )
    mo.Html(
        f"""
        <div style="font-family:{THEME["font"]}; background:{THEME["card"]};
              border:1px solid {THEME["line"]}; border-radius:16px; padding:22px 24px; margin:22px 0;">
          <h2 style="margin:0 0 4px; color:{THEME["navy"]}; font-size:20px;">Results so far (v1)</h2>
          <p style="margin:0 0 14px; color:{THEME["muted"]}; font-size:14px;">
            Measured on a frozen held-out test set plus adversarial safety suites.</p>
          <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <tr style="text-align:left; color:{THEME["muted"]}; font-size:11px;
                  text-transform:uppercase; letter-spacing:.06em;">
              <th style="padding:6px 10px;">Metric</th>
              <th style="padding:6px 10px; text-align:center;">Un-tuned</th>
              <th style="padding:6px 10px; text-align:center;">Fine-tuned v1</th>
              <th style="padding:6px 10px;">Notes</th></tr>
            {rows}
          </table>
          <div style="margin-top:16px; background:{THEME["accent_soft"]};
                border-left:3px solid {THEME["accent"]}; border-radius:0 8px 8px 0;
                padding:12px 14px; color:{THEME["navy"]}; font-size:13.5px; line-height:1.55;">
            <b>The improvement loop:</b> fine-tuning fixed structure (0→100%) and fabrication
            (80→100%). The evaluation then caught two safety gaps on adversarial cases — the model
            had learned surface forms, not behaviours. We traced it to the training data, widened
            the adversarial variety, and <b>v2 is regenerating now</b> to close them.
            <b>Nothing is approved for release until PII and faithfulness hit 100%.</b>
          </div>
          <p style="margin:12px 0 0; color:{THEME["muted"]}; font-size:12.5px; line-height:1.5;">
            <b>Caveats:</b> these use a fast automated grader (a rigorous model-judged pass is
            pending), and the test set is synthetic — real-world validation follows once
            de-identified real notes are available.</p>
        </div>
        """
    )
    return


@app.cell
def _(THEME, mo):
    # ---- Deployment & cost --------------------------------------------------
    mo.Html(
        f"""
        <div style="font-family:{THEME["font"]}; background:{THEME["card"]};
              border:1px solid {THEME["line"]}; border-radius:16px; padding:22px 24px; margin:22px 0;">
          <h2 style="margin:0 0 14px; color:{THEME["navy"]}; font-size:20px;">Deployment &amp; cost</h2>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px;">
            <div>
              <div style="font-size:11px; font-weight:700; letter-spacing:.08em;
                    text-transform:uppercase; color:{THEME["accent"]};">Runs on-device</div>
              <ul style="margin:8px 0 0; padding-left:18px; color:{THEME["ink"]};
                    font-size:14px; line-height:1.6;">
                <li><b>Any laptop / desktop:</b> yes today (≈6&nbsp;GB, 4-bit).</li>
                <li><b>Phone:</b> the smaller shrunk model (1.5–4B, ≈1–2&nbsp;GB) is
                    on-device-viable; the 8B is laptop-class.</li>
                <li>No internet required — fully offline.</li>
              </ul>
            </div>
            <div>
              <div style="font-size:11px; font-weight:700; letter-spacing:.08em;
                    text-transform:uppercase; color:{THEME["accent"]};">If hosted on cloud GPU</div>
              <ul style="margin:8px 0 0; padding-left:18px; color:{THEME["ink"]};
                    font-size:14px; line-height:1.6;">
                <li>One small GPU instance: <b>~$0.50–1.00 / hour</b>
                    (~$200–730/mo, less with spot/reserved).</li>
                <li>Batched serving → thousands of notes/day →
                    <b>a fraction of a cent per note</b>.</li>
                <li>Compare: a frontier API charges per call <i>and</i> sends data off-site.</li>
              </ul>
            </div>
          </div>
        </div>
        """
    )
    return


@app.cell
def _(THEME, mo):
    mo.Html(
        f"""<h2 style="font-family:{THEME["font"]}; color:{THEME["navy"]};
              margin:30px 0 4px;">A few examples</h2>
        <p style="font-family:{THEME["font"]}; color:{THEME["muted"]}; margin:0;
              font-size:14px;">Messy worker input on the left → the structured, compliance-ready
              note on the right.</p>"""
    )
    return


@app.cell
def _(EXAMPLES, mo, render_card):
    mo.Html("".join(render_card(ex) for ex in EXAMPLES))
    return


@app.cell
def _(THEME, mo):
    mo.Html(
        f"""<h2 style="font-family:{THEME["font"]}; color:{THEME["navy"]};
              margin:30px 0 4px;">Try it live</h2>
        <p style="font-family:{THEME["font"]}; color:{THEME["muted"]}; margin:0 0 8px;
              font-size:14px;">Type a rough note and draft one with the fine-tuned local
              model. (First run loads the model — a few seconds.)</p>"""
    )
    return


@app.cell
def _(mo):
    note_input = mo.ui.text_area(
        placeholder="e.g. saw PRT-0042 today, 60 min, cooking at home, went ok",
        rows=3,
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="Draft note")
    mo.vstack([note_input, run_btn])
    return note_input, run_btn


@app.cell
def _(load_model, mo, note_input, render_card, run_btn):
    if not run_btn.value or not note_input.value.strip():
        live = mo.md("")
    else:
        try:
            model = load_model()
            note = model.draft_note(note_input.value.strip())
            if not note:
                live = mo.md("> The model did not return a parseable note. Try rephrasing.")
            else:
                live = mo.Html(
                    render_card(
                        {
                            "tag": "Live",
                            "title": "Your note",
                            "input": note_input.value.strip(),
                            "note": note,
                            "notice": "Drafted on-device by the fine-tuned model. A reviewer "
                            "checks and signs off before anything is submitted.",
                        }
                    )
                )
        except Exception as exc:  # keep the notebook robust during a demo
            live = mo.md(
                f"> **Live model unavailable** ({type(exc).__name__}). The curated examples "
                f"above don't need the model.\n>\n> `{str(exc)[:160]}`"
            )
    live
    return


@app.cell
def _():
    # Load the fine-tuned model once and reuse across button clicks.
    import functools

    @functools.lru_cache(maxsize=1)
    def load_model():
        from eval.model_under_test import HFModel

        return HFModel("Qwen/Qwen3-8B", adapter="runs/adapters/qwen3-8b-v1")

    return (load_model,)


if __name__ == "__main__":
    app.run()
