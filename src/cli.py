"""CLI entry point for the ndis package."""

import json
import pathlib
import sys

import typer

from ndis.deidentify import deidentify_text
from ndis.synth_generate import STRATUM_PLAN, generate_dataset

cli = typer.Typer()


@cli.command("synth")
def synth(count: int = 20, output: str | None = None) -> None:
    """Generate synthetic NDIS case note data."""
    n_per = max(1, round(count / len(STRATUM_PLAN)))
    plan: dict[str, int] = {s: n_per for s in STRATUM_PLAN}
    records = generate_dataset(plan)

    for i, record in enumerate(records[:5]):
        typer.echo(f"✓ Record {i + 1}: {record['stratum']}")
    if len(records) > 5:
        typer.echo(f"... and {len(records) - 5} more records")

    if output is not None:
        out_dir = pathlib.Path(output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        typer.echo(f"Wrote {len(records)} records to {output}")

    typer.echo("Stratum distribution:")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["stratum"]] = counts.get(record["stratum"], 0) + 1
    for k, v in sorted(counts.items()):
        typer.echo(f"  {k}: {v}")


@cli.command("deid")
def deid(
    input_text: str | None = typer.Argument(None),
    input_file: str | None = typer.Option(None, "--file", "-f"),
) -> None:
    """De-identify text by redacting PII patterns."""
    if input_text is None and input_file is None:
        raise typer.BadParameter("Provide either INPUT_TEXT or --file")

    if input_file is not None:
        input_path = pathlib.Path(input_file)
        if not input_path.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        text = input_path.read_text(encoding="utf-8")
    else:
        text = input_text

    assert isinstance(text, str), "text must be a non-None str"
    redacted, found_types = deidentify_text(text)

    if found_types:
        typer.echo(f"Found PII types: {', '.join(found_types)}")
    else:
        typer.echo("No PII detected.")
    typer.echo(redacted)


@cli.command("draft")
def draft(
    input_text: str | None = typer.Argument(None, help="Worker's rough note. Omit to read stdin."),
    file: str | None = typer.Option(
        None, "--file", "-f", help="Read the worker note from a file."
    ),
    mode: str = typer.Option("hf", help="hf = fine-tuned local model; openai = Ollama base."),
    base_model: str = typer.Option("Qwen/Qwen3-8B", help="Base model id/path (hf mode)."),
    adapter: str = typer.Option(
        "runs/adapters/qwen3-8b-v1", help="LoRA adapter dir (hf mode); ignored if missing."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print only the raw JSON note."),
) -> None:
    """Draft a structured NDIS case note from a worker's rough input."""
    if file is not None:
        text = pathlib.Path(file).read_text(encoding="utf-8")
    elif input_text is not None:
        text = input_text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise typer.BadParameter("Provide INPUT_TEXT, --file, or pipe text via stdin.")
    text = text.strip()
    if not text:
        raise typer.BadParameter("Empty input.")

    # Build the model (heavy imports kept local so other CLI commands stay light).
    if mode == "hf":
        from eval.model_under_test import HFModel

        adapter_dir: str | None = adapter
        if adapter and not pathlib.Path(adapter).exists():
            typer.echo(f"[draft] adapter '{adapter}' not found — using base model only.", err=True)
            adapter_dir = None
        typer.echo(
            f"[draft] loading {base_model}{f' + {adapter_dir}' if adapter_dir else ''} ...",
            err=True,
        )
        model: object = HFModel(base_model, adapter=adapter_dir)
    elif mode == "openai":
        from eval.model_under_test import OpenAIModel

        ollama_model = "qwen3:8b" if base_model == "Qwen/Qwen3-8B" else base_model
        model = OpenAIModel(model=ollama_model)
    else:
        raise typer.BadParameter("mode must be 'hf' or 'openai'.")

    note = model.draft_note(text)  # type: ignore[attr-defined]

    if as_json:
        typer.echo(json.dumps(note, indent=2, ensure_ascii=False))
        return

    # Rendered, in-order view + a quick compliance read.
    from ndis.config_loader import get_config
    from ndis.notes import is_gap

    if not note:
        typer.echo("⚠ model returned no parseable JSON note.", err=True)
        raise typer.Exit(1)

    typer.echo("\n── Drafted NDIS case note ───────────────────────────")
    for field in get_config().required_fields.structure_order:
        value = note.get(field, "<MISSING>")
        flag = "  ⟵ [not recorded]" if is_gap(value) else ""
        typer.echo(f"  {field:<20} {value}{flag}")

    from eval.rubric import score_pii, score_structure

    st = score_structure(note)
    pii = score_pii(note)
    typer.echo("\n── Compliance check ─────────────────────────────────")
    typer.echo(f"  structure:  {'✓ ok' if st.passed else '✗ ' + _structure_reason(st)}")
    typer.echo(f"  pii-clean:  {'✓ ok' if pii.is_clean else '✗ leaked ' + str(pii.leaked_items)}")
    typer.echo("\n(full JSON: re-run with --json)")


def _structure_reason(st: object) -> str:
    bits = []
    if getattr(st, "missing_fields", None):
        bits.append(f"missing={st.missing_fields}")  # type: ignore[attr-defined]
    if getattr(st, "gap_fields", None):
        bits.append(f"unflagged-gaps={st.gap_fields}")  # type: ignore[attr-defined]
    if getattr(st, "invalid_fields", None):
        bits.append(f"bad-types={st.invalid_fields}")  # type: ignore[attr-defined]
    if not getattr(st, "order_ok", True):
        bits.append("wrong-order")
    return ", ".join(bits) or "failed"


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
