"""CLI entry point for the ndis package."""

import json
import pathlib

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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
