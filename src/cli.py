"""CLI entry point for the ndis package."""

import json
import pathlib

import typer
from rich.console import Console
from rich.table import Table

from ndis.synth_generate import generate_dataset

cli = typer.Typer()


@cli.command("synth")
def synth(count: int = 20, output: str | None = None) -> None:
    """Generate synthetic NDIS case note data."""
    console = Console()
    records = generate_dataset(count)

    for i, record in enumerate(records[:5]):
        console.print(f"[green]✓[/green] Record {i+1}: {record['stratum']}")
    if len(records) > 5:
        console.print(f"[yellow]... and {len(records) - 5} more records[/yellow]")

    if output is not None:
        out_dir = pathlib.Path(output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        console.print(f"[green]Wrote[/green] {len(records)} records to [bold]{output}[/bold]")

    table = Table(title="Stratum Distribution")
    table.add_column("Stratum", style="cyan")
    table.add_column("Count", style="green")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["stratum"]] = counts.get(record["stratum"], 0) + 1
    for k, v in sorted(counts.items()):
        table.add_row(k, str(v))
    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
