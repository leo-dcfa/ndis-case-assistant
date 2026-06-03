"""CLI entry point for the ndis package."""

import typer

cli = typer.Typer()


@cli.command("synth")
def synth(count: int = 20, output: str | None = None) -> None:
    """Generate synthetic NDIS case note data."""
    from ndis.synth_generate import generate_dataset
    from rich.console import Console
    from rich.table import Table

    console = Console()

    records = generate_dataset(count)

    for i, r in enumerate(records[:5]):
        console.print(f"[green]✓[/green] Record {i+1}: {r['stratum']}")
    if len(records) > 5:
        console.print(f"[yellow]... and {len(records) - 5} more records[/yellow]")

    if output:
        import pathlib
        out_dir = pathlib.Path(output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for r in records:
                import json
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        console.print(f"[green]Wrote[/green] {len(records)} records to [bold]{output}[/bold]")

    table = Table(title="Stratum Distribution")
    table.add_column("Stratum", style="cyan")
    table.add_column("Count", style="green")
    counts: dict[str, int] = {}
    for r in records:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    for k, v in sorted(counts.items()):
        table.add_row(k, str(v))
    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
