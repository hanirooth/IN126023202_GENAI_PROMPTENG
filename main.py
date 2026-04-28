import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from graph import run_query
from ingest import ingest_pdf
from config import CHROMA_PATH

console = Console()

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]RAG Customer Support Assistant[/bold cyan]\n"
        "[dim]Powered by LangGraph + ChromaDB + HITL[/dim]",
        border_style="cyan"
    ))

def main():
    print_banner()

    # Auto-ingest if ChromaDB is empty
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        console.print("[yellow]No knowledge base found. Running ingestion...[/yellow]")
        ingest_pdf()
    else:
        console.print("[green]✓ Knowledge base loaded.[/green]\n")

    console.print("Type [bold]'exit'[/bold] to quit | [bold]'reingest'[/bold] to reload PDF\n")

    while True:
        try:
            query = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        if query.lower() == "reingest":
            ingest_pdf()
            continue

        console.print()
        final_response, confidence, route = run_query(query)

        # Display result
        route_color = "green" if route == "answer" else "yellow"
        table = Table(show_header=False, box=None, padding=(0,1))
        table.add_row("[dim]Route:[/dim]",      f"[{route_color}]{route.upper()}[/{route_color}]")
        table.add_row("[dim]Confidence:[/dim]", f"{confidence:.0%}")
        console.print(table)

        console.print(Panel(
            final_response,
            title="[bold]Assistant[/bold]",
            border_style=route_color
        ))
        console.print()

if __name__ == "__main__":
    main()