from rich.console import Console
from rich.panel import Panel

console = Console()

def escalate_to_human(query: str, reason: str) -> str:
    """Simulate HITL escalation via CLI input."""
    console.print(Panel.fit(
        f"[bold yellow]⚠ ESCALATION TRIGGERED[/bold yellow]\n"
        f"[red]Reason:[/red] {reason}\n"
        f"[blue]User Query:[/blue] {query}",
        title="Human-in-the-Loop"
    ))
    console.print("[bold green]HUMAN AGENT:[/] Please type your response below:")
    human_response = input(">>> ").strip()

    if not human_response:
        human_response = "I've escalated your query to our support team. They will contact you within 24 hours."

    return human_response

def log_escalation(query: str, reason: str, response: str):
    with open("escalations.log", "a") as f:
        f.write(f"QUERY: {query}\nREASON: {reason}\nRESPONSE: {response}\n{'-'*50}\n")