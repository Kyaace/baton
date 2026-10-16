"""Interactive Terminal User Interface (TUI) for Baton.

Visualizes active LLM engine status, priority queue DAG, and generated artifacts in real time.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from baton.db import Database
from baton.engine_manager import EngineManager
from baton.models import Artifact, Job, Task, TaskStatus, TaskType, Tier, format_duration
from baton.orchestrator import Orchestrator

console = Console()


class BatonTUI:
    """Rich interactive dashboard and control center for Baton."""

    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator()
        self.selected_artifact: Optional[Artifact] = None

    def render_header(self) -> Panel:
        """Renders engine status and active tier badge with live elapsed time."""
        status = self.orchestrator.engine_mgr.get_status()
        active = status["active_tier"].upper()

        running_tasks = [t for t in self.orchestrator.db.get_all_tasks() if t.status == TaskStatus.IN_PROGRESS]
        if running_tasks:
            rt = running_tasks[0]
            prog = self.orchestrator.get_active_progress(rt.task_id)
            if prog:
                el_str = prog.formatted_elapsed
                tok_str = f"{prog.tokens_generated} tok" if prog.tokens_generated > 0 else ""
                spd_str = f"~{prog.tokens_per_second:.2f} t/s" if prog.tokens_per_second > 0 else ""
            elif rt.started_at:
                el_sec = max((datetime.now() - rt.started_at).total_seconds(), 0.0)
                el_str = format_duration(el_sec)
                tok_str = f"{rt.tokens_generated} tok" if rt.tokens_generated > 0 else ""
                spd_str = ""
            else:
                el_str = "starting"
                tok_str = ""
                spd_str = ""

            prog_parts = [p for p in [el_str, tok_str, spd_str] if p]
            prog_info = f" | {rt.task_id[:16]} [{ ' | '.join(prog_parts) }]"
            if rt.tier == Tier.SYSTEM_1:
                tier_badge = f"[bold white on green] RUNNING: S1 (Qwen3:14B){prog_info} [/]"
            else:
                tier_badge = f"[bold white on magenta] RUNNING: S2 (DeepSeek-V4-Flash){prog_info} [/]"
        elif active == "SYSTEM_1":
            tier_badge = "[bold white on green] ACTIVE: SYSTEM 1 (Qwen3:14B) [/]"
        elif active == "SYSTEM_2":
            tier_badge = "[bold white on magenta] ACTIVE: SYSTEM 2 (DeepSeek-V4-Flash) [/]"
        else:
            tier_badge = "[bold black on yellow] IDLE (Mode A Standby) [/]"

        ollama_stat = (
            "[green]ONLINE[/]" if status["ollama"]["online"] else "[dim]STOPPED[/]"
        )
        colibri_stat = (
            "[green]ONLINE[/]" if status["colibri"]["online"] else "[dim]STOPPED[/]"
        )

        router_status = status.get("router", {})
        if router_status.get("online"):
            engine_stat_line = "Router (:8080): [green]ONLINE (Ornith-35B + Qwen3-14B)[/]   |   Mode: [cyan]Router Mode (Dynamic Paging)[/]"
        else:
            engine_stat_line = f"Ollama (:11434): {ollama_stat}   |   Colibrì (:8000): {colibri_stat}   |   Mode: [cyan]Mode A (Interleaved)[/]"

        header_text = Text.from_markup(
            f"[bold cyan]BATON[/] [dim]|[/] Dual-Speed System 1 / System 2 Orchestrator\n"
            f"Hardware: Intel i5-13500 | 128 GB RAM | RTX 3060 12GB\n"
            f"{engine_stat_line}"
        )

        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=3)
        table.add_column(justify="right", ratio=2)
        table.add_row(header_text, Align.right(Text.from_markup(f"\n{tier_badge}")))

        return Panel(table, style="cyan", box=box.ROUNDED)

    def render_queue_table(self) -> Panel:
        """Renders the DAG priority queue and tasks."""
        tasks = self.orchestrator.db.get_all_tasks()
        table = Table(box=box.SIMPLE_HEAVY, expand=True)

        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Tier", style="bold")
        table.add_column("Type", style="cyan")
        table.add_column("Pri", justify="right", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Title", style="white")
        table.add_column("Deps", style="dim")
        table.add_column("Elapsed", justify="right", style="dim")

        if not tasks:
            table.add_row("-", "-", "-", "-", "[dim]EMPTY[/]", "No tasks in queue. Submit a goal to start.", "-", "-")
        else:
            for t in tasks:
                # Status styling
                if t.status == TaskStatus.READY:
                    stat_badge = "[bold green]READY[/]"
                elif t.status == TaskStatus.IN_PROGRESS:
                    stat_badge = "[bold yellow on dark_blue]RUNNING[/]"
                elif t.status == TaskStatus.COMPLETED:
                    stat_badge = "[blue]DONE[/]"
                elif t.status == TaskStatus.FAILED:
                    stat_badge = "[bold red]FAILED[/]"
                else:
                    stat_badge = "[dim]BLOCKED[/]"

                if t.task_type == TaskType.FAST_VERIFY:
                    tier_str = "[green]S1 (Verify)[/]"
                elif t.tier == Tier.SYSTEM_1:
                    tier_str = "[green]S1 (Recon)[/]"
                else:
                    tier_str = "[magenta]S2 (Deep)[/]"
                deps_str = ",".join(t.depends_on) if t.depends_on else "-"

                if t.status == TaskStatus.IN_PROGRESS:
                    prog = self.orchestrator.get_active_progress(t.task_id)
                    if prog:
                        spd = f" (~{prog.tokens_per_second:.2f} t/s)" if prog.tokens_per_second > 0 else ""
                        elapsed_str = f"[bold yellow]{prog.formatted_elapsed}{spd}[/]"
                    elif t.started_at:
                        el_sec = max((datetime.now() - t.started_at).total_seconds(), 0.0)
                        elapsed_str = f"[bold yellow]{format_duration(el_sec)}[/]"
                    else:
                        elapsed_str = "[yellow]running[/]"
                elif t.status == TaskStatus.COMPLETED:
                    elapsed_str = f"[blue]{format_duration(t.elapsed_seconds)}[/]" if t.elapsed_seconds > 0 else "done"
                else:
                    elapsed_str = "-"

                table.add_row(
                    t.task_id[:14],
                    tier_str,
                    t.task_type.value,
                    str(t.priority),
                    stat_badge,
                    t.title[:32],
                    deps_str[:12],
                    elapsed_str,
                )

        return Panel(table, title="[bold white]Task Priority Queue & DAG State[/]", border_style="blue", box=box.ROUNDED)

    def render_artifact_panel(self) -> Panel:
        """Renders the latest or selected artifact preview."""
        artifacts = self.orchestrator.db.get_artifacts()

        if not artifacts:
            return Panel(
                "[dim]No artifacts generated yet.\nCompleted System 1 dossiers and System 2 code will appear here.[/]",
                title="[bold white]Artifact & Dossier Inspector[/]",
                border_style="magenta",
                box=box.ROUNDED,
            )

        active_art = self.selected_artifact or artifacts[0]
        content = "Preview not available."
        path = Path(active_art.file_path)
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                # Show first 30 lines
                lines = raw.splitlines()[:30]
                content = "\n".join(lines)
                if len(raw.splitlines()) > 30:
                    content += "\n\n... [Truncated for preview]"
            except Exception as e:
                content = f"Error reading file: {e}"

        info_header = (
            f"[bold cyan]Artifact:[/] {active_art.title}\n"
            f"[dim]Type:[/] {active_art.artifact_type.value} | [dim]Path:[/] {active_art.file_path}\n"
            f"[dim]Available Artifacts ({len(artifacts)} total):[/] {', '.join(a.title[:15] for a in artifacts[:4])}\n"
            + "─" * 50
            + "\n"
        )

        # Check if syntax highlighting is appropriate
        if active_art.file_path.endswith(".py"):
            renderable = Syntax(content, "python", theme="monokai", line_numbers=True)
            return Panel(
                Columns([Text.from_markup(info_header), renderable]),
                title=f"[bold white]Artifact: {active_art.title}[/]",
                border_style="magenta",
                box=box.ROUNDED,
            )
        else:
            return Panel(
                f"{info_header}\n{content}",
                title=f"[bold white]Dossier: {active_art.title}[/]",
                border_style="magenta",
                box=box.ROUNDED,
            )

    def display(self) -> None:
        """Renders the full dashboard."""
        console.clear()
        console.print(self.render_header())
        console.print(self.render_queue_table())
        console.print(self.render_artifact_panel())

    def run_cli_loop(self) -> None:
        """Main interactive loop."""
        while True:
            self.display()
            console.print("\n[bold cyan]Controls:[/] "
                          "[bold][1][/] Step Queue  "
                          "[bold][2][/] Run All  "
                          "[bold][3][/] New Project Goal  "
                          "[bold][4][/] Add S1 Recon Task  "
                          "[bold][5][/] Add S2 Synth Task  "
                          "[bold][6][/] Inspect Artifacts  "
                          "[bold][r][/] Retry Failed  "
                          "[bold][q][/] Quit")

            try:
                choice = Prompt.ask("Choose action", default="1").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break

            if choice in ("q", "quit", "exit"):
                console.print("[yellow]Exiting Baton. Preserving database state.[/]")
                break
            elif choice == "1":
                console.print("[dim]Executing next runnable task...[/]")
                task = self.orchestrator.step()
                if not task:
                    console.print("[yellow]No runnable tasks in queue.[/]")
                    time.sleep(1.0)
            elif choice == "2":
                console.print("[dim]Running queue until idle...[/]")
                completed = self.orchestrator.run_all(max_steps=20)
                console.print(f"[green]Completed {len(completed)} tasks.[/]")
                time.sleep(1.5)
            elif choice == "3":
                self._handle_new_goal()
            elif choice == "4":
                self._handle_add_s1_task()
            elif choice == "5":
                self._handle_add_s2_task()
            elif choice == "6":
                self._handle_inspect_artifacts()
            elif choice == "r":
                reset_ids = self.orchestrator.db.retry_failed_tasks()
                console.print(f"[green]Reset {len(reset_ids)} failed task(s) to ready/blocked: {', '.join(reset_ids)}[/]")
                time.sleep(1.5)

    def _handle_new_goal(self) -> None:
        project = Prompt.ask("Project Title", default="Solar Ephemeris Model")
        goal = Prompt.ask(
            "Project Goal Specification",
            default="Extract orbital parameters from Wikipedia and synthesize a clean Python SolarEphemeris module."
        )
        plan_with_s2 = Confirm.ask("Use System 2 for automated DAG planning?", default=False)
        self.orchestrator.submit_goal(project, goal, plan_with_system2=plan_with_s2)

    def _handle_add_s1_task(self) -> None:
        jobs = self.orchestrator.db.list_jobs()
        if not jobs:
            job = self.orchestrator.db.create_job("General Workspace", "Ad-hoc tasks")
            jid = job.job_id
        else:
            jid = jobs[0].job_id

        title = Prompt.ask("Task Title", default="Recon Earth Orbital Parameters")
        tool = Prompt.ask("Tool (wiki_search / wiki_read_lead / wiki_read_section / wiki_infobox)", default="wiki_infobox")
        article = Prompt.ask("Wikipedia Article Title", default="Earth")
        section = ""
        if tool == "wiki_read_section":
            section = Prompt.ask("Section Name", default="Chemical composition")

        params = {"title": article}
        if section:
            params["section"] = section

        self.orchestrator.add_manual_task(
            job_id=jid,
            title=title,
            tier=Tier.SYSTEM_1,
            tool=tool,
            params=params,
            priority=5,
        )

    def _handle_add_s2_task(self) -> None:
        jobs = self.orchestrator.db.list_jobs()
        if not jobs:
            job = self.orchestrator.db.create_job("General Workspace", "Ad-hoc tasks")
            jid = job.job_id
        else:
            jid = jobs[0].job_id

        title = Prompt.ask("Task Title", default="Derive Solar Ephemeris Module")
        prompt = Prompt.ask(
            "Prompt / Instructions",
            default="Using upstream dossiers, write a robust solar_ephemeris.py script with numpy."
        )
        deps_str = Prompt.ask("Depends on Task IDs (comma-separated, optional)", default="")
        deps = [d.strip() for d in deps_str.split(",") if d.strip()]

        self.orchestrator.add_manual_task(
            job_id=jid,
            title=title,
            tier=Tier.SYSTEM_2,
            prompt=prompt,
            depends_on=deps,
            priority=1,
        )

    def _handle_inspect_artifacts(self) -> None:
        artifacts = self.orchestrator.db.get_artifacts()
        if not artifacts:
            console.print("[yellow]No artifacts found.[/]")
            time.sleep(1.0)
            return

        table = Table(title="Generated Artifacts")
        table.add_column("#", justify="right")
        table.add_column("Title")
        table.add_column("Type")
        table.add_column("Path")
        for idx, a in enumerate(artifacts, 1):
            table.add_row(str(idx), a.title, a.artifact_type.value, a.file_path)

        console.print(table)
        sel = Prompt.ask("Enter artifact number to view (or Enter to cancel)", default="")
        if sel.isdigit() and 1 <= int(sel) <= len(artifacts):
            self.selected_artifact = artifacts[int(sel) - 1]


def main() -> None:
    tui = BatonTUI()
    tui.run_cli_loop()


if __name__ == "__main__":
    main()
