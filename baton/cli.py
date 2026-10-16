"""CLI entrypoint for Baton orchestrator."""

import argparse
import sys
from pathlib import Path

from baton.db import Database
from baton.models import TaskStatus, TaskType, Tier
from baton.orchestrator import Orchestrator
from baton.tui import BatonTUI, console


def main() -> None:
    parser = argparse.ArgumentParser(description="Baton: Dual-Speed System 1 / System 2 Orchestrator")
    subparsers = parser.add_subparsers(dest="command")

    # TUI
    subparsers.add_parser("tui", help="Launch interactive terminal dashboard")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Launch interactive chat with System 1 Copilot")
    chat_parser.add_argument("--model", default=None, help="Model name (default: qwen3 for router)")
    chat_parser.add_argument("--coder", action="store_true", help="Shortcut for --model qwen2.5-coder:14b")
    chat_parser.add_argument("--url", dest="backend_url", default=None, help="llama-server router URL (default: http://127.0.0.1:8080)")
    chat_parser.add_argument("--router-url", dest="backend_url", default=None, help=argparse.SUPPRESS)
    chat_parser.add_argument("--ollama-url", dest="backend_url", default=None, help=argparse.SUPPRESS)
    chat_parser.add_argument(
        "--research",
        action="store_true",
        help="Launch research-focused agent mode (excludes baton queue mutation tools from context)",
    )
    chat_parser.add_argument(
        "--bundles",
        nargs="*",
        default=None,
        help="Explicit list of tool bundles to activate (e.g. --bundles core wikipedia newsboat)",
    )

    # Status
    subparsers.add_parser("status", help="Print queue and engine status")

    # Step
    subparsers.add_parser("step", help="Execute the next runnable task")

    # Run
    run_parser = subparsers.add_parser("run", help="Run the queue until idle")
    run_parser.add_argument("--max-steps", type=int, default=20, help="Maximum tasks to execute")
    run_parser.add_argument(
        "--wait",
        action="store_true",
        help="Sleep and wait for scheduled alarms/tasks instead of exiting when idle",
    )
    run_parser.add_argument(
        "--max-wait",
        type=float,
        default=None,
        help="Maximum seconds to wait for scheduled tasks when --wait is enabled",
    )

    # Submit
    submit_parser = subparsers.add_parser("submit", help="Submit a new project goal")
    submit_parser.add_argument("--project", required=True, help="Project name")
    submit_parser.add_argument("--goal", required=True, help="Goal description")
    submit_parser.add_argument("--plan", action="store_true", help="Use System 2 for automated DAG planning")

    # Artifacts
    subparsers.add_parser("artifacts", help="List generated artifacts")

    # Retry
    subparsers.add_parser("retry", help="Reset failed tasks to ready/blocked")

    # Stop
    subparsers.add_parser("stop", help="Stop/cancel active running task")

    # Delete
    del_parser = subparsers.add_parser("delete", help="Delete a task from the queue")
    del_parser.add_argument("task_id", help="Task ID to delete")

    # Verify
    verify_parser = subparsers.add_parser("verify", help="Queue or run a System 1 verification pass on a task")
    verify_parser.add_argument("task_id", help="Upstream task ID to verify")
    verify_parser.add_argument("--run-now", action="store_true", help="Execute immediately instead of just queuing")

    # New task
    new_parser = subparsers.add_parser("new", help="Manually create a new task")
    new_parser.add_argument("--title", required=True, help="Task title")
    new_parser.add_argument("--tier", choices=["system_1", "system_2"], default="system_2", help="System tier")
    new_parser.add_argument(
        "--type",
        dest="task_type",
        choices=["fast_recon", "deep_synthesis", "fast_verify"],
        default=None,
        help="Task type (default: fast_recon for S1, deep_synthesis for S2)",
    )
    new_parser.add_argument("--prompt", default="", help="Task prompt instructions")
    new_parser.add_argument("--deps", nargs="*", default=[], help="Dependency task IDs")
    new_parser.add_argument("--priority", type=int, default=5, help="Task priority")
    new_parser.add_argument("--tool", default=None, help="Tool name (for System 1)")
    new_parser.add_argument(
        "--schedule",
        default=None,
        help="Schedule delay or ISO/time-of-day timestamp (e.g. '30m', '14:30', '2026-10-13T10:00:00')",
    )

    args = parser.parse_args()

    orchestrator = Orchestrator()

    if args.command == "tui" or args.command is None:
        try:
            from baton.tui_textual import BatonApp
            app = BatonApp(orchestrator=orchestrator)
            app.run()
        except Exception as e:
            console.print(f"[yellow]Launching fallback Rich console: {e}[/]")
            tui = BatonTUI(orchestrator=orchestrator)
            tui.run_cli_loop()
    elif args.command == "chat":
        from baton.chat import BatonChatSession
        model = "qwen2.5-coder:14b" if args.coder else args.model
        initial_bundles = None
        if args.bundles:
            initial_bundles = set(args.bundles)
        elif args.research:
            # Dedicated research mode: load shelf, library (Wikipedia + ZIM archives), and newsboat without Baton queue tools
            initial_bundles = {"core", "library", "newsboat"}

        backend_url = args.backend_url
        chat_session = BatonChatSession(
            orchestrator=orchestrator,
            model_name=model,
            ollama_url=backend_url,
            initial_bundles=initial_bundles,
        )
        chat_session.run_repl()

    elif args.command == "status":
        tui = BatonTUI(orchestrator=orchestrator)
        console.print(tui.render_header())
        console.print(tui.render_queue_table())

    elif args.command == "step":
        console.print("[cyan]Stepping queue...[/]")
        task = orchestrator.step()
        if task:
            console.print(f"[green]Completed task:[/] {task.task_id} ({task.title})")
        else:
            console.print("[yellow]No runnable tasks in queue.[/]")

    elif args.command == "run":
        console.print(f"[cyan]Running queue (up to {args.max_steps} steps, wait={args.wait})...[/]")
        completed = orchestrator.run_all(
            max_steps=args.max_steps,
            wait_for_scheduled=args.wait,
            max_wait_seconds=args.max_wait,
        )
        console.print(f"[green]Finished! Completed {len(completed)} tasks.[/]")

    elif args.command == "submit":
        job = orchestrator.submit_goal(args.project, args.goal, plan_with_system2=args.plan)
        console.print(f"[green]Job created:[/] {job.job_id} ({job.title})")

    elif args.command == "artifacts":
        artifacts = orchestrator.db.get_artifacts()
        if not artifacts:
            console.print("[dim]No artifacts found.[/]")
        else:
            for a in artifacts:
                console.print(f"- [bold cyan]{a.title}[/] ({a.artifact_type.value}): [dim]{a.file_path}[/]")

    elif args.command == "retry":
        resets = orchestrator.db.retry_failed_tasks()
        console.print(f"[green]Reset {len(resets)} failed task(s) to ready/blocked: {', '.join(resets)}[/]")

    elif args.command == "stop":
        stopped_id = orchestrator.stop_current_task()
        if stopped_id:
            console.print(f"[yellow]Stopped task:[/] [bold]{stopped_id}[/]")
        else:
            console.print("[dim]No active running tasks found (runners signaled to stop).[/]")

    elif args.command == "delete":
        deleted = orchestrator.delete_task(args.task_id)
        if deleted:
            console.print(f"[green]Successfully deleted task:[/] {args.task_id}")
        else:
            console.print(f"[red]Task not found or could not be deleted:[/] {args.task_id}")

    elif args.command == "verify":
        target = orchestrator.db.get_task(args.task_id)
        if not target:
            console.print(f"[red]Error:[/] Task '{args.task_id}' not found.")
            sys.exit(1)
        v_task = orchestrator.create_task(
            title=f"Verify: {target.title}",
            tier=Tier.SYSTEM_1,
            task_type=TaskType.FAST_VERIFY,
            prompt_template=f"Verify syntax and execute test suites on artifacts from '{target.title}' ({target.task_id}).",
            depends_on=[target.task_id],
            priority=target.priority + 1,
            job_id=target.job_id,
        )
        console.print(f"[green]Queued verification task:[/] [bold]{v_task.task_id}[/] (depends on {target.task_id})")
        if args.run_now:
            console.print(f"[cyan]Executing verification task '{v_task.task_id}' now...[/]")
            res = orchestrator.step(task_id=v_task.task_id)
            if res and res.status == TaskStatus.COMPLETED:
                console.print(f"[bold green]Verification completed![/] Artifact: {res.artifact_path}")
            else:
                console.print(f"[red]Verification ended with status:[/] {res.status if res else 'None'}")

    elif args.command == "new":
        t_type = None
        if args.task_type:
            t_type = TaskType(args.task_type)
        scheduled_at = None
        if getattr(args, "schedule", None):
            from baton.tools.time_tool import parse_time_spec
            try:
                scheduled_at = parse_time_spec(args.schedule)
            except Exception as e:
                console.print(f"[red]Invalid --schedule specification:[/] {e}")
                sys.exit(1)
        task = orchestrator.create_task(
            title=args.title,
            tier=Tier(args.tier),
            task_type=t_type,
            prompt_template=args.prompt,
            depends_on=args.deps,
            priority=args.priority,
            tool_name=args.tool,
            scheduled_at=scheduled_at,
        )
        sched_msg = f", scheduled: {task.scheduled_at.isoformat(sep=' ')}" if task.scheduled_at else ""
        console.print(
            f"[green]Created task:[/] [bold]{task.task_id}[/] "
            f"({task.tier.value}, {task.task_type.value}, priority {task.priority}, status: {task.status.value}{sched_msg})"
        )


if __name__ == "__main__":
    main()
