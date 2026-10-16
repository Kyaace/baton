"""Textual interactive Terminal User Interface (TUI) for Baton.

Features:
- Arrow-key DataTable queue navigation
- Main inspection window: full Markdown/code artifact viewer for completed tasks;
  detailed prompt/parameter breakdown for pending/blocked tasks
- Space: Edit task modal (Enter to save, Escape to cancel)
- Enter: Run the selected task if ready
- 's' / 'S': Stop/interrupt active running task
- 't': Toggle Auto-Tick mode (continuous autonomous execution)
- 'r': Retry/reset failed tasks
- 'q': Quit
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    SelectionList,
    Static,
    TextArea,
)

from baton.db import Database
from baton.engine_manager import EngineManager
from baton.models import Artifact, Job, Task, TaskProgress, TaskStatus, TaskType, Tier, format_duration
from baton.orchestrator import Orchestrator


MODAL_CSS = """
EditTaskModal, NewTaskModal {
    align: center middle;
}
#dialog {
    width: 85%;
    height: 90%;
    border: thick $primary;
    background: $surface;
    padding: 1 2;
}
.modal-title {
    color: $accent;
    text-style: bold;
    margin-bottom: 1;
}
.field-label {
    color: $accent;
    text-style: bold;
    margin-top: 1;
}
#prompt-area {
    height: 7;
    margin-bottom: 1;
}
#upstream-select {
    height: 5;
    border: solid $secondary;
    margin-bottom: 1;
}
#embed-row {
    margin-top: 0;
    margin-bottom: 1;
}
#btn-row {
    margin-top: 1;
    align: right middle;
}
"""


class EditTaskModal(ModalScreen):
    """Modal screen for editing a selected task."""

    CSS = MODAL_CSS

    def __init__(self, task: Task, db: Database):
        super().__init__()
        self.target_task = task
        self.db = db

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label(f"Edit Task: {self.target_task.task_id}", classes="modal-title")

            with Horizontal():
                with Vertical():
                    yield Label("Title:", classes="field-label")
                    yield Input(value=self.target_task.title, id="inp-title")
                with Vertical():
                    yield Label("Tier:", classes="field-label")
                    yield Select(
                        [
                            ("System 1 (Scout & Verifier)", "system_1"),
                            ("System 2 (Deep Synthesizer)", "system_2"),
                        ],
                        value=self.target_task.tier.value,
                        id="sel-tier",
                        allow_blank=False,
                    )
                with Vertical():
                    yield Label("Priority (1=High, 10=Low):", classes="field-label")
                    yield Input(value=str(self.target_task.priority), id="inp-priority")

            # Upstream dependencies & artifacts selection
            all_tasks = self.db.get_all_tasks()
            options = []
            for t in all_tasks:
                if t.task_id == self.target_task.task_id:
                    continue
                has_artifact = bool(self.db.get_task_artifact_text(t.task_id))
                art_indicator = " [Artifact Ready]" if has_artifact else ""
                label = f"[{t.status.value.upper()}] {t.title} ({t.task_id}){art_indicator}"
                is_selected = t.task_id in self.target_task.depends_on
                options.append((label, t.task_id, is_selected))

            if options:
                yield Label("Link Upstream Tasks (Include Artifacts / Dossiers):", classes="field-label")
                yield SelectionList[str](*options, id="upstream-select")
                with Horizontal(id="embed-row"):
                    yield Button("Insert Selected Artifacts into Prompt Area", id="btn-embed-artifacts", variant="default")

            yield Label("Prompt / Instructions:", classes="field-label")
            yield TextArea(self.target_task.prompt_template or "", id="prompt-area")

            with Horizontal():
                with Vertical():
                    yield Label("Tool Name (optional for S1):", classes="field-label")
                    yield Input(value=self.target_task.tool_name or "", id="inp-tool")
                with Vertical():
                    yield Label("Tool Params (JSON):", classes="field-label")
                    params_str = json.dumps(self.target_task.tool_params or {})
                    yield Input(value=params_str, id="inp-params")

            with Horizontal(id="btn-row"):
                yield Button("Save Changes [Enter]", variant="primary", id="btn-save")
                yield Button("Cancel [Esc]", variant="default", id="btn-cancel")

    @on(Button.Pressed, "#btn-embed-artifacts")
    def on_embed_artifacts(self) -> None:
        try:
            sel_list = self.query_one("#upstream-select", SelectionList)
            selected_ids = sel_list.selected
        except Exception:
            self.notify("No upstream tasks selected.", severity="warning")
            return

        if not selected_ids:
            self.notify("Select at least one task in the list above first.", severity="warning")
            return

        prompt_area = self.query_one("#prompt-area", TextArea)
        current_text = prompt_area.text
        appended_parts = []
        count = 0

        for dep_id in selected_ids:
            art_text = self.db.get_task_artifact_text(dep_id)
            dep_task = self.db.get_task(dep_id)
            title = dep_task.title if dep_task else dep_id
            if art_text:
                appended_parts.append(f"\n\n## Briefing Finding: {title}\n{art_text.strip()}")
                count += 1
            else:
                appended_parts.append(f"\n\n## Upstream Dependency: {title} ({dep_id})\n[No artifact generated yet]")

        if appended_parts:
            prompt_area.text = (current_text + "".join(appended_parts)).strip()
            self.notify(f"Inserted artifact text from {count} task(s) into prompt area.", severity="information")

    @on(Button.Pressed, "#btn-save")
    def on_save(self) -> None:
        title = self.query_one("#inp-title", Input).value.strip()
        prompt = self.query_one("#prompt-area", TextArea).text.strip()
        try:
            tier_str = self.query_one("#sel-tier", Select).value
            tier = Tier(tier_str) if tier_str != Select.NULL else self.target_task.tier
        except Exception:
            tier = self.target_task.tier

        tool = self.query_one("#inp-tool", Input).value.strip() or None
        params_raw = self.query_one("#inp-params", Input).value.strip()
        pri_raw = self.query_one("#inp-priority", Input).value.strip()

        try:
            params = json.loads(params_raw) if params_raw else {}
        except Exception:
            params = self.target_task.tool_params

        try:
            priority = int(pri_raw)
        except Exception:
            priority = self.target_task.priority

        try:
            selected_deps = list(self.query_one("#upstream-select", SelectionList).selected)
        except Exception:
            selected_deps = self.target_task.depends_on

        self.db.update_task(
            task_id=self.target_task.task_id,
            title=title,
            tier=tier,
            prompt_template=prompt,
            tool_name=tool,
            tool_params=params,
            priority=priority,
            depends_on=selected_deps,
        )
        self.dismiss(True)

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)


class NewTaskModal(ModalScreen):
    """Modal screen for manually creating a new task."""

    CSS = MODAL_CSS

    def __init__(self, db: Database, initial_tier: Tier = Tier.SYSTEM_2):
        super().__init__()
        self.db = db
        self.initial_tier = initial_tier

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label("Create New Task", classes="modal-title")

            with Horizontal():
                with Vertical():
                    yield Label("Title:", classes="field-label")
                    yield Input(placeholder="e.g. Estimate Planetary Positions", id="inp-title")
                with Vertical():
                    yield Label("Tier:", classes="field-label")
                    yield Select(
                        [
                            ("System 1 (Scout & Verifier)", "system_1"),
                            ("System 2 (Deep Synthesizer)", "system_2"),
                        ],
                        value=self.initial_tier.value,
                        id="sel-tier",
                        allow_blank=False,
                    )
                with Vertical():
                    yield Label("Priority (1=High, 10=Low):", classes="field-label")
                    yield Input(value="1" if self.initial_tier == Tier.SYSTEM_2 else "10", id="inp-priority")

            # Upstream dependencies & artifacts selection
            all_tasks = self.db.get_all_tasks()
            options = []
            for t in all_tasks:
                has_artifact = bool(self.db.get_task_artifact_text(t.task_id))
                art_indicator = " [Artifact Ready]" if has_artifact else ""
                label = f"[{t.status.value.upper()}] {t.title} ({t.task_id}){art_indicator}"
                options.append((label, t.task_id, False))

            if options:
                yield Label("Link Upstream Tasks (Include Artifacts / Dossiers):", classes="field-label")
                yield SelectionList[str](*options, id="upstream-select")
                with Horizontal(id="embed-row"):
                    yield Button("Insert Selected Artifacts into Prompt Area", id="btn-embed-artifacts", variant="default")

            yield Label("Prompt / Instructions:", classes="field-label")
            yield TextArea("", id="prompt-area")

            with Horizontal():
                with Vertical():
                    yield Label("Tool Name (optional for S1):", classes="field-label")
                    yield Input(placeholder="e.g. wiki_read_section", id="inp-tool")
                with Vertical():
                    yield Label("Tool Params (JSON):", classes="field-label")
                    yield Input(placeholder='{"title": "Earth"}', id="inp-params")

            with Horizontal(id="btn-row"):
                yield Button("Create Task [Enter]", variant="primary", id="btn-save")
                yield Button("Cancel [Esc]", variant="default", id="btn-cancel")

    @on(Button.Pressed, "#btn-embed-artifacts")
    def on_embed_artifacts(self) -> None:
        try:
            sel_list = self.query_one("#upstream-select", SelectionList)
            selected_ids = sel_list.selected
        except Exception:
            self.notify("No upstream tasks available.", severity="warning")
            return

        if not selected_ids:
            self.notify("Select at least one task in the list above first.", severity="warning")
            return

        prompt_area = self.query_one("#prompt-area", TextArea)
        current_text = prompt_area.text
        appended_parts = []
        count = 0

        for dep_id in selected_ids:
            art_text = self.db.get_task_artifact_text(dep_id)
            dep_task = self.db.get_task(dep_id)
            title = dep_task.title if dep_task else dep_id
            if art_text:
                appended_parts.append(f"\n\n## Briefing Finding: {title}\n{art_text.strip()}")
                count += 1
            else:
                appended_parts.append(f"\n\n## Upstream Dependency: {title} ({dep_id})\n[No artifact generated yet]")

        if appended_parts:
            prompt_area.text = (current_text + "".join(appended_parts)).strip()
            self.notify(f"Inserted artifact text from {count} task(s) into prompt area.", severity="information")

    @on(Button.Pressed, "#btn-save")
    def on_save(self) -> None:
        title = self.query_one("#inp-title", Input).value.strip()
        if not title:
            self.notify("Please provide a title for the new task.", severity="error")
            return

        prompt = self.query_one("#prompt-area", TextArea).text.strip()
        try:
            tier_str = self.query_one("#sel-tier", Select).value
            tier = Tier(tier_str) if tier_str != Select.NULL else Tier.SYSTEM_2
        except Exception:
            tier = Tier.SYSTEM_2

        tool = self.query_one("#inp-tool", Input).value.strip() or None
        params_raw = self.query_one("#inp-params", Input).value.strip()
        pri_raw = self.query_one("#inp-priority", Input).value.strip()

        try:
            params = json.loads(params_raw) if params_raw else {}
        except Exception:
            params = None

        try:
            priority = int(pri_raw)
        except Exception:
            priority = 1 if tier == Tier.SYSTEM_2 else 10

        try:
            selected_deps = list(self.query_one("#upstream-select", SelectionList).selected)
        except Exception:
            selected_deps = []

        is_verify = tier == Tier.SYSTEM_1 and (
            (tool and any(k in tool for k in ("python_", "pytest_", "sandbox_", "verify")))
            or "verify" in title.lower()
        )
        task_type = TaskType.FAST_VERIFY if is_verify else (TaskType.FAST_RECON if tier == Tier.SYSTEM_1 else TaskType.DEEP_SYNTHESIS)

        self.db.create_task(
            title=title,
            tier=tier,
            task_type=task_type,
            prompt_template=prompt,
            tool_name=tool,
            tool_params=params,
            priority=priority,
            depends_on=selected_deps,
        )
        self.dismiss(True)

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)


class BatonApp(App):
    """Full-featured Textual application for Baton."""

    CSS = """
    Screen {
        background: $background;
    }
    #engine-bar {
        height: 3;
        background: $panel;
        padding: 0 1;
        content-align: left middle;
        border-bottom: heavy $primary;
    }
    #main-container {
        height: 1fr;
    }
    #queue-pane {
        height: 45%;
        border: round $primary;
        margin: 0 1;
    }
    #inspector-pane {
        height: 55%;
        border: round $accent;
        margin: 0 1;
        padding: 1;
    }
    #artifact-scroll {
        height: 100%;
    }
    DataTable {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("enter", "run_selected", "Run Selected Task", priority=True),
        Binding("n", "new_task", "New Task", priority=True),
        Binding("v", "verify_selected", "Verify Task", priority=True),
        Binding("s", "stop_task", "Stop Task", priority=True),
        Binding("S", "stop_task", "Stop Task", priority=True, show=False),
        Binding("space", "edit_selected", "Edit Task", priority=True),
        Binding("d", "delete_selected", "Delete Task", priority=True),
        Binding("r", "retry_failed", "Retry Failed", priority=True),
        Binding("t", "toggle_auto_tick", "Toggle Auto-Tick", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        super().__init__()
        self.orchestrator = orchestrator or Orchestrator()
        self.selected_task_id: Optional[str] = None
        self.auto_tick_enabled: bool = False
        self.is_running_task: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="engine-bar")
        with Vertical(id="main-container"):
            with Vertical(id="queue-pane"):
                yield DataTable(id="queue-table")
            with VerticalScroll(id="inspector-pane"):
                yield Markdown(id="inspector-content")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize DataTable and initial state."""
        table = self.query_one("#queue-table", DataTable)
        table.cursor_type = "row"
        table.add_column("ID", key="id")
        table.add_column("Tier", key="tier")
        table.add_column("Type", key="type")
        table.add_column("Pri", key="pri")
        table.add_column("Status", key="status")
        table.add_column("Title", key="title")
        table.add_column("Deps", key="deps")
        table.add_column("Elapsed", key="elapsed")
        self.refresh_all()
        # Set up 1.0 second polling interval for crisp live timers and streaming text
        self.set_interval(1.0, self.on_poll_tick)

    def update_engine_bar(self) -> None:
        """Updates the status banner with live engine states and running task timer."""
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
                tier_badge = f"[bold white on dark_green] RUNNING: S1 (Qwen3:14B){prog_info} [/]"
            else:
                tier_badge = f"[bold white on dark_magenta] RUNNING: S2 (Ornith-35B){prog_info} [/]"
        elif active == "SYSTEM_1":
            tier_badge = "[bold white on dark_green] ACTIVE: SYSTEM 1 (Qwen3:14B) [/]"
        elif active == "SYSTEM_2":
            tier_badge = "[bold white on dark_magenta] ACTIVE: SYSTEM 2 (Ornith-35B) [/]"
        else:
            tier_badge = "[bold black on dark_orange] IDLE (Standby) [/]"

        router_status = status.get("router", {})
        router_up = router_status.get("online", False)
        if router_up:
            engine_summary = "Router (:8080): [bold green]ONLINE (Ornith-35B + Qwen3-14B)[/]"
        else:
            engine_summary = "Router (:8080): [red]OFFLINE[/]"
        mode_label = "Router Mode"

        auto_badge = "[bold white on blue] AUTO-TICK: ON [/]" if self.auto_tick_enabled else "[dim]AUTO-TICK: OFF[/]"

        markup = (
            f" [bold cyan]BATON[/] | {mode_label}  |  "
            f"{engine_summary}   "
            f"{tier_badge}  {auto_badge}"
        )
        self.query_one("#engine-bar", Static).update(markup)

    def refresh_all(self) -> None:
        """Refreshes engine bar, queue table, and inspector."""
        self.update_engine_bar()
        self.refresh_table()

    def _get_status_and_elapsed(self, t: Task) -> tuple[str, str]:
        status_style = {
            TaskStatus.READY: "[bold green]READY[/]",
            TaskStatus.IN_PROGRESS: "[bold yellow]RUNNING[/]",
            TaskStatus.COMPLETED: "[blue]DONE[/]",
            TaskStatus.FAILED: "[bold red]FAILED[/]",
            TaskStatus.BLOCKED: "[dim]BLOCKED[/]",
        }.get(t.status, str(t.status.value))

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
        return status_style, elapsed_str

    def refresh_table(self) -> None:
        """Populates or updates the DataTable with current tasks from SQLite."""
        table = self.query_one("#queue-table", DataTable)
        tasks = self.orchestrator.db.get_all_tasks()

        existing_row_keys = [str(k.value) for k in table.rows.keys()] if hasattr(table, "rows") else []
        current_task_ids = [t.task_id for t in tasks]

        if existing_row_keys == current_task_ids and len(tasks) > 0:
            for t in tasks:
                st, el = self._get_status_and_elapsed(t)
                try:
                    table.update_cell(t.task_id, "status", st)
                    table.update_cell(t.task_id, "elapsed", el)
                except Exception:
                    pass
        else:
            current_cursor = table.cursor_row
            table.clear()
            for t in tasks:
                if t.task_type == TaskType.FAST_VERIFY:
                    tier_str = "S1 (Verify)"
                elif t.tier == Tier.SYSTEM_1:
                    tier_str = "S1 (Recon)"
                else:
                    tier_str = "S2 (Deep)"
                st, el = self._get_status_and_elapsed(t)
                deps_str = ",".join(t.depends_on) if t.depends_on else "-"
                table.add_row(
                    t.task_id[:16],
                    tier_str,
                    t.task_type.value,
                    str(t.priority),
                    st,
                    t.title[:35],
                    deps_str[:12],
                    el,
                    key=t.task_id,
                )

            if tasks:
                if current_cursor is not None and current_cursor < len(tasks):
                    table.move_cursor(row=current_cursor)
                else:
                    table.move_cursor(row=0)
                self.selected_task_id = tasks[table.cursor_row].task_id
                self.update_inspector(tasks[table.cursor_row])
            else:
                self.query_one("#inspector-content", Markdown).update(
                    "# No Tasks in Queue\n*Press 'n' to add a goal or use CLI.*"
                )

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Triggered on arrow-key navigation in DataTable."""
        if event.row_key and event.row_key.value:
            task_id = event.row_key.value
            self.selected_task_id = task_id
            task = self.orchestrator.db.get_task(task_id)
            if task:
                self.update_inspector(task)

    def update_inspector(self, task: Task) -> None:
        """Updates the main inspection window based on task completion or in-progress streaming."""
        content_widget = self.query_one("#inspector-content", Markdown)

        if task.status == TaskStatus.COMPLETED and task.artifact_path:
            # Task is completed -> Display the generated artifact!
            art_file = Path(task.artifact_path)
            if art_file.exists():
                try:
                    raw_text = art_file.read_text(encoding="utf-8")
                    trunc_banner = ""
                    if task.result_payload and "TRUNCATED" in task.result_payload:
                        trunc_banner = "> ⚠️ **Notice: Output reached token/context ceiling and was truncated.**\n\n---\n\n"
                    content_widget.update(trunc_banner + raw_text)
                    return
                except Exception as e:
                    raw_text = f"Error reading artifact: {e}"
            else:
                raw_text = f"Artifact recorded at `{task.artifact_path}` but file was not found."
            content_widget.update(raw_text)

        elif task.status == TaskStatus.IN_PROGRESS:
            # Task is actively generating -> Display live thinking and draft progress!
            prog = self.orchestrator.get_active_progress(task.task_id)
            if prog:
                elapsed_sec = prog.elapsed_seconds
                tok_cnt = prog.tokens_generated
                spd_str = f"{prog.tokens_per_second:.2f} tok/s" if prog.tokens_per_second > 0 else "evaluating..."
                reasoning_txt = prog.reasoning_text
                content_txt = prog.content_text
            elif task.started_at:
                elapsed_sec = max((datetime.now() - task.started_at).total_seconds(), 0.0)
                tok_cnt = task.tokens_generated
                spd_str = "evaluating..."
                reasoning_txt = ""
                content_txt = ""
            else:
                elapsed_sec = 0.0
                tok_cnt = 0
                spd_str = "starting..."
                reasoning_txt = ""
                content_txt = ""

            elapsed_formatted = format_duration(elapsed_sec)
            tier_label = "System 2 (Ornith-35B)" if task.tier == Tier.SYSTEM_2 else "System 1 (Qwen3:14B)"

            parts = [
                f"# ⏳ [IN PROGRESS] {task.title}\n",
                f"- **Task ID:** `{task.task_id}`",
                f"- **Tier:** `{task.tier.value}` ({tier_label})",
                f"- **Elapsed:** `{elapsed_formatted}`",
                f"- **Tokens Generated:** `{tok_cnt}` ({spd_str})\n",
                "---\n",
            ]

            if reasoning_txt:
                quoted = "\n".join(f"> {line}" for line in reasoning_txt.splitlines()[-40:])
                parts.append("### 🧠 Thinking Monologue (Chain-of-Thought in Progress)")
                parts.append(quoted + "\n")
            elif task.tier == Tier.SYSTEM_2:
                parts.append("### 🧠 Thinking Monologue (System 2 Deep Synthesis)")
                parts.append("> *Engine is computing prefill & evaluating initial attention matrices...*\n")

            if content_txt:
                parts.append("### 📝 Drafting Output Artifact in Progress")
                parts.append(f"```python\n{content_txt}\n```\n")
            else:
                parts.append("### 📝 Drafting Output Artifact")
                parts.append("*Drafting will begin once thinking phase completes...*\n")

            parts.append("---\n*Live progress updating every second. Press **s** to stop task, or **Space** to inspect/edit.*")
            content_widget.update("\n".join(parts))

        else:
            # Task is pending, blocked, running, or failed -> Display task breakdown!
            deps_list = ", ".join(task.depends_on) if task.depends_on else "None"
            tool_info = f"**Tool:** `{task.tool_name}`\n\n**Parameters:** `{json.dumps(task.tool_params or {})}`" if task.tool_name else "**Tool:** None (Pure reasoning/synthesis)"
            error_block = f"\n\n### Error Details\n`{task.error_message}`" if task.error_message else ""

            body = (
                f"# Task Details: {task.title}\n\n"
                f"- **Task ID:** `{task.task_id}`\n"
                f"- **Tier:** `{task.tier.value}` ({'System 1 Recon' if task.tier == Tier.SYSTEM_1 else 'System 2 Deep'})\n"
                f"- **Type:** `{task.task_type.value}`\n"
                f"- **Status:** `{task.status.value.upper()}`\n"
                f"- **Priority:** `{task.priority}`\n"
                f"- **Dependencies:** `{deps_list}`\n\n"
                f"### Tool Execution\n{tool_info}\n\n"
                f"### Prompt / Instructions\n```\n{task.prompt_template or 'No prompt template defined.'}\n```"
                f"{error_block}\n\n"
                f"---\n*Press **Enter** to run, **s** to stop, **Space** to edit, or **d** to delete task.*"
            )
            content_widget.update(body)

    # -------------------------------------------------------------------------
    # Key Actions & Work Orders
    # -------------------------------------------------------------------------

    def action_new_task(self) -> None:
        """'n': Open modal to manually create a new task."""
        def on_modal_close(created: bool) -> None:
            if created:
                self.notify("New task created successfully.", severity="information")
                self.refresh_all()

        self.push_screen(NewTaskModal(self.orchestrator.db), on_modal_close)

    def action_edit_selected(self) -> None:
        """Space: open task editor modal."""
        if not self.selected_task_id:
            self.notify("No task selected to edit. Use arrow keys or press 'n' to create one.", severity="warning")
            return
        task = self.orchestrator.db.get_task(self.selected_task_id)
        if not task:
            return

        def on_modal_close(saved: bool) -> None:
            if saved:
                self.notify("Task updated successfully.")
                self.refresh_all()

        self.push_screen(EditTaskModal(task, self.orchestrator.db), on_modal_close)

    def action_delete_selected(self) -> None:
        """'d': Delete the selected task from the queue."""
        if not self.selected_task_id:
            self.notify("No task selected to delete.", severity="warning")
            return

        task = self.orchestrator.db.get_task(self.selected_task_id)
        if not task:
            self.notify("Selected task was not found.", severity="warning")
            return

        if task.status == TaskStatus.IN_PROGRESS:
            self.notify(f"Cannot delete actively running task '{task.task_id}'.", severity="error")
            return

        deleted = self.orchestrator.delete_task(task.task_id)
        if deleted:
            self.notify(f"Deleted task '{task.task_id}'.", severity="information")
            self.selected_task_id = None
            self.refresh_all()
        else:
            self.notify(f"Failed to delete task '{task.task_id}'.", severity="error")

    def action_verify_selected(self) -> None:
        """'v': Queues an automated System 1 verification pass on the selected task."""
        if not self.selected_task_id:
            self.notify("Select a task first to queue verification.", severity="warning")
            return

        task = self.orchestrator.db.get_task(self.selected_task_id)
        if not task:
            return

        verify_title = f"Verify: {task.title}"
        v_task = self.orchestrator.create_task(
            title=verify_title,
            tier=Tier.SYSTEM_1,
            task_type=TaskType.FAST_VERIFY,
            prompt_template=f"Verify syntax and execute test suites on artifacts from '{task.title}' ({task.task_id}).",
            depends_on=[task.task_id],
            priority=task.priority + 1,
            job_id=task.job_id,
        )
        self.notify(f"Queued verification task '{v_task.task_id}' for '{task.task_id}'.", severity="information")
        self.refresh_all()

    def action_run_selected(self) -> None:
        """Enter: run selected task (if ready) or step queue."""
        if self.is_running_task or self.orchestrator.db.get_in_progress_tasks():
            self.notify("An engine task is already running in the background.", severity="warning")
            return

        if not self.selected_task_id:
            self.action_step_queue()
            return

        task = self.orchestrator.db.get_task(self.selected_task_id)
        if not task:
            return

        if task.status == TaskStatus.READY:
            self.run_task_worker(task.task_id)
        elif task.status == TaskStatus.BLOCKED:
            self.notify(f"Task is BLOCKED: waiting on dependencies: {task.depends_on}", severity="warning")
        elif task.status == TaskStatus.COMPLETED:
            self.notify("Task is already COMPLETED. Press Space to edit or 'r' to reset.", severity="information")
        elif task.status == TaskStatus.FAILED:
            self.notify("Task FAILED. Press 'r' to reset or Space to edit.", severity="error")

    def action_stop_task(self) -> None:
        """'s' / 'S': Interrupt and stop the currently executing task."""
        stopped_id = self.orchestrator.stop_current_task()
        self.is_running_task = False
        if stopped_id:
            self.notify(f"Stopped task '{stopped_id}'.", severity="warning")
        else:
            self.notify("No active task was running (runners signaled to cancel).", severity="information")
        self.refresh_all()

    @work(thread=True)
    def run_task_worker(self, task_id: str) -> None:
        """Asynchronously executes a ready task in a background worker."""
        self.is_running_task = True
        self.app.call_from_thread(self.notify, f"Starting task '{task_id}'...", severity="information")
        self.app.call_from_thread(self.refresh_all)

        try:
            completed = self.orchestrator.step(task_id=task_id)
        finally:
            self.is_running_task = False

        if completed and completed.status == TaskStatus.COMPLETED:
            self.app.call_from_thread(self.notify, f"Completed '{completed.title}'!", severity="information")
        elif completed and completed.status == TaskStatus.FAILED:
            is_stopped = "Stopped by user" in (completed.error_message or "")
            sev = "warning" if is_stopped else "error"
            self.app.call_from_thread(
                self.notify, f"Task '{task_id}' {completed.error_message or 'failed'}.", severity=sev
            )
        else:
            self.app.call_from_thread(self.notify, f"Task '{task_id}' halted or not runnable.", severity="warning")
        self.app.call_from_thread(self.refresh_all)

    @work(thread=True)
    def action_step_queue(self) -> None:
        """Executes next runnable task in priority order."""
        if self.is_running_task or self.orchestrator.db.get_in_progress_tasks():
            self.notify("An engine task is already running in the background.", severity="warning")
            return
        self.is_running_task = True
        try:
            completed = self.orchestrator.step()
        finally:
            self.is_running_task = False

        if completed and completed.status == TaskStatus.COMPLETED:
            self.app.call_from_thread(self.notify, f"Completed: {completed.title}")
        elif completed and completed.status == TaskStatus.FAILED:
            is_stopped = "Stopped by user" in (completed.error_message or "")
            sev = "warning" if is_stopped else "error"
            self.app.call_from_thread(
                self.notify, f"Task {completed.error_message or 'failed'}.", severity=sev
            )
        else:
            self.app.call_from_thread(self.notify, "No runnable tasks in queue.", severity="warning")
        self.app.call_from_thread(self.refresh_all)

    def action_retry_failed(self) -> None:
        """'r': reset failed tasks."""
        resets = self.orchestrator.db.retry_failed_tasks()
        if resets:
            self.notify(f"Reset {len(resets)} failed task(s) to ready/blocked.")
        else:
            self.notify("No failed tasks found.")
        self.refresh_all()

    def action_toggle_auto_tick(self) -> None:
        """'t': toggle autonomous queue progression."""
        self.auto_tick_enabled = not self.auto_tick_enabled
        state_str = "ENABLED" if self.auto_tick_enabled else "DISABLED"
        self.notify(f"Auto-Tick {state_str}.", severity="information")
        self.update_engine_bar()

    def on_poll_tick(self) -> None:
        """Periodic background poll running every second."""
        self.update_engine_bar()
        self.refresh_table()

        if self.selected_task_id:
            task = self.orchestrator.db.get_task(self.selected_task_id)
            if task and task.status == TaskStatus.IN_PROGRESS:
                self.update_inspector(task)

        if self.auto_tick_enabled and not self.is_running_task:
            if not self.orchestrator.db.get_in_progress_tasks():
                runnable = self.orchestrator.db.get_runnable_tasks()
                if runnable:
                    self.run_task_worker(runnable[0].task_id)


def main() -> None:
    app = BatonApp()
    app.run()


if __name__ == "__main__":
    main()
