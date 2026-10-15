"""SQLite persistence and deterministic DAG priority queue for Baton."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from baton.models import (
    Artifact,
    ArtifactType,
    DAGPlan,
    EphemeralNote,
    Job,
    JobStatus,
    Task,
    TaskPlan,
    TaskStatus,
    TaskType,
    Tier,
)

DEFAULT_DB_PATH = Path("baton.db")


class Database:
    """Deterministic SQLite DAG state engine and priority queue."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, timeout: float = 5.0):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self.get_connection() as conn:
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT CHECK(status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    tier TEXT CHECK(tier IN ('system_1', 'system_2')) NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT CHECK(status IN ('blocked', 'ready', 'in_progress', 'completed', 'failed')) DEFAULT 'blocked',
                    prompt_template TEXT,
                    tool_name TEXT,
                    tool_params TEXT,
                    result_payload TEXT,
                    artifact_path TEXT,
                    priority INTEGER DEFAULT 0,
                    tokens_generated INTEGER DEFAULT 0,
                    elapsed_seconds REAL DEFAULT 0.0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    scheduled_at TIMESTAMP,
                    note_id INTEGER,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY(task_id, depends_on_task_id),
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
                    FOREIGN KEY(depends_on_task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_preview TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ephemeral_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    dismissed INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_tier_status ON tasks(tier, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id);
                CREATE INDEX IF NOT EXISTS idx_notes_active ON ephemeral_notes(dismissed, expires_at);
            """)

            # Migration check: ensure scheduled_at and note_id exist on existing tables before creating index
            table_info = [r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
            if table_info and "scheduled_at" not in table_info:
                conn.execute("ALTER TABLE tasks ADD COLUMN scheduled_at TIMESTAMP")
            if table_info and "note_id" not in table_info:
                conn.execute("ALTER TABLE tasks ADD COLUMN note_id INTEGER")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_scheduled ON tasks(status, scheduled_at);")


    # -------------------------------------------------------------------------
    # Jobs Management
    # -------------------------------------------------------------------------

    def create_job(self, title: str, goal: str, job_id: Optional[str] = None) -> Job:
        job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, title, goal, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, title, goal, JobStatus.PENDING.value, now),
            )
        return Job(job_id=job_id, title=title, goal=goal, status=JobStatus.PENDING, created_at=now)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return Job(
                job_id=row["job_id"],
                title=row["title"],
                goal=row["goal"],
                status=JobStatus(row["status"]),
                created_at=row["created_at"],
                completed_at=row["completed_at"],
            )

    def list_jobs(self) -> list[Job]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [
                Job(
                    job_id=r["job_id"],
                    title=r["title"],
                    goal=r["goal"],
                    status=JobStatus(r["status"]),
                    created_at=r["created_at"],
                    completed_at=r["completed_at"],
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # DAG Ingestion & Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_dag(tasks: list[TaskPlan]) -> None:
        """Ensures the DAG contains no cycles using Kahn's algorithm."""
        task_ids = {t.id for t in tasks}
        in_degree = {t.id: 0 for t in tasks}
        adj: dict[str, list[str]] = {t.id: [] for t in tasks}

        for t in tasks:
            for dep in t.depends_on:
                if dep in task_ids:
                    in_degree[t.id] += 1
                    adj[dep].append(t.id)

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(tasks):
            raise ValueError("Cycle detected in DAG plan specifications")

    def ingest_dag_plan(self, job_id: str, plan: DAGPlan) -> list[Task]:
        """Decomposes a DAGPlan into persistent tasks and dependency edges."""
        self.validate_dag(plan.tasks)

        created_tasks: list[Task] = []
        now = datetime.now()

        with self.get_connection() as conn:
            # Update job status to running
            conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, job_id),
            )

            for tp in plan.tasks:
                initial_status = (
                    TaskStatus.READY if len(tp.depends_on) == 0 else TaskStatus.BLOCKED
                )
                params_json = json.dumps(tp.params) if tp.params else None

                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, job_id, title, tier, task_type, status,
                        prompt_template, tool_name, tool_params, priority, scheduled_at, note_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tp.id,
                        job_id,
                        tp.title,
                        tp.tier.value,
                        tp.type.value,
                        initial_status.value,
                        tp.prompt or tp.description,
                        tp.tool,
                        params_json,
                        tp.priority,
                        tp.scheduled_at,
                        tp.note_id,
                        now,
                    ),
                )

                for dep_id in tp.depends_on:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_dependencies (task_id, depends_on_task_id) VALUES (?, ?)",
                        (tp.id, dep_id),
                    )

                created_tasks.append(
                    Task(
                        task_id=tp.id,
                        job_id=job_id,
                        title=tp.title,
                        tier=tp.tier,
                        task_type=tp.type,
                        status=initial_status,
                        prompt_template=tp.prompt or tp.description,
                        tool_name=tp.tool,
                        tool_params=tp.params,
                        depends_on=tp.depends_on,
                        priority=tp.priority,
                        scheduled_at=tp.scheduled_at,
                        note_id=tp.note_id,
                        created_at=now,
                    )
                )

        return created_tasks

    # -------------------------------------------------------------------------
    # Priority Queue & Task State Machine
    # -------------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[Task]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            deps = [
                r["depends_on_task_id"]
                for r in conn.execute(
                    "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                    (task_id,),
                ).fetchall()
            ]
            return self._row_to_task(row, deps)

    def get_runnable_tasks(
        self, job_id: Optional[str] = None, tier: Optional[Tier] = None
    ) -> list[Task]:
        """Returns ready tasks ordered by priority (DESC) and creation time (ASC).
        Filters out tasks scheduled for the future.
        """
        now = datetime.now()
        query = "SELECT * FROM tasks WHERE status = 'ready' AND (scheduled_at IS NULL OR scheduled_at <= ?)"
        params: list[Any] = [now]

        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        if tier:
            query += " AND tier = ?"
            params.append(tier.value)

        query += " ORDER BY priority DESC, created_at ASC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            tasks: list[Task] = []
            for r in rows:
                deps = [
                    d["depends_on_task_id"]
                    for d in conn.execute(
                        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                        (r["task_id"],),
                    ).fetchall()
                ]
                tasks.append(self._row_to_task(r, deps))
            return tasks

    def get_scheduled_tasks(
        self, job_id: Optional[str] = None, tier: Optional[Tier] = None
    ) -> list[Task]:
        """Returns ready tasks that are deferred into the future."""
        now = datetime.now()
        query = "SELECT * FROM tasks WHERE status = 'ready' AND scheduled_at > ?"
        params: list[Any] = [now]

        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        if tier:
            query += " AND tier = ?"
            params.append(tier.value)

        query += " ORDER BY scheduled_at ASC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            tasks: list[Task] = []
            for r in rows:
                deps = [
                    d["depends_on_task_id"]
                    for d in conn.execute(
                        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                        (r["task_id"],),
                    ).fetchall()
                ]
                tasks.append(self._row_to_task(r, deps))
            return tasks

    def get_next_scheduled_task(
        self, job_id: Optional[str] = None, tier: Optional[Tier] = None
    ) -> Optional[Task]:
        """Returns the earliest scheduled future task."""
        scheduled = self.get_scheduled_tasks(job_id=job_id, tier=tier)
        return scheduled[0] if scheduled else None

    def set_task_schedule(
        self,
        task_id: str,
        scheduled_at: Optional[datetime],
        note_id: Optional[int] = None,
    ) -> bool:
        """Sets or clears the scheduled execution time for a task, optionally linking a note_id."""
        with self.get_connection() as conn:
            if note_id is not None:
                cursor = conn.execute(
                    "UPDATE tasks SET scheduled_at = ?, note_id = ? WHERE task_id = ?",
                    (scheduled_at, note_id, task_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE tasks SET scheduled_at = ? WHERE task_id = ?",
                    (scheduled_at, task_id),
                )
            return cursor.rowcount > 0

    def get_in_progress_tasks(self) -> list[Task]:
        """Returns all tasks currently marked IN_PROGRESS."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE status = 'in_progress'").fetchall()
            tasks: list[Task] = []
            for r in rows:
                deps = [
                    d["depends_on_task_id"]
                    for d in conn.execute(
                        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                        (r["task_id"],),
                    ).fetchall()
                ]
                tasks.append(self._row_to_task(r, deps))
            return tasks

    def claim_next_task(
        self, tier: Optional[Tier] = None, job_id: Optional[str] = None
    ) -> Optional[Task]:
        """Atomically claims the highest priority ready task."""
        runnable = self.get_runnable_tasks(job_id=job_id, tier=tier)
        if not runnable:
            return None

        chosen = runnable[0]
        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, started_at = ? WHERE task_id = ? AND status = 'ready'",
                (TaskStatus.IN_PROGRESS.value, now, chosen.task_id),
            )
            # Ensure job is marked running
            conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, chosen.job_id),
            )

        chosen.status = TaskStatus.IN_PROGRESS
        chosen.started_at = now
        return chosen

    def claim_specific_task(self, task_id: str) -> Optional[Task]:
        """Atomically claims a specific ready task."""
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.READY:
            return None

        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, started_at = ? WHERE task_id = ? AND status = 'ready'",
                (TaskStatus.IN_PROGRESS.value, now, task_id),
            )
            conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, task.job_id),
            )

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = now
        return task

    def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        prompt_template: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_params: Optional[dict[str, Any]] = None,
        priority: Optional[int] = None,
        tier: Optional[Tier | str] = None,
        depends_on: Optional[list[str]] = None,
        scheduled_at: Optional[datetime | str] = None,
        clear_schedule: bool = False,
        note_id: Optional[int] = None,
        clear_note: bool = False,
    ) -> None:
        """Updates editable fields of an existing task and synchronizes dependencies and readiness."""
        updates: list[str] = []
        params: list[Any] = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if prompt_template is not None:
            updates.append("prompt_template = ?")
            params.append(prompt_template)
        if tool_name is not None:
            updates.append("tool_name = ?")
            params.append(tool_name)
        if tool_params is not None:
            updates.append("tool_params = ?")
            params.append(json.dumps(tool_params))
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        if tier is not None:
            tier_val = tier.value if isinstance(tier, Tier) else str(tier)
            updates.append("tier = ?")
            params.append(tier_val)
        if clear_schedule:
            updates.append("scheduled_at = NULL")
        elif scheduled_at is not None:
            updates.append("scheduled_at = ?")
            params.append(scheduled_at)
        if clear_note:
            updates.append("note_id = NULL")
        elif note_id is not None:
            updates.append("note_id = ?")
            params.append(note_id)

        with self.get_connection() as conn:
            if updates:
                params.append(task_id)
                conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?", params)

            if depends_on is not None:
                deps_clean = [d for d in depends_on if d != task_id]
                conn.execute("DELETE FROM task_dependencies WHERE task_id = ?", (task_id,))
                valid_clean: list[str] = []
                if deps_clean:
                    placeholders = ",".join("?" for _ in deps_clean)
                    existing_rows = conn.execute(
                        f"SELECT task_id FROM tasks WHERE task_id IN ({placeholders})", deps_clean
                    ).fetchall()
                    existing_set = {r["task_id"] for r in existing_rows}
                    valid_clean = [d for d in deps_clean if d in existing_set]

                for dep_id in valid_clean:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_dependencies (task_id, depends_on_task_id) VALUES (?, ?)",
                        (task_id, dep_id),
                    )

                # Recalculate status if task is currently ready or blocked
                curr = conn.execute("SELECT status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
                if curr and curr["status"] in ("ready", "blocked"):
                    unmet = conn.execute(
                        """
                        SELECT COUNT(*) as count FROM task_dependencies td
                        JOIN tasks t ON td.depends_on_task_id = t.task_id
                        WHERE td.task_id = ? AND t.status != 'completed'
                        """,
                        (task_id,),
                    ).fetchone()["count"]
                    new_status = "blocked" if unmet > 0 else "ready"
                    conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (new_status, task_id))

    def create_task(
        self,
        title: str,
        tier: Tier | str,
        task_type: Optional[TaskType | str] = None,
        prompt_template: str = "",
        tool_name: Optional[str] = None,
        tool_params: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
        priority: int = 5,
        job_id: Optional[str] = None,
        task_id: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
        note_id: Optional[int] = None,
    ) -> Task:
        """Manually creates a new task with dependencies and inserts it into the queue."""
        tier_obj = Tier(tier) if isinstance(tier, str) else tier
        if task_type is None:
            type_obj = TaskType.DEEP_SYNTHESIS if tier_obj == Tier.SYSTEM_2 else TaskType.FAST_RECON
        elif isinstance(task_type, str):
            type_obj = TaskType(task_type)
        else:
            type_obj = task_type

        # Find or create target job
        if not job_id:
            jobs = self.list_jobs()
            if jobs:
                job_id = jobs[0].job_id
            else:
                job_id = f"job_{uuid.uuid4().hex[:8]}"
                self.create_job(job_id=job_id, title="Interactive Project", goal="Interactive project queue")

        # Generate unique task_id if not given
        if not task_id:
            slug = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")[:24]
            candidate_id = f"task_{slug}" if slug else f"task_{uuid.uuid4().hex[:8]}"
            with self.get_connection() as conn:
                existing = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (candidate_id,)).fetchone()
                if existing:
                    candidate_id = f"{candidate_id}_{uuid.uuid4().hex[:4]}"
            task_id = candidate_id

        deps = [d for d in (depends_on or []) if d != task_id]
        now = datetime.now()

        with self.get_connection() as conn:
            valid_deps: list[str] = []
            unmet = 0
            if deps:
                placeholders = ",".join("?" for _ in deps)
                existing_rows = conn.execute(
                    f"SELECT task_id, status FROM tasks WHERE task_id IN ({placeholders})", deps
                ).fetchall()
                valid_deps = [r["task_id"] for r in existing_rows]
                unmet = sum(1 for r in existing_rows if r["status"] != "completed")

            initial_status = TaskStatus.BLOCKED if unmet > 0 else TaskStatus.READY
            params_json = json.dumps(tool_params) if tool_params else None

            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, job_id, title, tier, task_type, status,
                    prompt_template, tool_name, tool_params, priority, scheduled_at, note_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    job_id,
                    title,
                    tier_obj.value,
                    type_obj.value,
                    initial_status.value,
                    prompt_template,
                    tool_name,
                    params_json,
                    priority,
                    scheduled_at,
                    note_id,
                    now,
                ),
            )

            for dep_id in valid_deps:
                conn.execute(
                    "INSERT OR IGNORE INTO task_dependencies (task_id, depends_on_task_id) VALUES (?, ?)",
                    (task_id, dep_id),
                )

        return Task(
            task_id=task_id,
            job_id=job_id,
            title=title,
            tier=tier_obj,
            task_type=type_obj,
            status=initial_status,
            prompt_template=prompt_template,
            tool_name=tool_name,
            tool_params=tool_params,
            depends_on=valid_deps,
            priority=priority,
            scheduled_at=scheduled_at,
            note_id=note_id,
            created_at=now,
        )

    def get_task_artifact_text(self, task_id: str) -> Optional[str]:
        """Returns the full text content of a task's generated artifact or payload."""
        task = self.get_task(task_id)
        if not task:
            return None

        # 1. Check artifact_path on disk
        if task.artifact_path:
            p = Path(task.artifact_path)
            paths_to_try = [p] if p.is_absolute() else [p, Path("baton") / p]
            for path_try in paths_to_try:
                if path_try.exists():
                    try:
                        return path_try.read_text(encoding="utf-8")
                    except Exception:
                        pass

        # 2. Check result_payload
        if task.result_payload and task.result_payload.strip():
            return task.result_payload.strip()

        # 3. Check artifacts table
        artifacts = self.get_artifacts(task_id=task_id)
        for art in artifacts:
            if art.file_path:
                p = Path(art.file_path)
                paths_to_try = [p] if p.is_absolute() else [p, Path("baton") / p]
                for path_try in paths_to_try:
                    if path_try.exists():
                        try:
                            return path_try.read_text(encoding="utf-8")
                        except Exception:
                            pass
            if art.content_preview and art.content_preview.strip():
                return art.content_preview.strip()

        return None

    def delete_task(self, task_id: str) -> bool:
        """Deletes a task from the database, cleans dependencies and artifacts, and unlocks unblocked dependents."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT job_id FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return False

            # Find dependent tasks that were waiting on this task
            dependent_tasks = [
                r["task_id"]
                for r in conn.execute(
                    "SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ?",
                    (task_id,),
                ).fetchall()
            ]

            # Delete dependency links
            conn.execute("DELETE FROM task_dependencies WHERE task_id = ? OR depends_on_task_id = ?", (task_id, task_id))

            # Delete associated artifacts
            conn.execute("DELETE FROM artifacts WHERE task_id = ?", (task_id,))

            # Delete the task record
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

            # Cascade: if any blocked task now has all remaining dependencies completed (or none left), unlock it to ready
            for dep_id in dependent_tasks:
                rem_unmet = conn.execute(
                    """
                    SELECT COUNT(*) as count FROM task_dependencies td
                    JOIN tasks t ON td.depends_on_task_id = t.task_id
                    WHERE td.task_id = ? AND t.status != 'completed'
                    """,
                    (dep_id,),
                ).fetchone()["count"]

                if rem_unmet == 0:
                    conn.execute(
                        "UPDATE tasks SET status = 'ready' WHERE task_id = ? AND status = 'blocked'",
                        (dep_id,),
                    )

            return True

    def mark_task_completed(
        self,
        task_id: str,
        result_payload: str = "",
        artifact_path: Optional[str] = None,
        tokens_generated: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Marks a task completed and unlocks downstream tasks whose dependencies are met."""
        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE tasks SET
                    status = ?,
                    result_payload = ?,
                    artifact_path = ?,
                    tokens_generated = ?,
                    elapsed_seconds = ?,
                    completed_at = ?
                WHERE task_id = ?
                """,
                (
                    TaskStatus.COMPLETED.value,
                    result_payload,
                    artifact_path,
                    tokens_generated,
                    elapsed_seconds,
                    now,
                    task_id,
                ),
            )

            # Find all downstream tasks that depend on this task
            downstream = conn.execute(
                "SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ?",
                (task_id,),
            ).fetchall()

            for row in downstream:
                d_id = row["task_id"]
                # Check if all dependencies for d_id are completed
                unmet = conn.execute(
                    """
                    SELECT COUNT(*) as count FROM task_dependencies td
                    JOIN tasks t ON td.depends_on_task_id = t.task_id
                    WHERE td.task_id = ? AND t.status != 'completed'
                    """,
                    (d_id,),
                ).fetchone()["count"]

                if unmet == 0:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE task_id = ? AND status = 'blocked'",
                        (TaskStatus.READY.value, d_id),
                    )

            # Check if all tasks in the job are completed
            task_job = conn.execute(
                "SELECT job_id FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task_job:
                jid = task_job["job_id"]
                remaining = conn.execute(
                    "SELECT COUNT(*) as count FROM tasks WHERE job_id = ? AND status != 'completed'",
                    (jid,),
                ).fetchone()["count"]
                if remaining == 0:
                    conn.execute(
                        "UPDATE jobs SET status = ?, completed_at = ? WHERE job_id = ?",
                        (JobStatus.COMPLETED.value, now, jid),
                    )

    def get_downstream_tasks(self, task_id: str) -> list[str]:
        """Returns a list of task_ids that directly depend on the given task_id."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ?",
                (task_id,),
            ).fetchall()
            return [r["task_id"] for r in rows]

    def rewire_dependencies(self, old_dep_id: str, new_dep_id: str) -> int:
        """
        Rewires task dependencies so any task waiting on old_dep_id
        now depends on new_dep_id instead. Prevents self-dependency and duplicates.
        """
        with self.get_connection() as conn:
            # Delete any existing pairs that would conflict with the update
            conn.execute(
                """
                DELETE FROM task_dependencies
                WHERE depends_on_task_id = ? AND task_id IN (
                    SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ?
                )
                """,
                (old_dep_id, new_dep_id),
            )
            cursor = conn.execute(
                """
                UPDATE task_dependencies
                SET depends_on_task_id = ?
                WHERE depends_on_task_id = ? AND task_id != ?
                """,
                (new_dep_id, old_dep_id, new_dep_id),
            )
            return cursor.rowcount

    def mark_task_failed(self, task_id: str, error_message: str) -> None:
        """Marks task failed with an error message."""
        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, error_message = ?, completed_at = ? WHERE task_id = ?",
                (TaskStatus.FAILED.value, error_message, now, task_id),
            )

    def retry_failed_tasks(self, job_id: Optional[str] = None) -> list[str]:
        """Resets failed tasks back to 'ready' (if dependencies are met) or 'blocked'."""
        with self.get_connection() as conn:
            query = "SELECT task_id, job_id FROM tasks WHERE status = 'failed'"
            params: list[Any] = []
            if job_id:
                query += " AND job_id = ?"
                params.append(job_id)
            rows = conn.execute(query, params).fetchall()
            reset_ids: list[str] = []

            for r in rows:
                tid = r["task_id"]
                jid = r["job_id"]
                # Check unmet dependencies
                unmet = conn.execute(
                    """
                    SELECT COUNT(*) as count FROM task_dependencies td
                    JOIN tasks t ON td.depends_on_task_id = t.task_id
                    WHERE td.task_id = ? AND t.status != 'completed'
                    """,
                    (tid,),
                ).fetchone()["count"]

                new_status = TaskStatus.READY.value if unmet == 0 else TaskStatus.BLOCKED.value
                conn.execute(
                    """
                    UPDATE tasks SET
                        status = ?,
                        error_message = NULL,
                        started_at = NULL,
                        completed_at = NULL
                    WHERE task_id = ?
                    """,
                    (new_status, tid),
                )
                conn.execute(
                    "UPDATE jobs SET status = ?, completed_at = NULL WHERE job_id = ?",
                    (JobStatus.RUNNING.value, jid),
                )
                reset_ids.append(tid)

            return reset_ids

    def get_job_tasks(self, job_id: str) -> list[Task]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE job_id = ? ORDER BY created_at ASC", (job_id,)
            ).fetchall()
            tasks: list[Task] = []
            for r in rows:
                deps = [
                    d["depends_on_task_id"]
                    for d in conn.execute(
                        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                        (r["task_id"],),
                    ).fetchall()
                ]
                tasks.append(self._row_to_task(r, deps))
            return tasks

    def get_all_tasks(self) -> list[Task]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at ASC").fetchall()
            tasks: list[Task] = []
            for r in rows:
                deps = [
                    d["depends_on_task_id"]
                    for d in conn.execute(
                        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                        (r["task_id"],),
                    ).fetchall()
                ]
                tasks.append(self._row_to_task(r, deps))
            return tasks

    # -------------------------------------------------------------------------
    # Upstream Dossiers & Context Budgeting
    # -------------------------------------------------------------------------

    def get_upstream_dossiers(self, task_id: str) -> list[dict[str, Any]]:
        """Retrieves results and artifacts from all completed upstream dependencies."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT t.task_id, t.title, t.task_type, t.tool_name, t.result_payload, t.artifact_path, t.completed_at
                FROM task_dependencies td
                JOIN tasks t ON td.depends_on_task_id = t.task_id
                WHERE td.task_id = ? AND t.status = 'completed'
                ORDER BY t.completed_at ASC, t.task_id ASC
                """,
                (task_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Artifact Management
    # -------------------------------------------------------------------------

    def record_artifact(
        self,
        task_id: str,
        job_id: str,
        artifact_type: ArtifactType,
        title: str,
        file_path: str,
        content_preview: Optional[str] = None,
    ) -> Artifact:
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, task_id, job_id, artifact_type, title, file_path, content_preview, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    task_id,
                    job_id,
                    artifact_type.value,
                    title,
                    file_path,
                    content_preview,
                    now,
                ),
            )
        return Artifact(
            artifact_id=artifact_id,
            task_id=task_id,
            job_id=job_id,
            artifact_type=artifact_type,
            title=title,
            file_path=file_path,
            content_preview=content_preview,
            created_at=now,
        )

    def get_artifacts(
        self, job_id: Optional[str] = None, task_id: Optional[str] = None
    ) -> list[Artifact]:
        query = "SELECT * FROM artifacts WHERE 1=1"
        params: list[Any] = []
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " ORDER BY created_at DESC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                Artifact(
                    artifact_id=r["artifact_id"],
                    task_id=r["task_id"],
                    job_id=r["job_id"],
                    artifact_type=ArtifactType(r["artifact_type"]),
                    title=r["title"],
                    file_path=r["file_path"],
                    content_preview=r["content_preview"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # Helper Parser
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_task(row: sqlite3.Row, deps: list[str]) -> Task:
        params = json.loads(row["tool_params"]) if row["tool_params"] else None
        scheduled_at = None
        if "scheduled_at" in row.keys() and row["scheduled_at"]:
            val = row["scheduled_at"]
            if isinstance(val, str):
                try:
                    scheduled_at = datetime.fromisoformat(val)
                except Exception:
                    scheduled_at = None
            else:
                scheduled_at = val

        return Task(
            task_id=row["task_id"],
            job_id=row["job_id"],
            title=row["title"],
            tier=Tier(row["tier"]),
            task_type=TaskType(row["task_type"]),
            status=TaskStatus(row["status"]),
            prompt_template=row["prompt_template"],
            tool_name=row["tool_name"],
            tool_params=params,
            result_payload=row["result_payload"],
            artifact_path=row["artifact_path"],
            depends_on=deps,
            priority=row["priority"],
            tokens_generated=row["tokens_generated"],
            elapsed_seconds=row["elapsed_seconds"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            scheduled_at=scheduled_at,
            note_id=row["note_id"] if "note_id" in row.keys() else None,
            error_message=row["error_message"],
        )

    # -------------------------------------------------------------------------
    # Ephemeral Notes (Mid-term Scratchpad)
    # -------------------------------------------------------------------------

    def create_note(self, title: str, content: str, ttl_hours: float = 24.0) -> EphemeralNote:
        now = datetime.now()
        expires_at = now + timedelta(hours=ttl_hours)
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ephemeral_notes (title, content, created_at, updated_at, expires_at, dismissed)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (title, content, now, now, expires_at),
            )
            note_id = cursor.lastrowid
        return EphemeralNote(
            id=note_id,
            title=title,
            content=content,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            dismissed=False,
        )

    def get_active_notes(self) -> list[EphemeralNote]:
        """Returns all non-dismissed, non-expired notes ordered by updated_at DESC."""
        now = datetime.now()
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ephemeral_notes
                WHERE dismissed = 0 AND expires_at > ?
                ORDER BY updated_at DESC
                """,
                (now,),
            ).fetchall()
            return [self._row_to_note(r) for r in rows]

    def get_note(self, note_id: int, apply_mercy_hour: bool = True) -> Optional[EphemeralNote]:
        """
        Retrieves a note by ID.
        If apply_mercy_hour is True and the note is active with < 1 hour remaining,
        extends expires_at by 1 hour from now (Mercy 1-Hour Rule).
        """
        now = datetime.now()
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM ephemeral_notes WHERE id = ?", (note_id,)).fetchone()
            if not row:
                return None
            note = self._row_to_note(row)
            if note.dismissed:
                return note

            # Mercy 1-Hour Rule: if active and has < 1 hour left, extend by 1 hour
            if apply_mercy_hour and note.expires_at > now:
                time_left = (note.expires_at - now).total_seconds()
                if time_left < 3600:
                    new_expiry = now + timedelta(hours=1)
                    conn.execute(
                        "UPDATE ephemeral_notes SET expires_at = ? WHERE id = ?",
                        (new_expiry, note_id),
                    )
                    note.expires_at = new_expiry
            return note

    def append_note(self, note_id: int, text: str, ttl_hours: float = 24.0) -> Optional[EphemeralNote]:
        """
        Appends text to an active note, updates updated_at, and resets the TTL (default 24h).
        Returns None if note does not exist or is dismissed.
        """
        note = self.get_note(note_id, apply_mercy_hour=False)
        if not note or note.dismissed:
            return None

        separator = "\n\n" if note.content.strip() else ""
        new_content = f"{note.content}{separator}{text}"
        now = datetime.now()
        new_expiry = now + timedelta(hours=ttl_hours)

        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE ephemeral_notes
                SET content = ?, updated_at = ?, expires_at = ?
                WHERE id = ?
                """,
                (new_content, now, new_expiry, note_id),
            )

        note.content = new_content
        note.updated_at = now
        note.expires_at = new_expiry
        return note

    def dismiss_note(self, note_id: int) -> bool:
        """Explicitly dismisses an ephemeral note."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE ephemeral_notes SET dismissed = 1 WHERE id = ?",
                (note_id,),
            )
            return cursor.rowcount > 0

    def keepalive_note(self, note_id: int, ttl_hours: float = 24.0) -> Optional[EphemeralNote]:
        """Refreshes the TTL of an active note without appending content."""
        note = self.get_note(note_id, apply_mercy_hour=False)
        if not note or note.dismissed:
            return None

        now = datetime.now()
        new_expiry = now + timedelta(hours=ttl_hours)
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE ephemeral_notes SET expires_at = ?, updated_at = ? WHERE id = ?",
                (new_expiry, now, note_id),
            )
        note.expires_at = new_expiry
        note.updated_at = now
        return note

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> EphemeralNote:
        def parse_dt(v):
            if not v:
                return None
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v)
                except Exception:
                    return None
            return v

        return EphemeralNote(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
            expires_at=parse_dt(row["expires_at"]) or datetime.now(),
            dismissed=bool(row["dismissed"]),
        )
