"""Deterministic Mediator and Relay Loop for Baton.

Coordinates priority queue state transitions, Mode A mutual exclusion engine switching,
and dispatching work orders to System 1 and System 2.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from baton.db import Database
from baton.engine_manager import EngineManager
from baton.models import (
    ArtifactType,
    DAGPlan,
    Job,
    JobStatus,
    Task,
    TaskCancelledError,
    TaskPlan,
    TaskProgress,
    TaskStatus,
    TaskType,
    Tier,
    VerificationFailedError,
)
from baton.system1 import System1Runner
from baton.system2 import System2Runner

logger = logging.getLogger("baton.orchestrator")


class Orchestrator:
    """Deterministic conductor: schedules tasks, switches engines, and records artifacts."""

    def __init__(
        self,
        db: Optional[Database] = None,
        engine_manager: Optional[EngineManager] = None,
        system1: Optional[System1Runner] = None,
        system2: Optional[System2Runner] = None,
        progress_dir: Optional[Path] = None,
        auto_verify: bool = True,
        auto_repair: bool = True,
        tier_sticky: bool = True,
        starvation_threshold: Optional[int] = 10,
    ):
        self.db = db or Database()
        self.engine_mgr = engine_manager or EngineManager()
        self.system1 = system1 or System1Runner(ollama_url=getattr(self.engine_mgr, "ollama_url", None))
        self.system2 = system2 or System2Runner(colibri_url=getattr(self.engine_mgr, "colibri_url", None))
        self.progress_dir = progress_dir or Path("baton/artifacts/progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.auto_verify = auto_verify
        self.auto_repair = auto_repair
        self.tier_sticky = tier_sticky
        self.starvation_threshold = starvation_threshold
        self.consecutive_tier_count: int = 0
        self.active_progress: dict[str, TaskProgress] = {}
        self.active_task_id: Optional[str] = None
        self.on_task_update: Optional[Callable[[Task], None]] = None
        self.on_progress_update: Optional[Callable[[TaskProgress], None]] = None

    # -------------------------------------------------------------------------
    # Project & Job Submission
    # -------------------------------------------------------------------------

    def submit_goal(self, project: str, goal: str, plan_with_system2: bool = True) -> Job:
        """Creates a new job container. Optionally prompts System 2 for upfront DAG planning."""
        job = self.db.create_job(title=project, goal=goal)

        if plan_with_system2:
            logger.info("Submitting goal to System 2 for macro-planning...")
            self.engine_mgr.switch_to_tier(Tier.SYSTEM_2)
            try:
                try:
                    dag_plan = self.system2.plan_project(project, goal, system1_repair_runner=self.system1)
                except TypeError:
                    dag_plan = self.system2.plan_project(project, goal)
                self.db.ingest_dag_plan(job.job_id, dag_plan)
                logger.info("Ingested %d tasks from System 2 plan.", len(dag_plan.tasks))
            except Exception as e:
                logger.error("Failed to generate System 2 plan: %s", e)
                # Create a fallback single-step task
                fallback_plan = DAGPlan(
                    project=project,
                    goal=goal,
                    tasks=[
                        TaskPlan(
                            id=f"plan_fallback_{job.job_id}",
                            title=f"Direct Execution: {project}",
                            tier=Tier.SYSTEM_1,
                            prompt=goal,
                        )
                    ],
                )
                self.db.ingest_dag_plan(job.job_id, fallback_plan)

        return job

    def add_manual_task(
        self,
        job_id: str,
        title: str,
        tier: Tier,
        task_id: Optional[str] = None,
        prompt: Optional[str] = None,
        tool: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
        priority: int = 0,
        scheduled_at: Optional[datetime] = None,
    ) -> Task:
        """Directly inject a user-specified task into the priority queue."""
        plan = DAGPlan(
            project="Manual Injection",
            goal=title,
            tasks=[
                TaskPlan(
                    id=task_id or f"task_{title.lower().replace(' ', '_')[:16]}",
                    title=title,
                    tier=tier,
                    prompt=prompt,
                    tool=tool,
                    params=params,
                    depends_on=depends_on or [],
                    priority=priority,
                    scheduled_at=scheduled_at,
                )
            ],
        )
        tasks = self.db.ingest_dag_plan(job_id, plan)
        return tasks[0]

    # -------------------------------------------------------------------------
    # Single-Step Execution
    # -------------------------------------------------------------------------

    def get_active_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Retrieves real-time streaming progress from memory or disk cache."""
        if task_id in self.active_progress:
            return self.active_progress[task_id]
        prog_file = self.progress_dir / f"{task_id}.json"
        return TaskProgress.load_from_file(prog_file)

    def delete_task(self, task_id: str) -> bool:
        """Deletes a task from the database and purges active progress records."""
        prog_file = self.progress_dir / f"{task_id}.json"
        if prog_file.exists():
            try:
                prog_file.unlink()
            except Exception:
                pass
        self.active_progress.pop(task_id, None)
        return self.db.delete_task(task_id)

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
    ) -> Task:
        """Creates a new task in the orchestrator database."""
        return self.db.create_task(
            title=title,
            tier=tier,
            task_type=task_type,
            prompt_template=prompt_template,
            tool_name=tool_name,
            tool_params=tool_params,
            depends_on=depends_on,
            priority=priority,
            job_id=job_id,
            task_id=task_id,
            scheduled_at=scheduled_at,
        )

    def get_task_artifact_text(self, task_id: str) -> Optional[str]:
        """Retrieves artifact text for a task."""
        return self.db.get_task_artifact_text(task_id)

    def stop_current_task(self) -> Optional[str]:
        """Cancels and interrupts any actively executing System 1 or System 2 task."""
        logger.info("Stop requested by user. Cancelling active runners...")
        self.system1.cancel_active()
        self.system2.cancel_active()

        stopped_id = self.active_task_id

        # Check database for in_progress tasks
        in_progress = self.db.get_in_progress_tasks()
        for t in in_progress:
            logger.info("Marking in-progress task '%s' as stopped/failed.", t.task_id)
            self.db.mark_task_failed(t.task_id, "Stopped by user")
            if t.task_id in self.active_progress:
                prog = self.active_progress[t.task_id]
                prog.is_completed = True
                prog.error = "Stopped by user"
                prog.save_to_file(self.progress_dir / f"{t.task_id}.json")
            if not stopped_id:
                stopped_id = t.task_id

        self.active_task_id = None
        self.consecutive_tier_count = 0
        return stopped_id

    def select_next_task(self, runnable: list[Task]) -> Optional[Task]:
        """
        Selects the next task from runnable candidates using Tier-Sticky Scheduling.

        To minimize costly VRAM context-switch thrashing (unloading/reloading large models):
        - If tier_sticky is disabled, returns highest priority runnable task (scalar priority).
        - If the system is cold (engine_mgr.active_tier is None), prioritizes System 1 tasks
          (recon and verification) to unblock downstream nodes before spinning up Colibrì.
        - When an engine is warm, drains ready tasks for that tier up to `starvation_threshold`.
        - If the warm tier exceeds starvation_threshold and tasks for the other tier are waiting,
          yields to the other tier to prevent starvation.
        """
        if not runnable:
            return None

        if not self.tier_sticky:
            return runnable[0]

        active_tier = getattr(self.engine_mgr, "active_tier", None)

        # 1. Cold start: prefer System 1 to drain recon/verification tasks first
        if active_tier is None:
            sys1_candidates = [t for t in runnable if t.tier == Tier.SYSTEM_1]
            if sys1_candidates:
                return sys1_candidates[0]
            return runnable[0]

        # 2. Warm engine active: partition runnable tasks
        warm_candidates = [t for t in runnable if t.tier == active_tier]
        other_candidates = [t for t in runnable if t.tier != active_tier]

        # If no tasks remain for the warm tier, we must switch to other tier
        if not warm_candidates:
            return other_candidates[0]

        # If no tasks exist for the other tier, keep executing warm tier
        if not other_candidates:
            return warm_candidates[0]

        # Both warm and other candidates exist: check starvation threshold
        if (
            self.starvation_threshold is not None
            and self.consecutive_tier_count >= self.starvation_threshold
        ):
            logger.info(
                "Active tier '%s' reached starvation threshold (%d consecutive tasks). Yielding to '%s'.",
                active_tier.value if isinstance(active_tier, Tier) else active_tier,
                self.consecutive_tier_count,
                other_candidates[0].tier.value if isinstance(other_candidates[0].tier, Tier) else other_candidates[0].tier,
            )
            return other_candidates[0]

        return warm_candidates[0]

    def step(self, task_id: Optional[str] = None) -> Optional[Task]:
        """
        Pulls the next runnable task (or a specific ready task), manages engine transitions,
        executes the task, records artifacts, and unlocks downstream tasks.
        """
        # Concurrency exclusion: if any task is already IN_PROGRESS, refuse to start a concurrent step
        in_prog = self.db.get_in_progress_tasks()
        if in_prog:
            logger.warning(
                "Cannot step: task '%s' is already in progress.", in_prog[0].task_id
            )
            return None

        if task_id:
            target_task = self.db.get_task(task_id)
            if not target_task or target_task.status != TaskStatus.READY:
                logger.warning("Target task '%s' is not in READY state.", task_id)
                return None
        else:
            runnable = self.db.get_runnable_tasks()
            if not runnable:
                return None
            target_task = self.select_next_task(runnable)
            if not target_task:
                return None

        logger.info(
            "Executing task: '%s' (%s, priority %d)",
            target_task.task_id,
            target_task.tier.value,
            target_task.priority,
        )

        # Mode A: Switch engine if required
        prev_tier = getattr(self.engine_mgr, "active_tier", None)
        if prev_tier != target_task.tier:
            logger.info("Transitioning active engine to %s...", target_task.tier.value)
            switched = self.engine_mgr.switch_to_tier(target_task.tier)
            if not switched:
                logger.error("Failed to switch engine to %s.", target_task.tier.value)
                return None
            self.consecutive_tier_count = 1
        else:
            self.consecutive_tier_count += 1

        # Claim task atomically
        task = self.db.claim_specific_task(target_task.task_id)
        if not task:
            return None

        self.active_task_id = task.task_id

        if self.on_task_update:
            self.on_task_update(task)

        # Retrieve all completed upstream dossiers
        upstream = self.db.get_upstream_dossiers(task.task_id)

        # Initialize and publish initial progress
        prog = TaskProgress(
            task_id=task.task_id,
            tier=task.tier,
            started_at=time.time(),
        )
        self.active_progress[task.task_id] = prog
        prog.save_to_file(self.progress_dir / f"{task.task_id}.json")

        def _on_progress(p: TaskProgress) -> None:
            self.active_progress[task.task_id] = p
            p.save_to_file(self.progress_dir / f"{task.task_id}.json")
            if self.on_progress_update:
                self.on_progress_update(p)

        try:
            if task.tier == Tier.SYSTEM_1:
                try:
                    summary, art_path, tokens, elapsed = self.system1.execute_task(
                        task, upstream, progress_callback=_on_progress
                    )
                except TypeError:
                    summary, art_path, tokens, elapsed = self.system1.execute_task(task, upstream)

                if art_path:
                    is_verify = task.task_type.value == "fast_verify"
                    art_type = ArtifactType.REPORT if is_verify else ArtifactType.DOSSIER
                    prefix = "Verification" if is_verify else "Dossier"
                    self.db.record_artifact(
                        task_id=task.task_id,
                        job_id=task.job_id,
                        artifact_type=art_type,
                        title=f"{prefix}: {task.title}",
                        file_path=art_path,
                        content_preview=summary,
                    )
            else:  # Tier.SYSTEM_2
                try:
                    summary, art_path, tokens, elapsed = self.system2.synthesize(
                        task, upstream, progress_callback=_on_progress
                    )
                except TypeError:
                    summary, art_path, tokens, elapsed = self.system2.synthesize(task, upstream)

                if art_path:
                    self.db.record_artifact(
                        task_id=task.task_id,
                        job_id=task.job_id,
                        artifact_type=ArtifactType.CODE,
                        title=f"Synthesis: {task.title}",
                        file_path=art_path,
                        content_preview=summary,
                    )
                    # Automatically ensure downstream System 1 verification
                    if self.auto_verify and (art_path.endswith(".py") or task.task_type == TaskType.DEEP_SYNTHESIS):
                        self._ensure_downstream_verification(task, art_path)

            active_prog = self.active_progress.get(task.task_id, prog)
            active_prog.is_completed = True
            active_prog.tokens_generated = tokens
            active_prog.save_to_file(self.progress_dir / f"{task.task_id}.json")

            self.db.mark_task_completed(
                task_id=task.task_id,
                result_payload=summary,
                artifact_path=art_path,
                tokens_generated=tokens,
                elapsed_seconds=elapsed,
            )
            task.status = TaskStatus.COMPLETED
            task.result_payload = summary
            task.artifact_path = art_path

        except VerificationFailedError as e:
            logger.warning("Verification failed for task '%s': %s", task.task_id, e)
            active_prog = self.active_progress.get(task.task_id, prog)
            active_prog.is_completed = True
            active_prog.error = str(e)
            active_prog.save_to_file(self.progress_dir / f"{task.task_id}.json")
            self.db.mark_task_failed(task.task_id, str(e))
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

            # Escalate: enqueue System 2 repair task!
            if self.auto_repair:
                self._enqueue_system2_repair(task, e)

        except TaskCancelledError as e:
            logger.info("Task '%s' was stopped by user.", task.task_id)
            active_prog = self.active_progress.get(task.task_id, prog)
            active_prog.is_completed = True
            active_prog.error = "Stopped by user"
            active_prog.save_to_file(self.progress_dir / f"{task.task_id}.json")
            self.db.mark_task_failed(task.task_id, "Stopped by user")
            task.status = TaskStatus.FAILED
            task.error_message = "Stopped by user"

        except Exception as e:
            logger.error("Error executing task '%s': %s", task.task_id, e)
            active_prog = self.active_progress.get(task.task_id, prog)
            active_prog.is_completed = True
            active_prog.error = str(e)
            active_prog.save_to_file(self.progress_dir / f"{task.task_id}.json")
            self.db.mark_task_failed(task.task_id, str(e))
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

        finally:
            self.active_task_id = None

        if self.on_task_update:
            self.on_task_update(task)

        return task

    def _ensure_downstream_verification(self, task: Task, artifact_path: str) -> Optional[Task]:
        """
        When a System 2 synthesis task completes and outputs Python code,
        ensures a downstream System 1 FAST_VERIFY task is in place.
        Rewires existing downstream dependents so they wait on verification.
        """
        downstream_ids = self.db.get_downstream_tasks(task.task_id)
        # Check if an existing downstream task is already a FAST_VERIFY task
        for d_id in downstream_ids:
            dt = self.db.get_task(d_id)
            if dt and dt.task_type == TaskType.FAST_VERIFY:
                return dt

        verify_task_id = f"verify_{task.task_id}"

        # 1. Create the FAST_VERIFY task first so it exists in tasks table
        v_task = self.db.create_task(
            task_id=verify_task_id,
            job_id=task.job_id,
            title=f"Verify: {task.title}",
            tier=Tier.SYSTEM_1,
            task_type=TaskType.FAST_VERIFY,
            prompt_template=f"Verify syntax and execute test suites on '{task.title}' ({artifact_path}).",
            depends_on=[task.task_id],
            priority=task.priority + 1,
        )

        # 2. Rewire existing downstream tasks so they depend on verify_task_id instead of task.task_id
        self.db.rewire_dependencies(old_dep_id=task.task_id, new_dep_id=verify_task_id)
        logger.info("Automatically injected downstream verification task '%s' for '%s'.", verify_task_id, task.task_id)
        return v_task

    def _enqueue_system2_repair(self, verify_task: Task, error: VerificationFailedError) -> Task:
        """
        Escalates a failed verification pass to System 2 by enqueuing a DEEP_SYNTHESIS repair task.
        Rewires downstream dependents to wait on the repair task.
        """
        repair_id = f"repair_{verify_task.task_id}"

        prompt = (
            f"Automated verification failed and System 1 local repairs were exhausted (Fast-Patch Ladder Tier 2 Escalation).\n\n"
            f"### Failure Diagnostic Dossier:\n{error.message}\n\n"
            f"### Condensed Traceback:\n```text\n{error.failure_trace[:2000]}\n```\n\n"
            f"Rethink the algorithm and regenerate the complete, working implementation."
        )

        # 1. Create the repair task first so it exists in tasks table
        repair_task = self.db.create_task(
            task_id=repair_id,
            job_id=verify_task.job_id,
            title=f"Repair: {verify_task.title}",
            tier=Tier.SYSTEM_2,
            task_type=TaskType.DEEP_SYNTHESIS,
            prompt_template=prompt,
            depends_on=[],  # Ready immediately for System 2
            priority=verify_task.priority + 2,
        )

        # 2. Rewire any downstream tasks that were waiting on verify_task to now wait on repair_task
        self.db.rewire_dependencies(old_dep_id=verify_task.task_id, new_dep_id=repair_id)
        logger.info("Enqueued System 2 repair task '%s' following failed verification.", repair_id)
        return repair_task

    def run_all(
        self,
        max_steps: int = 50,
        wait_for_scheduled: bool = False,
        max_wait_seconds: Optional[float] = None,
    ) -> list[Task]:
        """Drives the queue until no runnable tasks remain.
        If wait_for_scheduled is True, sleeps and waits for upcoming scheduled tasks.
        """
        completed: list[Task] = []
        for _ in range(max_steps):
            task = self.step()
            if not task:
                if wait_for_scheduled:
                    next_scheduled = self.db.get_next_scheduled_task()
                    if next_scheduled and next_scheduled.scheduled_at:
                        now = datetime.now()
                        delta = (next_scheduled.scheduled_at - now).total_seconds()
                        if delta > 0:
                            if max_wait_seconds is not None and delta > max_wait_seconds:
                                logger.info(
                                    "Next scheduled task exceeds max wait ceiling (%0.1fs > %0.1fs).",
                                    delta,
                                    max_wait_seconds,
                                )
                                break
                            logger.info(
                                "Sleeping %0.1fs for scheduled task '%s' at %s...",
                                delta,
                                next_scheduled.task_id,
                                next_scheduled.scheduled_at,
                            )
                            deadline = time.time() + delta
                            while time.time() < deadline:
                                if getattr(self, "_cancel_requested", False):
                                    break
                                time.sleep(min(1.0, deadline - time.time()))
                            continue
                break
            completed.append(task)
        return completed
