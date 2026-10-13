"""Data models and schemas for Baton orchestrator."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class Tier(str, Enum):
    SYSTEM_1 = "system_1"
    SYSTEM_2 = "system_2"


class TaskType(str, Enum):
    MACRO_PLAN = "macro_plan"
    FAST_RECON = "fast_recon"
    FAST_VERIFY = "fast_verify"
    DEEP_SYNTHESIS = "deep_synthesis"


class TaskStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EngineType(str, Enum):
    OLLAMA = "ollama"
    COLIBRI = "colibri"


class EngineStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


class ArtifactType(str, Enum):
    DOSSIER = "dossier"
    CODE = "code"
    SPEC = "spec"
    REPORT = "report"
    DAG_PLAN = "dag_plan"


class TaskCancelledError(Exception):
    """Raised when a task or generation stream is manually cancelled/stopped."""
    pass


class VerificationFailedError(Exception):
    """Raised when an automated verification pass fails and local System 1 repairs fail."""

    def __init__(
        self,
        message: str,
        artifact_path: Optional[str] = None,
        failure_trace: str = "",
        target_code: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.artifact_path = artifact_path
        self.failure_trace = failure_trace
        self.target_code = target_code


class TaskPlan(BaseModel):
    """Specification for a task emitted by System 2 planner or human input."""
    id: str
    title: str
    type: TaskType = TaskType.FAST_RECON
    tier: Tier = Tier.SYSTEM_1
    tool: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    depends_on: list[str] = Field(default_factory=list)
    prompt: Optional[str] = None
    description: Optional[str] = None
    output_artifact: Optional[str] = None
    priority: int = 0
    scheduled_at: Optional[datetime] = None


class DAGPlan(BaseModel):
    """A directed acyclic graph plan emitted by System 2 Macro-Planner."""
    project: str
    goal: str
    tasks: list[TaskPlan]


class BriefingDossier(BaseModel):
    """Structured, token-dense briefing dossier produced by System 1 reconnaissance."""
    summary: str = Field(description="One-sentence executive summary")
    core_facts: list[str] = Field(default_factory=list, description="Core verified facts, constants, and parameters")
    equations_or_apis: list[str] = Field(default_factory=list, description="Exact equations, formulas, type signatures, or API interfaces")
    open_questions: list[str] = Field(default_factory=list, description="Unresolved ambiguities or edge cases")
    source_citations: list[str] = Field(default_factory=list, description="Sources, files, or citations referenced")

    def to_compact_markdown(self) -> str:
        """Renders the dossier into an ultra-compact, token-dense format stripping conversational fluff."""
        parts = [f"## Summary: {self.summary.strip()}"]
        if self.core_facts:
            facts = "\n".join(f"- {f.strip()}" for f in self.core_facts if f.strip())
            parts.append(f"### Core Facts\n{facts}")
        if self.equations_or_apis:
            eqs = "\n".join(
                f"- `{eq.strip()}`" if not eq.strip().startswith("`") and not eq.strip().startswith("```") else f"- {eq.strip()}"
                for eq in self.equations_or_apis if eq.strip()
            )
            parts.append(f"### Equations & APIs\n{eqs}")
        if self.open_questions:
            questions = "\n".join(f"- {q.strip()}" for q in self.open_questions if q.strip())
            parts.append(f"### Open Questions\n{questions}")
        if self.source_citations:
            citations = "\n".join(f"- {c.strip()}" for c in self.source_citations if c.strip())
            parts.append(f"### Citations\n{citations}")
        return "\n\n".join(parts)


class VerificationDossier(BaseModel):
    """Structured verification dossier produced by System 1 code verifier."""
    verdict: str = Field(description="VERIFIED or FAILED")
    syntax_valid: bool = Field(default=True, description="Whether AST syntax validation passed")
    tests_passed: int = Field(default=0, description="Count of passed tests")
    tests_failed: int = Field(default=0, description="Count of failed tests")
    test_errors: int = Field(default=0, description="Count of test errors or crashes")
    key_findings: list[str] = Field(default_factory=list, description="Key verification findings or failure traces")
    remediation_points: list[str] = Field(default_factory=list, description="Actionable remediation steps if failed")

    def to_compact_markdown(self) -> str:
        status_line = (
            f"## Verdict: {self.verdict.strip().upper()} "
            f"(Syntax: {'PASS' if self.syntax_valid else 'FAIL'} | "
            f"Tests: {self.tests_passed} passed, {self.tests_failed} failed, {self.test_errors} errors)"
        )
        parts = [status_line]
        if self.key_findings:
            parts.append("### Findings\n" + "\n".join(f"- {f.strip()}" for f in self.key_findings if f.strip()))
        if self.remediation_points:
            parts.append("### Remediation\n" + "\n".join(f"- {r.strip()}" for r in self.remediation_points if r.strip()))
        return "\n\n".join(parts)


class RFDossier(BaseModel):
    """Structured RF signal intelligence dossier."""
    band_name: str = Field(description="Frequency band name or frequency range")
    noise_floor_db: Optional[float] = Field(default=None, description="Average noise floor in dBm")
    carriers_detected: list[str] = Field(default_factory=list, description="Detected carriers with frequency, power, and SNR")
    demodulated_transmissions: list[str] = Field(default_factory=list, description="Demodulated packets or signals")
    recommendations: list[str] = Field(default_factory=list, description="Channel parameters or processing recommendations")

    def to_compact_markdown(self) -> str:
        nf_str = f" (Noise Floor: {self.noise_floor_db:.1f} dBm)" if self.noise_floor_db is not None else ""
        parts = [f"## RF Survey: {self.band_name.strip()}{nf_str}"]
        if self.carriers_detected:
            parts.append("### Detected Carriers\n" + "\n".join(f"- {c.strip()}" for c in self.carriers_detected if c.strip()))
        if self.demodulated_transmissions:
            parts.append("### Demodulated Transmissions\n" + "\n".join(f"- {t.strip()}" for t in self.demodulated_transmissions if t.strip()))
        if self.recommendations:
            parts.append("### Recommendations\n" + "\n".join(f"- {r.strip()}" for r in self.recommendations if r.strip()))
        return "\n\n".join(parts)


class Task(BaseModel):
    """Represents a discrete task tracked in SQLite orchestrator.db."""
    task_id: str
    job_id: str
    title: str
    tier: Tier
    task_type: TaskType
    status: TaskStatus = TaskStatus.BLOCKED
    prompt_template: Optional[str] = None
    tool_name: Optional[str] = None
    tool_params: Optional[dict[str, Any]] = None
    result_payload: Optional[str] = None
    artifact_path: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)
    priority: int = 0
    tokens_generated: int = 0
    elapsed_seconds: float = 0.0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    error_message: Optional[str] = None


class Job(BaseModel):
    """High-level container for a project or macro goal."""
    job_id: str
    title: str
    goal: str
    status: JobStatus = JobStatus.PENDING
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Artifact(BaseModel):
    """Production artifact produced by System 1 or System 2."""
    artifact_id: str
    task_id: str
    job_id: str
    artifact_type: ArtifactType
    title: str
    file_path: str
    content_preview: Optional[str] = None
    created_at: Optional[datetime] = None


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration string."""
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_sec = int(seconds)
    mins = total_sec // 60
    secs = total_sec % 60
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours = mins // 60
    rem_mins = mins % 60
    return f"{hours}h {rem_mins:02d}m {secs:02d}s"


class TaskProgress(BaseModel):
    """Real-time live progress tracking for active streaming tasks."""
    task_id: str
    tier: Tier
    started_at: float
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    reasoning_text: str = ""
    content_text: str = ""
    last_update: float = 0.0
    is_completed: bool = False
    finish_reason: Optional[str] = None
    is_truncated: bool = False
    error: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        import time
        ref = self.last_update if self.is_completed and self.last_update > 0 else time.time()
        return max(ref - self.started_at, 0.0)

    @property
    def formatted_elapsed(self) -> str:
        return format_duration(self.elapsed_seconds)

    def save_to_file(self, path: Path) -> None:
        """Atomically saves progress state to a JSON file."""
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load_from_file(cls, path: Path) -> Optional[TaskProgress]:
        """Loads progress state from a JSON file, handling partial writes safely."""
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        try:
            content = p.read_text(encoding="utf-8")
            return cls.model_validate_json(content)
        except Exception:
            return None

