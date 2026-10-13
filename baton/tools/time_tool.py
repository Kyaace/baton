"""Time and Alarm scheduling tool for Baton System 1."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from baton.db import Database

logger = logging.getLogger("baton.tools.time")


def parse_time_spec(spec: str, base_time: Optional[datetime] = None) -> datetime:
    """Parses relative deltas, absolute ISO-8601 timestamps, or time-of-day strings.

    Supported patterns:
    - Relative deltas: '30s', '15m', '2h', '1d', '+30m', 'in 15 minutes', 'in 2 hours', '1h 30m'
    - Absolute ISO-8601: '2026-10-13T14:30:00', '2026-10-13 14:30:00', '2026-10-13T14:30:00Z'
    - Time-of-day: '14:30', '14:30:00', '2:30pm', '09:00 AM' (advances to tomorrow if already passed)
    - Unix epoch: '1791900300'
    """
    clean = spec.strip().lower()
    base = base_time or datetime.now()

    # 1. Numeric epoch seconds
    if clean.isdigit() and len(clean) >= 9:
        try:
            return datetime.fromtimestamp(float(clean))
        except (ValueError, OverflowError):
            pass

    # 2. Relative natural phrases ("in 10 minutes", "in 2 hours", etc.)
    if clean.startswith("in "):
        clean = clean[3:].strip()
    if clean.startswith("+"):
        clean = clean[1:].strip()

    # Parse compact relative compound units: e.g. "1h 30m", "45s", "10 min", "2 hours"
    relative_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(s(?:ec(?:ond)?s?)?|m(?:in(?:ute)?s?)?|h(?:(?:ou)?rs?)?|d(?:ays?)?|w(?:eeks?)?)",
        clean,
    )
    if relative_matches:
        delta = timedelta()
        for val_str, unit_str in relative_matches:
            val = float(val_str)
            u = unit_str[0]
            if u == "s":
                delta += timedelta(seconds=val)
            elif u == "m":
                delta += timedelta(minutes=val)
            elif u == "h":
                delta += timedelta(hours=val)
            elif u == "d":
                delta += timedelta(days=val)
            elif u == "w":
                delta += timedelta(weeks=val)
        return base + delta

    # 3. Time of day: e.g. "14:30", "14:30:00", "2:30pm", "09:00 AM"
    tod_match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?$", clean)
    if tod_match:
        hour = int(tod_match.group(1))
        minute = int(tod_match.group(2))
        sec = int(tod_match.group(3)) if tod_match.group(3) else 0
        ampm = tod_match.group(4)

        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        target = base.replace(hour=hour, minute=minute, second=sec, microsecond=0)
        # If target time has already elapsed today, advance by 1 day
        if target <= base:
            target += timedelta(days=1)
        return target

    # 4. Standard ISO-8601 parsing
    iso_clean = spec.strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(iso_clean)
        # If timezone-aware with UTC, convert to local naive for SQLite comparison consistency
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    raise ValueError(
        f"Could not parse time specification: '{spec}'. "
        "Use relative ('30m', 'in 2 hours'), time-of-day ('14:30', '2:30pm'), or ISO-8601 ('2026-10-13T14:30:00')."
    )


class TimeTool:
    """Tool Shelf bundle providing wall-clock temporal grounding and task alarm scheduling."""

    def __init__(self, db: Optional[Database] = None, orchestrator: Optional[Any] = None):
        self.orchestrator = orchestrator
        self.db = db or (orchestrator.db if orchestrator else Database())

    def get_current_time(self, format: Optional[str] = None) -> dict[str, Any]:
        """Returns structured real-world local and UTC time data."""
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)

        tz_name = now_local.tzname() or "Local"
        offset_seconds = now_local.utcoffset().total_seconds() if now_local.utcoffset() else 0
        hours, remainder = divmod(abs(int(offset_seconds)), 3600)
        minutes = remainder // 60
        sign = "+" if offset_seconds >= 0 else "-"
        offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"

        return {
            "utc_iso": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iso_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local_iso": now_local.isoformat(),
            "iso_local": now_local.isoformat(),
            "local_human": now_local.strftime("%A, %B %d, %Y %I:%M:%S %p"),
            "timezone": tz_name,
            "timezone_name": tz_name,
            "utc_offset": offset_str,
            "epoch_seconds": int(time.time()),
            "epoch_timestamp": int(time.time()),
            "formatted": f"Local: {now_local.strftime('%A, %B %d, %Y %I:%M:%S %p')} ({tz_name}, {offset_str}) | UTC: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        }

    def format_time_markdown(self) -> str:
        """Returns current time formatted as clean Markdown."""
        info = self.get_current_time()
        return (
            "### Current System Time\n"
            f"- **Local Time:** {info['local_human']} ({info['timezone']}, {info['utc_offset']})\n"
            f"- **UTC Time:** {info['utc_iso']}\n"
            f"- **ISO-8601 Local:** `{info['local_iso']}`\n"
            f"- **Unix Timestamp:** `{info['epoch_seconds']}`"
        )

    def schedule_task_alarm(
        self,
        task_id: str,
        when: Optional[str] = None,
        delay_or_timestamp: Optional[str] = None,
    ) -> str:
        """Schedules or defers an existing task in the queue to a future time."""
        task = self.db.get_task(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found in database."

        target_spec = when or delay_or_timestamp
        if not target_spec:
            return "Error: Missing time specification ('when' or 'delay_or_timestamp')."

        target_dt = parse_time_spec(target_spec)
        now = datetime.now()
        delta_s = (target_dt - now).total_seconds()

        if delta_s <= 0:
            # Setting schedule in the past or immediately
            self.db.set_task_schedule(task_id, None)
            return f"Task '{task_id}' scheduled for immediate execution (schedule cleared)."

        self.db.set_task_schedule(task_id, target_dt)

        countdown_str = self._format_delta(delta_s)
        return (
            f"✅ **Alarm Scheduled for Task:** `{task.task_id}` ({task.title})\n"
            f"- **Execution Time:** `{target_dt.strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"- **Wake Countdown:** In {countdown_str} ({int(delta_s)} seconds)\n"
            "- **Status:** Deferred until alarm trigger."
        )

    def create_scheduled_task(
        self,
        title: str,
        when: Optional[str] = None,
        delay_or_timestamp: Optional[str] = None,
        prompt: Optional[str] = None,
        tier: str = "system_1",
        task_type: Optional[str] = None,
        tool: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        priority: int = 5,
        job_id: Optional[str] = None,
    ) -> str:
        """Creates a new task already scheduled for a future wake-up time."""
        target_spec = when or delay_or_timestamp
        if not target_spec:
            return "Error: Missing time specification ('when' or 'delay_or_timestamp')."

        target_dt = parse_time_spec(target_spec)
        now = datetime.now()
        delta_s = (target_dt - now).total_seconds()

        task = self.db.create_task(
            title=title,
            tier=tier,
            task_type=task_type,
            prompt_template=prompt or title,
            tool_name=tool,
            tool_params=params,
            priority=priority,
            job_id=job_id,
            scheduled_at=target_dt,
        )

        countdown_str = self._format_delta(delta_s)
        return (
            f"✅ **New Scheduled Task Created:** `{task.task_id}`\n"
            f"- **Title:** {task.title}\n"
            f"- **Tier:** `{task.tier.value}`\n"
            f"- **Scheduled Wake:** `{target_dt.strftime('%Y-%m-%d %H:%M:%S')}` (in {countdown_str})\n"
            f"- **Status:** `{task.status.value}` (Waiting for alarm)"
        )

    def list_alarms(self) -> str:
        """Lists all pending scheduled tasks with countdowns."""
        tasks = self.db.get_scheduled_tasks()
        if not tasks:
            return "No scheduled alarms in queue. All ready tasks are set for immediate execution."

        now = datetime.now()
        lines = [
            f"### Active Task Alarms ({len(tasks)} scheduled)",
            "| Task ID | Tier | Title | Target Time | Countdown |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for t in tasks:
            if not t.scheduled_at:
                continue
            delta = (t.scheduled_at - now).total_seconds()
            cd_str = self._format_delta(delta) if delta > 0 else "Ready now"
            sched_str = t.scheduled_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"| `{t.task_id}` | `{t.tier.value}` | {t.title} | `{sched_str}` | {cd_str} |"
            )

        return "\n".join(lines)

    def cancel_alarm(self, task_id: str, run_now: bool = True) -> str:
        """Clears the scheduled time on a task."""
        task = self.db.get_task(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."

        if not task.scheduled_at:
            return f"Task '{task_id}' has no scheduled alarm attached."

        self.db.set_task_schedule(task_id, None)
        action_msg = "Task is now immediately runnable." if run_now else "Schedule cleared."
        return f"Alarm cancelled for task `{task_id}`. {action_msg}"

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Deterministic dispatcher for named tool calls."""
        name = tool_name.lower().strip()
        if name in ("get_current_time", "current_time", "time"):
            return self.format_time_markdown()
        elif name in ("schedule_task_alarm", "schedule_alarm", "set_alarm"):
            return self.schedule_task_alarm(
                task_id=params["task_id"],
                when=params.get("when") or params.get("delay_or_timestamp"),
            )
        elif name in ("create_scheduled_task", "new_scheduled_task"):
            return self.create_scheduled_task(
                title=params["title"],
                when=params.get("when") or params.get("delay_or_timestamp"),
                prompt=params.get("prompt"),
                tier=params.get("tier", "system_1"),
                task_type=params.get("task_type"),
                tool=params.get("tool"),
                params=params.get("params"),
                priority=params.get("priority", 5),
                job_id=params.get("job_id"),
            )
        elif name in ("list_alarms", "alarms"):
            return self.list_alarms()
        elif name in ("cancel_alarm", "clear_alarm"):
            return self.cancel_alarm(
                task_id=params["task_id"],
                run_now=params.get("run_now", True),
            )
        else:
            return f"Error: Unknown Time tool '{tool_name}'."

    @staticmethod
    def _format_delta(seconds: float) -> str:
        if seconds <= 0:
            return "0s"
        total = int(seconds)
        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        mins, secs = divmod(rem, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if mins > 0:
            parts.append(f"{mins}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)
