"""Baton System 1 Interactive Chat Copilot.

Provides a fast, zero-latency conversational interface using Ollama function calling (tools)
to inspect the queue, draft and add tasks, review artifacts, and query offline knowledge.

Features a modular 'Tool Shelf' architecture:
- 'core': Baseline tool shelf manager (list_tool_bundles, activate_tool_bundle, deactivate_tool_bundle)
- 'baton': Baton queue & task operations (add_task, delete_task, retry_failed_tasks, queue_status, step_queue, etc.)
- 'wikipedia': Offline Kiwix ZIM Wikipedia lookup
- 'newsboat': Offline RSS news monitoring & article reader
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from baton.db import Database
from baton.engine_manager import EngineManager
from baton.models import Task, TaskStatus, Tier
from baton.orchestrator import Orchestrator
from baton.tools.docs_tool import DocsTool
from baton.tools.newsboat import NewsboatTool
from baton.tools.sandbox import SandboxTool
from baton.tools.sdr import SdrTool
from baton.tools.time_tool import TimeTool
from baton.tools.vault import VaultTool
from baton.tools.zim import ZimTool

logger = logging.getLogger("baton.chat")
console = Console()

# -----------------------------------------------------------------------------
# 1. Baseline Shelf Manager Schema ('core' bundle)
# -----------------------------------------------------------------------------

SHELF_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tool_bundles",
            "description": "Inspects the Tool Shelf: lists available tool bundles, their active/inactive status, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_tool_bundle",
            "description": (
                "Pulls a tool bundle from the shelf and activates it on your active toolbelt. "
                "Once activated, its tools become immediately available to call in subsequent turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bundle_name": {
                        "type": "string",
                        "enum": ["baton", "library", "newsboat", "sandbox", "sdr", "time", "docs", "vault", "wikipedia"],
                        "description": "Name of the tool bundle to activate ('baton', 'library', 'newsboat', 'sandbox', 'sdr', 'time', 'docs', or 'vault').",
                    },
                },
                "required": ["bundle_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deactivate_tool_bundle",
            "description": "Puts a tool bundle back on the shelf, removing its tools from your active toolbelt to save context tokens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bundle_name": {
                        "type": "string",
                        "enum": ["baton", "library", "newsboat", "sandbox", "sdr", "time", "docs", "vault", "wikipedia"],
                        "description": "Name of the tool bundle to deactivate ('baton', 'library', 'newsboat', 'sandbox', 'sdr', 'time', 'docs', or 'vault').",
                    },
                },
                "required": ["bundle_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_help",
            "description": (
                "Provides help, parameter descriptions, accepted aliases, and working examples "
                "for any tool command. If you are ever unsure how to call a tool or encounter an error, "
                "call this tool for immediate guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to get help for (e.g. 'add_task', 'start_orchestration', 'activate_tool_bundle', 'step_queue', 'wiki_read_section').",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 2. Baton Queue & Task Operations Schema ('baton' bundle)
# -----------------------------------------------------------------------------

BATON_QUEUE_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "Lists tasks in the Baton priority queue. Can filter by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "ready", "in_progress", "completed", "failed"],
                        "description": "Filter by task status. If omitted, lists all tasks.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tasks to return (default: 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task",
            "description": "Gets full details for a specific task by task_id, including prompt, dependencies, result payload, and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The unique ID of the task to inspect.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": (
                "Adds a new task to the Baton priority queue. "
                "Tasks can be assigned to 'system_1' (fast recon or verification) or 'system_2' (deep synthesis)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, clear title for the task.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["system_1", "system_2"],
                        "description": (
                            "Execution tier: 'system_1' for fast Wikipedia/ZIM recon, RSS news digestion, or verification; "
                            "'system_2' for deep macro-synthesis (DeepSeek-V4-Flash overnight runner)."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed prompt instructions for the model executing this task. "
                            "For System 2 tasks, specify the complete mathematical or software requirements. "
                            "For System 1 tasks, specify what facts, formulas, or constants to extract."
                        ),
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task_ids that must complete before this task becomes ready to execute.",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["fast_recon", "fast_verify", "deep_synthesis"],
                        "description": "Classification of the task (default: fast_recon for system_1, deep_synthesis for system_2).",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Tool name for System 1 tasks (e.g. 'wiki_read_section', 'rss_search', 'rss_latest').",
                    },
                    "tool_params": {
                        "type": "object",
                        "description": "Dictionary of parameters passed to the tool.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority integer (1-10, default: 5). Higher priority runs first.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional custom task_id slug (e.g. 'recon_orbital_formulas').",
                    },
                },
                "required": ["title", "tier", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "Deletes a task from the Baton queue by task_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to delete.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retry_failed_tasks",
            "description": "Resets all failed tasks in the queue back to ready (if dependencies are satisfied) or blocked.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_status",
            "description": "Returns counts of tasks in each state (ready, blocked, completed, failed) and active engine status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step_queue",
            "description": "Executes the next runnable task in the queue, or a specific task if task_id is given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional specific ready task_id to execute.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_orchestration",
            "description": (
                "Starts or resumes execution of the Baton task queue. "
                "Deterministically steps runnable tasks in priority order (up to max_steps), "
                "handling engine execution and recording artifacts. "
                "Call this immediately after queueing tasks when the user requests project execution or building."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum number of ready tasks to execute in this run (default: 5).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_artifacts",
            "description": "Lists production artifacts produced by completed tasks (briefing dossiers and synthesized code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of artifacts to list (default: 15).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": "Reads the content of an artifact file or previous task result payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID whose artifact or result to read.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Direct file path to read (alternative to task_id).",
                    },
                },
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 3. Wikipedia ZIM Knowledge Schema ('wikipedia' bundle)
# -----------------------------------------------------------------------------

WIKIPEDIA_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "zim_list_corpora",
            "description": "Lists all currently mounted and discovered offline .zim archives on disk with their active corpus aliases, filenames, and sizes. Always call this tool to see what archives are available before querying.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Searches mounted offline .zim archives for articles/documentation matching a query. Supports targeting a specific mounted corpus via 'corpus' (e.g. 'wikipedia', 'wikimed', 'python') or across all mounted archives via corpus='all'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms.",
                    },
                    "corpus": {
                        "type": "string",
                        "description": "Target corpus alias from zim_list_corpora (e.g. 'wikipedia', 'wikimed', 'python', or 'all').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of search results (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_read_lead",
            "description": "Reads the lead summary section of an offline Wikipedia article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact title of the Wikipedia article.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_read_section",
            "description": "Extracts the text of a specific named section from an offline Wikipedia article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact title of the Wikipedia article.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Heading name of the section to read.",
                    },
                },
                "required": ["title", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_infobox",
            "description": "Extracts structured key-value constants and attributes from the infobox of an offline Wikipedia article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact title of the Wikipedia article.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_toc",
            "description": "Retrieves the table of contents and section structure for an offline Wikipedia article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact title of the Wikipedia article.",
                    },
                },
                "required": ["title"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 4. Newsboat RSS Schema ('newsboat' bundle)
# -----------------------------------------------------------------------------

NEWSBOAT_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rss_list_feeds",
            "description": "Lists subscribed RSS feeds in the offline Newsboat database with article and unread counts.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rss_latest",
            "description": "Fetches the newest RSS articles across subscribed feeds, with IDs, titles, publication dates, and links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "feed": {
                        "type": "string",
                        "description": "Optional feed title filter (e.g. 'Ars Technica', 'Nature', 'NASA').",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "If true, only returns unread articles.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default: 10).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rss_search",
            "description": "Full-text search across article titles and content in the offline Newsboat RSS database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or topic (e.g. 'satellite', 'quantum', 'battery').",
                    },
                    "feed": {
                        "type": "string",
                        "description": "Optional feed title filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matching articles to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rss_read_article",
            "description": "Reads the full clean text/markdown content of an RSS article using its numeric ID from rss_latest or rss_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The numeric article ID (e.g. 1, 42).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of article text to retrieve (default: 3500).",
                    },
                },
                "required": ["id"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 5. Sandboxed Python REPL & Verification Schema ('sandbox' bundle)
# -----------------------------------------------------------------------------

SANDBOX_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python_eval",
            "description": "Executes Python code in an isolated sandbox with strict timeout and process-tree termination. Returns stdout, stderr, and execution duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code or snippet to execute.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename to stage script as in sandbox (default: '_snippet.py').",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30).",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_syntax_check",
            "description": "Fast (<1ms) AST syntax validation of Python source code without executing it. Returns line numbers and syntax error diagnostics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to validate.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pytest_run",
            "description": "Executes pytest test suite in the sandbox and parses structured pass/fail metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Test file path or directory within the sandbox (default: '.').",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 60).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_write_file",
            "description": "Stages a source code or test file into the isolated sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative filename within the sandbox.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File text content.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_read_file",
            "description": "Reads a file from the isolated sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative filename within the sandbox.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to read (default: 3500).",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sandbox_list_files",
            "description": "Lists all files currently staged in the isolated sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 6. RTL-SDR RF Telemetry & Spectrum Scanner Schema ('sdr' bundle)
# -----------------------------------------------------------------------------

SDR_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sdr_device_info",
            "description": "Inspects connected RTL-SDR hardware: verifies tuner presence (e.g. R820T/R828D), sample rates, frequency range, and host USB passthrough status. Note: R820T tuner is standard hardware; transient PLL lock warnings during initial probe are normal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "wsl", "tcp"],
                        "description": "Execution backend to probe (default: 'auto').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdr_spectrum_scan",
            "description": (
                "Performs a frequency sweep across a specified band using rtl_power to detect active radio carriers, "
                "peak power, and SNR above noise floor. For NOAA Weather Radio, sweep 162.35 to 162.60 MHz (channels: "
                "162.400, 162.425, 162.450, 162.475, 162.500, 162.525, 162.550 MHz) with gain_db=35 or 40."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_freq_mhz": {
                        "type": "number",
                        "description": "Start frequency in MHz (e.g. 118.0 for aviation, 144.0 for 2m ham, 162.35 for NOAA, 433.0 for ISM).",
                    },
                    "stop_freq_mhz": {
                        "type": "number",
                        "description": "Stop frequency in MHz (e.g. 137.0, 148.0, 162.60, 435.0).",
                    },
                    "bin_size_khz": {
                        "type": "number",
                        "description": "FFT bin resolution in kHz (default: 25.0).",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Duration of sweep in seconds (1-30, default: 5).",
                    },
                    "gain_db": {
                        "type": "number",
                        "description": "RF tuner gain in dB (0 for auto, or 20.0-49.6 dB. Recommended: 35-45 dB for weak or indoor signals, default: 0).",
                    },
                    "min_snr_db": {
                        "type": "number",
                        "description": "Minimum SNR in dB above noise floor to qualify as a signal peak (default: 2.0).",
                    },
                },
                "required": ["start_freq_mhz", "stop_freq_mhz"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdr_record_rf",
            "description": "Tunes to a specific frequency and captures a brief demodulated audio (FM/AM) or baseband burst with bounded duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency_mhz": {
                        "type": "number",
                        "description": "Target center frequency in MHz (e.g. 137.1 for NOAA-19, 144.39 for APRS, 162.4 for NOAA Weather Radio).",
                    },
                    "modulation": {
                        "type": "string",
                        "enum": ["fm", "am", "wfm", "raw"],
                        "description": "Demodulation mode ('fm' for narrow FM, 'am' for airband, 'wfm' for broadcast FM, 'raw' for baseband).",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Capture duration in seconds (1-15, default: 5).",
                    },
                    "output_filename": {
                        "type": "string",
                        "description": "Relative output filename in baton/artifacts/sdr/ (will be saved as valid .wav file).",
                    },
                    "gain_db": {
                        "type": "number",
                        "description": "RF tuner gain in dB (0 for auto, or 20.0-49.6 dB. Recommended: 35-45 dB for weak or indoor signals, default: 0).",
                    },
                },
                "required": ["frequency_mhz"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdr_decode_telemetry",
            "description": (
                "Listens for digital sensor telemetry packets (smart utility meters, weather stations, TPMS, remote keyfobs) "
                "using rtl_433 and returns structured JSON packets. Supports regional band presets ('us_smart_meters' for 915 MHz, "
                "'us_tpms' for 315 MHz, 'eu_sensors' for 433.92 MHz, 'us_survey' for multi-band hopping)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency_mhz": {
                        "type": "number",
                        "description": "Listening frequency in MHz (default: 915.0 for US smart meters or 433.92 for EU ISM).",
                    },
                    "band_preset": {
                        "type": "string",
                        "enum": ["us_smart_meters", "us_tpms", "eu_sensors", "us_survey"],
                        "description": "Regional sensor preset ('us_smart_meters' = 915M, 'us_tpms' = 315M, 'eu_sensors' = 433.92M, 'us_survey' = hopping).",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Listening duration in seconds (1-30, default: 10).",
                    },
                    "gain_db": {
                        "type": "number",
                        "description": "RF tuner gain in dB (default: 40.0 dB).",
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Optional protocol filter for rtl_433 (e.g. 'ERT-SCM' or device ID).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdr_adsb_tracking",
            "description": "Listens for Mode-S ADS-B transponder squitters at 1090 MHz using dump1090-mutability and extracts real-time aircraft positions, callsigns, altitudes, speeds, and signal strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Listening duration in seconds (1-30, default: 5).",
                    },
                    "gain_db": {
                        "type": "number",
                        "description": "RF tuner gain in dB (0 for auto, default: 45.0 dB for high sensitivity 1090 MHz reception).",
                    },
                    "max_aircraft": {
                        "type": "integer",
                        "description": "Maximum number of aircraft to return in report (default: 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdr_decode_packet",
            "description": "Demodulates and decodes digital RF packet transmissions (APRS on 144.390 MHz, POCSAG pager on 152/454/929 MHz, FLEX, or NOAA EAS) using rtl_fm piped to multimon-ng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency_mhz": {
                        "type": "number",
                        "description": "Carrier frequency in MHz (default: 144.390 for APRS, 152.0 for pager, 162.55 for NOAA EAS).",
                    },
                    "protocol": {
                        "type": "string",
                        "enum": ["aprs", "pocsag", "flex", "eas", "all"],
                        "description": "Packet protocol decoder ('aprs', 'pocsag', 'flex', 'eas', 'all', default: 'aprs').",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Listening duration in seconds (1-30, default: 10).",
                    },
                    "gain_db": {
                        "type": "number",
                        "description": "RF tuner gain in dB (default: 40.0 dB).",
                    },
                },
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 7. Time & Alarm Scheduling Schema ('time' bundle)
# -----------------------------------------------------------------------------

TIME_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current local system time, UTC time, timezone name/offset, and Unix timestamp.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task_alarm",
            "description": "Defers or schedules an existing task in the queue to wake up and execute at a specific future time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the existing task to schedule (e.g. 'task_123').",
                    },
                    "when": {
                        "type": "string",
                        "description": "When to run the task. Accepts relative delays ('30m', 'in 2 hours', '1h 30m') or absolute ISO times ('2026-10-13T14:30:00', '14:30').",
                    },
                },
                "required": ["task_id", "when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_scheduled_task",
            "description": "Creates a new task in the queue scheduled for a future wake-up execution time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the task.",
                    },
                    "when": {
                        "type": "string",
                        "description": "Target execution time ('30m', 'in 2 hours', '14:30', or ISO-8601).",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Detailed instructions/prompt for the task.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["system_1", "system_2"],
                        "description": "Execution tier ('system_1' or 'system_2'). Defaults to 'system_1'.",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Optional specific tool name to execute.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority (higher runs earlier once ready, default 5).",
                    },
                },
                "required": ["title", "when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_alarms",
            "description": "Lists all pending scheduled task alarms with remaining countdowns.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_alarm",
            "description": "Clears a scheduled alarm on a task, optionally making it immediately runnable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to clear schedule for.",
                    },
                    "run_now": {
                        "type": "boolean",
                        "description": "If True (default), makes the task immediately runnable.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 8. Offline API Documentation Schema ('docs' bundle)
# -----------------------------------------------------------------------------

DOCS_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "docs_list_libraries",
            "description": "Lists available offline API docsets (Dash/Zeal format) with total indexed symbol counts.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "docs_search",
            "description": "Searches offline API docset symbols (e.g. classes, functions, methods, modules).",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "Library name (e.g. 'python', 'pydantic', 'numpy', 'scipy', 'textual').",
                    },
                    "query": {
                        "type": "string",
                        "description": "Symbol name or search query.",
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Optional symbol type filter (e.g. 'Class', 'Function', 'Method', 'Module').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10).",
                    },
                },
                "required": ["library", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "docs_read_symbol",
            "description": "Extracts clean Markdown documentation and code signatures for a specific symbol or document path in an offline docset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "Library name.",
                    },
                    "symbol_or_path": {
                        "type": "string",
                        "description": "Exact symbol name or document path.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of documentation to extract (default 3500).",
                    },
                },
                "required": ["library", "symbol_or_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "docs_get_signature",
            "description": "Extracts the exact code signature and docstring summary for a library symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "Library name.",
                    },
                    "symbol_name": {
                        "type": "string",
                        "description": "Symbol name.",
                    },
                },
                "required": ["library", "symbol_name"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# 9. Personal Notes & Corpus Vault Schema ('vault' bundle)
# -----------------------------------------------------------------------------

VAULT_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vault_search",
            "description": "Executes high-speed BM25 ranked full-text search across local personal notes, markdown files, and past artifacts using SQLite FTS5.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms or phrases.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'notes', 'docs', 'specs', 'code').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_read_document",
            "description": "Reads the full or budgeted content of a local document from the vault by document ID or path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id_or_path": {
                        "type": "string",
                        "description": "Numeric document ID from vault_search or path substring.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to retrieve (default 3500).",
                    },
                },
                "required": ["doc_id_or_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_index_path",
            "description": "Recursively scans and indexes local markdown, text, or documentation files into the SQLite FTS5 vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to index.",
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": "File glob pattern (default: '*.md').",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category label (e.g. 'notes', 'specs', 'research').",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_status",
            "description": "Returns index statistics: document count, total indexed words, and category breakdown.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# Backward compatibility alias
CORE_TOOLS_SCHEMA = SHELF_TOOLS_SCHEMA

# -----------------------------------------------------------------------------
# Tool Shelf Registry
# -----------------------------------------------------------------------------

BUNDLE_ALIASES: dict[str, str] = {
    "wikipedia": "library",
    "wiki": "library",
    "zim": "library",
}

TOOL_BUNDLES: dict[str, dict[str, Any]] = {
    "core": {
        "name": "core",
        "title": "Tool Shelf Manager",
        "description": "Baseline tool shelf inspection, bundle discovery, and on-demand activation/deactivation.",
        "tools": SHELF_TOOLS_SCHEMA,
        "always_active": True,
    },
    "baton": {
        "name": "baton",
        "title": "Baton Queue & Task Operations",
        "description": "Queue mutation, task creation (add_task), queue stepping, task inspection, and failure recovery.",
        "tools": BATON_QUEUE_TOOLS_SCHEMA,
        "always_active": False,
    },
    "library": {
        "name": "library",
        "title": "Offline ZIM Archive Explorer & Multi-Corpus Reader",
        "description": "Offline Kiwix .zim archive explorer and reader. Dynamically searches and reads across any locally mounted .zim files on disk (Wikipedia, Python docs, WikiMed, etc.).",
        "tools": WIKIPEDIA_TOOLS_SCHEMA,
        "always_active": False,
    },
    "newsboat": {
        "name": "newsboat",
        "title": "Newsboat RSS Feed Ingestion",
        "description": "Offline RSS news monitoring, feed listing, article search, and full-text reading across 12+ tech/science feeds.",
        "tools": NEWSBOAT_TOOLS_SCHEMA,
        "always_active": False,
    },
    "sandbox": {
        "name": "sandbox",
        "title": "Sandboxed Python REPL & Verifier",
        "description": "Safe Python evaluation, AST syntax checking, file staging, and pytest execution in an isolated sandbox.",
        "tools": SANDBOX_TOOLS_SCHEMA,
        "always_active": False,
    },
    "sdr": {
        "name": "sdr",
        "title": "RTL-SDR RF Telemetry & Spectrum Scanner",
        "description": "Hardware control for RTL-SDR dongle: spectrum sweeps (rtl_power), RF signal capture (rtl_fm), sensor telemetry decoding (rtl_433), ADS-B aircraft radar (dump1090), and digital packet decoding (multimon-ng).",
        "tools": SDR_TOOLS_SCHEMA,
        "always_active": False,
    },
    "time": {
        "name": "time",
        "title": "Wall-Clock Time & Task Alarm Scheduler",
        "description": "Real-world UTC and local time, timezone information, and task alarm scheduling (deferring or waking up tasks at specific future times).",
        "tools": TIME_TOOLS_SCHEMA,
        "always_active": False,
    },
    "docs": {
        "name": "docs",
        "title": "Offline API & Technical Documentation (Dash/Zeal)",
        "description": "Offline symbol lookup, exact API method signatures, and HTML-to-markdown documentation extraction using local .docset folders.",
        "tools": DOCS_TOOLS_SCHEMA,
        "always_active": False,
    },
    "vault": {
        "name": "vault",
        "title": "Personal Knowledge & Local Notes Vault (SQLite FTS5)",
        "description": "High-speed BM25 ranked full-text search and incremental indexing across local markdown notes, past project artifacts, and documentation.",
        "tools": VAULT_TOOLS_SCHEMA,
        "always_active": False,
    },
}

# Full combined schema list for backwards compatibility
BATON_TOOLS_SCHEMA: list[dict[str, Any]] = (
    SHELF_TOOLS_SCHEMA
    + BATON_QUEUE_TOOLS_SCHEMA
    + WIKIPEDIA_TOOLS_SCHEMA
    + NEWSBOAT_TOOLS_SCHEMA
    + SANDBOX_TOOLS_SCHEMA
    + SDR_TOOLS_SCHEMA
    + TIME_TOOLS_SCHEMA
    + DOCS_TOOLS_SCHEMA
    + VAULT_TOOLS_SCHEMA
)

# -----------------------------------------------------------------------------
# Curated On-Demand Tool Cheatsheets (Accessed via tool_help)
# -----------------------------------------------------------------------------

CURATED_TOOL_CHEATSHEETS: dict[str, dict[str, Any]] = {
    "add_task": {
        "summary": "Adds a new task to the Baton priority queue with dependencies and execution tier.",
        "parameters": {
            "title": "(string, required) Short clear title for the task.",
            "tier": "(string, required) 'system_1' (fast recon/verify) or 'system_2' (deep synthesis). Aliases: system1, system2, sys1, sys2, 1, 2.",
            "prompt": "(string, required) Detailed requirements/prompt instructions. Aliases: description, instructions, details.",
            "depends_on": "(list[string], optional) Task IDs or task titles this task depends on. Aliases: dependencies.",
            "task_type": "(string, optional) 'fast_recon' | 'fast_verify' | 'deep_synthesis'.",
            "tool_name": "(string, optional) Recon tool to execute for System 1 tasks (e.g. 'wiki_read_section', 'docs_search').",
            "tool_params": "(object, optional) Parameters passed to the recon tool. Aliases: params, arguments.",
            "priority": "(integer, optional) Priority 1-10 (default: 5).",
        },
        "canonical_dag_example": [
            {
                "step": 1,
                "description": "Create System 1 recon task",
                "call": "add_task(title='Boids Algorithm Recon', tier='system_1', prompt='Extract flocking formulas for separation, alignment, cohesion', tool_name='wiki_read_section', tool_params={'title': 'Boids', 'section': 'Behavioral rules'})",
            },
            {
                "step": 2,
                "description": "Create System 2 synthesis task dependent on upstream recon (can use title or task_id)",
                "call": "add_task(title='Boids Simulation Script', tier='system_2', prompt='Write 20-boid Python simulation in 100x100 space with 5m sensing range based on upstream dossier', depends_on=['Boids Algorithm Recon'])",
            },
        ],
    },
    "activate_tool_bundle": {
        "summary": "Pulls a tool bundle from the shelf and activates it on your active toolbelt.",
        "parameters": {
            "bundle_name": "(string, required) Name of bundle to activate ('baton', 'library', 'newsboat', 'sandbox', 'sdr', 'time', 'docs', 'vault'). Aliases: tool_bundle, bundle.",
        },
        "example": "activate_tool_bundle(bundle_name='library')",
    },
    "deactivate_tool_bundle": {
        "summary": "Returns an active tool bundle to the shelf to free context tokens.",
        "parameters": {
            "bundle_name": "(string, required) Name of bundle to deactivate. Aliases: tool_bundle, bundle.",
        },
        "example": "deactivate_tool_bundle(bundle_name='library')",
    },
    "step_queue": {
        "summary": "Executes a single ready task from the priority queue and records artifacts.",
        "parameters": {
            "task_id": "(string, optional) Specific ready task_id to execute. If omitted, executes the highest priority ready task.",
        },
        "example": "step_queue()",
    },
    "start_orchestration": {
        "summary": "Deterministically executes runnable tasks in the queue (up to max_steps). Handles engine execution seamlessly in Python.",
        "parameters": {
            "max_steps": "(integer, optional) Maximum number of ready tasks to execute in sequence (default: 5).",
        },
        "example": "start_orchestration(max_steps=5)",
    },
    "list_tasks": {
        "summary": "Lists tasks in the queue with IDs, status, priority, and dependencies.",
        "parameters": {
            "status": "(string, optional) Filter by status ('ready', 'blocked', 'completed', 'failed', 'in_progress').",
            "limit": "(integer, optional) Maximum tasks to return (default: 20).",
        },
        "example": "list_tasks(status='ready')",
    },
    "tool_help": {
        "summary": "Provides documentation, accepted parameter aliases, usage instructions, and working examples for any tool.",
        "parameters": {
            "tool_name": "(string, required) Name of the tool to inspect (e.g. 'add_task', 'start_orchestration').",
        },
        "example": "tool_help(tool_name='add_task')",
    },
}

# -----------------------------------------------------------------------------
# System Prompt for System 1 Copilot with Tool Shelf Awareness
# -----------------------------------------------------------------------------

SYSTEM_COPILOT_PROMPT = """You are Baton System 1 Copilot: an interactive, agile pair-programming scout and task manager for the Baton Orchestrator.

Architecture:
- Baton orchestrates a Dual-Speed system:
  1. System 1 (You / Qwen 14B on Ollama @ ~28 tok/s): High-throughput recon, Wikipedia lookups, RSS news monitoring, verification, and task queue management.
  2. System 2 (DeepSeek-V4-Flash on Colibrì @ ~0.8 tok/s): Frontier 284B MoE model that runs deep macro-planning and overnight synthesis. System 2 is tool-blind and consumes briefing dossiers compiled by System 1.

TOOL SHELF ARCHITECTURE:
To prevent context window bloat and tool distraction, tools are organized into modular bundles on a 'shelf':
- 'core' (ACTIVE): Baseline shelf controls (list_tool_bundles, activate_tool_bundle, deactivate_tool_bundle, tool_help).
- 'baton' (ACTIVE in Chat): Baton queue management, task creation (add_task), queue stepping, start_orchestration, and artifact review.
- 'library' (ON SHELF): Offline Kiwix ZIM archive explorer and reader. Inspects, searches, and reads from any locally mounted .zim files on disk (general knowledge, Python documentation, medical references, etc.).
- 'newsboat' (ON SHELF): Offline RSS news monitoring across 12+ tech/science feeds (Ars Technica, Nature, NASA, MIT Tech Review, Krebs, etc.).
- 'sandbox' (ON SHELF): Isolated Python execution, AST syntax verification, and pytest test suite runner.
- 'sdr' (ON SHELF): RTL-SDR RF spectrum analysis, carrier detection, audio recording, and telemetry decoding.
- 'time' (ON SHELF): Real-world UTC and local system time, timezone grounding, and task alarm scheduling (deferring or waking up tasks in the future).
- 'docs' (ON SHELF): Offline API and technical documentation (Dash/Zeal .docset archives) for exact method signatures, classes, and docstrings.
- 'vault' (ON SHELF): Personal local notes and project artifact search powered by SQLite FTS5 with BM25 ranking.

If you need capabilities from an inactive bundle on the shelf, call `activate_tool_bundle(bundle_name="library" | "newsboat" | "sandbox" | "sdr" | "time" | "docs" | "vault")`.
Once activated, those tools immediately become available on your toolbelt for subsequent tool calls.
You can return a bundle to the shelf when finished using `deactivate_tool_bundle(bundle_name=...)` to free context tokens.

Your Roles & Capabilities:
1. Brainstorm with the user on project objectives, architecture, formulas, and required steps.
2. Query offline knowledge or news on demand to verify facts, equations, or news updates before drafting tasks.
3. Test code snippets or run syntax/pytest checks using the 'sandbox' bundle.
4. Help the user construct clean Directed Acyclic Graphs (DAGs) of tasks:
   - Create 'system_1' tasks for fact extraction, formulas, news digests, or verification.
   - Create 'system_2' tasks for deep mathematical derivations, multi-file code synthesis, or complete modules.
   - Set up dependencies using `depends_on` so System 2 tasks automatically receive upstream dossiers.
5. Start orchestration when ready: call `start_orchestration()` to execute queued tasks without waiting.

Guidelines:
- When the user requests task creation or project execution, call the appropriate tools immediately.
- The 'core' and 'baton' bundles are active on your toolbelt by default.
- If you are ever unsure of a tool's parameters or usage, call `tool_help(tool_name=...)`.
- Keep your conversational answers crisp, clear, and proactive.
- When adding tasks, ensure the prompt is thorough and specific with zero placeholders.
- Always be helpful, objective, and mathematically precise.
"""


class BatonChatSession:
    """Manages the interactive chat session, Ollama API communications, and tool dispatches."""

    def __init__(
        self,
        db: Optional[Database] = None,
        orchestrator: Optional[Orchestrator] = None,
        zim_tool: Optional[ZimTool] = None,
        newsboat_tool: Optional[NewsboatTool] = None,
        sandbox_tool: Optional[SandboxTool] = None,
        sdr_tool: Optional[SdrTool] = None,
        time_tool: Optional[TimeTool] = None,
        docs_tool: Optional[DocsTool] = None,
        vault_tool: Optional[VaultTool] = None,
        engine_mgr: Optional[EngineManager] = None,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        initial_bundles: Optional[set[str]] = None,
        logs_dir: Optional[Path] = None,
    ):
        self.db = db or Database()
        self.engine_mgr = engine_mgr or EngineManager()
        self.orchestrator = orchestrator or Orchestrator(db=self.db, engine_manager=self.engine_mgr)
        self._zim_tool = zim_tool
        self._newsboat_tool = newsboat_tool
        self._sandbox_tool = sandbox_tool
        self._sdr_tool = sdr_tool
        self._time_tool = time_tool
        self._docs_tool = docs_tool
        self._vault_tool = vault_tool
        self.ollama_url = (
            ollama_url
            or getattr(self.engine_mgr, "ollama_url", None)
            or os.getenv("BATON_OLLAMA_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("BATON_SYSTEM1_MODEL")
            or "qwen3:14b"
        )
        # Baton Chat has access to 'core' and 'baton' bundles by default.
        # Research agents can pass initial_bundles={'core', 'library'} to exclude queue tools.
        if initial_bundles is not None:
            self.active_bundles: set[str] = {BUNDLE_ALIASES.get(b.lower().strip(), b.lower().strip()) for b in initial_bundles}
        else:
            self.active_bundles = {"core", "baton"}
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_COPILOT_PROMPT}
        ]

        # Session Logging setup (logs to baton/logs/chat_session_<ts>.log and baton/logs/chat_latest.log)
        self.logs_dir = logs_dir if logs_dir is not None else Path("baton/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_path = self.logs_dir / f"chat_session_{ts_str}.log"
        self.latest_log_path = self.logs_dir / "chat_latest.log"

        header = (
            f"================================================================================\n"
            f"BATON CHAT SESSION INITIALIZED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model: {self.model_name} | Ollama URL: {self.ollama_url}\n"
            f"Active Bundles: {', '.join(sorted(self.active_bundles))}\n"
            f"Session Log: {self.session_log_path}\n"
            f"================================================================================\n"
        )
        try:
            self.session_log_path.write_text(header, encoding="utf-8")
            self.latest_log_path.write_text(header, encoding="utf-8")
        except Exception as e:
            logger.debug("Failed initializing chat log header: %s", e)

    def _log_event(self, event_type: str, payload: Any) -> None:
        """Appends a structured event entry to the session log and chat_latest.log."""
        if not self.logs_dir:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n[{timestamp}] [{event_type}]"]

        if isinstance(payload, str):
            lines.append(payload)
        elif isinstance(payload, (dict, list)):
            lines.append(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            lines.append(str(payload))

        entry_text = "\n".join(lines) + "\n"

        try:
            with open(self.session_log_path, "a", encoding="utf-8") as f:
                f.write(entry_text)
            with open(self.latest_log_path, "a", encoding="utf-8") as f:
                f.write(entry_text)
        except Exception as e:
            logger.debug("Failed writing to chat log: %s", e)

    @property
    def zim_tool(self) -> ZimTool:
        if self._zim_tool is None:
            try:
                self._zim_tool = ZimTool()
            except Exception as e:
                logger.warning("Could not initialize ZimTool: %s", e)
                self._zim_tool = None  # type: ignore[assignment]
        return self._zim_tool

    @property
    def newsboat_tool(self) -> NewsboatTool:
        if self._newsboat_tool is None:
            try:
                self._newsboat_tool = NewsboatTool()
            except Exception as e:
                logger.warning("Could not initialize NewsboatTool: %s", e)
                self._newsboat_tool = None  # type: ignore[assignment]
        return self._newsboat_tool

    @property
    def sandbox_tool(self) -> SandboxTool:
        if self._sandbox_tool is None:
            try:
                self._sandbox_tool = SandboxTool()
            except Exception as e:
                logger.warning("Could not initialize SandboxTool: %s", e)
                self._sandbox_tool = None  # type: ignore[assignment]
        return self._sandbox_tool

    @property
    def sdr_tool(self) -> SdrTool:
        if self._sdr_tool is None:
            try:
                self._sdr_tool = SdrTool()
            except Exception as e:
                logger.warning("Could not initialize SdrTool: %s", e)
                self._sdr_tool = None  # type: ignore[assignment]
        return self._sdr_tool

    @property
    def time_tool(self) -> TimeTool:
        if self._time_tool is None:
            try:
                self._time_tool = TimeTool(db=self.db)
            except Exception as e:
                logger.warning("Could not initialize TimeTool: %s", e)
                self._time_tool = None  # type: ignore[assignment]
        return self._time_tool

    @property
    def docs_tool(self) -> DocsTool:
        if self._docs_tool is None:
            try:
                self._docs_tool = DocsTool()
            except Exception as e:
                logger.warning("Could not initialize DocsTool: %s", e)
                self._docs_tool = None  # type: ignore[assignment]
        return self._docs_tool

    @property
    def vault_tool(self) -> VaultTool:
        if self._vault_tool is None:
            try:
                self._vault_tool = VaultTool()
            except Exception as e:
                logger.warning("Could not initialize VaultTool: %s", e)
                self._vault_tool = None  # type: ignore[assignment]
        return self._vault_tool

    def get_active_tools_schema(self) -> list[dict[str, Any]]:
        """Compiles the list of JSON tool schemas strictly for currently active bundles."""
        tools: list[dict[str, Any]] = []
        for bundle_key in self.active_bundles:
            bundle = TOOL_BUNDLES.get(bundle_key)
            if bundle:
                tools.extend(bundle["tools"])
        return tools

    # -------------------------------------------------------------------------
    # Tool Dispatcher
    # -------------------------------------------------------------------------

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Executes a named tool and returns a structured dictionary result."""
        name = name.strip()
        try:
            # Shelf Management Tools ('core')
            if name == "list_tool_bundles":
                shelf_items = []
                for b_name, b_info in TOOL_BUNDLES.items():
                    is_active = b_name in self.active_bundles
                    status_str = "ACTIVE" if is_active else "ON SHELF (Inactive)"
                    if b_info.get("always_active"):
                        status_str += " [Required]"
                    tool_names = [t["function"]["name"] for t in b_info["tools"]]
                    desc = b_info["description"]
                    if b_name == "library" and self.zim_tool:
                        corpora_list = self.zim_tool.list_corpora()
                        corpora_summary = ", ".join(c["corpus"] for c in corpora_list) or "none"
                        desc = f"Offline Kiwix .zim archive explorer ({len(corpora_list)} corpora currently mounted: {corpora_summary})."
                    shelf_items.append({
                        "bundle": b_name,
                        "title": b_info["title"],
                        "status": status_str,
                        "description": desc,
                        "tools": tool_names,
                    })
                return {"shelf": shelf_items, "active_bundles": list(self.active_bundles)}

            elif name == "tool_help":
                tool_name = (
                    args.get("tool_name")
                    or args.get("tool")
                    or args.get("name")
                    or ""
                ).strip().lower()
                if not tool_name:
                    return {
                        "error": "Missing required argument 'tool_name'. Usage: tool_help(tool_name='add_task')",
                        "curated_tools": list(CURATED_TOOL_CHEATSHEETS.keys()),
                    }

                # 1. Curated specific help / cheatsheet lookup
                if tool_name in CURATED_TOOL_CHEATSHEETS:
                    return {
                        "tool": tool_name,
                        "source": "curated_cheatsheet",
                        **CURATED_TOOL_CHEATSHEETS[tool_name],
                    }

                # 2. Dynamic docstring and JSON schema fallback
                schema_fn = None
                for item in BATON_TOOLS_SCHEMA:
                    fn = item.get("function", {})
                    if fn.get("name") == tool_name:
                        schema_fn = fn
                        break

                if schema_fn:
                    props = schema_fn.get("parameters", {}).get("properties", {})
                    param_details = {}
                    for p_name, p_info in props.items():
                        p_type = p_info.get("type", "any")
                        p_desc = p_info.get("description", "")
                        param_details[p_name] = f"({p_type}) {p_desc}"

                    # Also inspect Python method docstring if available
                    method_doc = ""
                    tool_method = getattr(self, tool_name, None)
                    if tool_method and tool_method.__doc__:
                        method_doc = tool_method.__doc__.strip()

                    res_help: dict[str, Any] = {
                        "tool": tool_name,
                        "source": "schema_fallback",
                        "description": schema_fn.get("description", ""),
                        "parameters": param_details,
                        "required": schema_fn.get("parameters", {}).get("required", []),
                    }
                    if method_doc:
                        res_help["docstring"] = method_doc
                    return res_help

                return {
                    "error": f"Tool '{tool_name}' not found.",
                    "active_tools": [t["function"]["name"] for t in self.get_active_tools_schema()],
                    "tip": "Use list_tool_bundles to view available bundles and their tools.",
                }

            elif name == "activate_tool_bundle":
                raw_name = (
                    args.get("bundle_name")
                    or args.get("tool_bundle")
                    or args.get("bundle")
                    or ""
                ).lower().strip()
                if not raw_name:
                    return {
                        "error": "Missing required argument 'bundle_name'. Usage: activate_tool_bundle(bundle_name='library')",
                        "available_bundles": list(TOOL_BUNDLES.keys()),
                    }
                bundle_name = BUNDLE_ALIASES.get(raw_name, raw_name)
                if bundle_name not in TOOL_BUNDLES:
                    return {
                        "error": f"Unknown bundle '{raw_name}'. Available bundles: {list(TOOL_BUNDLES.keys())}"
                    }
                self.active_bundles.add(bundle_name)
                activated_tools = [t["function"]["name"] for t in TOOL_BUNDLES[bundle_name]["tools"]]
                res: dict[str, Any] = {
                    "success": True,
                    "activated_bundle": bundle_name,
                    "message": f"Bundle '{bundle_name}' has been activated on your toolbelt.",
                    "available_tools": activated_tools,
                }

                # Dynamic status and discovered elements for active bundles
                if bundle_name == "library" and self.zim_tool:
                    corpora = [c["corpus"] for c in self.zim_tool.list_corpora()]
                    res["mounted_corpora"] = corpora
                    res["default_corpus"] = getattr(self.zim_tool, "default_corpus", "wikipedia")
                    res["corpora_count"] = len(corpora)
                elif bundle_name == "docs" and self.docs_tool:
                    libs = [l["library"] for l in self.docs_tool.list_libraries()]
                    res["available_libraries"] = libs
                    res["libraries_count"] = len(libs)
                elif bundle_name == "newsboat" and self.newsboat_tool:
                    feeds = [f["title"] for f in self.newsboat_tool.list_feeds()]
                    res["subscribed_feeds"] = feeds
                    res["feeds_count"] = len(feeds)
                elif bundle_name == "vault" and self.vault_tool:
                    st = self.vault_tool.get_status()
                    res["total_documents"] = st.get("documents_count", 0)
                    res["categories"] = list(st.get("categories", {}).keys())
                elif bundle_name == "time" and self.time_tool:
                    t_info = self.time_tool.get_current_time()
                    res["current_time"] = t_info.get("local_human", "")
                    res["timezone"] = t_info.get("timezone", "")
                elif bundle_name == "sdr" and self.sdr_tool:
                    dev = self.sdr_tool.get_device_info()
                    res["hardware_status"] = dev.get("status", "unknown")
                    res["tuner"] = dev.get("tuner", "none")
                elif bundle_name == "baton":
                    tasks = self.db.get_all_tasks()
                    res["total_tasks"] = len(tasks)
                    res["ready_tasks"] = len([t for t in tasks if t.status.value == "ready"])

                return res

            elif name == "deactivate_tool_bundle":
                raw_name = (
                    args.get("bundle_name")
                    or args.get("tool_bundle")
                    or args.get("bundle")
                    or ""
                ).lower().strip()
                if not raw_name:
                    return {
                        "error": "Missing required argument 'bundle_name'. Usage: deactivate_tool_bundle(bundle_name='...')",
                        "active_bundles": list(self.active_bundles),
                    }
                bundle_name = BUNDLE_ALIASES.get(raw_name, raw_name)
                if bundle_name == "core":
                    return {"error": "Cannot deactivate the 'core' bundle."}
                if bundle_name in self.active_bundles:
                    self.active_bundles.remove(bundle_name)
                    return {
                        "success": True,
                        "bundle": bundle_name,
                        "message": f"Bundle '{bundle_name}' has been returned to the shelf.",
                    }
                return {"error": f"Bundle '{bundle_name}' was not active."}

            # Baton Queue Tools ('baton')
            elif name == "list_tasks":
                status_filter = args.get("status")
                limit = args.get("limit", 20)
                tasks = self.db.get_all_tasks()
                if status_filter:
                    tasks = [t for t in tasks if t.status.value == status_filter]
                tasks = tasks[:limit]
                return {
                    "count": len(tasks),
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "title": t.title,
                            "tier": t.tier.value,
                            "status": t.status.value,
                            "priority": t.priority,
                            "depends_on": t.depends_on,
                            "elapsed_seconds": round(t.elapsed_seconds, 1),
                        }
                        for t in tasks
                    ],
                }

            elif name == "get_task":
                task_id = args["task_id"]
                task = self.db.get_task(task_id)
                if not task:
                    return {"error": f"Task '{task_id}' not found."}
                return {
                    "task_id": task.task_id,
                    "title": task.title,
                    "tier": task.tier.value,
                    "task_type": task.task_type.value,
                    "status": task.status.value,
                    "priority": task.priority,
                    "depends_on": task.depends_on,
                    "prompt": task.prompt_template,
                    "tool_name": task.tool_name,
                    "tool_params": task.tool_params,
                    "artifact_path": task.artifact_path,
                    "error_message": task.error_message,
                    "result_preview": (task.result_payload[:400] + "...") if task.result_payload else None,
                }

            elif name == "add_task":
                title = (args.get("title") or "").strip()
                if not title:
                    return {
                        "error": "Missing required parameter 'title'. Usage: add_task(title='...', tier='system_1', prompt='...')",
                        "tip": "Call tool_help(tool_name='add_task') for usage examples.",
                    }

                raw_tier = str(args.get("tier") or args.get("task_type") or "").strip().lower()
                clean_tier = re.sub(r"[^a-z0-9]", "", raw_tier)
                if clean_tier in ("system1", "sys1", "1"):
                    tier_str = "system_1"
                elif clean_tier in ("system2", "sys2", "2"):
                    tier_str = "system_2"
                else:
                    return {
                        "error": f"Invalid tier '{raw_tier}'. Valid tiers are 'system_1' (fast recon/verify) or 'system_2' (deep synthesis).",
                        "tip": "Call tool_help(tool_name='add_task') for usage examples.",
                    }

                prompt = (
                    args.get("prompt")
                    or args.get("description")
                    or args.get("instructions")
                    or args.get("details")
                    or ""
                )

                raw_deps = args.get("depends_on") or args.get("dependencies") or []
                if isinstance(raw_deps, str):
                    raw_deps = [raw_deps] if raw_deps.strip() else []

                # Fuzzy resolution of dependencies:
                # If dep is not an existing task_id, check if it matches an existing task title
                resolved_deps = []
                resolution_notes = []
                all_tasks = self.db.get_all_tasks()
                task_id_set = {t.task_id for t in all_tasks}
                title_map = {t.title.strip().lower(): t.task_id for t in all_tasks}

                for dep in raw_deps:
                    dep_str = str(dep).strip()
                    if not dep_str:
                        continue
                    if dep_str in task_id_set:
                        resolved_deps.append(dep_str)
                    elif dep_str.lower() in title_map:
                        matched_id = title_map[dep_str.lower()]
                        resolved_deps.append(matched_id)
                        resolution_notes.append({"from_title": dep_str, "to_id": matched_id})
                    else:
                        # Partial/contains match
                        matched = None
                        for t in all_tasks:
                            if dep_str.lower() in t.title.lower() or t.title.lower() in dep_str.lower():
                                matched = t.task_id
                                resolution_notes.append({"from_title": dep_str, "to_id": matched})
                                break
                        if matched:
                            resolved_deps.append(matched)
                        else:
                            resolved_deps.append(dep_str)

                task_type = args.get("task_type")
                tool_name = args.get("tool_name") or args.get("tool")
                tool_params = args.get("tool_params") or args.get("params") or args.get("arguments")
                priority = args.get("priority", 5)
                task_id = args.get("task_id")

                task = self.orchestrator.create_task(
                    title=title,
                    tier=tier_str,
                    task_type=task_type,
                    prompt_template=prompt,
                    tool_name=tool_name,
                    tool_params=tool_params,
                    depends_on=resolved_deps,
                    priority=priority,
                    task_id=task_id,
                )
                res_task: dict[str, Any] = {
                    "success": True,
                    "task_id": task.task_id,
                    "title": task.title,
                    "tier": task.tier.value,
                    "status": task.status.value,
                    "depends_on": task.depends_on,
                    "priority": task.priority,
                }
                if resolution_notes:
                    res_task["resolved_dependencies"] = resolution_notes
                return res_task

            elif name == "delete_task":
                task_id = args["task_id"]
                deleted = self.orchestrator.delete_task(task_id)
                return {"success": deleted, "task_id": task_id}

            elif name == "retry_failed_tasks":
                resets = self.db.retry_failed_tasks()
                return {
                    "reset_count": len(resets),
                    "reset_task_ids": [t.task_id for t in resets],
                }

            elif name == "list_artifacts":
                limit = args.get("limit", 15)
                artifacts = self.db.get_artifacts()[:limit]
                return {
                    "count": len(artifacts),
                    "artifacts": [
                        {
                            "artifact_id": a.artifact_id,
                            "task_id": a.task_id,
                            "title": a.title,
                            "type": a.artifact_type.value,
                            "file_path": a.file_path,
                            "preview": (a.content_preview[:200] + "...") if a.content_preview else None,
                        }
                        for a in artifacts
                    ],
                }

            elif name == "read_artifact":
                task_id = args.get("task_id")
                file_path = args.get("file_path")
                text = ""

                if task_id:
                    text = self.orchestrator.get_task_artifact_text(task_id) or ""
                    if not text:
                        task = self.db.get_task(task_id)
                        if task and task.result_payload:
                            text = task.result_payload

                if not text and file_path:
                    p = Path(file_path)
                    if p.exists() and p.is_file():
                        text = p.read_text(encoding="utf-8", errors="replace")

                if not text:
                    return {"error": f"No artifact text found for task_id={task_id} / file_path={file_path}"}

                max_chars = 3500
                is_truncated = len(text) > max_chars
                return {
                    "content": text[:max_chars],
                    "is_truncated": is_truncated,
                    "total_chars": len(text),
                }

            elif name == "queue_status":
                tasks = self.db.get_all_tasks()
                status_counts: dict[str, int] = {}
                for t in tasks:
                    status_counts[t.status.value] = status_counts.get(t.status.value, 0) + 1

                engine_status = self.engine_mgr.get_status()
                return {
                    "total_tasks": len(tasks),
                    "status_counts": status_counts,
                    "active_engine": engine_status.get("active_tier", "idle"),
                    "ollama_online": engine_status.get("ollama", {}).get("online", False),
                    "colibri_online": engine_status.get("colibri", {}).get("online", False),
                }

            elif name == "step_queue":
                task_id = args.get("task_id")
                completed_task = self.orchestrator.step(task_id=task_id)
                if completed_task:
                    return {
                        "success": True,
                        "task_id": completed_task.task_id,
                        "title": completed_task.title,
                        "status": completed_task.status.value,
                        "summary": completed_task.result_payload[:300] if completed_task.result_payload else None,
                        "artifact_path": completed_task.artifact_path,
                    }
                else:
                    return {
                        "success": False,
                        "message": "No runnable task found in READY state to execute.",
                    }

            elif name == "start_orchestration":
                max_steps = int(args.get("max_steps", 5))
                executed_tasks = []
                for step_idx in range(max_steps):
                    runnable = self.db.get_runnable_tasks()
                    if not runnable:
                        break
                    completed = self.orchestrator.step()
                    if completed:
                        executed_tasks.append({
                            "task_id": completed.task_id,
                            "title": completed.title,
                            "tier": completed.tier.value,
                            "status": completed.status.value,
                            "artifact_path": completed.artifact_path,
                        })
                    else:
                        break

                all_t = self.db.get_all_tasks()
                status_counts: dict[str, int] = {}
                for t in all_t:
                    status_counts[t.status.value] = status_counts.get(t.status.value, 0) + 1
                remaining_ready = status_counts.get("ready", 0)
                remaining_blocked = status_counts.get("blocked", 0)

                return {
                    "success": True,
                    "steps_executed": len(executed_tasks),
                    "executed_tasks": executed_tasks,
                    "queue_status": status_counts,
                    "remaining_ready": remaining_ready,
                    "remaining_blocked": remaining_blocked,
                    "message": (
                        f"Executed {len(executed_tasks)} task(s). {remaining_ready} ready, {remaining_blocked} blocked in queue."
                        if executed_tasks
                        else "No runnable tasks found in READY state."
                    ),
                }

            # Wikipedia Bundle Tools ('wikipedia')
            elif name == "wiki_search":
                if not self.zim_tool:
                    return {"error": "ZimTool is not available (no .zim file configured)."}
                return {"result": self.zim_tool.search(args["query"], limit=args.get("limit", 5))}

            elif name == "wiki_read_lead":
                if not self.zim_tool:
                    return {"error": "ZimTool is not available."}
                return {"result": self.zim_tool.read_lead(args["title"])}

            elif name == "wiki_read_section":
                if not self.zim_tool:
                    return {"error": "ZimTool is not available."}
                return {"result": self.zim_tool.read_section(args["title"], args["section"])}

            elif name == "wiki_infobox":
                if not self.zim_tool:
                    return {"error": "ZimTool is not available."}
                return {"result": self.zim_tool.get_infobox(args["title"])}

            elif name == "wiki_toc":
                if not self.zim_tool:
                    return {"error": "ZimTool is not available."}
                return {"result": self.zim_tool.get_toc(args["title"])}

            # Newsboat Bundle Tools ('newsboat')
            elif name == "rss_list_feeds":
                if not self.newsboat_tool:
                    return {"error": "NewsboatTool is not available (no cache.db found)."}
                return {"feeds": self.newsboat_tool.list_feeds()}

            elif name == "rss_latest":
                if not self.newsboat_tool:
                    return {"error": "NewsboatTool is not available."}
                return {
                    "articles": self.newsboat_tool.get_latest_articles(
                        feed_query=args.get("feed"),
                        unread_only=args.get("unread_only", False),
                        limit=args.get("limit", 10),
                    )
                }

            elif name == "rss_search":
                if not self.newsboat_tool:
                    return {"error": "NewsboatTool is not available."}
                return {
                    "results": self.newsboat_tool.search_articles(
                        args["query"],
                        feed_query=args.get("feed"),
                        limit=args.get("limit", 5),
                    )
                }

            elif name == "rss_read_article":
                if not self.newsboat_tool:
                    return {"error": "NewsboatTool is not available."}
                return self.newsboat_tool.read_article(
                    int(args["id"]), max_chars=args.get("max_chars", 3500)
                )

            # Sandbox Bundle Tools ('sandbox')
            elif name == "python_eval":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return self.sandbox_tool.eval_code(
                    code=args["code"],
                    filename=args.get("filename", "_snippet.py"),
                    timeout=args.get("timeout"),
                )

            elif name == "python_syntax_check":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return self.sandbox_tool.syntax_check(args["code"])

            elif name == "pytest_run":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return self.sandbox_tool.run_pytest(
                    target=args.get("target", "."),
                    timeout=args.get("timeout"),
                )

            elif name == "sandbox_write_file":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return self.sandbox_tool.write_file(args["filename"], args["content"])

            elif name == "sandbox_read_file":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return self.sandbox_tool.read_file(
                    args["filename"], max_chars=args.get("max_chars", 3500)
                )

            elif name == "sandbox_list_files":
                if not self.sandbox_tool:
                    return {"error": "SandboxTool is not available."}
                return {"files": self.sandbox_tool.list_files()}

            # SDR Bundle Tools ('sdr')
            elif name == "sdr_device_info":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                return self.sdr_tool.get_device_info(backend=args.get("backend", "auto"))

            elif name == "sdr_spectrum_scan":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                return self.sdr_tool.spectrum_scan(
                    start_freq_mhz=float(args["start_freq_mhz"]),
                    stop_freq_mhz=float(args["stop_freq_mhz"]),
                    bin_size_khz=float(args.get("bin_size_khz", 25.0)),
                    duration_seconds=int(args.get("duration_seconds", 5)),
                    gain_db=float(args.get("gain_db", 0.0)),
                    min_snr_db=float(args.get("min_snr_db", 2.0)),
                )

            elif name == "sdr_record_rf":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                return self.sdr_tool.record_rf(
                    frequency_mhz=float(args["frequency_mhz"]),
                    modulation=args.get("modulation", "fm"),
                    duration_seconds=int(args.get("duration_seconds", 5)),
                    output_filename=args.get("output_filename"),
                    gain_db=float(args.get("gain_db", 0.0)),
                )

            elif name == "sdr_decode_telemetry":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                freq = float(args["frequency_mhz"]) if "frequency_mhz" in args else None
                return self.sdr_tool.decode_telemetry(
                    frequency_mhz=freq,
                    duration_seconds=int(args.get("duration_seconds", 10)),
                    protocol=args.get("protocol"),
                    band_preset=args.get("band_preset"),
                    gain_db=float(args.get("gain_db", 40.0)),
                )

            elif name == "sdr_adsb_tracking":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                return self.sdr_tool.adsb_tracking(
                    duration_seconds=int(args.get("duration_seconds", 5)),
                    gain_db=float(args.get("gain_db", 45.0)),
                    max_aircraft=int(args.get("max_aircraft", 20)),
                )

            elif name == "sdr_decode_packet":
                if not self.sdr_tool:
                    return {"error": "SdrTool is not available."}
                return self.sdr_tool.decode_digital_packet(
                    frequency_mhz=float(args.get("frequency_mhz", 144.390)),
                    protocol=args.get("protocol", "aprs"),
                    duration_seconds=int(args.get("duration_seconds", 10)),
                    gain_db=float(args.get("gain_db", 40.0)),
                )

            # Time & Alarm Bundle Tools ('time')
            elif name in (
                "get_current_time",
                "schedule_task_alarm",
                "create_scheduled_task",
                "list_alarms",
                "cancel_alarm",
            ):
                if not self.time_tool:
                    return {"error": "TimeTool is not available."}
                return {"result": self.time_tool.execute(name, args)}

            # Docs Bundle Tools ('docs')
            elif name in (
                "docs_list_libraries",
                "docs_search",
                "docs_read_symbol",
                "docs_get_signature",
            ):
                if not self.docs_tool:
                    return {"error": "DocsTool is not available."}
                return {"result": self.docs_tool.execute(name, args)}

            # Vault Bundle Tools ('vault')
            elif name in (
                "vault_search",
                "vault_read_document",
                "vault_index_path",
                "vault_status",
            ):
                if not self.vault_tool:
                    return {"error": "VaultTool is not available."}
                return {"result": self.vault_tool.execute(name, args)}

            # Multi-ZIM Corpora listing
            elif name in ("zim_list_corpora", "list_corpora"):
                if not self.zim_tool:
                    return {"error": "ZimTool is not available."}
                return {"result": self.zim_tool.execute(name, args)}

            else:
                return {"error": f"Unknown tool name: '{name}'"}

        except Exception as e:
            logger.error("Error executing tool '%s': %s", name, e, exc_info=True)
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Conversational Loop with Native Function Calling
    # -------------------------------------------------------------------------

    def send_message(
        self,
        user_input: str,
        on_tool_call: Optional[Callable[[str, dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> str:
        """Sends user message to Ollama with active tools, dynamically updated by the shelf."""
        self._log_event("USER", user_input)
        self.messages.append({"role": "user", "content": user_input})

        max_turns = 10
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1
            # Retrieve currently active tools based on the shelf state
            active_tools = self.get_active_tools_schema()

            payload = {
                "model": self.model_name,
                "messages": self.messages,
                "tools": active_tools,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                },
            }

            try:
                res = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.ConnectionError:
                err_msg = (
                    "Could not connect to Ollama at http://127.0.0.1:11434. "
                    "Make sure Ollama is running (run '.\\scripts\\start_ollama.bat')."
                )
                self._log_event("ERROR", err_msg)
                return err_msg
            except Exception as e:
                err_msg = f"Error communicating with Ollama: {e}"
                self._log_event("ERROR", err_msg)
                return err_msg

            message = data.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking")
            tool_calls = message.get("tool_calls", [])

            # Extract model thinking if present
            if thinking:
                self._log_event("AGENT_THINKING", str(thinking).strip())
            elif content:
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    self._log_event("AGENT_THINKING", think_match.group(1).strip())

            # If no tools called, we have our final conversational response
            if not tool_calls:
                # Check for degenerate thinking-only turn (model thought but didn't speak or call tools)
                eval_count = data.get("eval_count", 0)
                if not content.strip() and (thinking or eval_count > 0) and turn_count < max_turns:
                    logger.info(
                        "Detected empty generation with thinking/eval_count=%s. Issuing internal nudge.",
                        eval_count,
                    )
                    self.messages.append({
                        "role": "user",
                        "content": "Please proceed by calling the required tool or answering the request.",
                    })
                    continue

                self.messages.append({"role": "assistant", "content": content})
                self._log_event("AGENT_RESPONSE", content)
                telemetry = {
                    "model": self.model_name,
                    "active_bundles": list(sorted(self.active_bundles)),
                    "total_duration_s": round(data.get("total_duration", 0) / 1e9, 3) if "total_duration" in data else None,
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                }
                self._log_event("TELEMETRY", telemetry)
                return content

            # Record assistant turn with tool calls
            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            # Execute all requested tool calls
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})

                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                self._log_event("TOOL_CALL", {"tool": fn_name, "arguments": args})

                if on_tool_call:
                    on_tool_call(fn_name, args)

                # Execute
                tool_res = self.execute_tool(fn_name, args)

                self._log_event("TOOL_RESULT", {"tool": fn_name, "result": tool_res})

                if on_tool_result:
                    on_tool_result(fn_name, tool_res)

                # Append tool result message
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": json.dumps(tool_res, ensure_ascii=False),
                }
                if "name" in fn:
                    tool_msg["name"] = fn["name"]
                if "id" in tc:
                    tool_msg["tool_call_id"] = tc["id"]

                self.messages.append(tool_msg)

        max_msg = "I completed the maximum number of tool executions for this turn. How would you like to proceed?"
        self._log_event("AGENT_RESPONSE", max_msg)
        return max_msg

    # -------------------------------------------------------------------------
    # Quick Direct Slash Command Handlers
    # -------------------------------------------------------------------------

    def handle_slash_command(self, cmd_line: str) -> bool:
        """Handles fast direct slash commands (<1ms feedback). Returns True if handled."""
        self._log_event("SLASH_COMMAND", cmd_line.strip())
        parts = cmd_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            console.print("[dim]Exiting Baton Chat. Have a productive session![/]")
            sys.exit(0)

        elif cmd == "/clear":
            console.clear()
            return True

        elif cmd in ("/help", "/h"):
            if arg:
                help_res = self.execute_tool("tool_help", {"tool_name": arg.strip()})
                if "error" in help_res:
                    console.print(f"[red]{help_res['error']}[/]")
                else:
                    console.print(f"[bold magenta]Tool Help:[/] [bold cyan]{help_res['tool']}[/] [dim]({help_res.get('source', '')})[/]")
                    if "summary" in help_res:
                        console.print(f"[white]{help_res['summary']}[/]\n")
                    elif "description" in help_res:
                        console.print(f"[white]{help_res['description']}[/]\n")
                    if "parameters" in help_res:
                        console.print("[bold]Parameters:[/]")
                        for p, d in help_res["parameters"].items():
                            console.print(f"  • [cyan]{p}[/]: {d}")
                    if "example" in help_res:
                        console.print(f"\n[bold]Example:[/] [green]{help_res['example']}[/]")
                    elif "canonical_dag_example" in help_res:
                        console.print("\n[bold]DAG Example:[/]")
                        for s in help_res["canonical_dag_example"]:
                            console.print(f"  Step {s['step']} [dim]({s.get('description', '')})[/]: [green]{s['call']}[/]")
                    if "docstring" in help_res:
                        console.print(f"\n[dim]{help_res['docstring']}[/]")
            else:
                self.print_help()
            return True

        elif cmd == "/toolhelp":
            if not arg:
                console.print("[yellow]Usage: /toolhelp <tool_name> (e.g. /toolhelp add_task)[/]")
            else:
                help_res = self.execute_tool("tool_help", {"tool_name": arg.strip()})
                console.print(json.dumps(help_res, indent=2))
            return True

        elif cmd in ("/start", "/run"):
            console.print("[cyan]Starting orchestration...[/]")
            steps = int(arg) if arg.isdigit() else 5
            res = self.execute_tool("start_orchestration", {"max_steps": steps})
            executed = res.get("executed_tasks", [])
            if executed:
                for t in executed:
                    console.print(f"[green]✓ Completed task:[/] [bold]{t['task_id']}[/] ({t['title']}) [{t['tier']}]")
                console.print(f"[dim]{res.get('message', '')}[/]")
            else:
                console.print(f"[yellow]{res.get('message', 'No runnable tasks in READY state.')}[/]")
            return True

        elif cmd in ("/shelf", "/bundles"):
            self.print_shelf()
            return True

        elif cmd == "/bundle":
            raw_target = arg.strip().lower()
            target = BUNDLE_ALIASES.get(raw_target, raw_target)
            if not target:
                self.print_shelf()
            elif target == "all":
                self.active_bundles = set(TOOL_BUNDLES.keys())
                console.print(f"[green]All tool bundles activated:[/] [bold]{', '.join(self.active_bundles)}[/]")
            elif target == "core":
                self.active_bundles = {"core"}
                console.print("[yellow]Reset active toolbelt to core only.[/]")
            elif target in TOOL_BUNDLES:
                if target in self.active_bundles:
                    if target != "core":
                        self.active_bundles.remove(target)
                        console.print(f"[yellow]Returned bundle '{target}' to shelf.[/]")
                    else:
                        console.print("[dim]Core bundle cannot be deactivated.[/]")
                else:
                    self.active_bundles.add(target)
                    console.print(f"[green]Activated bundle '{target}' from shelf.[/]")
            else:
                console.print(f"[red]Unknown bundle '{raw_target}'. Available: {list(TOOL_BUNDLES.keys())}[/]")
            return True

        elif cmd in ("/tasks", "/queue"):
            self.print_task_table(arg)
            return True

        elif cmd in ("/status", "/s"):
            self.print_status()
            return True

        elif cmd in ("/artifacts", "/a"):
            self.print_artifacts()
            return True

        elif cmd == "/step":
            console.print("[cyan]Stepping queue...[/]")
            task = self.orchestrator.step(task_id=arg if arg else None)
            if task:
                console.print(f"[green]✓ Completed task:[/] [bold]{task.task_id}[/] ({task.title})")
            else:
                console.print("[yellow]No runnable tasks in READY state.[/]")
            return True

        elif cmd == "/model":
            if arg:
                self.model_name = arg.strip()
                console.print(f"[green]Active model switched to:[/] [bold]{self.model_name}[/]")
            else:
                console.print(f"[cyan]Current model:[/] [bold]{self.model_name}[/]")
            return True

        elif cmd == "/sdr":
            if self.sdr_tool:
                console.print("[cyan]Probing RTL-SDR hardware...[/]")
                info = self.sdr_tool.get_device_info()
                status_color = "green" if info.get("status") == "connected" else ("yellow" if "needs_attach" in info.get("status", "") else "cyan")
                console.print(f"[{status_color} bold]RTL-SDR Status:[/] [{status_color}]{info.get('status')}[/] | [bold]Tuner:[/] {info.get('tuner', 'Unknown')}")
                if "message" in info:
                    console.print(f"[dim]{info['message']}[/]")
            else:
                console.print("[red]SdrTool is not available.[/]")
            return True

        elif cmd in ("/adsb", "/aircraft"):
            if self.sdr_tool:
                dur = int(arg) if arg.isdigit() else 5
                console.print(f"[cyan]Scanning 1090 MHz for ADS-B aircraft ({dur}s)...[/]")
                briefing = self.sdr_tool.execute("sdr_adsb_tracking", {"duration_seconds": dur})
                console.print(Markdown(briefing))
            else:
                console.print("[red]SdrTool is not available.[/]")
            return True

        elif cmd in ("/telemetry", "/meters"):
            if self.sdr_tool:
                preset = arg.strip() if arg else "us_smart_meters"
                console.print(f"[cyan]Listening for telemetry ({preset}, 10s)...[/]")
                briefing = self.sdr_tool.execute("sdr_decode_telemetry", {"band_preset": preset, "duration_seconds": 10})
                console.print(Markdown(briefing))
            else:
                console.print("[red]SdrTool is not available.[/]")
            return True

        elif cmd == "/time":
            if self.time_tool:
                console.print(Markdown(self.time_tool.format_time_markdown()))
            else:
                console.print("[red]TimeTool is not available.[/]")
            return True

        elif cmd in ("/alarms", "/alarm"):
            if self.time_tool:
                console.print(Markdown(self.time_tool.list_alarms()))
            else:
                console.print("[red]TimeTool is not available.[/]")
            return True

        elif cmd == "/docs":
            if not self.docs_tool:
                console.print("[red]DocsTool is not available.[/]")
            elif not rest:
                console.print(Markdown(self.docs_tool.format_libraries_markdown()))
            else:
                parts = rest.split(maxsplit=1)
                lib = parts[0]
                q = parts[1] if len(parts) > 1 else ""
                if not q:
                    console.print(f"[yellow]Usage: /docs <library> <query>[/]")
                else:
                    console.print(Markdown(self.docs_tool.format_search_markdown(library=lib, query=q)))
            return True

        elif cmd == "/vault":
            if not self.vault_tool:
                console.print("[red]VaultTool is not available.[/]")
            elif not rest:
                console.print(Markdown(self.vault_tool.format_status_markdown()))
            else:
                console.print(Markdown(self.vault_tool.format_search_markdown(query=rest)))
            return True

        elif cmd in ("/corpora", "/zim"):
            if not self.zim_tool:
                console.print("[red]ZimTool is not available.[/]")
            else:
                console.print(Markdown(self.zim_tool.format_corpora_markdown()))
            return True

        return False

    # -------------------------------------------------------------------------
    # Visual Printers
    # -------------------------------------------------------------------------

    def print_welcome(self) -> None:
        active_str = ", ".join(self.active_bundles)
        console.print(
            Panel.fit(
                f"[bold cyan]🏏 Baton System 1 Copilot[/] [dim]({self.model_name})[/]\n"
                "[dim]Zero-lag conversational task planner, offline knowledge library & news scout.[/]\n"
                f"[dim]Active Toolbelt:[/] [bold green]{active_str}[/]  |  [dim]Shelf:[/] [yellow]/shelf[/], [yellow]/bundle <name>[/]\n"
                f"[dim]Live Session Log:[/] [bold cyan]{self.latest_log_path}[/] [dim]({self.session_log_path.name})[/]\n"
                "[dim]Commands:[/] [bold yellow]/tasks[/], [bold yellow]/status[/], [bold yellow]/artifacts[/], [bold yellow]/step[/], [bold yellow]/help[/], [bold yellow]/exit[/]",
                border_style="cyan",
            )
        )

    def print_shelf(self) -> None:
        table = Table(title="Baton Tool Shelf", show_header=True, header_style="bold magenta")
        table.add_column("Bundle", style="cyan bold")
        table.add_column("Status")
        table.add_column("Description", style="white")
        table.add_column("Included Tools", style="dim")

        for b_name, b_info in TOOL_BUNDLES.items():
            is_active = b_name in self.active_bundles
            status = "[green bold]● ACTIVE[/]" if is_active else "[dim]○ ON SHELF[/]"
            if b_info.get("always_active"):
                status += " [dim](core)[/]"
            tools = ", ".join(t["function"]["name"] for t in b_info["tools"])
            table.add_row(b_name, status, b_info["description"], tools)

        console.print(table)
        console.print("[dim]Use [bold yellow]/bundle <name>[/] to toggle a bundle onto your active toolbelt.[/]\n")

    def print_help(self) -> None:
        table = Table(title="Baton Chat Commands & Tools", show_header=True, header_style="bold magenta")
        table.add_column("Command / Capability", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("/shelf or /bundles", "Inspect tool shelf and active toolbelt bundles")
        table.add_row("/bundle <name>", "Activate/deactivate bundle (e.g. /bundle baton, /bundle library, /bundle sandbox, /bundle newsboat, /bundle docs, /bundle vault, /bundle all)")
        table.add_row("/tasks [status]", "Instantly view task table (filter by ready, blocked, completed, failed)")
        table.add_row("/status", "Show engine statuses (Ollama, Colibrì) and queue counts")
        table.add_row("/artifacts", "List generated briefing dossiers and synthesis artifacts")
        table.add_row("/step [task_id]", "Step the queue and execute highest priority ready task")
        table.add_row("/start or /run [max_steps]", "Start or resume autonomous orchestration of ready tasks (default: 5 steps)")
        table.add_row("/help [tool_name] or /toolhelp <name>", "Inspect parameters, accepted aliases, and examples for any tool")
        table.add_row("/model <name>", f"Change active Ollama model (current: {self.model_name})")
        table.add_row("/sdr", "Probe RTL-SDR hardware and tuner status")
        table.add_row("/adsb [seconds]", "Scan Mode-S ADS-B 1090 MHz aircraft radar")
        table.add_row("/telemetry [preset]", "Listen for sensor telemetry (us_smart_meters, us_tpms, eu_sensors)")
        table.add_row("/time", "Display current system time (local & UTC)")
        table.add_row("/alarms", "List scheduled task alarms and countdowns")
        table.add_row("/corpora or /zim", "List all mounted offline ZIM archives and loaded corpora")
        table.add_row("/docs [library] [query]", "Search offline API documentation (Dash/Zeal docsets)")
        table.add_row("/vault [query]", "Search personal notes & local corpus vault (SQLite FTS5)")
        table.add_row("/clear", "Clear terminal screen")
        table.add_row("/exit or /quit", "Leave chat mode")
        table.add_row("[dim]Autonomous Shelf[/]", "[dim]Agent automatically calls activate_tool_bundle('baton'|'library'|'sandbox'|'newsboat'|'sdr'|'time'|'docs'|'vault') on demand.[/]")
        console.print(table)

    def print_task_table(self, filter_status: str = "") -> None:
        tasks = self.db.get_all_tasks()
        if filter_status:
            tasks = [t for t in tasks if t.status.value.lower() == filter_status.lower().strip()]

        if not tasks:
            console.print("[yellow]No tasks found matching criteria.[/]")
            return

        table = Table(title="Baton Task Queue", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bold")
        table.add_column("Tier", style="dim")
        table.add_column("Title")
        table.add_column("Status")
        table.add_column("Deps", style="dim")
        table.add_column("Pri", justify="right")
        table.add_column("Time", justify="right")

        for t in tasks:
            status_style = {
                TaskStatus.READY: "green bold",
                TaskStatus.IN_PROGRESS: "cyan bold",
                TaskStatus.BLOCKED: "yellow",
                TaskStatus.COMPLETED: "blue",
                TaskStatus.FAILED: "red bold",
            }.get(t.status, "white")

            tier_badge = "S1" if t.tier == Tier.SYSTEM_1 else "S2"
            deps_str = ", ".join(t.depends_on) if t.depends_on else "-"
            table.add_row(
                t.task_id,
                tier_badge,
                t.title,
                f"[{status_style}]{t.status.value.upper()}[/]",
                deps_str,
                str(t.priority),
                f"{t.elapsed_seconds:.1f}s" if t.elapsed_seconds > 0 else "-",
            )
        console.print(table)

    def print_status(self) -> None:
        status = self.execute_tool("queue_status", {})
        counts = status.get("status_counts", {})
        console.print(f"[bold]Total Tasks:[/] {status.get('total_tasks', 0)}")
        console.print(
            f"[green]READY:[/] {counts.get('ready', 0)} | "
            f"[yellow]BLOCKED:[/] {counts.get('blocked', 0)} | "
            f"[blue]COMPLETED:[/] {counts.get('completed', 0)} | "
            f"[red]FAILED:[/] {counts.get('failed', 0)}"
        )
        console.print(
            f"[cyan]Ollama:[/] {'[green]ONLINE[/]' if status.get('ollama_online') else '[red]OFFLINE[/]'} | "
            f"[cyan]Colibrì:[/] {'[green]ONLINE[/]' if status.get('colibri_online') else '[red]OFFLINE[/]'}"
        )
        active_str = ", ".join(self.active_bundles)
        console.print(f"[magenta]Active Tool Bundles:[/] [bold]{active_str}[/]")

    def print_artifacts(self) -> None:
        arts = self.db.get_artifacts()
        if not arts:
            console.print("[dim]No artifacts generated yet.[/]")
            return

        table = Table(title="Baton Artifacts", show_header=True, header_style="bold green")
        table.add_column("Type", style="dim")
        table.add_column("Title", style="bold")
        table.add_column("File Path")
        table.add_column("Date", style="dim")

        for a in arts:
            table.add_row(
                a.artifact_type.value,
                a.title,
                a.file_path,
                str(a.created_at).split(".")[0] if a.created_at else "-",
            )
        console.print(table)

    # -------------------------------------------------------------------------
    # Main Interactive REPL
    # -------------------------------------------------------------------------

    def run_repl(self) -> None:
        """Runs the interactive zero-latency readline chat loop."""
        # Ensure Ollama is online if possible
        if not self.engine_mgr.is_ollama_online():
            console.print("[yellow]Ollama is currently offline. Starting Ollama daemon...[/]")
            started = self.engine_mgr.start_ollama()
            if not started:
                console.print(
                    "[bold red]Warning: Could not connect to Ollama. "
                    "Make sure Ollama is running at http://127.0.0.1:11434[/]"
                )

        self.print_welcome()
        session: PromptSession = PromptSession(history=InMemoryHistory())

        def _on_tool_call(fn: str, args: dict[str, Any]) -> None:
            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
            console.print(f"[dim]🔧 Calling tool:[/] [bold cyan]{fn}[/]([dim]{args_str}[/])")

        def _on_tool_result(fn: str, res: dict[str, Any]) -> None:
            if "error" in res:
                console.print(f"[red]  ✕ Tool Error:[/] {res['error']}")
            elif fn == "activate_tool_bundle" and res.get("success"):
                console.print(f"[green]  ★ Tool Shelf:[/] Activated bundle '[bold]{res.get('activated_bundle')}[/]'")
            elif fn == "deactivate_tool_bundle" and res.get("success"):
                console.print(f"[yellow]  ★ Tool Shelf:[/] Returned bundle '[bold]{res.get('bundle')}[/]' to shelf")
            elif "success" in res and res.get("task_id"):
                console.print(f"[green]  ✓ Task:[/] [bold]{res['task_id']}[/] ({res.get('status', 'OK')})")
            elif "count" in res:
                console.print(f"[dim]  ✓ Result: {res['count']} items returned[/]")
            elif "feeds" in res:
                console.print(f"[dim]  ✓ Result: {len(res['feeds'])} feeds listed[/]")
            elif "articles" in res:
                console.print(f"[dim]  ✓ Result: {len(res['articles'])} articles retrieved[/]")
            elif "results" in res:
                console.print(f"[dim]  ✓ Result: {len(res['results'])} matches found[/]")
            else:
                preview = str(res)[:100].replace("\n", " ")
                console.print(f"[dim]  ✓ Result:[/] [white]{preview}...[/]")

        while True:
            try:
                user_input = session.prompt(HTML("<b><ansicyan>You</ansicyan></b> <ansibrightblack>›</ansibrightblack> ")).strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if self.handle_slash_command(user_input):
                        continue

                console.print(f"[bold cyan]Copilot ({self.model_name})[/] [dim]thinking...[/]")
                response = self.send_message(
                    user_input,
                    on_tool_call=_on_tool_call,
                    on_tool_result=_on_tool_result,
                )

                console.print()
                console.print(Markdown(response))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Use /exit or Ctrl+D to quit.[/]")
            except EOFError:
                console.print("\n[dim]Exiting Baton Chat...[/]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/] {e}")


def main() -> None:
    session = BatonChatSession()
    session.run_repl()


if __name__ == "__main__":
    main()
