# Baton: Dual-Speed System 1 / System 2 Orchestrator

> **Baton** conducts an asymmetric orchestra of local models—acting as both the maestro's baton coordinating tiers and the relay baton passed between runners.

---

## 1. Architectural Overview

Baton resolves the hardware and cognitive trade-off between fast local copilots and frontier MoE models on a single consumer desktop (Intel i5-13500, 128 GB RAM, NVIDIA RTX 3060 12GB):

* **System 1 (Fast Recon Scout & Verifier):** Powered by **Qwen3:14B** via `llama-server` router (~28–33 tok/s). Rapidly navigates offline knowledge databases (Kiwix Wikipedia ZIM) with autonomous typo/miss self-recovery, and synthesizes multi-megabyte datasets into **300-word structured briefing dossiers**.
* **System 2 (Deep Macro-Planner & Overnight Synthesizer):** Powered by **Ornith-1.5-35B-A3B MoE** via `llama-server` (~15.5 tok/s). Employs decisive `<think>` reasoning protocols, automatic reasoning code salvage fallbacks, and multi-turn continuation recovery. Tool-blind to protect stability and context budgets.
* **The Deterministic Mediator:** Pure Python SQLite-backed priority queue (`baton.db`). Manages DAG state transitions (`blocked` $\rightarrow$ `ready` $\rightarrow$ `in_progress` $\rightarrow$ `completed`), bounds context budgets to $\le$ 1,200 tokens, and records production artifacts.
* **Multi-Model Router & VRAM Envelope:** Single `llama-server` instance on port 8080 dynamically pages models in/out of the RTX 3060 12GB VRAM using Q8_0 KV quantization and single-slot allocations, preventing PCIe paging stalls. Model weights remain memory-mapped in 128GB host RAM.

---

## 2. Quickstart & Usage

### A. Launch Interactive Textual TUI Dashboard
```powershell
.\baton.cmd
# or
.\baton.cmd tui
```
**Interactive Textual Capabilities:**
* **Arrow Keys (`Up` / `Down`):** Move through the Task Priority Queue table.
* **Smart Inspector Window:**
  * If a task is **in progress**: Displays live thinking monologue (`reasoning_content`) and drafting code with real-time tokens/sec throughput and duration.
  * If a task is **completed**: Automatically renders the full markdown briefing dossier or generated code with scroll support. Displays warning badges if the output was truncated by context limit.
  * If a task is **pending/blocked**: Displays the full breakdown of the prompt, parameters, tool configuration, and dependencies.
* **`Enter`:** Runs the selected task (if `READY`) or steps the queue.
* **`n`:** Opens the **New Task Modal** to manually create tasks, select dependencies, and embed upstream artifacts into the prompt.
* **`v`:** **Verify Task**: Queues an automated System 1 verification pass on the selected task (stages upstream code in sandbox, runs AST check & pytest).
* **`Space`:** Opens the **Edit Task Modal** to modify title, instructions, tool names, parameters, priority, dependencies, or embed artifacts.
* **`s` / `S`:** **Stop Task**: Immediately interrupts and stops the active running generation without having to kill the server process.
* **`d`:** Deletes the selected task from the queue.
* **`r`:** Retries and resets failed tasks back to `READY` (if dependencies are met) or `BLOCKED`.
* **`t`:** Toggles **Auto-Tick Mode** on/off (automatically steps the queue every 2 seconds while you watch!).
* **`q`:** Quits the dashboard safely while preserving database state.

### B. Launch System 1 Chat Copilot (Zero-Lag Readline REPL)
```powershell
.\baton.cmd chat
# or with code-focused model:
.\baton.cmd chat --coder
# or research-only mode (excludes baton queue tools from context):
.\baton.cmd chat --research
```
**Interactive Chat Capabilities:**
* **Instant, Zero-Latency Terminal Input:** Powered by `prompt_toolkit` with full line-editing, history, and smooth pasting without TUI redraw latency.
* **Modular 'Tool Shelf' Architecture:** Tools are compartmentalized into distinct bundles to prevent context bloat and eliminate hallucinated tool calls on 14B models:
    - `'core'` (Always active): Baseline tool shelf manager (`list_tool_bundles`, `activate_tool_bundle`, `deactivate_tool_bundle`, `tool_help`).
    - `'baton'` (Active in Chat by default, on shelf for research agents): Queue mutation and monitoring (`add_task`, `delete_task`, `retry_failed_tasks`, `queue_status`, `step_queue`, `start_orchestration`, `list_tasks`, `get_task`, `list_artifacts`, `read_artifact`).
    - `'library'` (On shelf, alias: `'wikipedia'`): Offline multi-corpus ZIM archive explorer and reader (`zim_list_corpora`, `wiki_search`, `wiki_read_lead`, `wiki_read_section`, `wiki_infobox`, `wiki_toc`).
    - `'rss'` (On shelf, alias: `'newsboat'`): Offline RSS news monitoring via SQLite cache with publication time filtering (`rss_list_feeds`, `rss_latest`, `rss_search`, `rss_read_article`).
    - `'sandbox'` (On shelf): Isolated Python evaluation, AST syntax checking, file staging, and pytest test suite runner (`python_eval`, `python_syntax_check`, `pytest_run`, `sandbox_write_file`, `sandbox_read_file`, `sandbox_list_files`).
    - `'sdr'` (On shelf): RTL-SDR RF spectrum analysis, carrier detection, audio demodulation, and digital sensor telemetry decoding (`sdr_device_info`, `sdr_spectrum_scan`, `sdr_record_rf`, `sdr_decode_telemetry`, `sdr_adsb_tracking`, `sdr_decode_packet`).
    - `'time'` (On shelf): Real-world UTC and local time grounding, task alarm scheduling, and ephemeral 24h post-it scratchpad notes with Mercy 1-Hour rules and character caps (`get_current_time`, `schedule_task_alarm`, `create_scheduled_task`, `list_alarms`, `cancel_alarm`, `save_note`, `append_note`, `read_note`, `list_notes`, `dismiss_note`, `keepalive_note`).
    - `'docs'` (On shelf): Offline Dash/Zeal API docset symbol search, exact method signatures, and HTML-to-markdown documentation (`docs_list_libraries`, `docs_search`, `docs_read_symbol`, `docs_get_signature`).
    - `'vault'` (On shelf): High-speed personal notes & local artifact BM25 full-text search (`vault_search`, `vault_read_document`, `vault_index_path`, `vault_status`).
* **Research Agent Isolation:** Specialized System 1 recon agents (and `chat --research`) can be initialized without the `'baton'` bundle. This ensures research agents focus exclusively on factual recon without wasting token context or risking accidental queue modification.
* **Instant Direct Slash Commands (<1ms feedback):**
  * `/shelf` or `/bundles`: Inspect the tool shelf and active toolbelt bundles.
  * `/bundle <name>`: Toggle a bundle on/off the active toolbelt (e.g. `/bundle baton`, `/bundle library`, `/bundle rss`, `/bundle sandbox`, `/bundle sdr`, `/bundle time`, `/bundle docs`, `/bundle vault`, `/bundle all`).
  * `/notes`: View active ephemeral scratchpad post-it notes currently on desk with remaining time.
  * `/note <id>`: Read full text of an ephemeral scratchpad note.
  * `/dismiss <id>`: Clear an ephemeral scratchpad note off the desk immediately.
  * `/time`: Display current system time (local & UTC).
  * `/alarms`: List scheduled task alarms and countdowns.
  * `/sdr`: Probe connected RTL-SDR hardware and tuner diagnostics instantly.
  * `/tasks [status]`: View the current task priority table instantly without LLM latency.
  * `/status`: Display queue counts and engine readiness (llama-server router).
  * `/artifacts`: List produced dossiers and synthesis code.
  * `/step [task_id]`: Step and execute highest-priority ready task.
  * `/start [steps]`: Deterministically execute runnable tasks in sequence (default: 5 steps).
  * `/model <name>`: Switch active System 1 model (e.g. `/model qwen3`).
  * `/clear`: Clear terminal screen.
  * `/exit` or `/quit`: Exit chat.

### C. CLI Commands
```powershell
# 1. View queue and engine status table
.\baton.cmd status

# 2. Step the queue (execute highest-priority runnable task)
.\baton.cmd step

# 3. Stop active running task
.\baton.cmd stop

# 4. Run all ready tasks until the queue is idle
.\baton.cmd run --max-steps 20

# 5. Manually create a new task
.\baton.cmd new --title "Estimate Earth-Sun Distance" --tier system_2 --prompt "Write a Python module..."

# 6. Queue or execute an automated verification pass on an upstream task
.\baton.cmd verify task_calc_123 --run-now

# 7. Retry / reset failed tasks
.\baton.cmd retry

# 8. Submit a new project goal
.\baton.cmd submit --project "Solar Ephemeris" --goal "Extract orbital formulas and synthesize a vectorized Python predictor."

# 9. List generated artifacts (dossiers, reports, and code)
.\baton.cmd artifacts
```

---

## 3. Resilience & Long-Run Safeguards

* **Autonomous Recon Self-Recovery Loop:** System 1 automatically catches tool lookup misses, empty results, or typos (`_is_recon_miss`), queries ZIM suggestion indexes, and uses structured JSON self-repair to diagnose and re-dispatch the corrected tool and parameters without aborting the task.
* **System 2 Decisive Reasoning & Code Salvage Fallback:** Prompts instruct decisive planning in `<think>` (<1000 tokens) and immediate conclusion with `</think>`. If generation hits token limits before closing the reasoning block, the engine automatically extracts and salvages code blocks from the `reasoning_content` monologue.
* **Automated Downstream Verification Draining:** `start_orchestration` checks for downstream `FAST_VERIFY` tasks queued immediately after System 2 synthesis and executes them in the same run, preventing verification passes from being stranded even when `max_steps=1`.
* **Empty File Protection:** `read_artifact` detects 0-byte output files and alerts the user/agent, linking directly to `_thinking.md` when reasoning was captured.
* **Global Concurrency Exclusion & Cross-Process Locking:** Guarantees single-task execution across independent processes (TUI Auto-Tick, Chat Copilot `start_orchestration`, CLI). Prevents multiple tasks from entering `IN_PROGRESS` simultaneously and colliding on single-GPU inference backends.
* **HTTP 429 Exponential Backoff:** Wraps System 2 generation requests with automatic backoff (up to 3 retries over 3s, 6s, 12s) to gracefully absorb brief engine saturation or prompt cache evictions without task failure.
* **Inter-Step Orchestration Cooldown:** Enforces a 1.5s inter-task delay during `start_orchestration` batch runs to allow local LLM servers to release sockets, clear attention caches, and reset VRAM slots cleanly.
* **Degenerate Repetition Detection:** Detects infinite token loops (e.g. `00000...` or periodic repeating cycles) in the stream tail and aborts cleanly to preserve valid code.
* **Mid-Stream Task Cancellation:** Non-destructive cancellation (`cancel_active()`) via `<kbd>s</kbd>` or `.\baton.cmd stop` without having to terminate inference servers.
* **Context Ceiling Detection & Seamless Continuation:** Automatically detects when generation reaches the token budget (`finish_reason: "length"`), using trailing unclosed assistant messages (`resolve_generation_prompt`) or continuation recovery prompts to continue generating across turns without loss of context.

---

## 4. Directory Structure

```
baton/
├── __init__.py
├── models.py          # Pydantic data schemas (Job, Task, Tier, TaskStatus, DAGPlan, Artifact, TaskProgress)
├── db.py              # Deterministic SQLite DAG engine & priority queue (baton.db)
├── engine_manager.py  # Mode A mutual-exclusion supervisor for Ollama and Colibrì
├── system1.py         # Qwen3:14B recon runner with 300-word briefing dossier compiler
├── system2.py         # DeepSeek-V4 macro-planner, streaming deep synthesizer, and continuation engine
├── orchestrator.py    # Priority queue runner, cancellation coordinator, and relay loop
├── chat.py            # Zero-lag prompt_toolkit REPL with native Ollama function calling
├── tui_textual.py     # Reactive Textual TUI dashboard with live inspector and modals
├── cli.py             # CLI command dispatcher
├── tools/
│   ├── __init__.py
│   ├── zim.py         # Offline Kiwix ZIM multi-archive reader (Wikipedia, Python docs, etc.)
│   ├── newsboat.py    # Offline RSS news query engine with publication time filtering
│   ├── sandbox.py     # Sandboxed Python REPL & verification engine (eval, syntax check, pytest)
│   ├── sdr.py         # RTL-SDR RF spectrum analysis, carrier detection, audio demod, and telemetry
│   ├── time_tool.py   # Wall-clock time grounding, task alarms, and ephemeral scratchpad note system
│   ├── docs_tool.py   # Offline Dash/Zeal API docset symbol indexer & documentation reader
│   └── vault.py       # Personal local notes & artifact BM25 full-text search (SQLite FTS5)
├── data/
│   └── rss/           # Offline Newsboat SQLite cache (cache.db) & feed configuration (urls)
├── artifacts/
│   ├── dossiers/      # System 1 structured markdown briefing & verification dossiers
│   ├── synthesis/     # System 2 generated code, specifications, and proofs
│   ├── sandbox/       # Ephemeral isolated sandbox workspace for code execution
│   ├── sdr/           # Captured RF demodulations, audio snippets, and spectral sweeps
│   └── progress/      # Live streaming atomic JSON task progress snapshots
└── tests/
    ├── test_chat.py         # System 1 Copilot REPL, tool shelf, & function calling unit tests
    ├── test_newsboat.py     # Newsboat RSS offline query engine & time filter unit tests
    ├── test_sandbox.py      # Sandboxed Python execution, AST check, & pytest unit tests
    ├── test_sdr.py          # RTL-SDR spectrum sweep, demodulation, & telemetry unit tests
    ├── test_time.py         # Time grounding, alarm scheduler, & ephemeral notes unit tests
    ├── test_vault.py        # SQLite FTS5 BM25 search & document retrieval unit tests
    ├── test_zim.py          # Multi-ZIM corpus routing & metadata sanitation unit tests
    ├── test_db.py           # SQLite and DAG dependency tests
    ├── test_orchestrator.py # End-to-end relay loop and multi-tier pipeline unit tests
    ├── test_progress.py     # Live SSE streaming, repetition detection, cancellation, & continuation tests
    ├── test_system2_retry.py# System 2 Colibrì HTTP 429 backoff & retry recovery tests
    └── test_tui.py          # Textual TUI mounting, modals, and actions tests
```

---

## 5. Running Unit Tests

Run the complete Baton test suite:
```powershell
pytest baton/tests/ -v
# or
python -m pytest baton/tests/ -v
```
