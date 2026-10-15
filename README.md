# Baton

**Baton** is an autonomous, dual-speed local LLM orchestrator and pair-programming copilot designed to coordinate high-throughput agile scouts (System 1) and deep nocturnal synthesis models (System 2) under single-GPU resource constraints.

## Architecture & Modes

- **System 1 (Agile Scout / Copilot):** Fast reasoning and tool-calling model (e.g. Qwen3:14B via Ollama @ ~28 tok/s) equipped with modular tool bundles:
  - `core`: Tool Shelf discovery, on-demand bundle activation, and `tool_help`.
  - `baton`: Deterministic priority queue, DAG task creation (`add_task`), dependency wiring (`set_task_dependencies`), orchestration (`start_orchestration`), and artifact inspection.
  - `library`: Offline multi-corpus ZIM archive explorer and reader (Wikipedia, Python docs, WikiMed, etc.). Alias: `wikipedia`.
  - `rss`: Offline RSS news monitoring and reader across 12+ tech/science publications with time filtering (`since`). Alias: `newsboat`.
  - `sandbox`: Isolated Python AST verification, REPL evaluation, and pytest test suite execution.
  - `sdr`: RTL-SDR spectrum sweeps (`rtl_power`), carrier detection, audio recordings, and RF telemetry decoding.
  - `time`: Real-world UTC/local time grounding, timezone offsets, task alarm scheduling, and ephemeral 24h post-it scratchpad notes with Mercy 1-Hour rules and character caps (`save_note`, `append_note`, `read_note`, `list_notes`, `dismiss_note`, `keepalive_note`).
  - `docs`: Offline API documentation and exact method signatures (Dash/Zeal .docset archives).
  - `vault`: Personal notes and project artifact full-text search (SQLite FTS5 with BM25 ranking).
- **System 2 (Deep Nocturnal Synthesis):** High-capacity MoE reasoning model (Ornith-1.5-35B-A3B Q4_K_M via `llama-server` with Multi-Token Prediction and hybrid GPU/CPU expert offloading @ ~15.5 tok/s).
- **Mode A Mutual Exclusion & Auto-Restoration:** Dynamically stops and starts the respective model engines (Ollama for System 1, `llama-server` for System 2) to guarantee 100% exclusive GPU VRAM allocation, automatically restoring System 1 for the Chat Copilot when orchestration yields.
- **Cross-Process Concurrency Locking:** Enforces single-task execution across independent processes (TUI Auto-Tick, Chat Copilot `start_orchestration`, CLI) via database-level `IN_PROGRESS` mutual exclusion.
- **Colibrì / System 2 429 Exponential Backoff:** Automatic retry on HTTP 429 with exponential delays (3s, 6s, 12s) to gracefully absorb brief GPU saturation and prompt cache evictions.


## Configuration & Environment Variables

Baton is completely decoupled and can be customized via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BATON_OLLAMA_URL` | System 1 / Ollama API endpoint | `http://127.0.0.1:11434` |
| `BATON_COLIBRI_URL` | System 2 / OpenAI-compatible endpoint | `http://127.0.0.1:8000` |
| `BATON_SYSTEM1_MODEL`| Model name for System 1 | `qwen3:14b` |
| `BATON_SYSTEM2_MODEL`| Model name for System 2 | Auto-detected / `ornith-1.5-35b-a3b` |
| `BATON_OLLAMA_START_CMD` | Optional hook command to start Ollama | System binary fallback |
| `BATON_OLLAMA_STOP_CMD`  | Optional hook command to stop Ollama  | Process termination fallback |
| `BATON_COLIBRI_START_CMD`| Hook command to start System 2 | Auto-detects `scripts/run_ornith.bat` (or `.sh`) |
| `BATON_COLIBRI_STOP_CMD` | Hook command to stop System 2 | Auto-detects `scripts/stop_ornith.bat` (or `.sh`) |
| `BATON_NEWSBOAT_DB`  | Path to custom Newsboat SQLite cache | `baton/data/rss/cache.db` |
| `BATON_ZIM_PATH`     | Path to custom Kiwix Wikipedia `.zim` archive | Auto-detected / None |

### System 2 Launchers (`scripts/`)
- `scripts/run_ornith.bat` / `scripts/run_ornith.sh`: Launches `llama-server` with Ornith-1.5-35B-A3B Q4_K_M on port 8000, configuring 32k context, GPU offload (`-ngl 99`), MoE CPU expert offload (`-ot ".ffn_.*_exps.=CPU"`), and Multi-Token Prediction (`--spec-type draft-mtp`).
- `scripts/stop_ornith.bat` / `scripts/stop_ornith.sh`: Gracefully terminates `llama-server` and frees all VRAM before System 1 resumes.

## Quick Start

```powershell
# Run the interactive Copilot chat
baton chat

# Launch the Textual TUI dashboard
baton tui

# Run the test suite
pytest baton/tests
```
