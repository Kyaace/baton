# Baton

**Baton** is an autonomous, dual-speed local LLM orchestrator and pair-programming copilot designed to coordinate high-throughput agile scouts (System 1) and deep nocturnal synthesis models (System 2) under single-GPU resource constraints.

## Architecture & Modes

- **System 1 (Agile Scout / Copilot):** Fast reasoning and tool-calling model (e.g. Qwen3:14B via `llama-server` router or Ollama @ ~32 tok/s) equipped with modular tool bundles:
  - `core`: Tool Shelf discovery, on-demand bundle activation, and `tool_help`.
  - `baton`: Deterministic priority queue, DAG task creation (`add_task`), dependency wiring (`set_task_dependencies`), orchestration (`start_orchestration`), and artifact inspection.
  - `library`: Offline multi-corpus ZIM archive explorer and reader (Wikipedia, Python docs, WikiMed, etc.). Alias: `wikipedia`.
  - `rss`: Offline RSS news monitoring and reader across 12+ tech/science publications with time filtering (`since`). Alias: `newsboat`.
  - `sandbox`: Isolated Python AST verification, REPL evaluation, and pytest test suite execution.
  - `sdr`: RTL-SDR spectrum sweeps (`rtl_power`), carrier detection, audio recordings, and RF telemetry decoding.
  - `time`: Real-world UTC/local time grounding, timezone offsets, task alarm scheduling, and ephemeral 24h post-it scratchpad notes with Mercy 1-Hour rules and character caps (`save_note`, `append_note`, `read_note`, `list_notes`, `dismiss_note`, `keepalive_note`).
  - `docs`: Offline API documentation and exact method signatures (Dash/Zeal .docset archives).
  - `vault`: Personal notes and project artifact full-text search (SQLite FTS5 with BM25 ranking).
  - **Autonomous Recon Self-Recovery:** Automatically diagnoses tool lookup misses or typos via ZIM suggestion indexes and structured JSON self-repair, retrying with canonical parameters.
- **System 2 (Deep Nocturnal Synthesis):** High-capacity MoE reasoning model (Ornith-1.5-35B-A3B Q4_K_M via `llama-server` with Multi-Token Prediction and hybrid GPU/CPU expert offloading @ ~15.5 tok/s).
  - **Decisive Reasoning Protocol:** Enforces focused `<think>` planning (<1000 tokens) and decisive conclusion to avoid circular reasoning loops.
  - **Reasoning Code Salvage Fallback:** Parses and salvages drafted code blocks directly from reasoning monologues if token ceilings are reached before closing `</think>`.
  - **Streaming Monologue Capture:** Captures the full reasoning monologue atomically to `baton/artifacts/synthesis/<task_id>_thinking.md`.
- **Automated Verification Pipeline:** `start_orchestration` automatically executes downstream verification tasks (`fast_verify`) immediately following synthesis, ensuring code is verified without leaving verification abandoned.
- **Multi-Model Router Mode (llama-server :8080):** Serves both System 1 (`qwen3:14b`) and System 2 (`ornith-1.5-35b-a3b`) from a single `llama-server` process on port 8080 using `models_preset.ini`. Configured with `parallel=1`, `cache-type-k=q8_0`, and `cache-type-v=q8_0` to fit strictly within the 12GB VRAM envelope of an RTX 3060. Models dynamically page in/out of VRAM on demand (~1.1s eviction time).
- **Environment & Technical Reference:** See [`memory.md`](file:///e:/dev/Antigravity/baton/memory.md) for full hardware profiling, VRAM budget specifications, and operational guardrails.


## Configuration & Environment Variables

Baton is completely decoupled and can be customized via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BATON_ROUTER_URL` | Multi-Model llama-server router endpoint | `http://127.0.0.1:8080` |
| `BATON_SYSTEM1_MODEL`| Model name/alias for System 1 | `qwen3` |
| `BATON_SYSTEM2_MODEL`| Model name/alias for System 2 | `ornith` |
| `BATON_ROUTER_START_CMD` | Hook command to start multi-model router | Auto-detects `scripts/run_router.bat` |
| `BATON_ROUTER_STOP_CMD` | Hook command to stop multi-model router | Auto-detects `scripts/stop_router.bat` |
| `BATON_NEWSBOAT_DB`  | Path to custom Newsboat SQLite cache | `baton/data/rss/cache.db` |
| `BATON_ZIM_PATH`     | Path to custom Kiwix Wikipedia `.zim` archive | Auto-detected / None |

### Launchers & Scripts (`scripts/`)
- `scripts/run_router.bat` / `scripts/run_router.sh`: Launches `llama-server` in multi-model router mode on port 8080 with `models_preset.ini`, enabling dynamic VRAM/RAM swapping between Ornith-1.5-35B-A3B (MoE) and Qwen3:14B.
- `scripts/stop_router.bat` / `scripts/stop_router.sh`: Gracefully terminates the multi-model router.
- `scripts/reload_newsboat.bat` / `scripts/reload_newsboat.sh`: Fetches and caches offline RSS articles into the local SQLite store.

## Quick Start

```powershell
# Run the interactive Copilot chat
baton chat

# Launch the Textual TUI dashboard
baton tui

# Run the test suite
pytest baton/tests
```
