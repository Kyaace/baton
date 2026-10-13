# Baton

**Baton** is an autonomous, dual-speed local LLM orchestrator and pair-programming copilot designed to coordinate high-throughput agile scouts (System 1) and deep nocturnal synthesis models (System 2) under single-GPU resource constraints.

## Architecture & Modes

- **System 1 (Agile Scout / Copilot):** Fast reasoning and tool-calling model (e.g. Qwen3:14B via Ollama @ ~28 tok/s) equipped with modular tool bundles:
  - `core`: Tool Shelf discovery and on-demand bundle activation.
  - `baton`: Deterministic priority queue, DAG task creation, and artifact inspection.
  - `wikipedia`: Offline encyclopedic lookups using Kiwix ZIM archives.
  - `newsboat`: Offline RSS news monitoring and reader across 12+ tech/science publications.
  - `sandbox`: Isolated Python AST verification, REPL evaluation, and pytest test suite execution.
  - `sdr`: RTL-SDR spectrum sweeps (`rtl_power`), carrier detection, audio recordings, and RF telemetry decoding.
- **System 2 (Deep Macro-Planner / Synthesizer):** Frontier MoE model (e.g. DeepSeek-V4-Flash via Colibrì @ ~0.8 tok/s) that operates tool-blind, consuming briefing dossiers compiled by System 1 to synthesize structured task plans and complete codebase implementations.
- **Mode A Mutual Exclusion:** Dynamically stops and starts the respective model engines to guarantee 100% exclusive GPU VRAM allocation.

## Configuration & Environment Variables

Baton is completely decoupled and can be customized via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BATON_OLLAMA_URL` | System 1 / Ollama API endpoint | `http://127.0.0.1:11434` |
| `BATON_COLIBRI_URL` | System 2 / Colibrì OpenAI-compatible endpoint | `http://127.0.0.1:8000` |
| `BATON_SYSTEM1_MODEL`| Model name for System 1 | `qwen3:14b` |
| `BATON_SYSTEM2_MODEL`| Model name for System 2 | Auto-detected / DeepSeek-V4 |
| `BATON_OLLAMA_START_CMD` | Optional hook command to start Ollama | System binary fallback |
| `BATON_OLLAMA_STOP_CMD`  | Optional hook command to stop Ollama  | Process termination fallback |
| `BATON_COLIBRI_START_CMD`| Optional hook command to start Colibrì | None |
| `BATON_COLIBRI_STOP_CMD` | Optional hook command to stop Colibrì | CLI stop fallback |
| `BATON_NEWSBOAT_DB`  | Path to custom Newsboat SQLite cache | `baton/data/rss/cache.db` |

## Quick Start

```powershell
# Run the interactive Copilot chat
baton chat

# Launch the Textual TUI dashboard
baton tui

# Run the test suite
pytest baton/tests
```
