# Baton Environment Memory & Technical Reference

> **Target Audience:** Future AI coding sessions, pair-programming agents, and developers.  
> **Last Updated:** October 2026  
> **Project:** Baton (Autonomous Dual-Speed System 1 / System 2 Orchestrator)

---

## 1. Host Hardware & Environment Profile

| Subsystem | Specification / Configuration | Operational Notes |
| :--- | :--- | :--- |
| **OS** | Windows 11 Pro 64-bit | Primary platform for Baton, CLI, TUI, and local inference engines. |
| **CPU** | Intel Core i5-13500 | 14 cores (6 P-cores + 8 E-cores), 20 threads. Ideal for hybrid offloading. |
| **RAM** | **128 GB DDR4/DDR5** | Massive host RAM allows caching large MoE model layers (e.g. Ornith-35B, DeepSeek-V4). |
| **GPU** | **NVIDIA GeForce RTX 3060 12GB** | 12,288 MiB dedicated GDDR6 VRAM. Bandwidth: ~360 GB/s. PCIe Bandwidth: ~16 GB/s. |
| **VRAM Physics** | **Strict 12GB Budget** | Desktop apps take ~1.0–1.8 GB. Qwen3:14B takes ~9.1 GB. Free headroom: ~1.5–2.0 GB. **Never use parallel > 1 or unquantized 32k FP16 KV cache**, or memory pages to PCIe shared GPU memory and generation drops by 25x (~0.6 tok/s). |
| **Storage: `C:\`** | Samsung 970 EVO Plus 1TB NVMe | Model Weights: `C:\colibri_models\Ornith-1.5-35B-A3B\` and `C:\colibri_models\qwen3-14b\`. |
| **Storage: `E:\`** | Local Dev Drive | Workspace: `E:\dev\Antigravity\baton\` (Git repository, database, artifacts, tests). |
| **Python** | **Anaconda3 3.11.5** | Executable: `C:\ProgramData\Anaconda3\python.exe`. Used for all testing, pytest, and CLI execution. |
| **Shells** | PowerShell 7 / Windows Terminal | Launch via `.\baton.cmd` or `python -m baton.cli`. |

---

## 2. Model & Engine Architecture

### Multi-Model Router Mode (`llama-server` on port 8080)
- **Primary Engine:** A single `llama-server` router process serving both System 1 and System 2 using `models_preset.ini`.
- **System 1 Model:** `qwen3:14b` (Q4_K_M, ~9.1 GB, 8192 context window, Q8_0 KV cache).
- **System 2 Model:** `bartowski/Ornith-1.5-35B-A3B-GGUF:Q4_K_M` (35B MoE, 32768 context window, Q8_0 KV cache, hybrid CPU expert offload `.ffn_.*_exps.=CPU`, draft-mtp speculative decoding).
- **VRAM / RAM Swapping:** Models remain memory-mapped in 128GB host RAM and dynamically page in/out of the RTX 3060 12GB VRAM on demand with ~1.1s eviction time.
- **VRAM Configuration Safeguards in `models_preset.ini`:**
  ```ini
  version = 1

  [*]
  host = 127.0.0.1
  port = 8080
  flash-attn = true
  parallel = 1

  [bartowski/Ornith-1.5-35B-A3B-GGUF:Q4_K_M]
  model = C:\colibri_models\Ornith-1.5-35B-A3B\Ornith-1.5-35B-A3B-Q4_K_M.gguf
  alias = ornith,ornith-35b,Ornith-1.5-35B-A3B
  n-gpu-layers = 99
  override-tensor = .ffn_.*_exps.=CPU
  spec-type = draft-mtp
  ctx-size = 32768
  cache-type-k = q8_0
  cache-type-v = q8_0

  [qwen3:14b]
  model = C:\colibri_models\qwen3-14b\qwen3-14b-q4_k_m.gguf
  alias = qwen3,qwen3-14b,Qwen3:14B
  n-gpu-layers = 99
  ctx-size = 8192
  cache-type-k = q8_0
  cache-type-v = q8_0
  ```
- **Management Scripts:**
  - `scripts/run_router.bat`: Starts multi-model llama-server in background.
  - `scripts/stop_router.bat`: Gracefully stops the router.

---

## 3. Baton System Capabilities & Protocols

### A. System 1 (Agile Scout, Copilot, & Verifier)
- **Role:** Fast interactive copilot (~32 tok/s), DAG orchestrator, offline knowledge researcher, and automated code verifier.
- **Tool Shelf Bundles:**
  1. `core`: Baseline shelf manager (`list_tool_bundles`, `activate_tool_bundle`, `deactivate_tool_bundle`, `tool_help`).
  2. `baton`: Task priority queue, DAG creation, dependencies, stepping, orchestration (`add_task`, `start_orchestration`, `list_tasks`, `read_artifact`).
  3. `library` (`wikipedia`): Offline Kiwix ZIM multi-archive reader (`zim_list_corpora`, `wiki_search`, `wiki_read_lead`, `wiki_read_section`, `wiki_infobox`, `wiki_toc`).
  4. `rss` (`newsboat`): Offline RSS news search and extraction with timestamp filters (`rss_list_feeds`, `rss_latest`, `rss_search`, `rss_read_article`).
  5. `sandbox`: Isolated Python REPL, AST syntax validator, and pytest executor (`python_eval`, `python_syntax_check`, `pytest_run`, `sandbox_write_file`).
  6. `sdr`: RTL-SDR spectrum sweeps, carrier detection, audio recordings, telemetry decoding.
  7. `time`: Real-world UTC/local time, task alarm scheduling, 24h ephemeral desk notes (`get_current_time`, `schedule_task_alarm`, `save_note`, `read_note`, etc.).
  8. `docs`: Offline Dash/Zeal API docset symbol indexer and signature reader (`docs_search`, `docs_read_symbol`, `docs_get_signature`).
  9. `vault`: Local notes and artifact BM25 full-text search (`vault_search`, `vault_read_document`, `vault_index_path`).
- **Autonomous Recon Self-Recovery Loop:**
  - When a knowledge tool (ZIM Wikipedia, RSS, docs) returns a miss, typo, or empty content (`_is_recon_miss`), System 1 does not fail.
  - It automatically checks ZIM suggestion indexes and search candidates, runs a structured JSON diagnosis (`recovery_sys_prompt`), determines the corrected tool/parameters (e.g. correcting "Fall Equanox" -> "Autumnal equinox"), and re-queries the knowledge store.
  - Briefing dossiers explicitly record resolved typos and recovery metadata.

### B. System 2 (Deep Nocturnal Synthesizer)
- **Role:** Deep mathematical derivation, multi-file software synthesis, and specification proofs. Tool-blind to preserve context.
- **Reasoning Monologue Logging:** Emits live SSE tokens. The entire `<think>` reasoning monologue is captured in `baton/artifacts/synthesis/<task_id>_thinking.md`.
- **Decisive `<think>` Directives:** Prompt instructions mandate concise architectural planning (<1000 tokens) and decisive conclusion (`</think>`) to prevent circular reasoning loops.
- **Reasoning Code Salvage Fallback:** If the model hits token ceilings while generating inside reasoning without closing `</think>`, `System2Runner` parses and salvages Python code blocks directly from the reasoning monologue rather than generating an empty artifact.
- **Continuation Prompt Recovery:** If `output_code` is empty at token limit, the continuation request explicitly prompts the model to conclude thinking with `</think>` and emit the code.
- **Resilient File Naming:** Automatically names output artifacts `<task_id>.py` when code patterns or keywords are present.

### C. Orchestration & Verification Pipeline
- **Auto-Draining Downstream Verification:** In `start_orchestration`, whenever a System 2 synthesis task completes, the runner immediately checks for queued downstream `FAST_VERIFY` tasks and executes them in the same run, even if `max_steps=1` was passed.
- **Empty File Protection:** `read_artifact` detects 0-byte output files and alerts the user/agent, linking directly to `_thinking.md`.

---

## 4. Key CLI & Operational Commands

```powershell
# 1. Interactive Chat Copilot
.\baton.cmd chat

# 2. Reactive TUI Dashboard
.\baton.cmd tui

# 3. Queue Status & Inspection
.\baton.cmd status
.\baton.cmd artifacts

# 4. Step or Run Queue
.\baton.cmd step
.\baton.cmd run --max-steps 10

# 5. Stop Running Task Safely
.\baton.cmd stop

# 6. Run Test Suite
pytest baton/tests -v
```

---

## 5. Critical Development Guidelines & Gotchas

1. **VRAM Concurrency Rule:** Never increase `parallel` beyond `1` in `models_preset.ini`. Dual 32k contexts will exhaust the 12GB RTX 3060 and spill into Windows Shared GPU Memory over PCIe, causing a 25x-50x speed collapse.
2. **Orchestration Max Steps:** Do NOT pass `max_steps=1` when running synthesis or pipeline tasks; omit `max_steps` or use `5+` to allow chained tasks and automated downstream verification passes to complete.
3. **Artifact Files:** Generated dossiers reside in `baton/artifacts/dossiers/`, synthesis code in `baton/artifacts/synthesis/`, and atomic streaming progress snapshots in `baton/artifacts/progress/`.
4. **Testing Integrity:** All changes to `baton/chat.py`, `baton/system1.py`, `baton/system2.py`, or tools must maintain 100% pass rates across all 179+ pytest tests in `baton/tests/`.
