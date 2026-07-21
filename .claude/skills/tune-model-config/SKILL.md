---
name: tune-model-config
description: Configure the local pipeline's five model roles, their timeouts, and their num_ctx / VRAM budgets in src/local_llm/config.py (or the .port_sessions/local_llm_config.json / env overrides). Use when changing which model plays a role, fixing timeouts or KV-cache VRAM overflow, or wiring a new backend. Encodes the hard rule: never enable response streaming for a local runtime. Triggers: "change the planning model", "it keeps timing out", "VRAM overflow", "set num_ctx", "configure the local models".
---

# Tuning local model configuration

Config lives in `src/local_llm/config.py`. Precedence: **env var → per-model JSON → role default**. The JSON file is `<cwd>/.port_sessions/local_llm_config.json`.

## The five roles (defaults)

| Role            | Default model     | Timeout          | num_ctx | Env var                                    |
| --------------- | ----------------- | ---------------- | ------- | ------------------------------------------ |
| Coding          | `qwen3.6:35b-a3b` | 720s             | 131072  | `CODING_MODEL` (+ `OPENAI_MODEL` fallback) |
| Planning        | `qwen3:14b`       | 480s             | 16384   | `PLANNING_AND_ORCHESTRATION_MODEL`         |
| Review          | `phi4-mini`       | 120s             | 16384   | `REVIEW_MODEL`                             |
| Diagnosis       | `qwen3:14b`       | 480s             | 8192    | `DIAGNOSIS_MODEL`                          |
| Selection/route | `gemma3:4b`       | (server default) | —       | `MODEL_SELECTION_MODEL`                    |

Each role also honors `<ROLE>_BASE_URL` and `<ROLE>_API_KEY`. `LOCAL_LLM_NUM_CTX` overrides num_ctx for every role at once (per-model JSON still wins). Default backend: Ollama `http://localhost:11434/v1`.

## VRAM budgeting (target: RTX 3060, 12 GB)

num_ctx is sized so the KV cache fits in VRAM — overflow forces partial CPU offload, dropping to ~8–12 tok/s, which then blows the socket timeout. Documented in the config comments:

- Planning at 32K KV (~5.2 GB) overflowed → set to **16K** (~2.6 GB), keeping full-GPU ~50 tok/s.
- Diagnosis is single-turn/small → **8K**.
- Coding uses **128K** because long tool chains build huge histories; its KV spills to the 128 GB system RAM, which is acceptable.

When you change a model, re-check: weights size + KV cache at the chosen num_ctx must fit VRAM, or accept the slower offload and raise the role timeout to match.

## Hard rule — never stream a local runtime

Do **not** set `stream_model_responses=True` for any local LLM runtime.

- Streaming keeps qwen3 **think mode ON**. Coding tasks then reason past explicit bash instructions and substitute their own commands; planning hangs for ~20 min because thinking tokens stream indefinitely and the socket timeout never fires (data is flowing).
- The non-streaming path in `_build_payload` auto-applies `think:False` for qwen3, which makes the model follow instructions mechanically. Text-format `<tool_code>` output is then recovered by `openai_compat.py` (see `fix-tool-call-parsing`).

## Verify

```bash
python3 -m pytest tests/test_config_runtime.py -q
python3 -m src.main agent-local-llm "list the files here" --cwd . --planning-model qwen3:14b
```
