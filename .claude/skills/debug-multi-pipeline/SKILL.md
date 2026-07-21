---
name: debug-multi-pipeline
description: Debugging playbook for the claw-multi multi-model local pipeline — where session JSON and the task DB live, the route→plan→code→review→diagnose stages, and the common failure classes (empty responses, text-format tool calls, chat-mode drift, timeouts, output-file-missing). Use when an agent-local-llm run misbehaves, stalls, produces no output, or loops. Triggers: "the pipeline is stuck", "no tasks were queued", "the model won't call tools", "debug the local agent run".
---

# Debugging the claw-multi pipeline

## Pipeline stages (all in `src/local_llm/`)

1. **Route** (`router.py`) — a small model classifies the request as `coding` (exact file + exact edit named) or `planning` (everything else).
2. **Plan** (`executor.py` planning runtime) — decomposes the request into a task queue (`task_queue.py`, `queue_tool.py`). Planning gets **only** the `queue_task` tool.
3. **Code** — the coding model executes each task. The coding agent gets **only 7 core tools** (`_make_coding_registry` in `executor.py`); the rest are discoverable via the tool tree.
4. **Review** (`_run_review`) — a small model passes/fails the task, and checks whether `task.output_file` exists on disk.
5. **Diagnose** (`_run_diagnosis`) — on failure, generates a correction task, then loops back.

## Where to look first

- **Session tool results / transcript:** `.port_sessions/agent/<task_id>/` (session JSON — read the tool_use + tool result pairs and bash exit codes).
- **Task queue DB:** `.port_sessions/<session_id>/tasks.db` (SQLite — task states, types, output files, context).
- **Scratchpad + task.md:** `.port_sessions/scratchpad/<task_id>/task.md` (the task description written for the model before each run).
- **stderr logs** from the run: the `session_id=...` line, `Re-plan queued N task(s)`, and `diagnosis → correction` lines.

## Common failure classes and where they're handled

All detection/recovery lives in `executor.py` (nudge loop, `_looks_like_*` heuristics) and `../openai_compat.py` (tool-call parsing fallbacks).

| Symptom                                                                      | Likely cause                                                             | Where to check                                                                                       |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Coding model returns empty `content`, no tool calls                          | Too many tools in registry, or genuinely-empty final turn                | `_make_coding_registry` (must be 7 tools); `_EMPTY_OUTPUT_NUDGE`                                     |
| Model prints a tool call as text (`<bash>…`, `tool: {json}`, `name<{json}>`) | Text-format tool call not parsed                                         | `_extract_text_format_tool_calls` in `openai_compat.py` — see `.claude/skills/fix-tool-call-parsing` |
| Model explores then "chats" ("I see you've listed… what next?")              | Chat-mode drift; tool_calls>0 so nudge must still fire                   | `_CHAT_MODE_RE`, `_looks_like_chat_mode`                                                             |
| Planner queues 0 tasks                                                       | Planning model output stripped to thinking-only, or given too many tools | planning runtime; 0-task fallback to a single coding task                                            |
| `exit_code=-9` / "timed out" cascade                                         | A correction task re-ran the full test suite / a slow command            | `_DIAGNOSIS_SYSTEM_PROMPT` timeout rules                                                             |
| Output file never written                                                    | Model stopped after `exit_code=0`                                        | `_OUTPUT_FILE_MISSING_NUDGE`                                                                         |
| HTTP 500 from Ollama mid-run                                                 | Context overflow (heredoc file content, huge session)                    | diagnosis heredoc stripping; backend_error restore of pre-nudge output                               |

## Reproduce and isolate

- Re-run the exact prompt with `--max-tasks-per-session 1` to isolate a single task.
- Check the tool-call parser in isolation before editing the pipeline (`fix-tool-call-parsing`).
- Model-role sizing / VRAM timeouts belong to `tune-model-config`, not here.

The user maintains a detailed running log of fixed failure modes in project memory — consult it for the full history before re-diagnosing a known class.
