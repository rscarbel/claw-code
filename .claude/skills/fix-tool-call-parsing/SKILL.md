---
name: fix-tool-call-parsing
description: Add or fix a text-format tool-call fallback parser in src/openai_compat.py — the layer that recovers tool calls when a local model emits them as text (e.g. <bash>cmd</bash>, `tool: {json}`, `name<{json}>`, bare shell commands) instead of via the tool API. Use when a local model's tool call is being ignored (tool_calls=0) or misparsed, or when adding support for a new text format. Triggers: "the model output a tool call as text", "tool_calls is 0 but it clearly ran a command", "add a parser pattern", "openai_compat fallback".
---

# Fixing text-format tool-call parsing

Local models frequently emit tool calls as plain text rather than through the OpenAI tool API. `OpenAICompatClient.complete()` in `src/openai_compat.py` recovers these via `_extract_text_format_tool_calls`, a numbered chain of regex/heuristic patterns. There is also a **streaming** fallback in `src/local_llm/../agent_runtime.py` (`_query_model`) that calls the same extractor post-stream and then `session.patch_assistant_as_tool_calls(...)`.

## When to add a pattern

Add one only when a real captured session shows a model emitting a tool call in a text shape that **no existing pattern matches**, leaving `tool_calls == 0`. First confirm by reading the session JSON (see `debug-multi-pipeline`).

## How to add a pattern

1. In `src/openai_compat.py`, define a module-level compiled regex near the other `_*_RE` patterns.
2. Insert a numbered rule into `_extract_text_format_tool_calls` in priority order (specific formats before the greedy shell-command fallback).
3. Parse embedded JSON with `json.JSONDecoder().raw_decode(...)` at the `{` position (used by the existing `{json}` patterns) so trailing prose doesn't break parsing.
4. **Only emit a call when the extracted tool name is a known tool.** Gate greedy heuristics (bare shell commands) behind: bash is available AND the text reads like a command AND it isn't a bare interpreter name (`_BARE_INTERACTIVE_CMDS` — e.g. bare `bash`/`sh`/`python3` must NOT execute, they open a hanging interactive shell).
5. Multi-call formats (planner emitting several `queue_task: {…}`) must return **all** matches so the agent framework runs them in one turn.

## Guard against false positives

Reject prose like `I will run…`, `Let me…`, and multi-line text without shell continuations. Malformed XML bodies (`<bash>\n<cmd\n</bash>`) need artifact stripping (leading `<`). Test both the positive case and a prose negative.

## Detection vs parsing — two different files

- **Parsing** (recover the call): `openai_compat.py` `_extract_text_format_tool_calls`.
- **Detection** (notice the model is stuck and nudge it): `src/local_llm/executor.py` — `_looks_like_bare_tool_name`, `_looks_like_shell_command`, `_looks_like_chat_mode`, and the nudge builders. If a format should be _parsed_, fix it here; if the model should be _told to retry_, fix it there. Often both.

## Verify

```bash
python3 -m pytest tests/ -q -k openai_compat   # if a matching test exists
python3 -c "from src.openai_compat import _extract_text_format_tool_calls; print(_extract_text_format_tool_calls('<bash>ls -la</bash>', {'bash'}))"
```

Then re-run the failing prompt through `agent-local-llm` and confirm `tool_calls > 0`.
