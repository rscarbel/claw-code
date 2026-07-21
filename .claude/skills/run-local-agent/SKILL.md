---
name: run-local-agent
description: Run the claw-code-agent CLI — the multi-model local pipeline (agent-local-llm) or the single-agent modes (agent-prompt / agent-chat / agent-resume). Use whenever you need to actually invoke the agent, pick a backend (Ollama, vLLM, LiteLLM, OpenRouter), set model/env overrides, or resume a session. Triggers: "run the agent", "start claw", "run the local LLM pipeline", "try this prompt through the agent".
---

# Running claw-code-agent

The project is **pure standard library, zero dependencies**. The interpreter is `python3` (`python` is not on PATH). Always run from the repo root as a module.

```bash
python3 -m src.main <subcommand> ...
python3 -m src.main --help          # full subcommand list
```

## Two ways to run

### 1. Multi-model local pipeline (`agent-local-llm`)

The "claw-multi" pipeline: a routing model decides plan-vs-code, a planning model decomposes into a task queue, and a coding model executes each task with review/diagnosis loops. This is the primary research surface of this repo.

```bash
python3 -m src.main agent-local-llm "add unit tests for the parser" --cwd .
# session_id=... is printed to stderr; resume with:
python3 -m src.main agent-local-llm "" --resume-session-id <id>
```

Per-role model overrides are CLI flags: `--coding-model`, `--planning-model`, `--selection-model`, `--review-model`, `--diagnosis-model` (each also has `-base-url` / `-api-key` variants), plus `--max-tasks-per-session` and `--max-review-loops`. See `.claude/skills/tune-model-config` for what the roles are and how to size them.

### 2. Single-agent modes

```bash
python3 -m src.main agent-prompt "one-shot task" --cwd .
python3 -m src.main agent-chat --cwd .          # multi-turn REPL, /exit to quit
python3 -m src.main agent-resume <session-id>
```

## Backend selection

Default backend is **Ollama** at `http://localhost:11434/v1`. The runtime targets any OpenAI-compatible API, so vLLM / LiteLLM Proxy / OpenRouter all work by pointing the env vars at them.

```bash
# Ollama must be running with the models pulled:
ollama serve
ollama pull qwen3.6:35b-a3b       # coding model
```

The **coding model only** falls back to the generic `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` env vars, so a single-endpoint user can configure it that way. Other roles use role-specific env vars (see tune-model-config).

## Before assuming a run failed

A local-model run can take minutes — large hybrid-offload models (qwen3.6:35b-a3b) cold-load slowly (720s timeout). Silence is usually loading, not a hang. To debug an actual failure, use `.claude/skills/debug-multi-pipeline`.
