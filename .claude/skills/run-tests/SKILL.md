---
name: run-tests
description: Run and add tests for claw-code-agent. Pure-stdlib pytest suite under tests/, one test_<module>.py per src module, invoked with python3 -m pytest. Use whenever running the test suite, adding coverage for a change, or verifying a fix. Triggers: "run the tests", "add a test", "does this break anything", "verify with pytest".
---

# Testing claw-code-agent

Zero external dependencies — tests use `pytest` + the standard library only. The interpreter is `python3` (`python` is not on PATH).

```bash
python3 -m pytest tests/ -q                       # whole suite
python3 -m pytest tests/test_bash_security.py -q   # one file
python3 -m pytest tests/ -q -k "tool_pool"         # by keyword
python3 -m pytest tests/test_agent_runtime.py -q -x -vv   # stop at first failure, verbose
```

## Conventions

- Tests live in `tests/`, named `test_<module>.py`, mirroring `src/<module>.py`. Local-pipeline logic under `src/local_llm/` is covered by the corresponding runtime tests.
- Tests must **not** hit a live model backend. The pipeline model calls go through `OpenAICompatClient`; stub/patch it rather than requiring Ollama to be running.
- When you change behavior, add or update the matching `test_<module>.py`. Prefer a test that reproduces the bug first, then make it pass (per CLAUDE.md's goal-driven workflow).

## After a change

1. Run the file(s) for the module you touched.
2. Run the full suite before considering the work done.
3. For pipeline behavior that only manifests end-to-end (tool-call parsing, nudge loops), also drive a real prompt via `run-local-agent` and inspect the session JSON — unit tests alone won't exercise a local model's text quirks.
