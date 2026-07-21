# Claude Code skills for claw-code-agent

These are [Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) that help Claude Code work on **this** repository. Each `<name>/SKILL.md` is auto-discovered and invoked when its `description` matches the task.

> Not to be confused with `src/bundled_skills.py`, which is claw-code-agent's _own_ product feature — the skills its agent offers to a local model. See the `add-bundled-skill` skill for that.

| Skill                   | Use it when                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| `run-local-agent`       | Running the `agent-local-llm` pipeline or single-agent modes; picking a backend |
| `debug-multi-pipeline`  | An `agent-local-llm` run stalls, loops, or produces no output                   |
| `fix-tool-call-parsing` | A local model emits a tool call as text and it's ignored (`tool_calls=0`)       |
| `add-tool-to-tree`      | Exposing a new tool to the local models (tree vs 7-tool core registry)          |
| `tune-model-config`     | Changing model roles, fixing timeouts / VRAM overflow, sizing `num_ctx`         |
| `run-tests`             | Running or adding to the pytest suite                                           |
| `run-benchmarks`        | Evaluating a model/change with the local or standard eval suites                |
| `parity-audit`          | Tracking parity with the upstream Claude Code npm source                        |
| `add-bundled-skill`     | Extending claw-code-agent's own built-in skill catalog                          |
