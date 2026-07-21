---
name: add-bundled-skill
description: Add or edit a bundled skill in the repo's OWN skill feature — src/bundled_skills.py — the prompt-type skills the agent exposes to its model via the Skill tool (simplify, verify, debug, update-config). Use when extending claw-code-agent's built-in skill catalog. This is DISTINCT from .claude/skills/*/SKILL.md (which configure Claude Code working on this repo). Triggers: "add a bundled skill", "the agent should offer a X skill", "register a skill in bundled_skills".
---

# Adding a bundled skill (the repo's own skill system)

> Disambiguation: `.claude/skills/<name>/SKILL.md` files (including this one) tell **Claude Code** how to work on this repo. `src/bundled_skills.py` is a **product feature of claw-code-agent** — the skills its own agent offers to a local model. This skill is about the latter.

Bundled skills are prompt-type: each generates an AI prompt sent to the model, carries `when_to_use` guidance for auto-invocation, and can restrict `allowed_tools`. They mirror the npm `src/skills/bundled/` module.

## Anatomy (`src/bundled_skills.py`)

- `BundledSkill` dataclass (frozen): `name`, `description`, `when_to_use`, `aliases`, `allowed_tools`, `user_invocable`, `get_prompt`.
- `get_prompt(agent, args) -> str` — a function that builds the prompt. It has access to the live `LocalCodingAgent` (cwd, model_config, session, usage), so it can inject git diffs, diagnostics, etc. See `_simplify_prompt` (pulls `git diff`) and `_debug_prompt` (session stats).
- `BUNDLED_SKILLS` tuple — the registry. `format_skills_for_system_prompt()` renders it into the system-reminder listing the model reads; `find_bundled_skill(name)` resolves by name or alias.

## To add one

1. Write a `_<name>_prompt(agent, args) -> str` generator.
2. Append a `BundledSkill(...)` entry to `BUNDLED_SKILLS` with a precise `description` and `when_to_use` (these drive model discovery and auto-invocation).
3. Set `allowed_tools` to the minimum the skill needs; set `user_invocable=False` to hide it from the user-facing listing (like an internal-only skill).
4. Add `aliases` if it has alternate names.

## Verify

```bash
python3 -c "from src.bundled_skills import get_bundled_skills, find_bundled_skill, format_skills_for_system_prompt; \
print([s.name for s in get_bundled_skills()]); print(find_bundled_skill('<name>')); print(format_skills_for_system_prompt())"
```

Add/extend a test alongside the existing skill/slash-command tests (`tests/test_agent_slash_commands.py`, `tests/test_new_slash_commands.py`).
