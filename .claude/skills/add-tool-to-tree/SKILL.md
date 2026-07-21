---
name: add-tool-to-tree
description: Add a new agent tool to the local pipeline's discoverable tool tree (src/local_llm/tool_tree.py) and decide whether it belongs in the coding agent's 7-tool core registry. Use when exposing a new capability to the local coding/planning models, or reorganizing tool categories. Triggers: "add a tool for the agent", "register a new tool", "the model needs a X tool", "tool tree category".
---

# Adding a tool to the local pipeline

The local coding model degrades sharply with tool count — with all ~65 tools it returns empty responses; with ~7 it calls tools reliably. So tools are split:

- **Core registry (7 tools):** `_make_coding_registry()` in `src/local_llm/executor.py` — `bash`, `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`, `list_dir`. Always present for the coding agent.
- **Tool tree (everything else):** `src/local_llm/tool_tree.py` — the other ~58 tools, discoverable at runtime via `explore_tools(category?)` + `use_discovered_tool(name, args_json)`. The tree exposes only ~9 tools to the model (~600 tokens).

## Default: add to the tree, not the core

Only add to the core 7 if the tool is needed on essentially every coding task. Anything else goes in the tree. The tool_tree.py module docstring (top of file) has the canonical steps; follow it. In short:

1. Add a `ToolMeta` entry to `_TOOL_META` (name, description, arg schema).
2. Add the tool name to the `tools` tuple of the appropriate `CategoryNode` in `_CATEGORIES`.
3. To add a **new category**: define the `CategoryNode`, then reference its name from `_ROOT` (top-level) or a parent node's `subcategories` tuple.

## If it must be core

Add the name to the tuple returned by `_make_coding_registry()` in `executor.py`. Keep the count as small as possible — every added core tool measurably raises the empty-response risk. Both coding-agent instantiations (main loop and fresh-context restart) already call `_make_coding_registry()`, so editing that one function covers both.

## Verify

```bash
python3 -m pytest tests/ -q -k "tool"        # tool_pool / extended_tools / tree tests
python3 -m src.main tool-pool                 # inspect the assembled default tool pool
```

Then run a task through `agent-local-llm` that should trigger the tool and confirm the model discovers/calls it via the session JSON.
