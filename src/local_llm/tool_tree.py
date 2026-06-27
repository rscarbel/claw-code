"""
src/local_llm/tool_tree.py
Hierarchical tool directory for local-LLM coding agents.

WHY THIS EXISTS
---------------
The full default_tool_registry contains 65 tools (~6 000 tokens of tool-spec
overhead). Sending all of them in every API call causes qwen3.6:35b-a3b to
produce empty responses with no tool calls (confirmed by direct Ollama testing:
7 tools → proper call; 65 tools → content='', finish_reason='stop').

The solution: give the coding agent only 7 core tools (bash, read_file, etc.)
plus two meta-tools from this module:

    explore_tools(category?)             Navigate the directory to find a tool.
    use_discovered_tool(name, args_json) Execute any tool once you know its name.

The other 58 tools live in a tree of 9 top-level categories. The model calls
explore_tools() to navigate and use_discovered_tool() to execute — never seeing
more than ~10 options at a time.

===========================================================================
HOW TO EXTEND THIS FILE
===========================================================================

--- Adding a new tool to an existing category ---

1. Add an entry to _TOOL_META (near the bottom of this section):

       "my_new_tool": ToolMeta(
           description="One-line explanation of what this tool does.",
           args="required_param: str, optional_param?: int",
       ),

   The key must exactly match the tool name in default_tool_registry().
   The args string uses TypeScript-style notation (?: = optional).

2. Add the tool name to the `tools` tuple in the appropriate CategoryNode:

       "my_category": CategoryNode(
           description="...",
           tools=(..., "my_new_tool"),   # ← append here
       ),

   A tool may appear in multiple categories — just add its name to each.
   Maximum 10 items per node (subcategories or tools combined).

--- Adding a new terminal category (directly contains tools) ---

Add an entry to _CATEGORIES:

    "my_category": CategoryNode(
        description="Short description of what tools are here.",
        tools=("tool_a", "tool_b"),
    ),

Then register it as either a top-level category (add to _ROOT) or as a
sub-category (add to a parent CategoryNode's `subcategories` tuple).

--- Adding a new non-terminal category (groups sub-categories) ---

Define the sub-categories first (as terminal CategoryNodes), then:

    "my_group": CategoryNode(
        description="What this group of sub-categories covers.",
        subcategories=("sub_a", "sub_b"),
    ),

Add to _ROOT or a parent's subcategories. Max 10 subcategories per node.

--- Removing a node ---

Tool: delete from _TOOL_META, remove its name from its category's `tools` tuple.
Category: delete from _CATEGORIES, remove its name from its parent's
          `subcategories` tuple or from _ROOT.

===========================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agent_tools import AgentTool, ToolExecutionContext


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolMeta:
    """Metadata displayed when a model browses to a terminal category.

    description  One-line summary of what the tool does.
    args         TypeScript-style argument signature (e.g. "url: str, timeout?: int").
                 Write "(no arguments)" when the tool takes none.
    """
    description: str
    args: str


@dataclass(frozen=True)
class CategoryNode:
    """A node in the tool tree.

    Either `subcategories` or `tools` is non-empty — not both.

    subcategories  Names of child CategoryNodes (non-terminal node).
    tools          Names of leaf tools from _TOOL_META (terminal node).
    """
    description: str
    subcategories: tuple[str, ...] = field(default_factory=tuple)
    tools: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_terminal(self) -> bool:
        return bool(self.tools)


# ---------------------------------------------------------------------------
# Tool metadata
#
# Every discoverable (non-core) tool gets one entry here.
# Key   = exact tool name from default_tool_registry() in agent_tools.py.
# Value = ToolMeta(description, args).
#
# Core tools (bash, read_file, write_file, edit_file, glob_search,
# grep_search, list_dir) are omitted — they are already in the coding
# agent's registry and never need to be discovered.
# ---------------------------------------------------------------------------

_TOOL_META: dict[str, ToolMeta] = {

    # ── Web ──────────────────────────────────────────────────────────────────

    "web_fetch": ToolMeta(
        description="Fetch the content of a URL (HTML, JSON, plain text).",
        args="url: str",
    ),
    "web_search": ToolMeta(
        description="Search the web and return a list of results.",
        args="query: str",
    ),
    "search_status": ToolMeta(
        description="Show the active web-search provider and its status.",
        args="(no arguments)",
    ),
    "search_list_providers": ToolMeta(
        description="List all configured web-search providers.",
        args="(no arguments)",
    ),
    "search_activate_provider": ToolMeta(
        description="Activate a named web-search provider.",
        args="provider: str",
    ),

    # ── Notebooks ────────────────────────────────────────────────────────────

    "notebook_edit": ToolMeta(
        description="Edit a cell in a Jupyter .ipynb notebook.",
        args="path: str, cell_index: int, source: str, cell_type?: str, create_cell?: bool",
    ),

    # ── Code intelligence ────────────────────────────────────────────────────

    "LSP": ToolMeta(
        description="Run a language-server operation (hover, definition, diagnostics, …).",
        args="operation: str, path: str, line?: int, character?: int",
    ),
    "tool_search": ToolMeta(
        description="Search registered agent tools by keyword.",
        args="query: str",
    ),

    # ── MCP (Model Context Protocol) ─────────────────────────────────────────

    "mcp_list_tools": ToolMeta(
        description="List tools exposed by connected MCP servers.",
        args="(no arguments)",
    ),
    "mcp_call_tool": ToolMeta(
        description="Call a tool provided by an MCP server.",
        args="name: str, arguments?: dict",
    ),
    "mcp_list_resources": ToolMeta(
        description="List resources available from connected MCP servers.",
        args="(no arguments)",
    ),
    "mcp_read_resource": ToolMeta(
        description="Read a resource from an MCP server by URI.",
        args="uri: str",
    ),

    # ── Task management — running tasks ──────────────────────────────────────

    "task_create": ToolMeta(
        description="Create a new task in the task tracker.",
        args="title: str, description?: str, assignee?: str",
    ),
    "task_update": ToolMeta(
        description="Update fields on an existing task.",
        args="task_id: str, title?: str, description?: str, status?: str",
    ),
    "task_start": ToolMeta(
        description="Mark a task as in-progress.",
        args="task_id: str",
    ),
    "task_complete": ToolMeta(
        description="Mark a task as completed.",
        args="task_id: str",
    ),
    "task_block": ToolMeta(
        description="Mark a task as blocked (add an optional reason).",
        args="task_id: str, reason?: str",
    ),
    "task_cancel": ToolMeta(
        description="Cancel a task.",
        args="task_id: str",
    ),
    "task_next": ToolMeta(
        description="Get the next pending task from the queue.",
        args="(no arguments)",
    ),
    "TaskOutput": ToolMeta(
        description="Read the live output of a running background task.",
        args="task_id: str",
    ),
    "TaskStop": ToolMeta(
        description="Stop a running background task.",
        args="task_id: str",
    ),

    # ── Task management — querying ────────────────────────────────────────────

    "task_list": ToolMeta(
        description="List tasks, optionally filtered by status.",
        args="status?: str, assignee?: str",
    ),
    "task_get": ToolMeta(
        description="Get full details for a specific task.",
        args="task_id: str",
    ),

    # ── Planning ─────────────────────────────────────────────────────────────

    "plan_get": ToolMeta(
        description="Read the current structured plan.",
        args="(no arguments)",
    ),
    "update_plan": ToolMeta(
        description="Overwrite the current plan with new content.",
        args="content: str",
    ),
    "plan_clear": ToolMeta(
        description="Delete the current plan.",
        args="(no arguments)",
    ),
    "EnterPlanMode": ToolMeta(
        description="Switch the agent into plan-review mode (prevents execution).",
        args="(no arguments)",
    ),
    "ExitPlanMode": ToolMeta(
        description="Leave plan-review mode and resume normal execution.",
        args="(no arguments)",
    ),
    "todo_write": ToolMeta(
        description="Write or replace the session todo list.",
        args="todos: list[{content: str, status: str, priority: str}]",
    ),

    # ── Team ─────────────────────────────────────────────────────────────────

    "team_list": ToolMeta(
        description="List all teams.",
        args="(no arguments)",
    ),
    "team_get": ToolMeta(
        description="Get details for a specific team.",
        args="team_id: str",
    ),
    "team_create": ToolMeta(
        description="Create a new team.",
        args="name: str, description?: str",
    ),
    "team_delete": ToolMeta(
        description="Delete a team.",
        args="team_id: str",
    ),
    "send_message": ToolMeta(
        description="Post a message to a team channel.",
        args="team_id: str, message: str",
    ),
    "team_messages": ToolMeta(
        description="Read recent messages from a team channel.",
        args="team_id: str, limit?: int",
    ),

    # ── Remote connections ────────────────────────────────────────────────────

    "remote_status": ToolMeta(
        description="Show current remote-connection status.",
        args="(no arguments)",
    ),
    "remote_list_profiles": ToolMeta(
        description="List available remote connection profiles.",
        args="(no arguments)",
    ),
    "remote_connect": ToolMeta(
        description="Connect to a remote profile.",
        args="profile: str",
    ),
    "remote_disconnect": ToolMeta(
        description="Disconnect from the active remote profile.",
        args="(no arguments)",
    ),
    "remote_trigger": ToolMeta(
        description="Trigger a workflow on a remote agent.",
        args="workflow: str, inputs?: dict",
    ),

    # ── Git worktrees ─────────────────────────────────────────────────────────

    "worktree_status": ToolMeta(
        description="Show the current worktree status (branch, path, dirty?).",
        args="(no arguments)",
    ),
    "worktree_enter": ToolMeta(
        description="Enter a git worktree at the given path.",
        args="path: str",
    ),
    "worktree_exit": ToolMeta(
        description="Exit the current worktree and return to the main repo.",
        args="(no arguments)",
    ),

    # ── Account and configuration ─────────────────────────────────────────────

    "account_status": ToolMeta(
        description="Show the currently logged-in account and its limits.",
        args="(no arguments)",
    ),
    "account_list_profiles": ToolMeta(
        description="List all configured account profiles.",
        args="(no arguments)",
    ),
    "account_login": ToolMeta(
        description="Log in to an account profile.",
        args="profile?: str",
    ),
    "account_logout": ToolMeta(
        description="Log out of the active account.",
        args="(no arguments)",
    ),
    "config_list": ToolMeta(
        description="List all configuration key-value pairs.",
        args="(no arguments)",
    ),
    "config_get": ToolMeta(
        description="Get the value of a single configuration key.",
        args="key: str",
    ),
    "config_set": ToolMeta(
        description="Set a configuration key to a new value.",
        args="key: str, value: str",
    ),

    # ── Agents, workflows, and system ─────────────────────────────────────────

    "Agent": ToolMeta(
        description="Spawn a sub-agent to handle a delegated task.",
        args="description: str, prompt: str, subagent_type?: str",
    ),
    "Skill": ToolMeta(
        description="Invoke a registered skill by name.",
        args="skill: str, args?: str",
    ),
    "workflow_list": ToolMeta(
        description="List available named workflows.",
        args="(no arguments)",
    ),
    "workflow_get": ToolMeta(
        description="Get the definition of a named workflow.",
        args="name: str",
    ),
    "workflow_run": ToolMeta(
        description="Run a named workflow with optional inputs.",
        args="name: str, inputs?: dict",
    ),
    "ask_user_question": ToolMeta(
        description="Ask the user a clarifying question and wait for their answer.",
        args="question: str, options?: list[str]",
    ),
    "sleep": ToolMeta(
        description="Pause execution for a number of seconds.",
        args="seconds: float",
    ),
}


# ---------------------------------------------------------------------------
# Category tree
#
# Each CategoryNode is either:
#   terminal     — `tools` is non-empty; lists tool names from _TOOL_META.
#   non-terminal — `subcategories` is non-empty; lists child category names.
#
# Constraint: max 10 items per node (tools or subcategories combined).
# A tool name may appear in more than one terminal category.
# ---------------------------------------------------------------------------

_CATEGORIES: dict[str, CategoryNode] = {

    # ── Top-level categories ──────────────────────────────────────────────────

    "web": CategoryNode(
        description="Fetch web pages or search the internet",
        tools=(
            "web_fetch",
            "web_search",
            "search_status",
            "search_list_providers",
            "search_activate_provider",
        ),
    ),
    "notebooks": CategoryNode(
        description="Edit Jupyter .ipynb notebooks",
        tools=(
            "notebook_edit",
        ),
    ),
    "code_intel": CategoryNode(
        description="Language server (LSP) and in-registry tool search",
        tools=(
            "LSP",
            "tool_search",
        ),
    ),
    "mcp": CategoryNode(
        description="Model Context Protocol — external tool and resource servers",
        tools=(
            "mcp_list_tools",
            "mcp_call_tool",
            "mcp_list_resources",
            "mcp_read_resource",
        ),
    ),
    "tasks": CategoryNode(
        description="Task tracking and project planning",
        # Non-terminal: browse sub-categories for specific task tools.
        subcategories=(
            "task_run",
            "task_query",
            "planning",
        ),
    ),
    "team": CategoryNode(
        description="Team management and messaging",
        tools=(
            "team_list",
            "team_get",
            "team_create",
            "team_delete",
            "send_message",
            "team_messages",
        ),
    ),
    "remote": CategoryNode(
        description="Remote connections and git worktrees",
        # Non-terminal: browse sub-categories.
        subcategories=(
            "connections",
            "worktrees",
        ),
    ),
    "account_config": CategoryNode(
        description="Account authentication and app configuration",
        tools=(
            "account_status",
            "account_list_profiles",
            "account_login",
            "account_logout",
            "config_list",
            "config_get",
            "config_set",
        ),
    ),
    "orchestration": CategoryNode(
        description="Sub-agents, skills, workflows, user interaction, and sleep",
        tools=(
            "Agent",
            "Skill",
            "workflow_list",
            "workflow_get",
            "workflow_run",
            "ask_user_question",
            "sleep",
        ),
    ),

    # ── Sub-categories of "tasks" ─────────────────────────────────────────────

    "task_run": CategoryNode(
        description="Create, start, complete, block, cancel, or stop tasks",
        tools=(
            "task_create",
            "task_update",
            "task_start",
            "task_complete",
            "task_block",
            "task_cancel",
            "task_next",
            "TaskOutput",
            "TaskStop",
        ),
    ),
    "task_query": CategoryNode(
        description="List and look up existing tasks",
        tools=(
            "task_list",
            "task_get",
        ),
    ),
    "planning": CategoryNode(
        description="Plans, todo lists, and plan-mode toggles",
        tools=(
            "plan_get",
            "update_plan",
            "plan_clear",
            "EnterPlanMode",
            "ExitPlanMode",
            "todo_write",
        ),
    ),

    # ── Sub-categories of "remote" ────────────────────────────────────────────

    "connections": CategoryNode(
        description="Remote profile connections and workflow triggers",
        tools=(
            "remote_status",
            "remote_list_profiles",
            "remote_connect",
            "remote_disconnect",
            "remote_trigger",
        ),
    ),
    "worktrees": CategoryNode(
        description="Git worktree navigation",
        tools=(
            "worktree_status",
            "worktree_enter",
            "worktree_exit",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Root listing
#
# The 9 names shown when explore_tools() is called with no arguments.
# To add a new top-level category: define it in _CATEGORIES above, then
# append its key here. To promote a sub-category to top-level, move its
# name here and remove it from its parent's `subcategories` tuple.
# ---------------------------------------------------------------------------

_ROOT: tuple[str, ...] = (
    "web",
    "notebooks",
    "code_intel",
    "mcp",
    "tasks",
    "team",
    "remote",
    "account_config",
    "orchestration",
)


# ---------------------------------------------------------------------------
# Output formatters (internal helpers)
# ---------------------------------------------------------------------------

def _fmt_root() -> str:
    lines = [
        "Discoverable tool categories — call explore_tools(category=<name>) to browse:\n",
    ]
    for name in _ROOT:
        node = _CATEGORIES.get(name)
        if node is None:
            continue
        suffix = "  [has sub-categories]" if not node.is_terminal else ""
        lines.append(f"  {name:<18} — {node.description}{suffix}")
    lines.append(
        "\nOnce you find the tool you need, call:\n"
        '  use_discovered_tool(tool_name="<name>", arguments_json=\'{"arg": "value"}\')'
    )
    return "\n".join(lines)


def _fmt_category(name: str, node: CategoryNode) -> str:
    if node.is_terminal:
        lines = [f"Category: {name} — {node.description}\n"]
        for tool_name in node.tools:
            meta = _TOOL_META.get(tool_name)
            if meta is None:
                lines.append(f"  {tool_name}  (no metadata — check _TOOL_META)")
                continue
            lines.append(f"  {tool_name}")
            lines.append(f"    {meta.description}")
            lines.append(f"    args: {meta.args}")
            example_args = _example_args_json(tool_name, meta)
            lines.append(
                f'    call: use_discovered_tool(tool_name="{tool_name}",'
                f' arguments_json=\'{example_args}\')'
            )
            lines.append("")
        return "\n".join(lines).rstrip()
    else:
        lines = [
            f"Category: {name} — {node.description}",
            "",
            "Sub-categories — call explore_tools(category=<name>) to browse further:",
            "",
        ]
        for sub_name in node.subcategories:
            sub = _CATEGORIES.get(sub_name)
            desc = sub.description if sub else "(undefined)"
            lines.append(f"  {sub_name:<18} — {desc}")
        return "\n".join(lines)


def _example_args_json(tool_name: str, meta: ToolMeta) -> str:
    """Return a minimal JSON example string for the tool's arguments."""
    if meta.args == "(no arguments)":
        return "{}"
    # Extract the first required argument name (before "?" or ",")
    first = meta.args.split(",")[0].strip()
    arg_name = first.split(":")[0].strip().rstrip("?")
    # Pick a placeholder value based on the type hint
    arg_type = first.split(":", 1)[1].strip().lower() if ":" in first else "str"
    placeholder: Any
    if "int" in arg_type:
        placeholder = 0
    elif "float" in arg_type or "seconds" in arg_name:
        placeholder = 1.0
    elif "bool" in arg_type:
        placeholder = True
    elif "list" in arg_type or "dict" in arg_type:
        placeholder = []
    else:
        placeholder = "..."
    return json.dumps({arg_name: placeholder})


def _resolve_category(raw: str) -> tuple[str, CategoryNode] | None:
    """Look up a category by exact name, then by prefix, then by substring."""
    raw = raw.strip().lower()
    if raw in _CATEGORIES:
        return raw, _CATEGORIES[raw]
    # Prefix match
    for key, node in _CATEGORIES.items():
        if key.startswith(raw):
            return key, node
    # Substring match
    for key, node in _CATEGORIES.items():
        if raw in key or raw in node.description.lower():
            return key, node
    return None


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_explore_tools() -> AgentTool:
    """Return the explore_tools AgentTool for inclusion in a coding registry.

    Call with no arguments (or category="root") to list top-level categories.
    Call with a category name to see its tools or sub-categories.
    """
    def _handler(arguments: dict[str, Any], _ctx: ToolExecutionContext) -> str:
        raw = str(arguments.get("category") or "").strip()

        if not raw or raw.lower() in ("root", "all", "list"):
            return _fmt_root()

        result = _resolve_category(raw)
        if result is None:
            return (
                f'Category "{raw}" not found.\n\n'
                + _fmt_root()
            )
        cat_name, node = result
        return _fmt_category(cat_name, node)

    return AgentTool(
        name="explore_tools",
        description=(
            "Browse the tool directory. "
            "Call with no arguments to see top-level categories, "
            'or with category="<name>" to see tools or sub-categories inside it. '
            "Use use_discovered_tool() to execute a tool once you know its name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Category name to browse (e.g. \"web\", \"tasks\", \"mcp\"). "
                        "Omit or pass \"root\" to see all top-level categories."
                    ),
                },
            },
        },
        handler=_handler,
    )


def make_use_discovered_tool(full_registry: dict[str, AgentTool]) -> AgentTool:
    """Return the use_discovered_tool AgentTool for inclusion in a coding registry.

    Executes any tool from full_registry by name. The caller passes tool
    arguments as a JSON-encoded object string (arguments_json). This lets the
    model call any tool without those tools appearing in the top-level spec.

    full_registry should be the output of default_tool_registry() so that
    every tool handler has access to its usual ToolExecutionContext.
    """
    def _handler(arguments: dict[str, Any], ctx: ToolExecutionContext) -> str:
        tool_name = str(arguments.get("tool_name") or "").strip()
        raw_args = str(arguments.get("arguments_json") or "{}").strip()

        if not tool_name:
            return (
                'Error: tool_name is required. '
                'Call explore_tools() to browse available tools.'
            )

        tool = full_registry.get(tool_name)
        if tool is None:
            # Suggest categories the model could explore
            suggestions = ", ".join(f'"{n}"' for n in _ROOT[:5])
            return (
                f'Tool "{tool_name}" not found in the registry.\n'
                f'Call explore_tools() to browse. Top-level categories: {suggestions}, …\n'
                f'For shell commands, use the bash tool directly.'
            )

        try:
            tool_args: dict[str, Any] = {}
            if raw_args and raw_args != "{}":
                parsed = json.loads(raw_args)
                if not isinstance(parsed, dict):
                    raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
                tool_args = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            meta = _TOOL_META.get(tool_name)
            hint = f" Expected args: {meta.args}" if meta else ""
            return (
                f'Invalid arguments_json: {exc}.{hint}\n'
                f'Pass a JSON object string, e.g. arguments_json=\'{_example_args_json(tool_name, meta) if meta else "{}"}\''
            )

        result = tool.execute(tool_args, ctx)
        return result.content

    return AgentTool(
        name="use_discovered_tool",
        description=(
            "Execute any tool by name after finding it with explore_tools(). "
            "Pass the tool name and its arguments as a JSON object string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Exact tool name from explore_tools output.",
                },
                "arguments_json": {
                    "type": "string",
                    "description": (
                        'Tool arguments as a JSON object string, '
                        'e.g. \'{"url": "https://..."}\'. '
                        'Pass \'{}\'  for tools that take no arguments.'
                    ),
                },
            },
            "required": ["tool_name"],
        },
        handler=_handler,
    )
