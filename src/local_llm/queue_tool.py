from __future__ import annotations

import json
import re
from typing import Any

from ..agent_tools import AgentTool, ToolExecutionContext
from .task_queue import TaskQueue

_PARAMETERS = {
    'type': 'object',
    'properties': {
        'description': {
            'type': 'string',
            'description': (
                'What the task should accomplish. Required. '
                'Write as plain text: "Run `<cmd>`" for shell commands, '
                'or a sentence for code changes. '
                'This MUST be the "description" key — do NOT use "command" or "task".'
            ),
        },
        'task_type': {
            'type': 'string',
            'enum': ['coding', 'discovery'],
            'description': (
                'Use "discovery" when this task runs a command whose output is needed to '
                'plan the implementation steps — e.g. running a test suite to see coverage '
                'gaps, a linter to find type errors, or a build to see compilation failures. '
                'After a discovery task completes, its output_file triggers an automatic '
                're-planning step that queues the actual implementation tasks. '
                'Use "coding" (default) for all other tasks.'
            ),
        },
        'input_files': {
            'type': 'string',
            'description': 'Comma-separated relative paths of files this task needs as input.',
        },
        'output_file': {
            'type': 'string',
            'description': (
                'Relative path where this task should write its result. '
                'Required for discovery tasks — the re-planner reads this file.'
            ),
        },
        'context': {
            'type': 'string',
            'description': 'Short context string (max ~500 chars) needed by the task executor.',
        },
    },
    'required': ['description'],
}

_VALID_TASK_TYPES = frozenset({'coding', 'discovery'})

# Matches description: "..." or description: '...' in free-form text
_NESTED_DESC_RE = re.compile(r'["\']?description["\']?\s*:\s*["\']([^"\']{3,})["\']')
# Matches command: "..." or command: '...'
_NESTED_CMD_RE = re.compile(r'["\']?command["\']?\s*:\s*["\']([^"\']{3,})["\']')

_USAGE_HINT = (
    'Error: "description" is required and must be a non-empty string. '
    'Correct usage: queue_task(description="Run `make test`", task_type="coding"). '
    'Do NOT use "command", "task", or other field names in place of "description".'
)


def _recover_description(arguments: dict[str, Any]) -> str:
    """Try to recover a description when the model used the wrong field name."""
    # Check common alternate field names the model might use
    for key in ('task', 'name', 'title', 'summary'):
        val = str(arguments.get(key) or '').strip()
        if val and len(val) < 600:
            return val

    # Model sometimes puts the full spec as a JSON string in 'command'
    raw = str(arguments.get('command') or '').strip()
    if not raw:
        return ''

    # Try JSON parse first (covers valid JSON payloads)
    try:
        nested = json.loads(raw)
        if isinstance(nested, dict):
            for key in ('description', 'task', 'name', 'title'):
                val = str(nested.get(key) or '').strip()
                if val:
                    return val
            cmd = str(nested.get('command') or '').strip()
            if cmd:
                return f'Run `{cmd}`'
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex: handle unquoted-key JSON-like blobs (e.g. {description: "..."})
    m = _NESTED_DESC_RE.search(raw)
    if m:
        return m.group(1).strip()

    m = _NESTED_CMD_RE.search(raw)
    if m:
        cmd = m.group(1).strip()
        return f'Run `{cmd}`' if 'Run `' not in cmd else cmd

    # Last resort: if the raw value itself looks like a plain description, use it
    if not raw.startswith('{') and '\n' not in raw and len(raw) < 400:
        return raw

    return ''


def make_queue_task_tool(
    queue: TaskQueue,
    session_id: str,
    max_tasks_per_session: int,
) -> AgentTool:
    def _handler(arguments: dict[str, Any], _ctx: ToolExecutionContext) -> str:
        current_count = queue.count_session_tasks(session_id)
        if current_count >= max_tasks_per_session:
            return f'Error: session task limit ({max_tasks_per_session}) reached'

        description = str(arguments.get('description') or '').strip()
        if not description:
            description = _recover_description(arguments)
        if not description:
            return _USAGE_HINT

        task_type = str(arguments.get('task_type', 'coding'))
        if task_type not in _VALID_TASK_TYPES:
            task_type = 'coding'
        input_files = str(arguments.get('input_files', ''))
        output_file = str(arguments.get('output_file', ''))
        context = str(arguments.get('context', ''))[:500]

        # Reject duplicate discovery tasks — the model sometimes re-queues the same
        # discovery command after a failed read_file attempt on the not-yet-created output.
        if task_type == 'discovery' and output_file:
            existing_id = queue.find_by_output_file(session_id, output_file)
            if existing_id is not None:
                return (
                    f'A task already writing to {output_file!r} is queued (task {existing_id}). '
                    f'Do not queue it again — it will execute automatically. '
                    f'Queue implementation (coding) tasks instead, or use a different output_file.'
                )

        task_id = queue.add_task(
            session_id,
            description,
            task_type,
            input_files=input_files,
            output_file=output_file,
            context=context,
        )
        msg = f'Task {task_id} queued ({task_type}): {description[:80]}'
        if output_file:
            msg += f'\nNote: {output_file!r} will be written when this task executes — do not read it during planning.'
        return msg

    return AgentTool(
        name='queue_task',
        description=(
            'Add a task to the execution queue. The planning model uses this to '
            'decompose a request into ordered steps that will be executed sequentially. '
            'Use task_type="discovery" for tasks whose shell output is needed to plan '
            'the implementation (the output_file will trigger automatic re-planning).'
        ),
        parameters=_PARAMETERS,
        handler=_handler,
    )
