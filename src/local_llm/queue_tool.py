from __future__ import annotations

from typing import Any

from ..agent_tools import AgentTool, ToolExecutionContext
from .task_queue import TaskQueue

_PARAMETERS = {
    'type': 'object',
    'properties': {
        'description': {
            'type': 'string',
            'description': 'What the task should accomplish.',
        },
        'input_files': {
            'type': 'string',
            'description': 'Comma-separated relative paths of files this task needs as input.',
        },
        'output_file': {
            'type': 'string',
            'description': 'Relative path where this task should write its result.',
        },
        'context': {
            'type': 'string',
            'description': 'Short context string (max ~500 chars) needed by the task executor.',
        },
    },
    'required': ['description'],
}


def make_queue_task_tool(
    queue: TaskQueue,
    session_id: str,
    max_tasks_per_session: int,
) -> AgentTool:
    def _handler(arguments: dict[str, Any], _ctx: ToolExecutionContext) -> str:
        current_count = queue.count_session_tasks(session_id)
        if current_count >= max_tasks_per_session:
            return f'Error: session task limit ({max_tasks_per_session}) reached'
        description = str(arguments.get('description', ''))
        input_files = str(arguments.get('input_files', ''))
        output_file = str(arguments.get('output_file', ''))
        context = str(arguments.get('context', ''))[:500]
        task_id = queue.add_task(
            session_id,
            description,
            'coding',
            input_files=input_files,
            output_file=output_file,
            context=context,
        )
        return f'Task {task_id} queued: {description[:80]}'

    return AgentTool(
        name='queue_task',
        description=(
            'Add a task to the execution queue. The planning model uses this to '
            'decompose a request into ordered steps that will be executed sequentially.'
        ),
        parameters=_PARAMETERS,
        handler=_handler,
    )
