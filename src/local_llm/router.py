from __future__ import annotations

from typing import Literal

from ..agent_types import ModelConfig
from ..openai_compat import OpenAICompatClient, OpenAICompatError

_SYSTEM_PROMPT = (
    'You are a routing classifier. Reply with exactly one word: coding or planning.\n'
    'Reply coding ONLY if ALL three conditions are true:\n'
    '  1. The exact file to change is explicitly named in the request\n'
    '  2. The exact change to make is fully specified — no discovery, reading, or analysis needed first\n'
    '  3. It can be done in a single self-contained edit to that one file\n'
    'Reply planning for everything else. This includes ANY request that:\n'
    '  - Mentions "increase", "improve", "add", "enhance", or "fix" something without naming the exact file\n'
    '  - Spans multiple files or requires discovering which files to touch\n'
    '  - Requires running a command, reading output, or analysis before implementing\n'
    '  - Is vague, broad, or describes a goal rather than a specific edit\n'
    'Examples that are ALWAYS planning (not coding):\n'
    '  "increase test coverage", "improve performance", "refactor auth", '
    '  "add error handling", "fix the failing tests", "add logging throughout the app"\n'
    'Examples that could be coding:\n'
    '  "add a null check in validateUser() in src/auth.ts", '
    '  "rename field userId to user_id in src/types.ts"\n'
    'When in doubt, reply planning.'
)

RouteDecision = Literal['coding', 'planning']


def route_request(prompt: str, selection_model: ModelConfig) -> RouteDecision:
    client = OpenAICompatClient(selection_model)
    try:
        turn = client.complete(
            messages=[
                {'role': 'system', 'content': _SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ],
            tools=[],
        )
    except OpenAICompatError:
        return 'planning'

    response = turn.content.strip().lower()
    return 'coding' if response == 'coding' else 'planning'
