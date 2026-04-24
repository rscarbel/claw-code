from __future__ import annotations

from typing import Literal

from ..agent_types import ModelConfig
from ..openai_compat import OpenAICompatClient, OpenAICompatError

_SYSTEM_PROMPT = (
    'You are a routing classifier. Reply with exactly one word: coding or planning. '
    'Reply coding only if the request is fully self-contained: the exact target is '
    'specified, no discovery or file reading is needed first, and it can be completed '
    'in a single model call. Reply planning for everything else.'
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
