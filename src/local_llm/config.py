from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..agent_types import ModelConfig, ModelPricing

_DEFAULT_BASE_URL = 'http://localhost:11434/v1'
_DEFAULT_API_KEY = 'none'
_DEFAULT_MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
_MODEL_TIMEOUT = 720.0  # large models need time to cold-load and generate long responses
_DEFAULT_NUM_CTX = 32768  # full tool registry (~65 tools) + system prompt easily exceeds 4096


@dataclass(frozen=True)
class LocalLLMConfig:
    coding_model: ModelConfig
    planning_model: ModelConfig
    selection_model: ModelConfig
    max_tasks_per_session: int = 50
    max_review_loops: int = 3


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _build_model_config(
    model_env: str,
    base_url_env: str,
    api_key_env: str,
    base: dict,
    *,
    openai_fallback: bool = False,
    timeout_seconds: float | None = None,
    num_ctx: int = 0,
) -> ModelConfig:
    # openai_fallback=True for coding_model only: it's the primary model and may
    # be configured via generic OPENAI_* vars by users who run a single local endpoint.
    default_base_url = os.environ.get('OPENAI_BASE_URL', _DEFAULT_BASE_URL) if openai_fallback else _DEFAULT_BASE_URL
    default_api_key = os.environ.get('OPENAI_API_KEY', _DEFAULT_API_KEY) if openai_fallback else _DEFAULT_API_KEY
    default_model = os.environ.get('OPENAI_MODEL', _DEFAULT_MODEL) if openai_fallback else _DEFAULT_MODEL

    model = os.environ.get(model_env) or base.get('model') or default_model
    base_url = os.environ.get(base_url_env) or base.get('base_url') or default_base_url
    api_key = os.environ.get(api_key_env) or base.get('api_key') or default_api_key

    kwargs: dict = dict(model=model, base_url=base_url, api_key=api_key)
    if timeout_seconds is not None:
        kwargs['timeout_seconds'] = timeout_seconds
    resolved_num_ctx = _safe_int(base.get('num_ctx'), num_ctx)
    if resolved_num_ctx > 0:
        kwargs['num_ctx'] = resolved_num_ctx
    return ModelConfig(**kwargs)


def load_local_llm_config(cwd: Path) -> LocalLLMConfig:
    json_path = cwd / '.port_sessions' / 'local_llm_config.json'
    raw: dict = {}
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass

    default_num_ctx = _safe_int(os.environ.get('LOCAL_LLM_NUM_CTX', raw.get('num_ctx')), _DEFAULT_NUM_CTX)

    coding_model = _build_model_config(
        'CODING_MODEL',
        'CODING_MODEL_BASE_URL',
        'CODING_MODEL_API_KEY',
        raw.get('coding_model', {}),
        openai_fallback=True,
        timeout_seconds=_MODEL_TIMEOUT,
        num_ctx=default_num_ctx,
    )
    planning_model = _build_model_config(
        'PLANNING_AND_ORCHESTRATION_MODEL',
        'PLANNING_AND_ORCHESTRATION_MODEL_BASE_URL',
        'PLANNING_AND_ORCHESTRATION_MODEL_API_KEY',
        raw.get('planning_and_orchestration_model', {}),
        timeout_seconds=_MODEL_TIMEOUT,
        num_ctx=default_num_ctx,
    )
    selection_model = _build_model_config(
        'MODEL_SELECTION_MODEL',
        'MODEL_SELECTION_MODEL_BASE_URL',
        'MODEL_SELECTION_MODEL_API_KEY',
        raw.get('model_selection_model', {}),
    )

    max_tasks = _safe_int(os.environ.get('MAX_TASKS_PER_SESSION', raw.get('max_tasks_per_session')), 50)
    max_reviews = _safe_int(os.environ.get('MAX_REVIEW_LOOPS', raw.get('max_review_loops')), 3)

    return LocalLLMConfig(
        coding_model=coding_model,
        planning_model=planning_model,
        selection_model=selection_model,
        max_tasks_per_session=max_tasks,
        max_review_loops=max_reviews,
    )
