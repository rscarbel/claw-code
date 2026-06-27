from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..agent_types import ModelConfig, ModelPricing

_DEFAULT_BASE_URL = 'http://localhost:11434/v1'
_DEFAULT_API_KEY = 'none'

# Model defaults per role
_DEFAULT_CODING_MODEL = 'qwen3.6:35b-a3b'
_DEFAULT_PLANNING_MODEL = 'qwen3:14b'
_DEFAULT_REVIEW_MODEL = 'phi4-mini'
_DEFAULT_DIAGNOSIS_MODEL = 'qwen3:14b'
_DEFAULT_SELECTION_MODEL = 'gemma3:4b'

# Timeouts: large hybrid-offload models need time to cold-load.
# Small/medium models that fit in VRAM are much faster.
_LARGE_MODEL_TIMEOUT = 720.0   # qwen3.6:35b-a3b, qwen3:30b-a3b (hybrid offload)
_MEDIUM_MODEL_TIMEOUT = 480.0  # qwen3:14b (fits in VRAM; 480s = safety margin over 4K-token cap)
_SMALL_MODEL_TIMEOUT = 120.0   # phi4-mini (fits in VRAM, fast)

# Context windows per role.
# LOCAL_LLM_NUM_CTX env var overrides all; per-model JSON num_ctx overrides its role default.
#
# Coding: 128K — long tool-call chains build up large histories; KV cache spills to RAM
#   (fine with 128GB available).
# Planning: 16K — qwen3:14b has 9.3 GB weights; only ~2.7 GB VRAM left for KV cache.
#   KV cache at 32K (~5.2 GB) overflows VRAM → partial CPU offload → 8–12 tok/s → timeout.
#   16K keeps KV cache at ~2.6 GB, staying within VRAM for full GPU inference (~50 tok/s).
#   16K also fits system prompt + tool defs + planning prompt + multi-turn file exploration.
# Review: 16K — prompts are tiny (~4K max); generous headroom; phi4-mini fits KV in VRAM.
# Diagnosis: 8K — single-turn, small prompts; no multi-turn expansion needed.
# Selection: no num_ctx — routing prompt is tiny; use server default.
_CODING_NUM_CTX = 131072
_PLANNING_NUM_CTX = 16384
_REVIEW_NUM_CTX = 16384
_DIAGNOSIS_NUM_CTX = 8192


@dataclass(frozen=True)
class LocalLLMConfig:
    coding_model: ModelConfig
    planning_model: ModelConfig
    review_model: ModelConfig
    diagnosis_model: ModelConfig
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
    default_model: str,
    *,
    openai_fallback: bool = False,
    timeout_seconds: float | None = None,
    num_ctx: int = 0,
) -> ModelConfig:
    # openai_fallback=True for coding_model only: it's the primary model and may
    # be configured via generic OPENAI_* vars by users who run a single local endpoint.
    default_base_url = os.environ.get('OPENAI_BASE_URL', _DEFAULT_BASE_URL) if openai_fallback else _DEFAULT_BASE_URL
    default_api_key = os.environ.get('OPENAI_API_KEY', _DEFAULT_API_KEY) if openai_fallback else _DEFAULT_API_KEY
    if openai_fallback:
        default_model = os.environ.get('OPENAI_MODEL', default_model)

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

    # Global num_ctx override: LOCAL_LLM_NUM_CTX or top-level JSON num_ctx.
    # If set, it overrides all per-role defaults. Per-model JSON num_ctx still wins.
    global_ctx = _safe_int(os.environ.get('LOCAL_LLM_NUM_CTX', raw.get('num_ctx', 0)), 0)

    coding_model = _build_model_config(
        'CODING_MODEL',
        'CODING_MODEL_BASE_URL',
        'CODING_MODEL_API_KEY',
        raw.get('coding_model', {}),
        _DEFAULT_CODING_MODEL,
        openai_fallback=True,
        timeout_seconds=_LARGE_MODEL_TIMEOUT,
        num_ctx=global_ctx if global_ctx > 0 else _CODING_NUM_CTX,
    )
    planning_model = _build_model_config(
        'PLANNING_AND_ORCHESTRATION_MODEL',
        'PLANNING_AND_ORCHESTRATION_MODEL_BASE_URL',
        'PLANNING_AND_ORCHESTRATION_MODEL_API_KEY',
        raw.get('planning_and_orchestration_model', {}),
        _DEFAULT_PLANNING_MODEL,
        timeout_seconds=_MEDIUM_MODEL_TIMEOUT,
        num_ctx=global_ctx if global_ctx > 0 else _PLANNING_NUM_CTX,
    )
    review_model = _build_model_config(
        'REVIEW_MODEL',
        'REVIEW_MODEL_BASE_URL',
        'REVIEW_MODEL_API_KEY',
        raw.get('review_model', {}),
        _DEFAULT_REVIEW_MODEL,
        timeout_seconds=_SMALL_MODEL_TIMEOUT,
        num_ctx=global_ctx if global_ctx > 0 else _REVIEW_NUM_CTX,
    )
    diagnosis_model = _build_model_config(
        'DIAGNOSIS_MODEL',
        'DIAGNOSIS_MODEL_BASE_URL',
        'DIAGNOSIS_MODEL_API_KEY',
        raw.get('diagnosis_model', {}),
        _DEFAULT_DIAGNOSIS_MODEL,
        timeout_seconds=_MEDIUM_MODEL_TIMEOUT,
        num_ctx=global_ctx if global_ctx > 0 else _DIAGNOSIS_NUM_CTX,
    )
    selection_model = _build_model_config(
        'MODEL_SELECTION_MODEL',
        'MODEL_SELECTION_MODEL_BASE_URL',
        'MODEL_SELECTION_MODEL_API_KEY',
        raw.get('model_selection_model', {}),
        _DEFAULT_SELECTION_MODEL,
    )

    max_tasks = _safe_int(os.environ.get('MAX_TASKS_PER_SESSION', raw.get('max_tasks_per_session')), 50)
    max_reviews = _safe_int(os.environ.get('MAX_REVIEW_LOOPS', raw.get('max_review_loops')), 3)

    return LocalLLMConfig(
        coding_model=coding_model,
        planning_model=planning_model,
        review_model=review_model,
        diagnosis_model=diagnosis_model,
        selection_model=selection_model,
        max_tasks_per_session=max_tasks,
        max_review_loops=max_reviews,
    )
