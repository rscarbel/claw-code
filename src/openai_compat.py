from __future__ import annotations

import json
import re
from typing import Any, Iterator
from urllib import error, request

_THINK_RE = re.compile(r'<(?:ant)?[Tt]hink(?:ing)?>.*?</(?:ant)?[Tt]hink(?:ing)?>', re.DOTALL)
_TEXT_XML_TAG_RE = re.compile(r'<([a-zA-Z_]\w*)>(.*?)</\1>', re.DOTALL)
# Qwen3 <tool_call>{"name": "...", "arguments": {...}}</tool_call> format
_TOOL_CALL_TAG_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
# <tool_code>tool_name\nkey: value\n</tool_code> (Gemini-style fallback from some Qwen3 configs)
_TOOL_CODE_TAG_RE = re.compile(r'<tool_code>\s*([a-zA-Z_]\w*)\s*(.*?)</tool_code>', re.DOTALL)
# Qwen3 native tool call format that leaks through Ollama as special-token text:
# tool_name<｜fe_NNN>param_name<｜fe_NNN><｜fe_NNN>value<｜fe_NNN>...
_SPECIAL_TOKEN_TOOL_RE = re.compile(
    r'^([a-zA-Z_]\w*)<[|｜]fe_\d+>(\w+)<[|｜]fe_\d+><[|｜]fe_\d+>([^<|｜]+?)<[|｜]',
    re.DOTALL,
)
# <tool_name<argkey>param<argkey>value format (Qwen3 via some Ollama versions)
_ARGKEY_RE = re.compile(r'^<([a-zA-Z_]\w*)<argkey>(.*)', re.DOTALL)
# Claude-style: tool_name<invoke>\n<parameter name="key">val</parameter>\n</invoke>
_CLAUDE_INVOKE_RE = re.compile(
    r'\b([a-zA-Z_]\w*)\s*<invoke>(.*?)</invoke>', re.DOTALL
)
_CLAUDE_PARAM_RE = re.compile(r'<parameter\s+name=["\'](\w+)["\']>(.*?)</parameter>', re.DOTALL)
# YAML-like: tool_name\n[blank lines]\nkey: value\nkey2: value2
_YAML_TOOL_RE = re.compile(r'^([a-zA-Z_]\w*)\n+(\w+):\s*(.+)', re.DOTALL)
# bash: <cmd> anywhere on a line (not just at start of content)
_BASH_LINE_RE = re.compile(r'(?:^|\n)\s*bash:\s*(.+?)(?:\n|$)', re.IGNORECASE)
# Space-separated key=value: tool_name key=value (e.g. "bash command=nyc npm test")
_KV_SPACE_TOOL_RE = re.compile(r'^([a-zA-Z_]\w*)\s+(\w+)=(.+)$', re.DOTALL)
# bash shorthand: "bash <cmd>" with no key= prefix (e.g. "bash pwd")
_BASH_SHORTHAND_RE = re.compile(r'^bash\s+(?!\w+=)(.+)$', re.IGNORECASE | re.DOTALL)
# Markdown bash/sh code block: ```bash\ncommand\n```
_MARKDOWN_BASH_RE = re.compile(r'```(?:bash|sh|shell)\s*\n(.*?)(?:\n```|$)', re.DOTALL | re.IGNORECASE)
# Python-style function call: tool_name(key="value", ...)
_PYFUNC_RE = re.compile(r'\b([a-zA-Z_]\w*)\s*\(([^)]*)\)', re.DOTALL)
_PYFUNC_KV_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+))')


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub('', text).lstrip()


def _parse_pyfunc_args(args_str: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for m in _PYFUNC_KV_RE.finditer(args_str):
        val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4) or '')
        params[m.group(1)] = val
    if not params and args_str.strip():
        params = {'command': args_str.strip().strip('"\'').strip()}
    return params


def _extract_text_format_tool_calls(
    content: str,
    known_tool_names: set[str],
) -> list[Any]:  # list[ToolCall] — forward ref avoids circular import
    """Parse tool calls emitted as text by models that don't use the OpenAI tool call API."""
    if not content or not known_tool_names:
        return []
    stripped = content.strip()

    # 1. Qwen3 <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    tc_m = _TOOL_CALL_TAG_RE.search(stripped)
    if tc_m:
        try:
            obj = json.loads(tc_m.group(1).strip())
            if isinstance(obj, dict):
                name = str(obj.get('name') or '')
                if name in known_tool_names:
                    args = obj.get('arguments') or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {'command': args}
                    return [{'_text_tool': True, 'name': name, 'arguments': args}]
        except (json.JSONDecodeError, AttributeError):
            pass

    # 2. XML: search for any known tool name as an XML tag (handles nesting like <function_calls>)
    name_alt = '|'.join(re.escape(n) for n in sorted(known_tool_names, key=len, reverse=True))
    tool_xml_re = re.compile(rf'<({name_alt})>(.*?)</\1>', re.DOTALL)
    xml_m = tool_xml_re.search(stripped)
    if xml_m:
        name = xml_m.group(1)
        body = xml_m.group(2)
        params: dict[str, str] = {}
        for pm in _TEXT_XML_TAG_RE.finditer(body):
            params[pm.group(1)] = pm.group(2).strip()
        if not params and body.strip():
            params = {'command': body.strip()}
        return [{'_text_tool': True, 'name': name, 'arguments': params}]

    # 3. JSON action format: {"action": "tool_name", "key": "val"}
    json_start = stripped.find('{')
    if json_start != -1:
        try:
            obj = json.loads(stripped[json_start:])
            if isinstance(obj, dict):
                name = str(obj.get('action') or obj.get('tool') or obj.get('name') or '')
                if name in known_tool_names:
                    args = {k: v for k, v in obj.items() if k not in ('action', 'tool', 'name')}
                    return [{'_text_tool': True, 'name': name, 'arguments': args}]
        except json.JSONDecodeError:
            pass

    # 4. bash: <command> on any line
    bash_m = _BASH_LINE_RE.search(stripped)
    if bash_m and 'bash' in known_tool_names:
        cmd = bash_m.group(1).strip()
        if cmd:
            return [{'_text_tool': True, 'name': 'bash', 'arguments': {'command': cmd}}]

    # 4b. Markdown bash code block: ```bash\ncommand\n```
    md_m = _MARKDOWN_BASH_RE.search(stripped)
    if md_m and 'bash' in known_tool_names:
        cmd = md_m.group(1).strip()
        if cmd:
            return [{'_text_tool': True, 'name': 'bash', 'arguments': {'command': cmd}}]

    # 4c. Space-separated key=value: "bash command=nyc npm test" or "tool_name key=value"
    kv_m = _KV_SPACE_TOOL_RE.match(stripped)
    if kv_m:
        name = kv_m.group(1)
        if name in known_tool_names:
            key, value = kv_m.group(2), kv_m.group(3).strip()
            return [{'_text_tool': True, 'name': name, 'arguments': {key: value}}]

    # 4d. bash shorthand: "bash <cmd>" with no key= prefix (e.g. "bash pwd")
    bs_m = _BASH_SHORTHAND_RE.match(stripped)
    if bs_m and 'bash' in known_tool_names:
        cmd = bs_m.group(1).strip()
        if cmd:
            return [{'_text_tool': True, 'name': 'bash', 'arguments': {'command': cmd}}]

    # 5. Python-style function call: tool_name(key="value", ...)
    for m in _PYFUNC_RE.finditer(stripped):
        name = m.group(1)
        if name in known_tool_names:
            args_str = m.group(2).strip()
            return [{'_text_tool': True, 'name': name, 'arguments': _parse_pyfunc_args(args_str)}]

    # 6. <tool_code>tool_name\nkey: value\n</tool_code> (Gemini-style)
    tc_code_m = _TOOL_CODE_TAG_RE.search(stripped)
    if tc_code_m:
        name = tc_code_m.group(1)
        if name in known_tool_names:
            body = tc_code_m.group(2).strip()
            params: dict[str, str] = {}
            for line in body.splitlines():
                if ':' in line:
                    k, _, v = line.partition(':')
                    params[k.strip()] = v.strip()
            if not params and body:
                params = {'command': body}
            return [{'_text_tool': True, 'name': name, 'arguments': params}]

    # 7. <tool_name<argkey>param<argkey>value (Qwen3 via certain Ollama versions)
    ak_m = _ARGKEY_RE.match(stripped)
    if ak_m:
        name = ak_m.group(1)
        if name in known_tool_names:
            parts = ak_m.group(2).split('<argkey>')
            args: dict[str, str] = {}
            for i in range(0, len(parts) - 1, 2):
                key = parts[i].strip()
                val = parts[i + 1].strip() if i + 1 < len(parts) else ''
                if key:
                    args[key] = val
            return [{'_text_tool': True, 'name': name, 'arguments': args}]

    # 8. Claude-style: tool_name<invoke><parameter name="key">val</parameter></invoke>
    ci_m = _CLAUDE_INVOKE_RE.search(stripped)
    if ci_m:
        name = ci_m.group(1)
        if name in known_tool_names:
            body = ci_m.group(2)
            params = {m.group(1): m.group(2).strip() for m in _CLAUDE_PARAM_RE.finditer(body)}
            return [{'_text_tool': True, 'name': name, 'arguments': params}]

    # 9. YAML-like: tool_name\nkey: value
    yl_m = _YAML_TOOL_RE.match(stripped)
    if yl_m:
        name = yl_m.group(1)
        if name in known_tool_names:
            params = {}
            for line in stripped.splitlines()[1:]:
                if ':' in line:
                    k, _, v = line.partition(':')
                    params[k.strip()] = v.strip()
            if params:
                return [{'_text_tool': True, 'name': name, 'arguments': params}]

    # 9b. tool_name\n{JSON args} — model writes the tool name on one line then a JSON object
    if '\n' in stripped:
        first_line, _, rest = stripped.partition('\n')
        first_line = first_line.strip()
        if first_line in known_tool_names:
            rest = rest.strip()
            if rest.startswith('{'):
                try:
                    args = json.loads(rest)
                    if isinstance(args, dict):
                        return [{'_text_tool': True, 'name': first_line, 'arguments': args}]
                except json.JSONDecodeError:
                    pass

    # 10. Qwen3 native special-token format leaked as text
    st = _SPECIAL_TOKEN_TOOL_RE.match(stripped)
    if st:
        name, param, value = st.group(1), st.group(2), st.group(3).strip()
        if name in known_tool_names:
            return [{'_text_tool': True, 'name': name, 'arguments': {param: value}}]

    return []

from .agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
    UsageStats,
)


class OpenAICompatError(RuntimeError):
    """Raised when the local OpenAI-compatible backend returns an invalid response."""


def _join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip('/')
    return f'{base}/{suffix.lstrip("/")}'


def _normalize_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get('type') == 'text' and isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            if isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            parts.append(json.dumps(item, ensure_ascii=True))
        return ''.join(parts)
    return str(content)


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        raw_arguments = raw_arguments.strip()
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise OpenAICompatError(
                f'Invalid tool arguments returned by model: {raw_arguments!r}'
            ) from exc
        if not isinstance(parsed, dict):
            raise OpenAICompatError(
                f'Tool arguments must decode to an object, got {type(parsed).__name__}'
            )
        return parsed
    raise OpenAICompatError(
        f'Unsupported tool arguments payload: {type(raw_arguments).__name__}'
    )


def _optional_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _parse_usage(payload: Any) -> UsageStats:
    if not isinstance(payload, dict):
        return UsageStats()
    completion_details = payload.get('completion_tokens_details')
    if not isinstance(completion_details, dict):
        completion_details = {}
    return UsageStats(
        input_tokens=(
            _optional_int(payload.get('input_tokens'))
            or _optional_int(payload.get('prompt_tokens'))
            or _optional_int(payload.get('prompt_eval_count'))
        ),
        output_tokens=(
            _optional_int(payload.get('output_tokens'))
            or _optional_int(payload.get('completion_tokens'))
            or _optional_int(payload.get('eval_count'))
        ),
        cache_creation_input_tokens=_optional_int(
            payload.get('cache_creation_input_tokens')
        ),
        cache_read_input_tokens=_optional_int(payload.get('cache_read_input_tokens')),
        reasoning_tokens=(
            _optional_int(payload.get('reasoning_tokens'))
            or _optional_int(completion_details.get('reasoning_tokens'))
        ),
    )


def _build_response_format(
    schema: OutputSchemaConfig | None,
) -> dict[str, Any] | None:
    if schema is None:
        return None
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': schema.name,
            'schema': schema.schema,
            'strict': schema.strict,
        },
    }


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client for local model servers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        payload = self._request_json(
            self._build_payload(
                messages=messages,
                tools=tools,
                stream=False,
                output_schema=output_schema,
            )
        )
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatError('Local model backend returned no choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatError('Local model backend returned malformed choice data')

        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise OpenAICompatError('Local model backend returned no assistant message')

        content = _strip_thinking(_normalize_content(message.get('content')))
        tool_calls = self._parse_tool_calls_from_message(message)

        # Fallback: parse text-format tool calls for models (e.g. qwen3.6 via Ollama) that
        # emit XML/JSON/prefix tool calls as content instead of API-level tool_calls.
        if not tool_calls and content and tools:
            known_names = {
                t.get('function', {}).get('name', '')
                for t in tools
                if isinstance(t, dict)
            } - {''}
            text_calls = _extract_text_format_tool_calls(content, known_names)
            if text_calls:
                tool_calls = [
                    ToolCall(
                        id=f'text_call_{i}',
                        name=tc['name'],
                        arguments=tc['arguments'],
                    )
                    for i, tc in enumerate(text_calls)
                ]
                content = ''  # consumed by tool calls

        finish_reason = first_choice.get('finish_reason')
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        return AssistantTurn(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            raw_message=message,
            usage=_parse_usage(payload.get('usage')),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=True,
            output_schema=output_schema,
        )
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                yield StreamEvent(type='message_start')
                for event_payload in self._iter_sse_payloads(response):
                    yield from self._parse_stream_payload(event_payload)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(
                f'HTTP {exc.code} from local model backend: {detail}'
            ) from exc
        except error.URLError as exc:
            raise OpenAICompatError(
                f'Unable to reach local model backend at {self.config.base_url}: {exc.reason}'
            ) from exc

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode('utf-8')
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=body,
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(
                f'HTTP {exc.code} from local model backend: {detail}'
            ) from exc
        except error.URLError as exc:
            raise OpenAICompatError(
                f'Unable to reach local model backend at {self.config.base_url}: {exc.reason}'
            ) from exc

        try:
            payload = json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise OpenAICompatError('Local model backend returned invalid JSON') from exc
        if not isinstance(payload, dict):
            raise OpenAICompatError('Local model backend returned malformed JSON payload')
        return payload

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool,
        output_schema: OutputSchemaConfig | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'temperature': self.config.temperature,
            'stream': stream,
        }
        if stream:
            payload['stream_options'] = {'include_usage': True}
        response_format = _build_response_format(output_schema)
        if response_format is not None:
            payload['response_format'] = response_format
        # Qwen3 models default to thinking mode. For non-streaming requests (e.g. review)
        # the full response must buffer before any bytes arrive, so thinking causes
        # timeouts — suppress it. For streaming requests thinking tokens arrive
        # incrementally (no timeout) and are required for proper function-call emission
        # on qwen3.6: sending think:false causes the model to output Gemini-style
        # <tool_code> text blocks instead of OpenAI API-level tool calls.
        model_lower = self.config.model.lower()
        if 'qwen3' in model_lower and 'qwen2.5' not in model_lower and not stream:
            payload['think'] = False
        if self.config.num_ctx > 0:
            payload.setdefault('options', {})['num_ctx'] = self.config.num_ctx
        return payload

    def _parse_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get('tool_calls')
        if isinstance(raw_tool_calls, list):
            for idx, raw_call in enumerate(raw_tool_calls):
                if not isinstance(raw_call, dict):
                    raise OpenAICompatError('Malformed tool call payload from model')
                function_block = raw_call.get('function') or {}
                if not isinstance(function_block, dict):
                    raise OpenAICompatError('Malformed tool call function payload from model')
                name = function_block.get('name')
                if not isinstance(name, str) or not name:
                    raise OpenAICompatError('Tool call missing function name')
                call_id = raw_call.get('id')
                if not isinstance(call_id, str) or not call_id:
                    call_id = f'call_{idx}'
                arguments = _parse_tool_arguments(function_block.get('arguments'))
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        elif isinstance(message.get('function_call'), dict):
            function_call = message['function_call']
            name = function_call.get('name')
            if not isinstance(name, str) or not name:
                raise OpenAICompatError('Function call missing name')
            arguments = _parse_tool_arguments(function_call.get('arguments'))
            tool_calls.append(ToolCall(id='call_0', name=name, arguments=arguments))
        return tool_calls

    def _iter_sse_payloads(self, response: Any) -> Iterator[dict[str, Any]]:
        buffer: list[str] = []
        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode('utf-8', errors='replace')
            else:
                text = str(line)
            stripped = text.strip()
            if not stripped:
                if not buffer:
                    continue
                joined = '\n'.join(buffer)
                buffer.clear()
                if joined == '[DONE]':
                    break
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed SSE payload from model backend')
                yield payload
                continue
            if stripped.startswith('data:'):
                buffer.append(stripped[5:].strip())

        if buffer:
            joined = '\n'.join(buffer)
            if joined != '[DONE]':
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid trailing JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed trailing SSE payload from model backend')
                yield payload

    def _parse_stream_payload(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        usage = _parse_usage(payload.get('usage'))
        if usage.total_tokens:
            yield StreamEvent(
                type='usage',
                usage=usage,
                raw_event=payload,
            )

        choices = payload.get('choices')
        if not isinstance(choices, list):
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                delta = {}
            content = delta.get('content')
            if isinstance(content, str) and content:
                yield StreamEvent(
                    type='content_delta',
                    delta=content,
                    raw_event=choice,
                )
            tool_calls = delta.get('tool_calls')
            if isinstance(tool_calls, list):
                for raw_tool_call in tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    function_block = raw_tool_call.get('function')
                    if not isinstance(function_block, dict):
                        function_block = {}
                    yield StreamEvent(
                        type='tool_call_delta',
                        tool_call_index=(
                            raw_tool_call.get('index')
                            if isinstance(raw_tool_call.get('index'), int)
                            else 0
                        ),
                        tool_call_id=(
                            raw_tool_call.get('id')
                            if isinstance(raw_tool_call.get('id'), str)
                            else None
                        ),
                        tool_name=(
                            function_block.get('name')
                            if isinstance(function_block.get('name'), str)
                            else None
                        ),
                        arguments_delta=(
                            function_block.get('arguments')
                            if isinstance(function_block.get('arguments'), str)
                            else ''
                        ),
                        raw_event=raw_tool_call,
                    )
            finish_reason = choice.get('finish_reason')
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    finish_reason = str(finish_reason)
                yield StreamEvent(
                    type='message_stop',
                    finish_reason=finish_reason,
                    raw_event=choice,
                )
