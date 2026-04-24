from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from ..agent_runtime import LocalCodingAgent
from ..agent_tools import default_tool_registry
from ..agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from ..openai_compat import OpenAICompatClient, OpenAICompatError, _strip_thinking
from ..session_store import load_agent_session
from .config import LocalLLMConfig
from .queue_tool import make_queue_task_tool
from .router import route_request
from .task_queue import TaskQueue, TaskRecord


def _log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f'[claw-multi {ts}] {msg}', file=sys.stderr, flush=True)

_TOOL_USE_SYSTEM_ADDON = (
    'CRITICAL RULE: Your first response must always be a tool call. '
    'Never output text like "Let me explore..." or "I will..." before calling a tool. '
    'Call read_file, bash, write_file, or another tool immediately. '
    'Text-only responses are treated as task failures.\n'
    'If a shell command fails (non-zero exit code), do NOT stop to narrate or plan. '
    'Immediately call another tool to investigate or recover: check the error output, '
    'locate the executable, try an alternative invocation, or fix the root cause. '
    'Keep calling tools until the task succeeds or is provably impossible.'
)

_TOOL_USE_SYSTEM_ADDON_BASH_FIRST = (
    'CRITICAL RULE: Your FIRST tool call must be bash. '
    'Do NOT call read_file, glob, or any other tool before running bash. '
    'Execute the required shell command immediately — explore files only if the command fails. '
    'Text-only responses are treated as task failures.\n'
    'If the command fails, call another tool immediately to investigate or recover. '
    'Keep calling tools until the task succeeds or is provably impossible.'
)

_TOOL_NUDGE_PROMPT = (
    'You have not used any tools yet. '
    'Call a tool RIGHT NOW — read_file, bash, write_file, or glob. '
    'Do not output any text. Make a tool call immediately.'
)

_OUTPUT_FILE_MISSING_NUDGE = (
    'The required output file {path!r} was not created. '
    'Your task is NOT complete. '
    'Investigate why the file is missing and keep calling tools until it exists.'
)

# Tool names that, if emitted as the sole output, indicate the model tried but failed
# to produce a proper tool call.
_BARE_TOOL_NAMES = frozenset({
    'read_file', 'write_file', 'bash', 'glob', 'list_directory', 'search',
})

# Two patterns for "Run bash immediately with command: `...`":
# group 1 — backtick-quoted (preferred, captures full command including dots and pipes)
# group 2 — unquoted fallback (stops at semicolon/newline only)
# group 3 — Run 'cmd' / Run "cmd" / Run `cmd` shorthand
_SHELL_CMD_RE = re.compile(
    r'(?:'
    r'[Rr]un bash immediately with command[=:\s]+`([^`\n]+)`'  # backtick-quoted
    r'|[Rr]un bash immediately with command[=:\s]+([^\n;]{1,300})'  # unquoted
    r'|[Rr]un\s+[`\'"]([^`\'";\n]+)[`\'"]'                         # "Run 'cmd'" format
    r')',
)

# Matches text-format fake tool calls the model sometimes writes instead of calling the tool
_TEXT_FORMAT_BASH_RE = re.compile(r'\[/?bash\]|```bash', re.IGNORECASE)


def _extract_shell_command(description: str) -> str | None:
    """Extract a specific shell command from a task description, if clearly stated."""
    m = _SHELL_CMD_RE.search(description)
    if m:
        cmd = (m.group(1) or m.group(2) or m.group(3) or '').strip().strip('`').strip()
        return cmd if cmd else None
    return None


def _looks_like_text_format_bash(text: str) -> bool:
    return bool(_TEXT_FORMAT_BASH_RE.search(text))


def _looks_like_bare_tool_name(text: str) -> bool:
    return text.strip() in _BARE_TOOL_NAMES


_TEXT_FORMAT_NUDGE = (
    'You wrote a bash command as a text block instead of calling the bash tool. '
    'Call the bash TOOL now — use the tool interface, do not write code blocks or text.'
)


def _build_nudge_prompt(task: TaskRecord, clean_output: str = '') -> str:
    if _looks_like_text_format_bash(clean_output):
        return _TEXT_FORMAT_NUDGE
    cmd = _extract_shell_command(task.description)
    if cmd:
        return (
            f'You must call bash RIGHT NOW with command: {cmd!r}. '
            f'Do not output any text. Make the bash tool call immediately.'
        )
    return _TOOL_NUDGE_PROMPT

_REVIEW_SYSTEM_PROMPT = (
    'You are a code reviewer. A single step in a multi-step pipeline has just executed. '
    'Evaluate ONLY whether this specific task completed successfully. '
    'Do NOT fail the task because later steps (e.g. running analysis, reporting results) have not been done — other tasks handle those. '
    'Check: does the output match the task description? Are referenced files plausible? Are there any obvious errors? '
    'Reply with exactly PASS or FAIL followed by one sentence of feedback.'
)

_DIAGNOSIS_SYSTEM_PROMPT = (
    'You are diagnosing a failed software engineering task. '
    'Identify the specific root cause of failure and write ONE replacement task description '
    'that directly addresses it. Requirements:\n'
    '- Name the exact shell command, file path, or action needed\n'
    '- Include a concrete verification step (e.g. check a file exists, print output)\n'
    '- If the agent output was empty or showed only file reads, say explicitly: '
    '"Run bash immediately with command: <exact command>"\n'
    '- If tool results show a non-zero exit code or "command not found", the fix must '
    'address the actual error — locate the executable, use a different invocation, or '
    'install the missing dependency\n'
    '- If the same general approach has been tried more than once and still fails, '
    'the approach itself is wrong. Suggest exploring the project structure first to '
    'discover what tools, frameworks, and commands are actually used, then use the correct ones\n'
    'Output only the replacement task description — no preamble, no explanation.'
)


_SMALL_CONFIG_FILES = (
    'package.json', 'pyproject.toml', 'Makefile', 'makefile',
    'Cargo.toml', 'go.mod', 'build.gradle', 'pom.xml',
    'composer.json', 'Gemfile', '.ruby-version',
)
_MAX_CONFIG_BYTES = 1500

# Maps lock file / build file → (ecosystem label, canonical commands).
# Ordered within each ecosystem so the most specific indicator wins first.
_TOOLCHAIN_INDICATORS: tuple[tuple[str, str, str], ...] = (
    # JS/TS — bun before yarn/npm since bun projects may also have node_modules
    ('bun.lock',          'bun',     'bun add / bun run <script> / bun test / bun install'),
    ('bunfig.toml',       'bun',     'bun add / bun run <script> / bun test / bun install'),
    ('pnpm-lock.yaml',    'pnpm',    'pnpm add / pnpm run <script> / pnpm test'),
    ('yarn.lock',         'yarn',    'yarn add / yarn run <script> / yarn test'),
    ('package-lock.json', 'npm',     'npm install / npm run <script> / npm test'),
    # Python
    ('uv.lock',           'uv',      'uv add / uv run <cmd> / uv run pytest'),
    ('poetry.lock',       'poetry',  'poetry add / poetry run <cmd>'),
    ('Pipfile.lock',      'pipenv',  'pipenv install / pipenv run <cmd>'),
    # Rust
    ('Cargo.lock',        'cargo',   'cargo add / cargo build / cargo test / cargo run'),
    # Go
    ('go.sum',            'go',      'go get / go build ./... / go test ./...'),
    # JVM
    ('pom.xml',           'maven',   'mvn install / mvn test / mvn compile'),
    ('build.gradle',      'gradle',  './gradlew build / ./gradlew test'),
    ('build.gradle.kts',  'gradle',  './gradlew build / ./gradlew test'),
    # Ruby
    ('Gemfile.lock',      'bundler', 'bundle install / bundle exec <cmd>'),
    # C/C++ build systems (no single canonical tool; detect whichever is present)
    ('vcpkg.json',        'vcpkg',   'vcpkg install (package manager); build with cmake/make below'),
    ('conanfile.txt',     'conan',   'conan install . --build=missing; build with cmake/make below'),
    ('conanfile.py',      'conan',   'conan install . --build=missing; build with cmake/make below'),
    ('CMakeLists.txt',    'cmake',   'cmake -B build && cmake --build build; test: ctest --test-dir build'),
    ('meson.build',       'meson',   'meson setup build && ninja -C build; test: meson test -C build'),
    # PHP
    ('composer.lock',     'composer','composer install / composer require <pkg> / composer run-script test'),
    # Elixir
    ('mix.lock',          'mix',     'mix deps.get / mix compile / mix test'),
    # Swift
    ('Package.resolved',  'swift',   'swift build / swift test / swift package add <pkg>'),
    # Dart / Flutter
    ('pubspec.lock',      'dart',    'dart pub add <pkg> / dart test  (or: flutter pub add / flutter test)'),
    # Scala
    ('build.sbt',         'sbt',     'sbt compile / sbt test / sbt run'),
    # Haskell
    ('stack.yaml.lock',   'stack',   'stack build / stack test / stack run'),
)


_DOTNET_COMMANDS = 'dotnet build / dotnet test / dotnet run / dotnet add package <pkg>'


def _detect_toolchain(cwd: Path) -> list[str]:
    """Return one hint line per detected toolchain (skips duplicates by label)."""
    seen: set[str] = set()
    hints: list[str] = []
    for filename, label, commands in _TOOLCHAIN_INDICATORS:
        if (cwd / filename).exists() and label not in seen:
            seen.add(label)
            hints.append(f'{label}: {commands}')
    # C#/.NET: glob for *.csproj / *.sln since names are not fixed
    if 'dotnet' not in seen:
        try:
            if any(
                f.suffix in ('.csproj', '.sln')
                for f in cwd.iterdir()
                if f.is_file()
            ):
                hints.append(f'dotnet: {_DOTNET_COMMANDS}')
        except OSError:
            pass
    return hints


def _scan_project_context(cwd: Path) -> str:
    """Return a brief snapshot of the project structure for the planning prompt."""
    lines: list[str] = []

    toolchain = _detect_toolchain(cwd)
    if toolchain:
        lines.append('Toolchain (use these commands — do not assume defaults):')
        lines.extend(f'  {h}' for h in toolchain)

    try:
        entries = sorted(os.listdir(cwd))
        lines.append(f'Root files: {", ".join(entries[:60])}')
    except OSError:
        return '\n'.join(lines)

    for name in _SMALL_CONFIG_FILES:
        p = cwd / name
        if p.exists():
            try:
                text = p.read_text(encoding='utf-8', errors='replace')
                snippet = text[:_MAX_CONFIG_BYTES]
                if len(text) > _MAX_CONFIG_BYTES:
                    snippet += '\n[...truncated...]'
                lines.append(f'\n{name}:\n{snippet}')
            except OSError:
                pass

    return '\n'.join(lines)


def _base_runtime(cwd: Path, label: str = 'planning') -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        cwd=cwd,
        permissions=AgentPermissions(
            allow_file_write=True,
            allow_shell_commands=True,
        ),
        session_directory=(cwd / '.port_sessions' / 'agent' / label).resolve(),
        scratchpad_root=(cwd / '.port_sessions' / 'scratchpad' / label).resolve(),
        command_timeout_seconds=300.0,  # allow slow commands like npm install / test suites
    )


def _build_task_prompt(task: TaskRecord) -> str:
    cmd = _extract_shell_command(task.description)
    if cmd:
        parts = [
            f'Your first tool call must be bash with command: {cmd!r}',
            'Do NOT read any files before running this command.',
            f'Task: {task.description}',
        ]
    else:
        parts = [
            'Call a tool immediately to begin. Do not output any text first.',
            f'Task: {task.description}',
        ]
    if task.input_files:
        parts.append(f'Input files: {task.input_files}')
    if task.output_file:
        parts.append(f'Output file: {task.output_file}')
    if task.context:
        parts.append(f'Context: {task.context}')
    return '\n'.join(parts)


def _extract_tool_results(session_directory: Path, session_id: str) -> str:
    """Extract the last few bash commands and their outputs from the agent session."""
    session_file = session_directory / f'{session_id}.json'
    if not session_file.exists():
        return ''
    try:
        data = json.loads(session_file.read_text(encoding='utf-8'))
        messages = data if isinstance(data, list) else data.get('messages', [])
        # Pair up tool_use (commands) with tool (results)
        pairs: list[str] = []
        last_command: str = ''
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'assistant' and isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get('type') == 'tool_use':
                        inp = c.get('input', {})
                        cmd = inp.get('command', '') if isinstance(inp, dict) else ''
                        last_command = f'$ {cmd}' if cmd else f'[{c.get("name", "tool")}]'
            elif role == 'tool':
                raw = content
                if isinstance(raw, list):
                    raw = next(
                        (c.get('content', '') for c in raw
                         if isinstance(c, dict) and c.get('type') == 'tool_result'),
                        '',
                    )
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            raw = str(parsed.get('content', raw))
                    except (json.JSONDecodeError, ValueError):
                        pass
                entry = (f'{last_command}\n' if last_command else '') + raw[:1000]
                pairs.append(entry)
                last_command = ''
        return '\n---\n'.join(pairs[-4:])
    except Exception:
        return ''


def _run_diagnosis(
    task: TaskRecord,
    coding_model: ModelConfig,
    *,
    agent_output: str = '',
    reviewer_feedback: str = '',
    tool_results: str = '',
    attempt: int = 1,
    project_context: str = '',
) -> str:
    """Ask the model to diagnose a failure and return a targeted replacement task description."""
    client = OpenAICompatClient(coding_model)
    output_desc = agent_output[:1500] if agent_output else '(empty — agent made no tool calls)'
    user_content = (
        f'Original task: {task.description}\n\n'
        f'Agent output:\n{output_desc}\n\n'
        f'Reviewer feedback: {reviewer_feedback}'
    )
    if tool_results:
        user_content += f'\n\nTool execution results (bash commands + outputs):\n{tool_results[:1500]}'
    if project_context:
        user_content += f'\n\nProject context (use this to determine the correct tools/commands):\n{project_context[:800]}'
    if attempt >= 2:
        user_content += (
            f'\n\nWARNING: This task has failed {attempt} times in a row. '
            'The current approach is not working. '
            'Do NOT suggest the same type of command again. '
            'Instead, suggest exploring the project structure first to discover '
            'what tools, frameworks, and commands are actually used in this project.'
        )
    try:
        turn = client.complete(
            messages=[
                {'role': 'system', 'content': _DIAGNOSIS_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            tools=[],
        )
        result = _strip_thinking(turn.content).strip()
        return result[:500] if result else task.description
    except Exception:
        return task.description


def _run_review(
    task: TaskRecord,
    coding_model: ModelConfig,
    *,
    agent_output: str = '',
    tool_results: str = '',
    cwd: Path | None = None,
) -> tuple[bool, str]:
    client = OpenAICompatClient(coding_model)
    output_section = f'\n\nAgent output:\n{agent_output[:2000]}' if agent_output else ''
    if tool_results:
        output_section += f'\n\nTool execution results (what actually ran):\n{tool_results[:1500]}'
    output_note = ''
    if task.output_file and cwd:
        exists = (cwd / task.output_file).exists()
        output_note = f'\nExpected output file: {task.output_file} (exists on disk: {exists})'
    user_content = f'Task: {task.description}{output_note}{output_section}'
    try:
        turn = client.complete(
            messages=[
                {'role': 'system', 'content': _REVIEW_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            tools=[],
        )
    except Exception as exc:
        msg = f'Review skipped ({type(exc).__name__}: {exc}); treating as passed.'
        return True, msg

    response = turn.content.strip()
    passed = response.upper().startswith('PASS')
    return passed, response


def _thread_output_context(
    queue: TaskQueue,
    completed: TaskRecord,
    session_id: str,
    cwd: Path,
) -> None:
    """After a task completes, inject its output file contents into the next pending task's context."""
    if not completed.output_file:
        return
    output_path = cwd / completed.output_file
    if not output_path.exists():
        return
    next_task = queue.get_next_pending(session_id)
    if next_task is None:
        return
    try:
        content = output_path.read_text(encoding='utf-8', errors='replace')
        snippet = content[:300].rstrip()
        note = f'Previous task wrote {completed.output_file}:\n{snippet}'
        if len(content) > 300:
            note += '\n[...truncated...]'
        existing = next_task.context
        new_ctx = (note + ('\n\n' + existing if existing else ''))[:700]
        queue.update_context(next_task.id, new_ctx)
    except OSError:
        pass


class TaskExecutor:
    def __init__(self, config: LocalLLMConfig, cwd: Path) -> None:
        self.config = config
        self.cwd = cwd

    def run(self, prompt: str, session_id: str, *, resume: bool = False) -> str:
        db_path = self.cwd / '.port_sessions' / session_id / 'tasks.db'
        queue = TaskQueue(db_path)
        project_ctx = _scan_project_context(self.cwd)

        if resume:
            stuck = queue.reset_in_progress(session_id)
            _log(f'Resuming session {session_id} (reset {stuck} in-progress task(s) to pending)')
        else:
            route = route_request(prompt, self.config.selection_model)
            _log(f'Route decision: {route}')

            if route == 'coding':
                queue.add_task(session_id, prompt, 'coding', context=prompt[:500])
            else:
                planning_registry = {
                    **default_tool_registry(),
                    'queue_task': make_queue_task_tool(
                        queue, session_id, self.config.max_tasks_per_session
                    ),
                }
                planning_agent = LocalCodingAgent(
                    model_config=self.config.planning_model,
                    runtime_config=_base_runtime(self.cwd, 'planning'),
                    tool_registry=planning_registry,
                )
                _log('Planning phase started')
                ctx_section = f'\n\nProject context:\n{project_ctx}\n' if project_ctx else ''
                planning_agent.run(
                    f'Decompose the following request into tasks using the queue_task tool. '
                    f'Create one task per discrete step. Do not execute any tasks yourself.\n\n'
                    f'REQUIRED: Every task description must include the exact shell command '
                    f'to run, written as: \'Run `<exact command>`\'. '
                    f'Never write descriptions like "install dependencies" or "run tests" '
                    f'without the literal command. Use the project context to determine the '
                    f'correct commands for this specific project.'
                    f'{ctx_section}\n'
                    f'Request:\n\n{prompt}'
                )
                pending = queue.count_pending(session_id)
                _log(f'Planning complete — {pending} task(s) queued')

        outputs: list[str] = []
        task_num = 0
        while True:
            task = queue.get_next_pending(session_id)
            if task is None:
                break

            task_num += 1
            _log(f'Task {task.id} started: {task.description[:70]}')
            queue.update_status(task.id, 'in_progress')
            task_runtime = _base_runtime(self.cwd, str(task.id))
            system_addon = (
                _TOOL_USE_SYSTEM_ADDON_BASH_FIRST
                if _extract_shell_command(task.description)
                else _TOOL_USE_SYSTEM_ADDON
            )
            agent = LocalCodingAgent(
                model_config=self.config.coding_model,
                runtime_config=task_runtime,
                append_system_prompt=system_addon,
            )
            try:
                result = agent.run(_build_task_prompt(task))
                clean_output = _strip_thinking(result.final_output)
                # Nudge loop: re-prompt up to 3 times if the model refuses to call tools.
                for _ in range(3):
                    if not (
                        result.tool_calls == 0
                        or _looks_like_bare_tool_name(clean_output)
                        or _looks_like_text_format_bash(clean_output)
                    ):
                        break
                    if not result.session_id:
                        break
                    try:
                        stored = load_agent_session(
                            result.session_id,
                            directory=task_runtime.session_directory,
                        )
                        result = agent.resume(_build_nudge_prompt(task, clean_output), stored)
                        clean_output = _strip_thinking(result.final_output)
                    except Exception:
                        break
                # Fire a second nudge if the agent made tool calls but the required
                # output file still doesn't exist — the agent stopped too early.
                if (
                    task.output_file
                    and result.tool_calls > 0
                    and not (self.cwd / task.output_file).exists()
                    and result.session_id
                ):
                    try:
                        stored = load_agent_session(
                            result.session_id,
                            directory=task_runtime.session_directory,
                        )
                        nudge = _OUTPUT_FILE_MISSING_NUDGE.format(path=task.output_file)
                        result = agent.resume(nudge, stored)
                        clean_output = _strip_thinking(result.final_output)
                    except Exception:
                        pass
            except Exception as exc:
                queue.update_status(task.id, 'failed')
                _log(f'Task {task.id} failed (coding): {exc}')
                outputs.append(f'[ERROR] {task.description}: {exc}')
                continue

            review_tool_results = _extract_tool_results(
                task_runtime.session_directory,
                result.session_id or '',
            )
            passed, feedback = _run_review(
                task, self.config.coding_model,
                agent_output=clean_output,
                tool_results=review_tool_results,
                cwd=self.cwd,
            )
            if passed:
                queue.update_status(task.id, 'complete')
                _log(f'Task {task.id} complete')
                _thread_output_context(queue, task, session_id, self.cwd)
                outputs.append(result.final_output)
            else:
                count = queue.increment_review_count(task.id)
                queue.update_status(task.id, 'failed')
                _log(f'Task {task.id} failed review (attempt {count}): {feedback[:80]}')
                _log(f'Task {task.id} output was: {clean_output[:300]!r}')
                if count < self.config.max_review_loops and queue.count_session_tasks(session_id) < self.config.max_tasks_per_session:
                    # Diagnose the root cause and build a targeted correction task.
                    tool_results = review_tool_results
                    diagnosed = _run_diagnosis(
                        task, self.config.coding_model,
                        agent_output=clean_output, reviewer_feedback=feedback,
                        tool_results=tool_results,
                        attempt=count,
                        project_context=project_ctx,
                    )
                    _log(f'Task {task.id} diagnosis → correction: {diagnosed[:120]}')
                    correction_context = f'Previous attempt failed. Reviewer feedback: {feedback}'
                    if tool_results:
                        correction_context += f'\n\nWhat ran:\n{tool_results[:400]}'
                    if task.context:
                        correction_context += f'\n\nOriginal context: {task.context}'
                    # Insert immediately after this task so it runs before any
                    # subsequent planned tasks (which may depend on this one).
                    queue.add_correction(
                        session_id,
                        task.task_order,
                        diagnosed,
                        'coding',
                        input_files=task.input_files,
                        output_file=task.output_file,
                        context=correction_context[:500],
                        initial_review_count=count,
                    )
                else:
                    # Exhausted all retries — block every subsequent planned task
                    # that hasn't run yet, since they likely depend on this one.
                    blocked = queue.block_remaining(session_id, task.task_order)
                    if blocked:
                        _log(
                            f'Task {task.id} exhausted retries — '
                            f'blocked {blocked} subsequent task(s)'
                        )
                    outputs.append(f'[FAILED after {count} review(s)] {task.description}')

        blocked_tasks = queue.get_blocked(session_id)
        for bt in blocked_tasks:
            outputs.append(f'[BLOCKED — prerequisite failed] {bt.description}')

        _log(f'Session complete — {task_num} task(s) processed')
        return '\n\n---\n\n'.join(outputs) if outputs else '(no tasks executed)'
