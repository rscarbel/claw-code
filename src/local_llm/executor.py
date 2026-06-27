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
from ..openai_compat import OpenAICompatClient, _strip_thinking
from ..session_store import load_agent_session
from .config import LocalLLMConfig
from .queue_tool import make_queue_task_tool
from .router import route_request
from .task_queue import TaskQueue, TaskRecord
from .tool_tree import make_explore_tools, make_use_discovered_tool


def _log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f'[claw-multi {ts}] {msg}', file=sys.stderr, flush=True)

_TOOL_USE_SYSTEM_ADDON = (
    'CRITICAL RULE: Your first response must always be a tool call. '
    'Never output text like "Let me explore..." or "I will..." before calling a tool. '
    'Call read_file, bash, write_file, or another tool immediately. '
    'Text-only responses are treated as task failures.\n'
    'CRITICAL: The bash tool\'s "command" argument must be a valid shell command '
    '(e.g. `grep -n "foo" file.ts`, `bun test`, `find . -name "*.ts"`). '
    'NEVER pass a sentence or description as the command argument. '
    'If you want to write a file, use write_file or edit_file — not bash.\n'
    'CRITICAL: NEVER write file content via bash heredoc (cat > file <<EOF) or echo redirects. '
    'Use write_file to create files and edit_file to modify existing ones.\n'
    'If a shell command fails (non-zero exit code), do NOT stop to narrate or plan. '
    'Immediately call another tool to investigate or recover: check the error output, '
    'locate the executable, try an alternative invocation, or fix the root cause. '
    'Keep calling tools until the task succeeds or is provably impossible.\n'
    'If a tool returns "file not found", "no such file or directory", or similar: '
    'DO NOT stop or output text. Immediately call bash with '
    '`find . -maxdepth 4 -type f | head -50` or `ls -la <parent_dir>` '
    'to locate the correct path, then continue implementing the task using the actual path found.'
)

_TOOL_USE_SYSTEM_ADDON_BASH_FIRST = (
    'CRITICAL RULE: Your first action must be to execute the shell command specified in the task. '
    'Do NOT call read_file, glob, or any other tool before running that command. '
    'Text-only responses are treated as task failures.\n'
    'After the first command, continue calling tools as needed to complete all remaining '
    'steps in the task — do not stop after a single command if more work is required. '
    'If a command fails, call another tool immediately to investigate or recover. '
    'Keep calling tools until the task fully succeeds or is provably impossible.'
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

# Lines in tool output that carry actionable signal (errors, failures, assertion mismatches).
# Matched against each line of discovery output in _extract_discovery_signal().
_DISCOVERY_SIGNAL_RE = re.compile(
    r'(?:'
    r'\berror(?:\s+[A-Z]\d+|\s*:|\s+TS\d+|\[)'  # error:  error TS2345  error[E0001]
    r'|FAILED\b'                                  # pytest FAILED
    r'|FAIL:'                                     # Go --- FAIL:
    r'|^\s*(?:×|✕|✗)\s'                          # vitest/jest failure symbols
    r'|\bAssertionError\b'                        # Python assertion failures
    r'|\bExpected\b.{0,60}\bReceived\b'           # jest expect diff header
    r'|\bnot found\b|\bcannot find\b'             # module-not-found style errors
    r')',
    re.IGNORECASE | re.MULTILINE,
)

_DISCOVERY_OUTPUT_LIMIT = 8000
_DISCOVERY_CONTEXT_LINES = 8  # lines to keep around each signal line

# Appended to the re-planning agent's system prompt so qwen3:14b calls queue_task
# instead of falling back to chat-mode markdown explanations.
_REPLAN_TOOL_CALL_ADDON = (
    'CRITICAL: Your only job is to call queue_task for each affected file or error group. '
    'You only have the queue_task tool — no file reading or exploration tools are available. '
    'The discovery output already contains all file paths and errors you need. '
    'Do NOT output explanations, summaries, or markdown — call queue_task immediately. '
    'Text-only responses are treated as failures and will be discarded.'
)


def _extract_discovery_signal(output: str) -> str:
    """Return the actionable portion of tool output for the re-planning model.

    If the output fits within the limit, returns it unchanged. Otherwise,
    extracts lines matching error/failure patterns plus surrounding context so
    the re-planner sees complete error messages without boilerplate noise.
    Falls back to raw truncation when no signal lines are found.
    """
    if len(output) <= _DISCOVERY_OUTPUT_LIMIT:
        return output
    lines = output.splitlines()
    keep = [False] * len(lines)
    for i, line in enumerate(lines):
        if _DISCOVERY_SIGNAL_RE.search(line):
            lo = max(0, i - _DISCOVERY_CONTEXT_LINES)
            hi = min(len(lines), i + _DISCOVERY_CONTEXT_LINES + 1)
            for j in range(lo, hi):
                keep[j] = True
    extracted = '\n'.join(line for line, k in zip(lines, keep) if k)
    if not extracted.strip():
        # No recognizable error patterns — fall back to raw truncation
        truncated = output[:_DISCOVERY_OUTPUT_LIMIT]
        return truncated + f'\n[...{len(output) - _DISCOVERY_OUTPUT_LIMIT} chars truncated]'
    if len(extracted) <= _DISCOVERY_OUTPUT_LIMIT:
        return extracted
    truncated = extracted[:_DISCOVERY_OUTPUT_LIMIT]
    shown = sum(keep)
    return truncated + f'\n[...truncated — {shown}/{len(lines)} lines shown]'


# Tool names that, if emitted as the sole output, indicate the model tried but failed
# to produce a proper tool call.
_BARE_TOOL_NAMES = frozenset({
    'read_file', 'write_file', 'bash', 'glob', 'list_directory', 'search',
})

# Two patterns for "Run bash immediately with command: `...`":
# group 1 — backtick-quoted (preferred, captures full command including dots and pipes)
# group 2 — unquoted fallback (stops at semicolon/newline only)
# group 3 — Run `cmd` shorthand — backtick delimiter; inner quotes/semicolons are fine
# group 4 — Run 'cmd' shorthand — single-quote delimiter
# (old combined group used [^\`\'";\n] which stopped at inner " in e.g. find -name "*.ts",
#  truncating the extracted command and producing a wrong "first action" hint)
_SHELL_CMD_RE = re.compile(
    r'(?:'
    r'[Rr]un bash immediately with command[=:\s]+`([^`\n]+)`'  # group 1: backtick
    r'|[Rr]un bash immediately with command[=:\s]+([^\n;]{1,300})'  # group 2: unquoted
    r'|[Rr]un\s+`([^`\n]+)`'                                        # group 3: Run `cmd`
    r"|[Rr]un\s+'([^'\n]+)'"                                        # group 4: Run 'cmd'
    r')',
)

# Matches text-format fake tool calls the model sometimes writes instead of calling the tool
_TEXT_FORMAT_BASH_RE = re.compile(r'\[/?bash\]|```bash|</?bash>', re.IGNORECASE)

# Matches model responses that are asking for direction instead of completing the task,
# OR that describe planned future actions instead of executing them.
# Fires the nudge loop even when tool_calls > 0 (e.g. model explored files then switched
# to Q&A mode, or called one tool then stopped to announce next steps instead of doing them).
_CHAT_MODE_RE = re.compile(
    r'(?:what would you like|how (?:can|shall) I (?:help|proceed)|'
    r'what do you want|shall I|should I (?:proceed|continue)|'
    r'would you like me to|please (?:clarify|specify|let me know)|'
    r'tell me (?:what|if|how)|(?:let me know|feel free) (?:if|what|how)|'
    r'what(?:\'s| is) (?:your|the) (?:goal|preference|next step)|'
    r'(?:for example|such as)[:\s]*\n\s*[-•]|'
    # Planning-language: model announces what it WILL do instead of doing it
    r'[Ll]et me (?:explore|look|check|examine|investigate|start|begin|understand)|'
    r'[Ii]\'ll (?:start|begin) by\b|'
    r'[Ii] (?:need to|will) first\b|'
    # Explanatory mode: model describes the problem/fix instead of implementing it.
    # "Based on the errors..." / "Here are the most likely causes..."
    r'[Bb]ased on (?:the|your|these|this|those)\b|'
    r'[Hh]ere (?:are|is) (?:(?:the|a) )?(?:(?:most|possible|common|likely|very) )*'
    r'(?:cause|causes|step|steps|reason|reasons|approach|approaches)\b)',
    re.IGNORECASE,
)

_CHAT_MODE_NUDGE = (
    'You are an automated task executor, not a chat assistant. '
    'Do not describe what you see or ask what to do — execute the task. '
    'Call a tool RIGHT NOW to make progress on the task.'
)

_EMPTY_OUTPUT_NUDGE = (
    'Your last response was empty. The task is not yet complete. '
    'Continue executing: call write_file, edit_file, or bash to implement the required changes. '
    'Do not output text without calling a tool.'
)

# Explanatory-response detection: used by _build_chat_mode_nudge to distinguish "tutorial
# mode" (model described the fix) from generic chat mode (model asked for direction).
_EXPLANATORY_RESPONSE_RE = re.compile(
    r'(?:[Bb]ased on (?:the|your|these|this|those)\b'
    r'|[Hh]ere (?:are|is|\'s) (?:(?:the|a|an|my) )?(?:most |possible |common |likely )?'
    r'(?:cause|fix|step|issue|approach|solution|reason|error)\b'
    r'|[Tt]o fix (?:the|this|these|those)\b)',
    re.IGNORECASE,
)


def _looks_like_explanatory_response(text: str) -> bool:
    return bool(_EXPLANATORY_RESPONSE_RE.search(text))


def _build_chat_mode_nudge(task: TaskRecord, clean_output: str) -> str:
    task_reminder = task.description[:250]
    if _looks_like_file_not_found(clean_output):
        return (
            f'Your task is: {task_reminder}\n\n'
            + _FILE_NOT_FOUND_NUDGE
        )
    if _looks_like_explanatory_response(clean_output):
        return (
            'You described the errors but made no code changes. '
            'Call read_file to inspect the affected file, then call edit_file to apply the fix RIGHT NOW. '
            f'Your task: {task_reminder}\n'
            'Do not output any explanations — call a tool immediately to implement the change.'
        )
    return (
        'STOP. You are in chat mode but you must execute a task — not describe, ask, or plan.\n'
        f'Your task: {task_reminder}\n'
        'Call edit_file or write_file RIGHT NOW to make the required file change. '
        'If you need to locate the file first, call bash with `find . -maxdepth 4 -name "*.ts" | head -20`. '
        'Do not output any text. Make the tool call immediately.'
    )

_FILE_NOT_FOUND_NUDGE = (
    'A file path was not found. Call bash RIGHT NOW to locate the correct path: '
    '`find . -maxdepth 4 -type f | head -50` '
    'or `ls -la <parent_directory>`. '
    'Do not stop — find the actual path and then implement the task using it.'
)

_FILE_NOT_FOUND_RE = re.compile(
    r'(?:file not found|no such file|ENOENT|does not exist|not found)',
    re.IGNORECASE,
)


def _looks_like_file_not_found(text: str) -> bool:
    return bool(_FILE_NOT_FOUND_RE.search(text))


_HEREDOC_ANY_RE = re.compile(r'<<\s*[\'"]?E?OF[\'"]?', re.IGNORECASE)
_CAT_REDIRECT_RE = re.compile(r'cat\s*>+\s*(\S+)')


def _strip_heredoc_from_diagnosis(description: str) -> str:
    """Remove shell heredoc content from a diagnosed task description.

    The diagnosis model generates 'cat > path <<EOF\\n...\\nEOF' commands despite
    being instructed not to. These always fail: the \\n-escaped newlines in a
    single-line string prevent the shell from recognizing the EOF terminator.
    Strip the heredoc and rephrase as a write_file instruction instead.
    """
    if not _HEREDOC_ANY_RE.search(description):
        return description
    before = _HEREDOC_ANY_RE.split(description, maxsplit=1)[0]
    m = _CAT_REDIRECT_RE.search(before)
    if m:
        file_path = m.group(1)
        preamble = before[:m.start()].rstrip(' ;&|\\ ')
        if preamble:
            return f'{preamble} then create {file_path} with the required content using write_file'
        return f'Create {file_path} with the required content using write_file'
    stripped = before.rstrip(' ;&|\\ `').strip()
    return stripped if stripped else description


def _write_task_md(task: TaskRecord, scratchpad_root: Path) -> None:
    """Write the task description to task.md in the scratchpad root.

    qwen3.6 looks for task.md in the scratchpad directory. Writing it at the
    scratchpad root level means find . will discover it after the file-not-found
    nudge fires, giving the model a second path to the task description.
    """
    try:
        scratchpad_root.mkdir(parents=True, exist_ok=True)
        (scratchpad_root / 'task.md').write_text(
            f'# Task\n\n{task.description}\n',
            encoding='utf-8',
        )
    except OSError:
        pass


def _extract_shell_command(description: str) -> str | None:
    """Extract a specific shell command from a task description, if clearly stated."""
    m = _SHELL_CMD_RE.search(description)
    if m:
        cmd = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or '').strip().strip('`').strip()
        return cmd if cmd else None
    return None


def _looks_like_text_format_bash(text: str) -> bool:
    return bool(_TEXT_FORMAT_BASH_RE.search(text))


def _looks_like_bare_tool_name(text: str) -> bool:
    # Strip markdown formatting (*bash*, **bash**, `bash`) before checking
    stripped = text.strip().strip('*').strip('`').strip('_').strip()
    return stripped in _BARE_TOOL_NAMES


def _looks_like_chat_mode(text: str) -> bool:
    """Return True when the model is asking for direction instead of completing the task."""
    return bool(_CHAT_MODE_RE.search(text))


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
            f'Run this shell command now: {cmd!r}. '
            f'Do not output any text. Make the tool call immediately.'
        )
    return _TOOL_NUDGE_PROMPT


def _build_format_hint_nudge(task: TaskRecord) -> str:
    """Build a nudge that shows the exact <tool_call> text format when the model is stuck.

    Used when the model is echoing a bare tool name (e.g. '*bash*') instead of calling
    the tool. The <tool_call> format is already handled by _TOOL_CALL_TAG_RE in
    _extract_text_format_tool_calls, so the model just needs to reproduce this text.
    """
    cmd = _extract_shell_command(task.description)
    if cmd:
        call_json = json.dumps({'name': 'bash', 'arguments': {'command': cmd}})
        return (
            f'Your response was not a valid tool call. '
            f'Output this exact text and nothing else:\n'
            f'<tool_call>{call_json}</tool_call>'
        )
    return _TOOL_NUDGE_PROMPT

_REVIEW_SYSTEM_PROMPT = (
    'You are a code reviewer. A single step in a multi-step pipeline has just executed. '
    'Evaluate ONLY whether this specific task completed successfully. '
    'Do NOT fail the task because later steps (e.g. running analysis, reporting results) have not been done — other tasks handle those. '
    'Check: does the output match the task description? Are referenced files plausible? Are there any obvious errors?\n'
    'IMPORTANT — always reply PASS if ANY of these are true:\n'
    '  - Tool results show a test or typecheck command completing with exit_code=0 and no failure lines\n'
    '  - The agent made file changes (write_file or edit_file succeeded) and subsequent verification confirms the fix\n'
    '  - The agent correctly determined the stated problem does not exist in the codebase '
    '(e.g. the specific string or error was not found at the stated location) and the relevant tests pass\n'
    'IMPORTANT — always reply FAIL if:\n'
    '  - The agent output is empty and the task required writing or modifying files\n'
    '  - Tool results show only exploration commands (find, ls, cat, read_file, glob) '
    'with no file writes AND no passing test/typecheck run\n'
    '  - The task was to implement something but tool results show a non-zero exit code '
    'with no subsequent successful fix\n'
    '  - Tool results show a command output that contradicts the agent\'s claimed success '
    '(e.g. agent says "command succeeded" but tool result shows "command not found")\n'
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
    '- CRITICAL: If tool results show "timed out", "Command timed out", or "exit_code=-9", '
    'the command that timed out MUST NOT appear in the replacement task. '
    'Replace it with a faster targeted command: `find`, `ls`, `cat`, or a direct file write. '
    'NEVER include the full test suite (bun test, npm test, pytest, go test, etc.) '
    'in a correction task unless the original task was specifically about running tests.\n'
    '- CRITICAL: If the failure was "file not found", "no such file", or "not found": '
    'the replacement MUST start with a bash command to find the actual path '
    '(e.g. `find . -maxdepth 4 -name "filename*" | head -20 && ls -la src/`), '
    'immediately followed by the implementation step — all in one chained task description. '
    'Example: Run `find . -name "target*" | head -10 && ls -la src/` to locate '
    'the file, then implement the required change at the correct path found.\n'
    '- If the same general approach has been tried more than once and still fails, '
    'the approach itself is wrong. Suggest exploring the project structure first to '
    'discover what tools, frameworks, and commands are actually used, then use the correct ones\n'
    '- CRITICAL: If multiple shell commands are needed, chain them into ONE command using '
    '&& (e.g., Run `bun test --help && cat bunfig.toml && grep coverage package.json`). '
    'NEVER list commands as "Run X, then Y" or "Run X, Y, and Z" — always chain with &&.\n'
    '- CRITICAL: If the original task was an implementation task (write code, add tests, '
    'modify files), the replacement MUST also implement the change — not just run a '
    'discovery command. A replacement that only reports results without writing files '
    'does NOT accomplish the goal. If a discovery command is needed to inform the '
    'implementation, it must be the first step in a task that also writes the code.\n'
    '- CRITICAL: NEVER embed file content in the task description using heredoc syntax '
    '(<<EOF) or shell redirects with literal code (e.g. cat > file <<EOF\\n[content]). '
    'Embedding large file content inflates context and causes model failures. '
    'If a file must be created, write the task as: '
    '"Create src/foo.ts with [specific functions/tests described in words]" '
    'and let the agent use its write_file tool to write the content.\n'
    '- CRITICAL: If tool results show bash commands failing with "unexpected EOF", '
    '"unexpected end of file", or similar shell parse errors, the agent passed a sentence '
    'or description as a bash command instead of using the file-editing tools. '
    'The replacement task MUST say: "Read <file> then call edit_file to make the change. '
    'Do NOT use bash sed or cat commands to write the file — use edit_file." '
    'Describe the change to make in plain language (which lines to change, to what) '
    'so the agent can use edit_file instead of attempting bash file manipulation.\n'
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


# The full default_tool_registry has 65 tools (~6000 token overhead).
# qwen3.6:35b-a3b produces empty responses (no tool calls, finish_reason="stop")
# when given the full list — confirmed by direct Ollama testing. With 7 tools it
# returns proper tool_calls; with 65 it doesn't. Stripped to core coding tools only.
_CODING_CORE_TOOLS = frozenset({
    'bash', 'read_file', 'write_file', 'edit_file',
    'glob_search', 'grep_search', 'list_dir',
})


def _make_coding_registry() -> dict:
    """Return the tool registry for local LLM coding tasks.

    Exposes the 7 core coding tools directly plus two meta-tools for
    discovering and executing the remaining 58 tools on demand. Total: 9 tools
    (~600 tokens vs ~6 000 for the full registry), which keeps qwen3.6 from
    producing empty responses due to tool-spec context overflow.

    To expose an additional tool directly (without tree navigation), add its
    name to _CODING_CORE_TOOLS above — or add it to the tree in tool_tree.py.
    """
    full = default_tool_registry()
    registry = {k: v for k, v in full.items() if k in _CODING_CORE_TOOLS}
    registry['explore_tools'] = make_explore_tools()
    registry['use_discovered_tool'] = make_use_discovered_tool(full)
    return registry


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


def _planning_runtime(cwd: Path) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        cwd=cwd,
        permissions=AgentPermissions(
            allow_file_write=False,
            allow_shell_commands=False,
        ),
        session_directory=(cwd / '.port_sessions' / 'agent' / 'planning').resolve(),
        scratchpad_root=(cwd / '.port_sessions' / 'scratchpad' / 'planning').resolve(),
        command_timeout_seconds=300.0,
        # Non-streaming: _build_payload applies think:False for qwen3 models, which
        # prevents the 20-minute hangs caused by streaming + think mode generating
        # unbounded thinking tokens. The queue_task text-format fallback handles
        # any <tool_code> output from think:False mode.
    )


def _base_runtime(cwd: Path, label: str) -> AgentRuntimeConfig:
    # WARNING: Do NOT change stream_model_responses to True for qwen3 coding tasks.
    # With streaming, qwen3's think mode stays ON. The model then "reasons" around
    # explicit task instructions and substitutes its own commands (e.g. runs 'ls -la'
    # instead of the commanded 'bun test --coverage', or says "I'll start by exploring"
    # and loops identically on every nudge). Non-streaming auto-applies think:False via
    # _build_payload, which makes the model follow instructions mechanically. The
    # _extract_text_format_tool_calls fallback in complete() handles any text-format
    # tool calls that think:False mode produces.
    return AgentRuntimeConfig(
        cwd=cwd,
        max_turns=30,
        stream_model_responses=False,
        permissions=AgentPermissions(
            allow_file_write=True,
            allow_shell_commands=True,
        ),
        session_directory=(cwd / '.port_sessions' / 'agent' / label).resolve(),
        scratchpad_root=(cwd / '.port_sessions' / 'scratchpad' / label).resolve(),
        command_timeout_seconds=300.0,
    )


def _build_task_prompt(task: TaskRecord) -> str:
    cmd = _extract_shell_command(task.description)
    if cmd:
        parts = [
            f'First action — execute this shell command: {cmd!r}',
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


def _build_fresh_task_prompt(task: TaskRecord) -> str:
    """Task prompt for fresh-context restarts after bare-tool-name failures.

    Avoids repeating 'bash' — the word that locks the model into outputting
    '*bash*' on every subsequent turn. Uses neutral 'shell command' framing
    so the model can approach the task without the confusing feedback loop.
    """
    cmd = _extract_shell_command(task.description)
    if cmd:
        parts = [
            'Shell command to execute:',
            f'  {cmd}',
            '',
            'Use an available tool to run this command. Do not output text before calling the tool.',
        ]
    else:
        parts = [
            f'Task: {task.description}',
            'Call a tool immediately to begin. Do not output any text first.',
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
    diagnosis_model: ModelConfig,
    *,
    agent_output: str = '',
    reviewer_feedback: str = '',
    tool_results: str = '',
    attempt: int = 1,
    project_context: str = '',
) -> str:
    """Ask the model to diagnose a failure and return a targeted replacement task description."""
    client = OpenAICompatClient(diagnosis_model)
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
    review_model: ModelConfig,
    *,
    agent_output: str = '',
    tool_results: str = '',
    cwd: Path | None = None,
) -> tuple[bool, str]:
    client = OpenAICompatClient(review_model)
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


_PLANNING_READ_TOOLS = frozenset({'read_file', 'glob_search', 'list_dir', 'grep_search'})


def _make_planning_registry(
    queue: TaskQueue,
    session_id: str,
    max_tasks: int,
) -> dict:
    """Build the tool registry shared by both the initial planning step and discovery re-planning."""
    registry: dict = {
        'queue_task': make_queue_task_tool(queue, session_id, max_tasks),
    }
    for name, tool in default_tool_registry().items():
        if name in _PLANNING_READ_TOOLS:
            registry[name] = tool
    return registry


def _run_replan_after_discovery(
    task: TaskRecord,
    discovery_output: str,
    original_prompt: str,
    queue: TaskQueue,
    session_id: str,
    config: LocalLLMConfig,
    cwd: Path,
    project_ctx: str,
) -> int:
    """Queue implementation tasks based on a completed discovery task. Returns count of new tasks."""
    before = queue.count_session_tasks(session_id)
    if before >= config.max_tasks_per_session:
        _log(f'Task {task.id} (discovery): session task limit reached, skipping re-plan')
        return 0

    # Only queue_task — no exploration tools. The discovery output already contains
    # all file paths and error details the re-planner needs. Exploration tools cause
    # the model to wander after a failed queue_task call instead of retrying correctly.
    registry = {'queue_task': make_queue_task_tool(queue, session_id, config.max_tasks_per_session)}
    planning_agent = LocalCodingAgent(
        model_config=config.planning_model,
        runtime_config=_planning_runtime(cwd),
        tool_registry=registry,
        append_system_prompt=_REPLAN_TOOL_CALL_ADDON,
    )
    ctx_section = f'\n\nProject context:\n{project_ctx}' if project_ctx else ''
    plan_result = planning_agent.run(
        f'A discovery task has completed. Based on its output, queue implementation tasks '
        f'to fix what it found.\n\n'
        f'Original request: {original_prompt}\n\n'
        f'Discovery task: {task.description}\n\n'
        f'Discovery output:\n{_extract_discovery_signal(discovery_output)}\n'
        f'{ctx_section}\n\n'
        f'Queue one fix task per affected file or error group using queue_task. '
        f'Each task description must:\n'
        f'1. Name the exact file to modify\n'
        f'2. List the specific errors or gaps from the discovery output (with line numbers)\n'
        f'3. Instruct the coding agent to read the file and fix the errors\n\n'
        f'Do NOT prescribe the exact code change — let the coding agent determine the correct '
        f'fix after reading the file.\n'
        f'Do NOT queue tasks that only verify, check existence, or re-run analysis — '
        f'only queue tasks that modify files.\n\n'
        f'Always use the "description" parameter in queue_task, never "command".\n'
        f'Use task_type="coding" for all tasks.'
    )
    if plan_result.tool_calls == 0:
        raw_output = _strip_thinking(plan_result.final_output).strip()
        fallback_tasks = _parse_tasks_from_planning_output(raw_output)
        if fallback_tasks:
            for desc, task_type, output_file in fallback_tasks:
                if desc.strip():
                    queue.add_task(
                        session_id, desc, task_type,
                        output_file=output_file,
                        context=original_prompt[:500],
                    )
            _log(
                f'Task {task.id} (discovery): parsed {len(fallback_tasks)} task(s) '
                f'from text output (model did not use tool API)'
            )

    # If the re-planner produced no tasks (confused or failed), fall back to a single
    # coding task that passes the discovery output directly to the coding model.
    # The coding model is more reliable at execution than the planning model is at
    # translating structured error output into task calls when it has already failed.
    mid = queue.count_session_tasks(session_id)
    if mid <= before and before < config.max_tasks_per_session:
        signal = _extract_discovery_signal(discovery_output)
        fallback_desc = (
            f'{original_prompt}\n\n'
            f'Diagnostic output from running the project checks:\n\n'
            f'{signal[:2000]}'
        )
        queue.add_task(
            session_id, fallback_desc, 'coding',
            context='Re-planner queued 0 tasks — fallback from discovery output',
        )
        _log(f'Task {task.id} (discovery): re-plan produced 0 tasks — queued fallback coding task')
        mid = queue.count_session_tasks(session_id)

    # After implementation tasks are queued, add an automatic verification task that
    # re-runs the original discovery command to confirm the fixes succeeded end-to-end.
    if mid > before and mid < config.max_tasks_per_session:
        verify_cmd = _extract_shell_command(task.description)
        if verify_cmd:
            # Strip output redirection (| tee file) — verification only needs the exit status
            verify_cmd_clean = re.sub(r'\s*\|?\s*tee\s+\S+', '', verify_cmd).strip().rstrip('|').strip()
            if not verify_cmd_clean:
                verify_cmd_clean = verify_cmd
            queue.add_task(
                session_id,
                f'Run `{verify_cmd_clean}` to verify that all previous fixes succeeded. '
                f'The task passes only if the command exits with no errors or failures.',
                'coding',
                context=f'End-to-end verification for: {original_prompt[:300]}',
            )

    after = queue.count_session_tasks(session_id)
    return after - before


_VALID_TASK_TYPES = frozenset({'coding', 'discovery'})

# Matches "queue_task: {" to find text-format tool call blocks
_QUEUE_TASK_BLOCK_RE = re.compile(r'\bqueue_task\s*:\s*(?=\{)')

# Matches "Run `cmd`[, output_file="file"]" lines the planner sometimes emits
_RUN_CMD_LINE_RE = re.compile(
    r'Run\s+`([^`]+)`(?:[^\n]*?output_file\s*=\s*["\']([^"\']+)["\'])?',
    re.MULTILINE,
)


def _parse_tasks_from_planning_output(output: str) -> list[tuple[str, str, str]]:
    """Extract task specs from text the planner emitted instead of calling queue_task.

    Returns list of (description, task_type, output_file) tuples.
    Handles both pure-JSON output and text-format 'queue_task: {...}' blocks.
    """
    def _task_type(val: object) -> str:
        s = str(val or 'coding')
        return s if s in _VALID_TASK_TYPES else 'coding'

    def _item_to_spec(item: object) -> tuple[str, str, str] | None:
        if isinstance(item, str) and item.strip():
            return (item.strip(), 'coding', '')
        if isinstance(item, dict):
            desc = str(item.get('description') or '').strip()
            cmd = str(item.get('command') or '').strip()
            tt = _task_type(item.get('task_type'))
            out = str(item.get('output_file') or '')
            if cmd and 'Run `' not in desc:
                desc = f'Run `{cmd}` — {desc}' if desc else f'Run `{cmd}`'
            return (desc, tt, out) if desc else None
        return None

    # 1. Try parsing the whole output as JSON
    try:
        obj = json.loads(output.strip())
        items: list = []
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            for key in ('tasks', 'steps', 'plan'):
                if isinstance(obj.get(key), list):
                    items = obj[key]
                    break
            if not items and obj.get('description'):
                items = [obj]
        specs = [s for item in items for s in [_item_to_spec(item)] if s]
        if specs:
            return specs
    except json.JSONDecodeError:
        pass

    # 2. Parse text-format "queue_task: {...}" blocks (model skipped the tool API)
    decoder = json.JSONDecoder()
    specs = []
    for m in _QUEUE_TASK_BLOCK_RE.finditer(output):
        try:
            args, _ = decoder.raw_decode(output, m.end())
            if isinstance(args, dict):
                spec = _item_to_spec(args)
                if spec:
                    specs.append(spec)
        except (json.JSONDecodeError, ValueError):
            pass
    if specs:
        return specs

    # 3. Parse bare "Run `cmd`[, output_file="file"]" lines
    for m in _RUN_CMD_LINE_RE.finditer(output):
        cmd = m.group(1).strip()
        out_file = (m.group(2) or '').strip()
        if cmd:
            desc = f'Run `{cmd}`'
            task_type = 'discovery' if out_file else 'coding'
            specs.append((desc, task_type, out_file))
    return specs


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
                planning_registry = _make_planning_registry(
                    queue, session_id, self.config.max_tasks_per_session
                )
                planning_agent = LocalCodingAgent(
                    model_config=self.config.planning_model,
                    runtime_config=_planning_runtime(self.cwd),
                    tool_registry=planning_registry,
                )
                _log('Planning phase started')
                ctx_section = f'\n\nProject context:\n{project_ctx}\n' if project_ctx else ''
                try:
                    plan_result = planning_agent.run(
                        f'Decompose the following request into tasks using the queue_task tool. '
                        f'Create one task per discrete step. Do not execute any tasks yourself.\n\n'
                        f'You have read_file, glob_search, list_dir, and grep_search to explore the '
                        f'project before planning. Use them to find relevant source files, existing '
                        f'tests, and project structure — then queue specific targeted tasks.\n\n'
                        f'task_type field:\n'
                        f'- "coding" (default): a normal implementation step.\n'
                        f'- "discovery": use when you need the runtime output of a shell command to '
                        f'plan the implementation — e.g. running tests to see coverage gaps, a linter '
                        f'for type errors, or a build for compilation failures. The discovery task '
                        f'must set output_file so its output is captured. After it runs, an automatic '
                        f're-planning step will read that file and queue the implementation tasks. '
                        f'Only use discovery when static file reading cannot answer the question.\n'
                        f'IMPORTANT: When you queue a discovery task with output_file="X", the file X '
                        f'does NOT exist yet — it will only be written when the coding agent runs that task. '
                        f'Do NOT call read_file on a queued output_file during planning.\n\n'
                        f'REQUIRED: Every task description must be specific and actionable:\n'
                        f'- For shell commands: \'Run `<exact command>`\'\n'
                        f'- For code changes: name the exact file to create or modify and what to write\n'
                        f'Never write vague descriptions like "add tests" or "fix the bug" — '
                        f'say exactly which file and what change to make.\n\n'
                        f'NEVER queue a task whose description contains "queue tasks", "queue targeted", '
                        f'"analyze and queue", or "identify files and queue" — these are orchestration '
                        f'steps the coding agent cannot perform. Dynamic re-planning from runtime output '
                        f'is handled automatically by the discovery mechanism after discovery tasks complete. '
                        f'If you need runtime output to guide implementation, use a discovery task instead.\n\n'
                        f'CRITICAL — always use the "description" parameter, never "command":\n'
                        f'  # Discovery (shell output needed to plan next steps):\n'
                        f'  queue_task(description="Run `bun test --coverage 2>&1 | tee cov.txt`", '
                        f'task_type="discovery", output_file="cov.txt")\n'
                        f'  # Implementation (write or modify files):\n'
                        f'  queue_task(description="Create src/auth.test.ts with unit tests for '
                        f'login(), logout(), and refreshToken()", task_type="coding")\n\n'
                        f'CRITICAL — command format rules:\n'
                        f'1. For shell-command tasks: chain multiple commands with && inside a single '
                        f'Run `...` instruction. Never write "Run X, then Y" — write '
                        f'\'Run `X && Y`\' instead.\n'
                        f'2. For discovery tasks, always set output_file and redirect command output '
                        f'into it: e.g. \'Run `bun test --coverage 2>&1 | tee coverage.txt`\', '
                        f'output_file="coverage.txt".\n'
                        f'3. Do not install packages as a discovery step. Use the runtime\'s built-in '
                        f'coverage/lint flags first.\n'
                        f'{ctx_section}\n'
                        f'Request:\n\n{prompt}'
                    )
                    if plan_result.tool_calls == 0:
                        raw_output = _strip_thinking(plan_result.final_output).strip()
                        fallback_tasks = _parse_tasks_from_planning_output(raw_output)
                        if fallback_tasks:
                            for desc, task_type, output_file in fallback_tasks:
                                queue.add_task(
                                    session_id, desc, task_type,
                                    output_file=output_file,
                                    context=prompt[:500],
                                )
                            _log(
                                f'Parsed {len(fallback_tasks)} task(s) from planning output '
                                f'(model emitted text-format tool calls instead of using the tool API)'
                            )
                        else:
                            _log(
                                f'WARNING: Planning agent made no tool calls '
                                f'(stop_reason={plan_result.stop_reason}, turns={plan_result.turns}): '
                                f'{raw_output[:300]!r}'
                            )
                    pending = queue.count_pending(session_id)
                    _log(f'Planning complete — {pending} task(s) queued')
                    if pending == 0:
                        ctx = (project_ctx[:300] + '\n' + prompt if project_ctx else prompt)[:500]
                        _log('Planning produced 0 tasks — retrying with discovery directive')
                        # Retry with a short, focused prompt: ask the planning model to
                        # queue exactly one discovery task that surfaces the current state.
                        # This is intentionally model-agnostic — the planner infers the
                        # right diagnostic command from the project context rather than us
                        # hardcoding keyword→command mappings.
                        ctx_section = f'\n\nProject context:\n{project_ctx}' if project_ctx else ''
                        retry_agent = LocalCodingAgent(
                            model_config=self.config.planning_model,
                            runtime_config=_planning_runtime(self.cwd),
                            tool_registry=planning_registry,
                        )
                        try:
                            retry_result = retry_agent.run(
                                f'Request: {prompt}\n\n'
                                f'You must queue exactly one discovery task using queue_task.\n'
                                f'The task must run a diagnostic command that reveals the '
                                f'current state of the project (e.g. type checker, test runner, '
                                f'linter, build command) so the next step can fix what it finds.\n'
                                f'Use the project context to choose the correct command.\n'
                                f'Set output_file so the output is captured for re-planning.\n'
                                f'Example:\n'
                                f'  queue_task(description="Run `tsc --noEmit 2>&1 | head -80`",'
                                f' task_type="discovery", output_file="errors.txt")\n'
                                f'{ctx_section}\n\n'
                                f'Call queue_task now with one discovery task.'
                            )
                            if retry_result.tool_calls == 0:
                                raw_retry = _strip_thinking(retry_result.final_output).strip()
                                fallback_tasks = _parse_tasks_from_planning_output(raw_retry)
                                for desc, task_type, output_file in fallback_tasks:
                                    if desc.strip():
                                        queue.add_task(
                                            session_id, desc, task_type,
                                            output_file=output_file, context=ctx,
                                        )
                        except Exception:
                            pass
                        pending = queue.count_pending(session_id)
                        if pending == 0:
                            _log('Planning retry also produced 0 tasks — falling back to direct coding')
                            queue.add_task(session_id, prompt, 'coding', context=ctx)
                except Exception as exc:
                    _log(
                        f'WARNING: Planning failed ({type(exc).__name__}: {exc}) '
                        f'— falling back to direct coding'
                    )
                    queue.add_task(session_id, prompt, 'coding', context=prompt[:500])

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
                tool_registry=_make_coding_registry(),
            )
            _write_task_md(task, task_runtime.scratchpad_root)
            try:
                result = agent.run(_build_task_prompt(task))
                clean_output = _strip_thinking(result.final_output)
                # Nudge loop: re-prompt up to 3 times if the model refuses to call tools
                # or falls into chat mode (describing findings instead of executing).
                last_nudge = ''
                for nudge_iter in range(3):
                    in_chat_mode = _looks_like_chat_mode(clean_output)
                    is_empty_output = not clean_output.strip()
                    if not (
                        result.tool_calls == 0
                        or _looks_like_bare_tool_name(clean_output)
                        or _looks_like_text_format_bash(clean_output)
                        or in_chat_mode
                        or is_empty_output
                    ):
                        break
                    # If the required output file already exists and tools ran, the task
                    # is done — the model just narrated the result in chat mode. Nudging
                    # further only grows the context and risks a backend error.
                    if task.output_file and result.tool_calls > 0 and (self.cwd / task.output_file).exists():
                        break
                    if not result.session_id:
                        break
                    # If the model is echoing the nudge verbatim, further nudging is futile.
                    if last_nudge and clean_output.strip() == last_nudge.strip():
                        _log(f'Task {task.id}: model echoed nudge text — aborting nudge loop')
                        break
                    # Escalate to a concrete <tool_call> format example on the 2nd nudge
                    # when the model is stuck outputting a bare tool name (e.g. '*bash*').
                    # _extract_text_format_tool_calls already handles <tool_call> tags, so
                    # showing the model this exact format is the most reliable recovery.
                    if nudge_iter >= 1 and _looks_like_bare_tool_name(clean_output):
                        nudge = _build_format_hint_nudge(task)
                    elif in_chat_mode:
                        nudge = _build_chat_mode_nudge(task, clean_output)
                    elif is_empty_output:
                        nudge = _EMPTY_OUTPUT_NUDGE
                    else:
                        nudge = _build_nudge_prompt(task, clean_output)
                    last_nudge = nudge
                    try:
                        stored = load_agent_session(
                            result.session_id,
                            directory=task_runtime.session_directory,
                        )
                        pre_nudge_result, pre_nudge_output = result, clean_output
                        result = agent.resume(nudge, stored)
                        clean_output = _strip_thinking(result.final_output)
                        # If the backend errored on the nudge (e.g. Ollama HTTP 500 from
                        # a grown context), revert to the pre-nudge result so the reviewer
                        # evaluates the actual task output, not the error string.
                        if result.stop_reason == 'backend_error':
                            _log(f'Task {task.id}: backend error during nudge — reverting to pre-nudge result')
                            result, clean_output = pre_nudge_result, pre_nudge_output
                            break
                    except Exception:
                        break
                # Nudge up to 2 more times if the agent ran tools but the required output
                # file still doesn't exist — the agent stopped too early or went chat-mode
                # after making tool calls.
                for _ in range(2):
                    if not (
                        task.output_file
                        and result.tool_calls > 0
                        and not (self.cwd / task.output_file).exists()
                        and result.session_id
                    ):
                        break
                    try:
                        stored = load_agent_session(
                            result.session_id,
                            directory=task_runtime.session_directory,
                        )
                        nudge = _OUTPUT_FILE_MISSING_NUDGE.format(path=task.output_file)
                        pre_nudge_result, pre_nudge_output = result, clean_output
                        result = agent.resume(nudge, stored)
                        clean_output = _strip_thinking(result.final_output)
                        if result.stop_reason == 'backend_error':
                            _log(f'Task {task.id}: backend error during output-file nudge — reverting')
                            result, clean_output = pre_nudge_result, pre_nudge_output
                            break
                    except Exception:
                        break
            except Exception as exc:
                queue.update_status(task.id, 'failed')
                _log(f'Task {task.id} failed (coding): {exc}')
                outputs.append(f'[ERROR] {task.description}: {exc}')
                continue

            # Fresh-context restart: if the nudge loop couldn't break the model out of a
            # bare-tool-name loop (e.g. outputting '*bash*' on every turn), try once more
            # with a clean session and a rephrased prompt that avoids repeating 'bash'.
            # The poisoned context from multiple "call bash NOW" nudges is likely causing
            # the model to lock in; a fresh start with neutral wording often breaks it.
            if result.tool_calls == 0 and _looks_like_bare_tool_name(clean_output):
                _log(f'Task {task.id}: bare-tool-name stuck — trying fresh-context restart')
                try:
                    fresh_label = f'{task.id}_r'
                    fresh_runtime = _base_runtime(self.cwd, fresh_label)
                    fresh_agent = LocalCodingAgent(
                        model_config=self.config.coding_model,
                        runtime_config=fresh_runtime,
                        # Use the generic addon (not BASH_FIRST) to avoid re-triggering
                        # the '*bash*' loop via keyword repetition.
                        append_system_prompt=_TOOL_USE_SYSTEM_ADDON,
                        tool_registry=_make_coding_registry(),
                    )
                    fresh_result = fresh_agent.run(_build_fresh_task_prompt(task))
                    fresh_output = _strip_thinking(fresh_result.final_output)
                    if fresh_result.tool_calls > 0:
                        result = fresh_result
                        clean_output = fresh_output
                        task_runtime = fresh_runtime
                        agent = fresh_agent
                except Exception:
                    pass

            # Hard fail if the agent never called any tools — no review model needed.
            if result.tool_calls == 0:
                passed = False
                feedback = 'Agent made no tool calls — task was not attempted'
                review_tool_results = ''
                _log(f'Task {task.id} hard-failed: agent made no tool calls')
            else:
                review_tool_results = _extract_tool_results(
                    task_runtime.session_directory,
                    result.session_id or '',
                )
                # Discovery tasks: output file is a hard requirement — don't let the
                # review model decide. If it wasn't written, the task unambiguously failed.
                if task.task_type == 'discovery' and task.output_file:
                    output_path = self.cwd / task.output_file
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        passed = False
                        feedback = f'Discovery output file {task.output_file!r} was not written'
                        _log(f'Task {task.id} (discovery): output file missing — hard fail')
                    else:
                        passed, feedback = _run_review(
                            task, self.config.review_model,
                            agent_output=clean_output,
                            tool_results=review_tool_results,
                            cwd=self.cwd,
                        )
                else:
                    passed, feedback = _run_review(
                        task, self.config.review_model,
                        agent_output=clean_output,
                        tool_results=review_tool_results,
                        cwd=self.cwd,
                    )
            if passed:
                queue.update_status(task.id, 'complete')
                _log(f'Task {task.id} complete')
                if task.task_type == 'discovery' and task.output_file:
                    output_path = self.cwd / task.output_file
                    try:
                        discovery_output = output_path.read_text(encoding='utf-8', errors='replace')
                    except OSError:
                        discovery_output = ''
                    if discovery_output.strip():
                        _log(f'Task {task.id} (discovery): re-planning from {task.output_file!r}')
                        new_count = _run_replan_after_discovery(
                            task, discovery_output, prompt, queue, session_id,
                            self.config, self.cwd, project_ctx,
                        )
                        _log(f'Re-plan queued {new_count} implementation task(s)')
                    else:
                        _log(
                            f'Task {task.id} (discovery): {task.output_file!r} missing or empty, '
                            f'skipping re-plan'
                        )
                else:
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
                        task, self.config.diagnosis_model,
                        agent_output=clean_output, reviewer_feedback=feedback,
                        tool_results=tool_results,
                        attempt=count,
                        project_context=project_ctx,
                    )
                    diagnosed = _strip_heredoc_from_diagnosis(diagnosed)
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
