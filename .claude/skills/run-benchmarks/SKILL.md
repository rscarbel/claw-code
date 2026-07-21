---
name: run-benchmarks
description: Run the claw-code-agent benchmark systems — local task benchmarks (benchmarks/run.py) and standard eval suites (benchmarks/run_suite.py, e.g. HumanEval, MBPP, SWE-Bench, Aider, GSM8K). Use when evaluating a model or a pipeline change, running a smoke test, or adding a new suite. Triggers: "run the benchmarks", "benchmark this model", "run humaneval", "measure pass rate", "add a benchmark suite".
---

# Running benchmarks

Two systems live under `benchmarks/`, both run as modules from the repo root:

## 1. Standard evaluation suites (`run_suite.py`)
Point the OpenAI-compatible env vars at your endpoint first:
```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"   # Ollama, vLLM, etc.
export OPENAI_API_KEY="none"
export OPENAI_MODEL="qwen3.6:35b-a3b"

python3 -m benchmarks.run_suite --list
python3 -m benchmarks.run_suite --suite humaneval --limit 5 -v   # quick smoke test
python3 -m benchmarks.run_suite --all -o results.json
```
Suites are implemented in `benchmarks/suites/` (`humaneval`, `mbpp`, `swe_bench`, `aider`, `livecodebench`, `codeforces`, `gsm8k`, `math_bench`, `aime`, `gpqa`, `mmlu_pro`, `ifeval`, `bfcl`, `tau2`, …), each subclassing `suites/base.py`.

## 2. Local task benchmarks (`run.py`)
Custom tasks that exercise agent capabilities directly:
```bash
python3 -m benchmarks.run --list
python3 -m benchmarks.run --task file-create-basic
python3 -m benchmarks.run --category bugfix --verbose
```
Task definitions: `benchmarks/tasks/definitions.py`.

## Datasets
Larger datasets are downloaded, not committed. `benchmarks/data/*.jsonl` and `benchmarks/data/manifest.json` are gitignored; fetch via `python3 -m benchmarks.download_datasets`. Benchmark outputs (`benchmark_artifacts/`, `humaneval_results.json`, `tb2*.json`, `output_terminal/`) are also gitignored — don't commit them.

## Adding a suite
1. Add `benchmarks/suites/<name>.py` subclassing the base suite.
2. Register it in `benchmarks/suites/__init__.py`.
3. If it needs a dataset, wire it into `download_datasets.py`.
4. Add a test under `tests/` (see `test_benchmark_artifacts.py`, `test_benchmark_download_datasets.py`).
