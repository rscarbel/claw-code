---
name: parity-audit
description: Track and verify this repo's parity with the upstream Claude Code npm source using PARITY_CHECKLIST.md, TESTING_GUIDE.md, and the parity-audit CLI command. Use when adding a feature to match the npm implementation, updating parity status, or checking what's still unported. Triggers: "is X implemented", "update the parity checklist", "match the npm behavior", "what's left to port".
---

# Parity auditing

This project reimplements the Claude Code npm agent architecture in pure Python. Two living documents plus one command track how close the port is:

- **`PARITY_CHECKLIST.md`** — feature-by-feature implementation status vs the npm source. The source of truth for "is X done".
- **`TESTING_GUIDE.md`** — a concrete command for every implemented feature, organized by runtime surface. When you implement a feature, add its test command here.
- **`parity-audit` command** — compares the Python workspace against a locally-archived TypeScript snapshot when present:
  ```bash
  python3 -m src.main parity-audit
  ```
  The archived npm source is gitignored (`archive/`, `claude-code-sourcemap-main`), so this only produces a full diff on a machine that has the archive checked out; without it, rely on the checklist.

## Related inspection commands

```bash
python3 -m src.main subsystems      # list current Python modules
python3 -m src.main commands        # mirrored command entries from the snapshot
python3 -m src.main tools           # mirrored tool entries from the snapshot
python3 -m src.main command-graph   # command graph segmentation
```

## Workflow when adding a ported feature

1. Check `PARITY_CHECKLIST.md` for its current status and the intended npm behavior.
2. Implement it in `src/` matching the existing module layout (`<feature>_runtime.py`), pure stdlib.
3. Add a `tests/test_<module>.py`.
4. Add a concrete invocation to `TESTING_GUIDE.md`.
5. Update the entry in `PARITY_CHECKLIST.md`.
