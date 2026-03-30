# ForgePad

Command-driven analysis workbench. Parses text commands (e.g., `ttest yield mu=80`) and routes them to Forge computation packages. The execution engine behind SVEND's Analysis Workbench.

## Install

```bash
pip install forgepad
```

## Quick Start

```python
from forgepad.parser import parse
from forgepad.session import Session
from forgepad.executor import execute

session = Session()
cmd = parse("ttest yield mu=80")
result = execute(session, cmd)
print(result.output)
```

## Modules

| Module | Purpose |
|---|---|
| `parser` | Tokenizes command strings into `ParsedCommand` objects |
| `registry` | Maps verbs to `CommandDef` entries with metadata |
| `executor` | Resolves parsed commands against registry, dispatches to forge packages |
| `session` | `Session` state (loaded datasets, history) and `CommandResult` |
| `commands.data` | Data loading/manipulation commands |
| `commands.stats` | Statistical tests (t-test, ANOVA, etc.) |
| `commands.quality` | Quality tools (capability, control charts) |
| `commands.visualization` | Chart generation commands |
| `commands.advanced` | Advanced analysis commands |
| `commands.domain` | Domain-specific commands |
| `commands.regression` | Regression analysis commands |
| `charts.mapping` | Maps typed results to chart specifications |

## Dependencies

- `forgestat` >= 0.1.0
- `forgespc` >= 0.1.0
- `forgeviz` >= 0.1.0
- Optional: `forgedoe`, `forgesiop`, `forgesia`

## Tests

```bash
python3 -m pytest tests/ -q
```

42 tests covering parser, registry, executor, session, and all command modules.

## License

MIT
