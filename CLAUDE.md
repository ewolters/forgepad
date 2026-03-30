# ForgePad -- Command-Driven Analysis Workbench

## What It Is

Text command parser and dispatcher for the Forge ecosystem. Users type commands like `ttest yield mu=80` and ForgePad parses, routes, and executes them against forgestat/forgespc/forgeviz. SVEND's Analysis Workbench wraps this with a web UI.

## Architecture

```
forgepad/
  parser.py         -- Tokenizer: "ttest yield mu=80" -> ParsedCommand(verb, args, kwargs)
  registry.py       -- Verb -> CommandDef mapping. Decorator-based registration.
  executor.py       -- Resolves ParsedCommand + Session -> dispatches to forge package
  session.py        -- Session (loaded data, history) + CommandResult
  commands/         -- Command implementations by domain
    data.py         -- load, filter, describe
    stats.py        -- ttest, anova, correlation
    quality.py      -- capability, control_chart
    visualization.py -- plot, histogram, scatter
    advanced.py     -- advanced analyses
    domain.py       -- domain-specific commands
    regression.py   -- regression models
  charts/
    mapping.py      -- CommandResult -> ChartSpec translation
```

## Running Tests

```bash
cd ~/forgepad
python3 -m pytest tests/ -q          # 42 tests
python3 -m ruff check .              # lint
```

## Key Design Decisions

- **Lazy imports** -- forgestat/forgespc/forgeviz imported at call time, not module load. Keeps import fast and allows partial installs.
- **Parse/Route/Compute/Chart pipeline** -- each stage is independent and testable. Parser knows nothing about computation; executor knows nothing about rendering.
- **Registry is a singleton** -- command modules register via `@register` decorator at import time.
- **Session carries state** -- datasets, variables, and history live on Session. Commands are stateless functions that read/write Session.
- **No direct dependencies** -- forgepad itself is pure orchestration. All computation delegated to forge packages.
