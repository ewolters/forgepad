"""Session — persistent state across commands.

Holds loaded data, analysis history, named results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResult:
    """Result of executing a single command."""

    command: str = ""  # original command string
    verb: str = ""
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)  # typed result as dict
    charts: list[Any] = field(default_factory=list)  # ChartSpec objects
    summary: str = ""  # one-line text summary
    error: str = ""
    result_name: str | None = None  # if user assigned: ttest yield mu=80 -> t1


def _ensure_commands_loaded():
    """Import command modules to trigger @register decorators."""
    from .commands import data, stats, quality, visualization  # noqa: F401


class Session:
    """Stateful analysis session.

    Holds the loaded dataset and history of results.
    Commands can reference columns by name and previous results by alias.
    """

    _commands_loaded = False

    def __init__(self):
        if not Session._commands_loaded:
            _ensure_commands_loaded()
            Session._commands_loaded = True
        self.data: dict[str, list[float | str]] = {}
        self.columns: list[str] = []
        self.n_rows: int = 0
        self.history: list[CommandResult] = []
        self.named_results: dict[str, CommandResult] = {}
        self.metadata: dict[str, Any] = {}  # dataset name, source, etc.

    def load_data(self, data: dict[str, list]) -> None:
        """Load a dataset into the session.

        Args:
            data: Dict of column_name → values. All columns same length.
        """
        self.data = {k: list(v) for k, v in data.items()}
        self.columns = list(self.data.keys())
        self.n_rows = len(next(iter(self.data.values()))) if self.data else 0

    def get_column(self, name: str) -> list:
        """Get column values by name. Raises KeyError if not found."""
        if name not in self.data:
            # Case-insensitive fallback
            for col in self.columns:
                if col.lower() == name.lower():
                    return self.data[col]
            raise KeyError(f"Column '{name}' not found. Available: {', '.join(self.columns)}")
        return self.data[name]

    def get_numeric_column(self, name: str) -> list[float]:
        """Get column as floats, filtering non-numeric."""
        raw = self.get_column(name)
        result = []
        for v in raw:
            try:
                result.append(float(v))
            except (ValueError, TypeError):
                pass
        return result

    def get_groups(self, response: str, factor: str) -> dict[str, list[float]]:
        """Split response by factor levels into {level: [values]}."""
        resp = self.get_column(response)
        fact = self.get_column(factor)
        groups: dict[str, list[float]] = {}
        for r, f in zip(resp, fact):
            key = str(f)
            if key not in groups:
                groups[key] = []
            try:
                groups[key].append(float(r))
            except (ValueError, TypeError):
                pass
        return groups

    def store_result(self, result: CommandResult) -> None:
        """Add result to history and optionally store by name."""
        self.history.append(result)
        if result.result_name:
            self.named_results[result.result_name] = result

    def run(self, command_str: str) -> CommandResult:
        """Parse and execute a command string.

        Args:
            command_str: User input like "ttest yield mu=80 -> t1"

        Returns:
            CommandResult with data, charts, summary.
        """
        from .parser import parse
        from .executor import execute

        parsed = parse(command_str)
        result = execute(self, parsed)
        self.store_result(result)
        return result
