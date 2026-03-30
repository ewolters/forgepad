"""Command registry — maps verbs to forge functions.

Each command entry defines:
- verb: the command name
- aliases: alternative names
- handler: function(session, parsed) → CommandResult
- description: help text
- usage: example usage string
- category: grouping for help display
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CommandDef:
    """Definition of a registered command."""

    verb: str
    handler: Callable
    description: str = ""
    usage: str = ""
    category: str = "general"
    aliases: list[str] = field(default_factory=list)
    min_args: int = 0
    requires_data: bool = True


class Registry:
    """Command registry — singleton mapping verbs to handlers."""

    def __init__(self):
        self._commands: dict[str, CommandDef] = {}
        self._aliases: dict[str, str] = {}  # alias → canonical verb

    def register(self, cmd: CommandDef) -> None:
        """Register a command."""
        self._commands[cmd.verb] = cmd
        for alias in cmd.aliases:
            self._aliases[alias] = cmd.verb

    def get(self, verb: str) -> CommandDef | None:
        """Look up a command by verb or alias."""
        canonical = self._aliases.get(verb, verb)
        return self._commands.get(canonical)

    def list_commands(self, category: str | None = None) -> list[CommandDef]:
        """List all registered commands, optionally filtered by category."""
        cmds = list(self._commands.values())
        if category:
            cmds = [c for c in cmds if c.category == category]
        cmds.sort(key=lambda c: (c.category, c.verb))
        return cmds

    @property
    def categories(self) -> list[str]:
        return sorted(set(c.category for c in self._commands.values()))

    @property
    def verbs(self) -> list[str]:
        return sorted(list(self._commands.keys()) + list(self._aliases.keys()))


# Global registry instance
REGISTRY = Registry()


def register(verb: str, **kwargs):
    """Decorator to register a command handler."""
    def decorator(fn):
        cmd = CommandDef(verb=verb, handler=fn, **kwargs)
        REGISTRY.register(cmd)
        return fn
    return decorator
