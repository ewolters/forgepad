"""Command executor — routes parsed commands through the registry."""

from __future__ import annotations

from .parser import ParsedCommand
from .registry import REGISTRY
from .session import CommandResult, Session


def execute(session: Session, parsed: ParsedCommand) -> CommandResult:
    """Execute a parsed command against the session.

    Args:
        session: Current session with loaded data.
        parsed: Structured command from parser.

    Returns:
        CommandResult with data, charts, summary.
    """
    if not parsed.verb:
        return CommandResult(
            command=parsed.raw, verb="", success=False,
            error="Empty command",
        )

    cmd_def = REGISTRY.get(parsed.verb)
    if cmd_def is None:
        available = ", ".join(REGISTRY.verbs[:20])
        return CommandResult(
            command=parsed.raw, verb=parsed.verb, success=False,
            error=f"Unknown command: '{parsed.verb}'. Available: {available}",
        )

    if cmd_def.requires_data and not session.data:
        return CommandResult(
            command=parsed.raw, verb=parsed.verb, success=False,
            error=f"'{parsed.verb}' requires loaded data. Use session.load_data() first.",
        )

    try:
        result = cmd_def.handler(session, parsed)
        result.command = parsed.raw
        result.verb = parsed.verb
        result.result_name = parsed.result_name
        return result
    except KeyError as e:
        return CommandResult(
            command=parsed.raw, verb=parsed.verb, success=False,
            error=f"Column not found: {e}",
        )
    except Exception as e:
        return CommandResult(
            command=parsed.raw, verb=parsed.verb, success=False,
            error=str(e),
        )
