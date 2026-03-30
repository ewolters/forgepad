"""ForgePad — command-driven analysis workbench.

Bridges user commands to the Forge computation ecosystem.
Parse → Route → Compute → Chart → Return.

No web framework. No database. SVEND wraps this with views and templates.
"""

__version__ = "0.1.0"

__all__ = [
    # parser
    "ParsedCommand",
    "parse",
    # registry
    "CommandDef",
    "Registry",
    "register",
    # executor
    "execute",
    # session
    "CommandResult",
    "Session",
]
