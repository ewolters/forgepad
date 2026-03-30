"""Command parser — transforms user input into structured commands.

Syntax:
    verb arg1 arg2 key=value key2=value2 -> result_name

Examples:
    ttest yield mu=80
    anova yield ~ machine
    correlation temp pressure yield
    scatter temp yield
    capability yield lsl=75 usl=95
    regression yield ~ temp pressure temp*pressure
    describe
    imr yield -> chart1
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedCommand:
    """Structured representation of a user command."""

    raw: str
    verb: str = ""
    positional: list[str] = field(default_factory=list)
    named: dict[str, str] = field(default_factory=dict)
    formula: str | None = None  # "yield ~ temp pressure" → response ~ predictors
    response: str | None = None  # extracted from formula
    predictors: list[str] = field(default_factory=list)  # extracted from formula
    result_name: str | None = None  # "-> name" alias


def parse(command_str: str) -> ParsedCommand:
    """Parse a command string into structured form.

    Args:
        command_str: Raw user input.

    Returns:
        ParsedCommand with verb, args, named params, formula.
    """
    raw = command_str.strip()
    if not raw:
        return ParsedCommand(raw=raw)

    # Extract result name: "... -> name"
    result_name = None
    if " -> " in raw:
        raw, result_name = raw.rsplit(" -> ", 1)
        result_name = result_name.strip()

    # Also support "... as name"
    if result_name is None and " as " in raw:
        parts = raw.rsplit(" as ", 1)
        if len(parts) == 2 and re.match(r"^[a-zA-Z_]\w*$", parts[1].strip()):
            raw = parts[0]
            result_name = parts[1].strip()

    # Tokenize (respecting quoted strings)
    tokens = _tokenize(raw)
    if not tokens:
        return ParsedCommand(raw=command_str)

    verb = tokens[0].lower()
    rest = tokens[1:]

    # Check for formula syntax: "response ~ predictor1 predictor2"
    formula = None
    response = None
    predictors = []

    if "~" in rest:
        tilde_idx = rest.index("~")
        if tilde_idx > 0:
            response = rest[tilde_idx - 1]
            predictors = [t for t in rest[tilde_idx + 1:] if "=" not in t]
            formula = f"{response} ~ {' '.join(predictors)}"
            # Remove formula parts from rest
            rest = rest[:tilde_idx - 1] + [t for t in rest[tilde_idx + 1:] if "=" in t]

    # Separate positional and named args
    positional = []
    named = {}
    for token in rest:
        if "=" in token and not token.startswith("="):
            key, _, value = token.partition("=")
            named[key.lower()] = value
        else:
            positional.append(token)

    return ParsedCommand(
        raw=command_str,
        verb=verb,
        positional=positional,
        named=named,
        formula=formula,
        response=response,
        predictors=predictors,
        result_name=result_name,
    )


def _tokenize(s: str) -> list[str]:
    """Split on whitespace, respecting quoted strings."""
    tokens = []
    current = ""
    in_quote = False
    quote_char = ""

    for ch in s:
        if in_quote:
            if ch == quote_char:
                in_quote = False
            else:
                current += ch
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif ch == " " or ch == "\t":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch

    if current:
        tokens.append(current)

    return tokens
