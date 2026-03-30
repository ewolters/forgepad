"""Data inspection commands — info, describe, columns, head, tail, unique, missing."""

from __future__ import annotations

from ..registry import register
from ..session import CommandResult


@register("help", category="general", requires_data=False, description="Show available commands",
          usage="help [category]")
def cmd_help(session, parsed):
    from ..registry import REGISTRY
    cat = parsed.positional[0] if parsed.positional else None
    cmds = REGISTRY.list_commands(category=cat)

    lines = []
    current_cat = ""
    for c in cmds:
        if c.category != current_cat:
            current_cat = c.category
            lines.append(f"\n[{current_cat}]")
        aliases = f" ({', '.join(c.aliases)})" if c.aliases else ""
        lines.append(f"  {c.verb}{aliases} — {c.description}")

    return CommandResult(success=True, summary="\n".join(lines))


@register("info", category="data", description="Dataset summary",
          usage="info")
def cmd_info(session, parsed):
    name = session.metadata.get("name", "unnamed")
    return CommandResult(
        success=True,
        summary=f"Dataset: {name} | {session.n_rows} rows × {len(session.columns)} columns",
        data={"name": name, "n_rows": session.n_rows, "n_cols": len(session.columns),
              "columns": session.columns},
    )


@register("columns", aliases=["cols"], category="data", description="List all columns with types",
          usage="columns")
def cmd_columns(session, parsed):
    col_info = []
    for col in session.columns:
        vals = session.data[col]
        n_num = sum(1 for v in vals if isinstance(v, (int, float)))
        dtype = "numeric" if n_num > len(vals) * 0.5 else "text"
        col_info.append(f"  {col} ({dtype}, {len(vals)} values)")
    return CommandResult(success=True, summary="\n".join(col_info),
                         data={"columns": session.columns})


@register("head", category="data", description="First n rows", usage="head [n]")
def cmd_head(session, parsed):
    n = int(parsed.positional[0]) if parsed.positional else 5
    rows = {col: session.data[col][:n] for col in session.columns}
    lines = [" | ".join(session.columns)]
    for i in range(min(n, session.n_rows)):
        lines.append(" | ".join(str(rows[c][i]) for c in session.columns))
    return CommandResult(success=True, summary="\n".join(lines), data=rows)


@register("describe", aliases=["summary"], category="data", description="Descriptive statistics",
          usage="describe [column]")
def cmd_describe(session, parsed):
    from forgestat.exploratory.univariate import describe

    if parsed.positional:
        col = parsed.positional[0]
        vals = session.get_numeric_column(col)
        result = describe(vals)
        summary = (f"{col}: n={result.n}, mean={result.mean:.4f}, std={result.std:.4f}, "
                   f"median={result.median:.4f}, min={result.min:.4f}, max={result.max:.4f}")
        return CommandResult(success=True, summary=summary,
                             data={"column": col, "n": result.n, "mean": result.mean,
                                   "std": result.std, "median": result.median})
    else:
        # All numeric columns
        lines = []
        for col in session.columns:
            try:
                vals = session.get_numeric_column(col)
                if len(vals) < 2:
                    continue
                result = describe(vals)
                lines.append(f"{col}: mean={result.mean:.3f}, std={result.std:.3f}, "
                             f"n={result.n}")
            except Exception:
                pass
        return CommandResult(success=True, summary="\n".join(lines))


@register("missing", category="data", description="Missing value report", usage="missing")
def cmd_missing(session, parsed):
    lines = []
    for col in session.columns:
        vals = session.data[col]
        n_miss = sum(1 for v in vals if v is None or (isinstance(v, float) and v != v))
        pct = 100 * n_miss / len(vals) if vals else 0
        if n_miss > 0:
            lines.append(f"  {col}: {n_miss}/{len(vals)} ({pct:.1f}%)")
    if not lines:
        return CommandResult(success=True, summary="No missing values")
    return CommandResult(success=True, summary="Missing values:\n" + "\n".join(lines))


@register("unique", category="data", description="Unique values in a column",
          usage="unique <column>", min_args=1)
def cmd_unique(session, parsed):
    col = parsed.positional[0]
    vals = session.get_column(col)
    uniq = sorted(set(str(v) for v in vals))
    n = len(uniq)
    shown = uniq[:20]
    summary = f"{col}: {n} unique values"
    if n > 20:
        summary += " (showing first 20)"
    summary += "\n  " + ", ".join(shown)
    return CommandResult(success=True, summary=summary, data={"column": col, "n_unique": n})
