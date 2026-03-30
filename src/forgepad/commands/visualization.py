"""Visualization commands — chart-only commands that produce ChartSpecs directly."""

from __future__ import annotations

from ..registry import register
from ..session import CommandResult


@register("hist", aliases=["histogram"], category="chart",
          description="Histogram",
          usage="hist <column> [bins=<n>]")
def cmd_hist(session, parsed):
    from forgeviz.charts.distribution import histogram

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: hist <column>")

    vals = session.get_numeric_column(col)
    bins = int(parsed.named.get("bins", 20))
    spec = histogram(vals, bins=bins, title=col)
    return CommandResult(success=True, charts=[spec], summary=f"Histogram of {col} ({len(vals)} values)")


@register("scatter", aliases=["plot"], category="chart",
          description="Scatter plot",
          usage="scatter <x> <y>")
def cmd_scatter(session, parsed):
    from forgeviz.charts.scatter import scatter

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: scatter <x_col> <y_col>")

    x_col, y_col = parsed.positional[0], parsed.positional[1]
    x = session.get_numeric_column(x_col)
    y = session.get_numeric_column(y_col)
    spec = scatter(x, y, title=f"{y_col} vs {x_col}", x_label=x_col, y_label=y_col)
    return CommandResult(success=True, charts=[spec], summary=f"Scatter: {y_col} vs {x_col}")


@register("bar", category="chart",
          description="Bar chart",
          usage="bar <category_col> <value_col>")
def cmd_bar(session, parsed):
    from forgeviz.charts.generic import bar

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: bar <category_col> <value_col>")

    cat_col, val_col = parsed.positional[0], parsed.positional[1]
    cats = [str(v) for v in session.get_column(cat_col)]
    vals = session.get_numeric_column(val_col)

    # Aggregate: mean per category
    from collections import defaultdict
    agg = defaultdict(list)
    for c, v in zip(cats, vals):
        agg[c].append(v)
    labels = sorted(agg.keys())
    means = [sum(agg[k]) / len(agg[k]) for k in labels]

    spec = bar(labels, means, title=f"{val_col} by {cat_col}", x_label=cat_col, y_label=val_col)
    return CommandResult(success=True, charts=[spec], summary=f"Bar: {val_col} by {cat_col}")


@register("boxplot", aliases=["box"], category="chart",
          description="Box plot by factor",
          usage="boxplot <response> ~ <factor>")
def cmd_boxplot(session, parsed):
    from forgeviz.charts.distribution import box_plot

    if not parsed.response or not parsed.predictors:
        if len(parsed.positional) >= 2:
            # Fallback: boxplot response factor
            resp, fact = parsed.positional[0], parsed.positional[1]
        else:
            return CommandResult(success=False, error="Usage: boxplot <response> ~ <factor>")
    else:
        resp, fact = parsed.response, parsed.predictors[0]

    groups = session.get_groups(resp, fact)
    spec = box_plot(groups, title=f"{resp} by {fact}", x_label=fact, y_label=resp)
    return CommandResult(success=True, charts=[spec], summary=f"Box plot: {resp} by {fact}")


@register("line", category="chart",
          description="Line chart",
          usage="line <x_col> <y_col>")
def cmd_line(session, parsed):
    from forgeviz.charts.generic import line

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: line <x_col> <y_col>")

    x_col, y_col = parsed.positional[0], parsed.positional[1]
    x = session.get_numeric_column(x_col)
    y = session.get_numeric_column(y_col)
    spec = line(x, y, title=f"{y_col} vs {x_col}", x_label=x_col, y_label=y_col)
    return CommandResult(success=True, charts=[spec], summary=f"Line: {y_col} vs {x_col}")
