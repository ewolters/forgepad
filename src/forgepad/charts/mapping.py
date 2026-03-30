"""Chart mapping — typed results → ChartSpec lists.

Each result type maps to a function that produces the right charts.
This is the glue between forge computation and forgeviz rendering.
"""

from __future__ import annotations

from typing import Any


def charts_for(result: Any) -> list:
    """Generate ChartSpec list from a typed result.

    Dispatches on the result's class name. Returns empty list
    if no chart mapping exists for this result type.

    Args:
        result: A typed result from any forge package.

    Returns:
        List of ChartSpec objects.
    """
    type_name = type(result).__name__

    handler = _CHART_HANDLERS.get(type_name)
    if handler:
        try:
            return handler(result)
        except Exception:
            return []
    return []


def _ttest_charts(r) -> list:
    from forgeviz.charts.generic import bar
    from forgeviz.core.colors import STATUS_GREEN, STATUS_RED

    # Mean comparison bar
    if r.mean2 is not None:
        labels = ["Group 1", "Group 2"]
        values = [r.mean1, r.mean2]
    else:
        labels = ["Sample Mean", "H₀"]
        values = [r.mean1, r.mean1 - r.mean_diff]  # mu = mean - diff

    c = STATUS_GREEN if not r.significant else STATUS_RED
    spec = bar(labels, values, title=f"t={r.statistic:.3f}, p={r.p_value:.4f}",
               y_label="Mean", color=c)
    return [spec]


def _anova_charts(r) -> list:
    from forgeviz.charts.distribution import box_plot

    if r.group_means:
        spec = box_plot(
            {name: [] for name in r.group_means},  # empty box — forgeviz handles
            title=f"F={r.statistic:.2f}, p={r.p_value:.4f}",
        )
        # Add means as bar chart instead
        from forgeviz.charts.generic import bar
        spec = bar(
            list(r.group_means.keys()),
            list(r.group_means.values()),
            title=f"Group Means (F={r.statistic:.2f}, p={r.p_value:.4f})",
            y_label="Mean",
        )
        return [spec]
    return []


def _correlation_charts(r) -> list:
    from forgeviz.charts.statistical import heatmap

    if r.matrix:
        names = list(r.matrix.keys())
        z = [[r.matrix[row].get(col, 0) for col in names] for row in names]
        spec = heatmap(names, names, z, title=f"{r.method.capitalize()} Correlation")
        return [spec]
    return []


def _regression_charts(r) -> list:
    from forgeviz.charts.scatter import scatter

    charts = []
    if r.fitted and r.residuals:
        # Fitted vs residuals
        charts.append(scatter(
            r.fitted, r.residuals,
            title="Residuals vs Fitted",
            x_label="Fitted Values",
            y_label="Residuals",
        ))
    return charts


def _capability_charts(r) -> list:
    """Process capability → histogram with spec limits."""
    # This would need the raw data, which isn't in the result
    # Return empty — the command handler adds charts directly
    return []


def _control_chart_charts(r) -> list:
    """SPC control chart result → I-chart."""
    from forgeviz.charts.control import control_chart
    try:
        spec = control_chart(r)
        return [spec]
    except Exception:
        return []


# Registry of type name → chart handler
_CHART_HANDLERS: dict[str, Any] = {
    "TTestResult": _ttest_charts,
    "AnovaResult": _anova_charts,
    "CorrelationResult": _correlation_charts,
    "RegressionResult": _regression_charts,
    "ControlChartResult": _control_chart_charts,
}
