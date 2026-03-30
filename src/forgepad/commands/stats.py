"""Statistical analysis commands — t-tests, ANOVA, correlation, chi-square, etc."""

from __future__ import annotations

from ..charts.mapping import charts_for
from ..registry import register
from ..session import CommandResult


@register("ttest", aliases=["t"], category="stats",
          description="T-test (one-sample, two-sample, or paired)",
          usage="ttest <col> mu=<value> | ttest <col1> <col2> | ttest <response> ~ <factor>")
def cmd_ttest(session, parsed):
    from forgestat.parametric.ttest import one_sample, two_sample, paired

    mu = parsed.named.get("mu")
    pair = parsed.named.get("paired", "").lower() in ("true", "yes", "1")

    if parsed.response and parsed.predictors:
        # Formula: ttest yield ~ group → two-sample
        groups = session.get_groups(parsed.response, parsed.predictors[0])
        group_names = list(groups.keys())
        if len(group_names) != 2:
            return CommandResult(success=False, error=f"Two-sample t-test needs exactly 2 groups, got {len(group_names)}")
        r = two_sample(groups[group_names[0]], groups[group_names[1]])
        summary = f"Welch t-test: t={r.statistic:.4f}, p={r.p_value:.4f}, d={r.effect_size:.3f} ({r.effect_label})"
        return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)

    cols = parsed.positional
    if len(cols) == 1 and mu is not None:
        # One-sample
        vals = session.get_numeric_column(cols[0])
        r = one_sample(vals, mu=float(mu))
        summary = f"One-sample t-test: t={r.statistic:.4f}, p={r.p_value:.4f}, d={r.effect_size:.3f} ({r.effect_label})"
        return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)

    elif len(cols) == 2:
        x1 = session.get_numeric_column(cols[0])
        x2 = session.get_numeric_column(cols[1])
        if pair:
            r = paired(x1, x2)
            summary = f"Paired t-test: t={r.statistic:.4f}, p={r.p_value:.4f}, d={r.effect_size:.3f}"
        else:
            r = two_sample(x1, x2)
            summary = f"Welch t-test: t={r.statistic:.4f}, p={r.p_value:.4f}, d={r.effect_size:.3f}"
        return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)

    return CommandResult(success=False, error="Usage: ttest <col> mu=<value> | ttest <col1> <col2>")


@register("anova", category="stats",
          description="One-way ANOVA",
          usage="anova <response> ~ <factor>")
def cmd_anova(session, parsed):
    from forgestat.parametric.anova import one_way_from_dict

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: anova <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    r = one_way_from_dict(groups)
    summary = f"One-way ANOVA: F={r.statistic:.4f}, p={r.p_value:.4f}, η²={r.effect_size:.3f} ({r.effect_label})"
    return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)


@register("correlation", aliases=["corr"], category="stats",
          description="Correlation matrix",
          usage="correlation <col1> <col2> [col3...] [method=pearson|spearman]")
def cmd_correlation(session, parsed):
    from forgestat.parametric.correlation import correlation

    cols = parsed.positional
    if len(cols) < 2:
        # Default: all numeric columns
        cols = [c for c in session.columns if len(session.get_numeric_column(c)) > 2]
        if len(cols) < 2:
            return CommandResult(success=False, error="Need at least 2 numeric columns")

    method = parsed.named.get("method", "pearson")
    data = {col: session.get_numeric_column(col) for col in cols}
    r = correlation(data, method=method)

    lines = [f"{method.capitalize()} correlation:"]
    for pair in r.pairs[:10]:
        sig = "*" if pair.p_value < 0.05 else ""
        lines.append(f"  {pair.var1} × {pair.var2}: r={pair.r:.3f}{sig} (p={pair.p_value:.4f})")

    return CommandResult(success=True, data={"pairs": [p.__dict__ for p in r.pairs], "matrix": r.matrix},
                         charts=charts_for(r), summary="\n".join(lines))


@register("normality", aliases=["norm"], category="stats",
          description="Normality test",
          usage="normality <column>")
def cmd_normality(session, parsed):
    from forgestat.core.assumptions import check_normality

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: normality <column>")

    vals = session.get_numeric_column(col)
    r = check_normality(vals, label=col)
    status = "NORMAL" if r.passed else "NOT NORMAL"
    summary = f"{col}: {status} — {r.test_name} {r.detail}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("regression", aliases=["reg", "ols"], category="regression",
          description="OLS regression",
          usage="regression <response> ~ <predictor1> <predictor2> ...")
def cmd_regression(session, parsed):
    from forgestat.regression.linear import ols

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: regression <y> ~ <x1> <x2> ...")

    y = session.get_numeric_column(parsed.response)
    X = []
    for pred in parsed.predictors:
        X.append(session.get_numeric_column(pred))

    import numpy as np
    X_arr = np.column_stack(X) if X else np.array([]).reshape(len(y), 0)

    r = ols(X_arr, y, feature_names=parsed.predictors)
    lines = [f"R²={r.r_squared:.4f}, Adj R²={r.adj_r_squared:.4f}, F={r.f_statistic:.2f} (p={r.f_p_value:.4f})"]
    for name, coef in r.coefficients.items():
        pv = r.p_values.get(name, 0)
        sig = "*" if pv < 0.05 else ""
        lines.append(f"  {name}: {coef:.4f} (p={pv:.4f}){sig}")
    return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary="\n".join(lines))


@register("equivalence", aliases=["tost"], category="stats",
          description="TOST equivalence test",
          usage="equivalence <col1> <col2> margin=<value>")
def cmd_equivalence(session, parsed):
    from forgestat.parametric.equivalence import tost

    if len(parsed.positional) < 2 or "margin" not in parsed.named:
        return CommandResult(success=False, error="Usage: equivalence <col1> <col2> margin=<value>")

    x1 = session.get_numeric_column(parsed.positional[0])
    x2 = session.get_numeric_column(parsed.positional[1])
    margin = float(parsed.named["margin"])
    r = tost(x1, x2, margin=margin)
    status = "EQUIVALENT" if r.equivalent else "NOT EQUIVALENT"
    summary = f"TOST: {status} (p={r.p_tost:.4f}, margin=±{margin})"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("proportion", aliases=["prop"], category="stats",
          description="Proportion test",
          usage="proportion successes=<n> n=<total> p0=<hypothesized>")
def cmd_proportion(session, parsed):
    from forgestat.parametric.proportion import one_proportion

    s = int(parsed.named.get("successes", parsed.named.get("s", 0)))
    n = int(parsed.named.get("n", 0))
    p0 = float(parsed.named.get("p0", 0.5))
    if n <= 0:
        return CommandResult(success=False, error="Usage: proportion successes=<n> n=<total> p0=<value>")

    r = one_proportion(s, n, p0=p0)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"One-proportion z-test: p̂={r.p_hat:.3f}, z={r.statistic:.3f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)
