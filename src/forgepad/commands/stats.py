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


# --- Chi-square & Fisher ---

@register("chi2", aliases=["chisquare"], category="stats",
          description="Chi-square test of independence",
          usage="chi2 <row_col> <col_col>")
def cmd_chi2(session, parsed):
    from forgestat.parametric.chi_square import chi_square_independence

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: chi2 <row_col> <col_col>")

    rows = session.get_column(parsed.positional[0])
    cols = session.get_column(parsed.positional[1])
    row_levels = sorted(set(str(r) for r in rows))
    col_levels = sorted(set(str(c) for c in cols))

    observed = []
    for rl in row_levels:
        row = []
        for cl in col_levels:
            count = sum(1 for r, c in zip(rows, cols) if str(r) == rl and str(c) == cl)
            row.append(count)
        observed.append(row)

    r = chi_square_independence(observed, row_labels=row_levels, col_labels=col_levels)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"χ²={r.statistic:.4f}, p={r.p_value:.4f}, V={r.cramers_v:.3f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)


@register("fisher", category="stats",
          description="Fisher's exact test (2×2)",
          usage="fisher <row_col> <col_col>")
def cmd_fisher(session, parsed):
    from forgestat.parametric.chi_square import fisher_exact

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: fisher <row_col> <col_col>")

    rows = session.get_column(parsed.positional[0])
    cols = session.get_column(parsed.positional[1])
    row_levels = sorted(set(str(r) for r in rows))
    col_levels = sorted(set(str(c) for c in cols))

    if len(row_levels) != 2 or len(col_levels) != 2:
        return CommandResult(success=False, error="Fisher exact requires exactly 2×2 table")

    table = []
    for rl in row_levels:
        row = [sum(1 for r, c in zip(rows, cols) if str(r) == rl and str(c) == cl) for cl in col_levels]
        table.append(row)

    r = fisher_exact(table)
    summary = f"Fisher exact: OR={r.statistic:.3f}, p={r.p_value:.4f}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- Variance tests ---

@register("f_test", aliases=["ftest"], category="stats",
          description="F-test for equality of variances",
          usage="f_test <col1> <col2>")
def cmd_f_test(session, parsed):
    from forgestat.parametric.variance import f_test

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: f_test <col1> <col2>")

    x1 = session.get_numeric_column(parsed.positional[0])
    x2 = session.get_numeric_column(parsed.positional[1])
    r = f_test(x1, x2)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"F-test: F={r.statistic:.4f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("variance", aliases=["levene", "bartlett"], category="stats",
          description="Test equal variances (Levene's or Bartlett's)",
          usage="variance <response> ~ <factor> [method=levene|bartlett]")
def cmd_variance(session, parsed):
    from forgestat.parametric.variance import variance_test

    if parsed.response and parsed.predictors:
        groups = session.get_groups(parsed.response, parsed.predictors[0])
    elif len(parsed.positional) >= 2:
        groups = {parsed.positional[i]: session.get_numeric_column(parsed.positional[i])
                  for i in range(len(parsed.positional))}
    else:
        return CommandResult(success=False, error="Usage: variance <response> ~ <factor>")

    method = parsed.named.get("method", "levene")
    arrays = list(groups.values())
    labels = list(groups.keys())
    r = variance_test(*arrays, labels=labels, method=method)
    sig = "UNEQUAL" if r.significant else "EQUAL"
    summary = f"{r.test_name}: stat={r.statistic:.4f}, p={r.p_value:.4f} — variances {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- Nonparametric ---

@register("mann_whitney", aliases=["mannwhitney", "mw"], category="nonparametric",
          description="Mann-Whitney U test",
          usage="mann_whitney <col1> <col2> | mann_whitney <response> ~ <factor>")
def cmd_mann_whitney(session, parsed):
    from forgestat.nonparametric.rank_tests import mann_whitney

    if parsed.response and parsed.predictors:
        groups = session.get_groups(parsed.response, parsed.predictors[0])
        names = list(groups.keys())
        if len(names) != 2:
            return CommandResult(success=False, error="Mann-Whitney needs exactly 2 groups")
        x1, x2 = groups[names[0]], groups[names[1]]
    elif len(parsed.positional) >= 2:
        x1 = session.get_numeric_column(parsed.positional[0])
        x2 = session.get_numeric_column(parsed.positional[1])
    else:
        return CommandResult(success=False, error="Usage: mann_whitney <col1> <col2>")

    r = mann_whitney(x1, x2)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"Mann-Whitney U={r.statistic:.1f}, p={r.p_value:.4f}, r={r.effect_size:.3f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("kruskal", aliases=["kruskal_wallis", "kw"], category="nonparametric",
          description="Kruskal-Wallis H test",
          usage="kruskal <response> ~ <factor>")
def cmd_kruskal(session, parsed):
    from forgestat.nonparametric.rank_tests import kruskal_wallis

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: kruskal <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    arrays = list(groups.values())
    labels = list(groups.keys())
    r = kruskal_wallis(*arrays, labels=labels)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"Kruskal-Wallis H={r.statistic:.4f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("wilcoxon", category="nonparametric",
          description="Wilcoxon signed-rank test",
          usage="wilcoxon <col1> <col2>")
def cmd_wilcoxon(session, parsed):
    from forgestat.nonparametric.rank_tests import wilcoxon_signed_rank

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: wilcoxon <col1> <col2>")

    x1 = session.get_numeric_column(parsed.positional[0])
    x2 = session.get_numeric_column(parsed.positional[1])
    r = wilcoxon_signed_rank(x1, x2)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"Wilcoxon W={r.statistic:.1f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("friedman", category="nonparametric",
          description="Friedman test (repeated measures nonparametric)",
          usage="friedman <col1> <col2> <col3> ...")
def cmd_friedman(session, parsed):
    from forgestat.nonparametric.rank_tests import friedman

    if len(parsed.positional) < 3:
        return CommandResult(success=False, error="Usage: friedman <col1> <col2> <col3> ...")

    arrays = [session.get_numeric_column(c) for c in parsed.positional]
    r = friedman(*arrays, labels=parsed.positional)
    sig = "SIGNIFICANT" if r.significant else "NOT SIGNIFICANT"
    summary = f"Friedman χ²={r.statistic:.4f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("runs", aliases=["runs_test"], category="nonparametric",
          description="Runs test for randomness",
          usage="runs <column>")
def cmd_runs(session, parsed):
    from forgestat.nonparametric.rank_tests import runs_test

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: runs <column>")

    vals = session.get_numeric_column(col)
    r = runs_test(vals)
    status = "NON-RANDOM" if r.significant else "RANDOM"
    summary = f"Runs test: z={r.statistic:.3f}, p={r.p_value:.4f} — {status}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- Post-hoc ---

@register("tukey", aliases=["tukey_hsd"], category="posthoc",
          description="Tukey HSD post-hoc",
          usage="tukey <response> ~ <factor>")
def cmd_tukey(session, parsed):
    from forgestat.posthoc.comparisons import tukey_hsd

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: tukey <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    arrays = list(groups.values())
    labels = list(groups.keys())
    r = tukey_hsd(*arrays, labels=labels)
    lines = [f"Tukey HSD ({len(r.comparisons)} comparisons):"]
    for c in r.comparisons:
        sig = "*" if c.significant else ""
        lines.append(f"  {c.group1} vs {c.group2}: diff={c.mean_diff:.3f}, p={c.p_value:.4f}{sig}")
    return CommandResult(success=True, data={"comparisons": [c.__dict__ for c in r.comparisons]},
                         summary="\n".join(lines))


@register("dunnett", category="posthoc",
          description="Dunnett's test vs control",
          usage="dunnett <response> ~ <factor> control=<group>")
def cmd_dunnett(session, parsed):
    from forgestat.posthoc.comparisons import dunnett

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: dunnett <response> ~ <factor> control=<group>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    control_name = parsed.named.get("control", list(groups.keys())[0])
    if control_name not in groups:
        return CommandResult(success=False, error=f"Control group '{control_name}' not found")

    control = groups.pop(control_name)
    treatment_names = list(groups.keys())
    treatments = list(groups.values())
    r = dunnett(control, *treatments, control_name=control_name, treatment_names=treatment_names)
    lines = [f"Dunnett vs '{control_name}':"]
    for c in r.comparisons:
        sig = "*" if c.significant else ""
        lines.append(f"  {c.group1}: diff={c.mean_diff:.3f}, p={c.p_value:.4f}{sig}")
    return CommandResult(success=True, data={"comparisons": [c.__dict__ for c in r.comparisons]},
                         summary="\n".join(lines))


@register("games_howell", aliases=["gameshowell", "gh"], category="posthoc",
          description="Games-Howell (unequal variances)",
          usage="games_howell <response> ~ <factor>")
def cmd_games_howell(session, parsed):
    from forgestat.posthoc.comparisons import games_howell

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: games_howell <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    r = games_howell(*groups.values(), labels=list(groups.keys()))
    lines = [f"Games-Howell ({len(r.comparisons)} comparisons):"]
    for c in r.comparisons:
        sig = "*" if c.significant else ""
        lines.append(f"  {c.group1} vs {c.group2}: diff={c.mean_diff:.3f}, p={c.p_value:.4f}{sig}")
    return CommandResult(success=True, data={"comparisons": [c.__dict__ for c in r.comparisons]},
                         summary="\n".join(lines))


@register("dunn", category="posthoc",
          description="Dunn's test (nonparametric post-hoc)",
          usage="dunn <response> ~ <factor>")
def cmd_dunn(session, parsed):
    from forgestat.posthoc.comparisons import dunn

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: dunn <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    r = dunn(*groups.values(), labels=list(groups.keys()))
    lines = [f"Dunn's test ({len(r.comparisons)} comparisons):"]
    for c in r.comparisons:
        sig = "*" if c.significant else ""
        lines.append(f"  {c.group1} vs {c.group2}: p={c.p_value:.4f}{sig}")
    return CommandResult(success=True, data={"comparisons": [c.__dict__ for c in r.comparisons]},
                         summary="\n".join(lines))


@register("scheffe", category="posthoc",
          description="Scheffé's test (conservative omnibus)",
          usage="scheffe <response> ~ <factor>")
def cmd_scheffe(session, parsed):
    from forgestat.posthoc.comparisons import scheffe

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: scheffe <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    r = scheffe(*groups.values(), labels=list(groups.keys()))
    lines = [f"Scheffé ({len(r.comparisons)} comparisons):"]
    for c in r.comparisons:
        sig = "*" if c.significant else ""
        lines.append(f"  {c.group1} vs {c.group2}: diff={c.mean_diff:.3f}, p={c.p_value:.4f}{sig}")
    return CommandResult(success=True, data={"comparisons": [c.__dict__ for c in r.comparisons]},
                         summary="\n".join(lines))
