"""Advanced commands — power, reliability, time series, Bayesian."""

from __future__ import annotations

from ..registry import register
from ..session import CommandResult


# --- Power & Sample Size ---

@register("power", category="power", requires_data=False,
          description="Power analysis",
          usage="power test=<ttest|anova|proportion|chi2> effect=<d> n=<size> | power=<target>")
def cmd_power(session, parsed):
    from forgestat.power.sample_size import power_t_test, power_anova, power_proportion, power_chi_square

    test = parsed.named.get("test", "ttest")
    effect = float(parsed.named.get("effect", parsed.named.get("d", 0.5)))
    n = int(parsed.named["n"]) if "n" in parsed.named else None
    target_power = float(parsed.named["power"]) if "power" in parsed.named else None

    if test in ("ttest", "t"):
        r = power_t_test(effect, n=n, power=target_power)
    elif test == "anova":
        k = int(parsed.named.get("k", 3))
        r = power_anova(effect, k, n_per_group=n, power=target_power)
    elif test in ("proportion", "prop"):
        p1 = float(parsed.named.get("p1", 0.6))
        p0 = float(parsed.named.get("p0", 0.5))
        r = power_proportion(p1, p0=p0, n=n, power=target_power)
    elif test == "chi2":
        df = int(parsed.named.get("df", 1))
        r = power_chi_square(effect, df, n=n, power=target_power)
    else:
        return CommandResult(success=False, error=f"Unknown test type: {test}")

    if n is not None:
        summary = f"Power ({test}): n={n}, d={effect}, power={r.power:.3f}"
    else:
        summary = f"Sample size ({test}): d={effect}, target power={target_power}, n={r.sample_size}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("sample_size", aliases=["samplesize", "ss"], category="power", requires_data=False,
          description="Sample size for CI precision",
          usage="sample_size std=<value> width=<margin> | sample_size prop=<value> width=<margin>")
def cmd_sample_size(session, parsed):
    from forgestat.power.sample_size import sample_size_for_ci

    width = float(parsed.named.get("width", parsed.named.get("margin", 1.0)))
    std = float(parsed.named["std"]) if "std" in parsed.named else None
    prop = float(parsed.named["prop"]) if "prop" in parsed.named else None

    if std is None and prop is None:
        return CommandResult(success=False, error="Usage: sample_size std=<value> width=<margin>")

    n = sample_size_for_ci(target_width=width, std=std, proportion=prop)
    summary = f"Required n = {n} for ±{width} margin"
    return CommandResult(success=True, data={"n": n, "margin": width}, summary=summary)


# --- Reliability ---

@register("weibull", category="reliability",
          description="Weibull distribution fit",
          usage="weibull <column>")
def cmd_weibull(session, parsed):
    from forgestat.reliability.distributions import weibull_fit

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: weibull <column>")

    vals = session.get_numeric_column(col)
    r = weibull_fit(vals)
    summary = f"Weibull: β={r.shape:.3f}, η={r.scale:.1f}, B10={r.b10_life:.1f}, mode={r.failure_mode}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("kaplan_meier", aliases=["km", "survival"], category="reliability",
          description="Kaplan-Meier survival curve",
          usage="kaplan_meier <time_col> [event_col]")
def cmd_kaplan_meier(session, parsed):
    from forgestat.reliability.survival import kaplan_meier

    if not parsed.positional:
        return CommandResult(success=False, error="Usage: kaplan_meier <time_col> [event_col]")

    times = session.get_numeric_column(parsed.positional[0])
    events = None
    if len(parsed.positional) >= 2:
        events = [bool(int(v)) for v in session.get_column(parsed.positional[1])]

    r = kaplan_meier(times, events)
    median = f"{r.median_survival:.1f}" if r.median_survival else "not reached"
    summary = f"KM: n={r.n_total}, events={r.n_events}, censored={r.n_censored}, median={median}"
    return CommandResult(success=True, data={"n_total": r.n_total, "n_events": r.n_events,
                                              "median": r.median_survival}, summary=summary)


@register("cox", aliases=["cox_ph"], category="reliability",
          description="Cox proportional hazards",
          usage="cox <time_col> <event_col> ~ <covariate1> <covariate2>")
def cmd_cox(session, parsed):
    from forgestat.reliability.cox import cox_ph

    if not parsed.response or not parsed.predictors or len(parsed.positional) < 1:
        return CommandResult(success=False, error="Usage: cox <time> <event> ~ <covar1> ...")

    # Parse: cox time event ~ covar1 covar2
    # positional[0] is the event column (time is response from formula)
    # Swap: in "cox time event ~ covars", response=time, positional[0]=event before ~
    # Actually let's use a simpler pattern
    if len(parsed.positional) >= 1:
        times = session.get_numeric_column(parsed.response)
        events = [bool(int(v)) for v in session.get_column(parsed.positional[0])]
        covariates = {p: session.get_numeric_column(p) for p in parsed.predictors}
    else:
        return CommandResult(success=False, error="Usage: cox <time> <event> ~ <covar1> ...")

    r = cox_ph(times, events, covariates)
    lines = [f"Cox PH (n={r.n}, events={r.n_events}):"]
    for name, hr in r.hazard_ratios.items():
        lines.append(f"  {name}: HR={hr:.3f}")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


# --- Time Series ---

@register("arima", category="timeseries",
          description="ARIMA forecast",
          usage="arima <column> [p=1 d=1 q=1 forecast=10]")
def cmd_arima(session, parsed):
    from forgestat.timeseries.forecasting import arima

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: arima <column>")

    vals = session.get_numeric_column(col)
    p = int(parsed.named.get("p", 1))
    d = int(parsed.named.get("d", 1))
    q = int(parsed.named.get("q", 1))
    steps = int(parsed.named.get("forecast", parsed.named.get("steps", 10)))

    r = arima(vals, order=(p, d, q), forecast_steps=steps)
    summary = f"ARIMA({p},{d},{q}): AIC={r.aic:.1f}, {len(r.forecast)} forecasts"
    if r.ljung_box_p is not None:
        summary += f", Ljung-Box p={r.ljung_box_p:.4f}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("acf", aliases=["pacf", "acf_pacf"], category="timeseries",
          description="ACF/PACF analysis",
          usage="acf <column> [lags=20]")
def cmd_acf(session, parsed):
    from forgestat.timeseries.correlation import acf_pacf

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: acf <column>")

    vals = session.get_numeric_column(col)
    lags = int(parsed.named.get("lags", 20))
    r = acf_pacf(vals, n_lags=lags)
    order = r.suggested_order
    summary = f"ACF/PACF: suggested order p={order.get('p', 0)}, q={order.get('q', 0)}"
    if r.ljung_box_p is not None:
        summary += f", Ljung-Box p={r.ljung_box_p:.4f}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("decompose", aliases=["decomposition"], category="timeseries",
          description="Seasonal decomposition",
          usage="decompose <column> [period=12 model=additive]")
def cmd_decompose(session, parsed):
    from forgestat.timeseries.decomposition import classical_decompose

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: decompose <column>")

    vals = session.get_numeric_column(col)
    period = int(parsed.named.get("period", 12))
    model = parsed.named.get("model", "additive")
    r = classical_decompose(vals, period=period, model=model)
    summary = (f"Decomposition ({model}): seasonal strength={r.seasonal_strength:.2f}, "
               f"trend={r.trend_direction} ({r.trend_change:+.2f})")
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("changepoint", aliases=["pelt"], category="timeseries",
          description="Change point detection (PELT)",
          usage="changepoint <column> [penalty=bic]")
def cmd_changepoint(session, parsed):
    from forgestat.timeseries.changepoint import pelt

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: changepoint <column>")

    vals = session.get_numeric_column(col)
    penalty = parsed.named.get("penalty", "bic")
    r = pelt(vals, penalty=penalty)
    cp_indices = [cp.index for cp in r.changepoints]
    summary = f"PELT: {len(r.changepoints)} changepoint(s) at indices {cp_indices}, {r.n_segments} segments"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("granger", category="timeseries",
          description="Granger causality test",
          usage="granger <x_col> <y_col> [max_lag=4]")
def cmd_granger(session, parsed):
    from forgestat.timeseries.causality import granger_causality

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: granger <x_col> <y_col>")

    x = session.get_numeric_column(parsed.positional[0])
    y = session.get_numeric_column(parsed.positional[1])
    max_lag = int(parsed.named.get("max_lag", parsed.named.get("lags", 4)))
    r = granger_causality(x, y, max_lag=max_lag)
    status = "CAUSES" if r.x_causes_y else "DOES NOT CAUSE"
    x_name, y_name = parsed.positional[0], parsed.positional[1]
    summary = f"Granger: {x_name} {status} {y_name} (best lag={r.best_lag}, p={r.best_p_value:.4f})"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- Bayesian ---

@register("bayes_ttest", aliases=["bttest"], category="bayesian", requires_data=True,
          description="Bayesian t-test",
          usage="bayes_ttest <col> mu=<value> | bayes_ttest <col1> <col2>")
def cmd_bayes_ttest(session, parsed):
    from forgestat.bayesian.tests import bayesian_ttest_one_sample, bayesian_ttest_two_sample

    cols = parsed.positional
    mu = parsed.named.get("mu")

    if len(cols) == 1 and mu is not None:
        vals = session.get_numeric_column(cols[0])
        r = bayesian_ttest_one_sample(vals, mu=float(mu))
    elif len(cols) == 2:
        x1 = session.get_numeric_column(cols[0])
        x2 = session.get_numeric_column(cols[1])
        r = bayesian_ttest_two_sample(x1, x2)
    else:
        return CommandResult(success=False, error="Usage: bayes_ttest <col> mu=<val> | bayes_ttest <c1> <c2>")

    ci = f"[{r.credible_interval[0]:.3f}, {r.credible_interval[1]:.3f}]" if r.credible_interval else ""
    summary = f"Bayesian t-test: BF₁₀={r.bf10:.2f} ({r.bf_label}), 95% CI {ci}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("bayes_proportion", aliases=["bprop"], category="bayesian", requires_data=False,
          description="Bayesian proportion inference",
          usage="bayes_proportion successes=<n> n=<total>")
def cmd_bayes_proportion(session, parsed):
    from forgestat.bayesian.tests import bayesian_proportion

    s = int(parsed.named.get("successes", parsed.named.get("s", 0)))
    n = int(parsed.named.get("n", 0))
    if n <= 0:
        return CommandResult(success=False, error="Usage: bayes_proportion successes=<n> n=<total>")

    r = bayesian_proportion(s, n)
    ci = f"[{r.credible_interval[0]:.3f}, {r.credible_interval[1]:.3f}]" if r.credible_interval else ""
    summary = f"Bayesian proportion: mean={r.posterior_mean:.3f}, 95% CI {ci}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)
