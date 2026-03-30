"""Regression commands — logistic, robust, stepwise, nonlinear, GLM, best subsets."""

from __future__ import annotations

import numpy as np

from ..registry import register
from ..session import CommandResult


@register("logistic", category="regression",
          description="Binary logistic regression",
          usage="logistic <response> ~ <predictor1> <predictor2> ...")
def cmd_logistic(session, parsed):
    from forgestat.regression.logistic import logistic_regression

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: logistic <y> ~ <x1> <x2> ...")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    r = logistic_regression(X, [int(v) for v in y], feature_names=parsed.predictors)
    lines = [f"Logistic: pseudo-R²={r.pseudo_r_squared:.4f}, AIC={r.aic:.1f}"]
    for name, coef in r.coefficients.items():
        oratio = r.odds_ratios.get(name, 0)
        lines.append(f"  {name}: β={coef:.4f}, OR={oratio:.3f}")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


@register("poisson", category="regression",
          description="Poisson regression (count data)",
          usage="poisson <response> ~ <predictor1> ...")
def cmd_poisson(session, parsed):
    from forgestat.regression.logistic import poisson_regression

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: poisson <y> ~ <x1> ...")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    r = poisson_regression(X, [int(v) for v in y], feature_names=parsed.predictors)
    lines = [f"Poisson: deviance={r.deviance:.2f}, AIC={r.aic:.1f}"]
    for name in parsed.predictors:
        irr = r.irr.get(name, 0)
        lines.append(f"  {name}: IRR={irr:.3f}")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


@register("robust", aliases=["robust_reg"], category="regression",
          description="Robust regression (Huber/bisquare)",
          usage="robust <response> ~ <predictors> [method=huber|bisquare]")
def cmd_robust(session, parsed):
    from forgestat.regression.robust import robust_regression

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: robust <y> ~ <x1> ...")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    method = parsed.named.get("method", "huber")
    r = robust_regression(X, y, feature_names=parsed.predictors, method=method)
    lines = [f"Robust ({method}): R²={r.r_squared:.4f}, {r.n_downweighted} downweighted"]
    for name, coef in r.coefficients.items():
        ols_c = r.ols_coefficients.get(name, 0)
        lines.append(f"  {name}: robust={coef:.4f}, OLS={ols_c:.4f}")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


@register("stepwise", category="regression",
          description="Stepwise regression",
          usage="stepwise <response> ~ <predictors> [method=forward|backward|both]")
def cmd_stepwise(session, parsed):
    from forgestat.regression.stepwise import stepwise

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: stepwise <y> ~ <x1> <x2> ...")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    method = parsed.named.get("method", "both")
    r = stepwise(X, y, feature_names=parsed.predictors, method=method)
    selected = ", ".join(r.selected_features) if r.selected_features else "none"
    r2 = r.final_model.r_squared if r.final_model else 0
    summary = f"Stepwise ({method}): selected [{selected}], R²={r2:.4f}"
    return CommandResult(success=True, data={"selected": r.selected_features, "r_squared": r2},
                         summary=summary)


@register("nonlinear", aliases=["nlin", "curvefit"], category="regression",
          description="Nonlinear curve fitting",
          usage="nonlinear <x> <y> model=<exponential|logistic|power|...>")
def cmd_nonlinear(session, parsed):
    from forgestat.regression.nonlinear import curve_fit

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: nonlinear <x_col> <y_col> model=<name>")

    x = session.get_numeric_column(parsed.positional[0])
    y = session.get_numeric_column(parsed.positional[1])
    model = parsed.named.get("model", "exponential")
    r = curve_fit(x, y, model=model)

    if not r.converged:
        return CommandResult(success=False, error=f"Model '{model}' did not converge")

    params = ", ".join(f"{k}={v:.4f}" for k, v in r.parameters.items())
    summary = f"Nonlinear ({model}): R²={r.r_squared:.4f}, RMSE={r.rmse:.4f}\n  {params}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("glm", category="regression",
          description="Generalized Linear Model",
          usage="glm <response> ~ <predictors> family=<gaussian|poisson|binomial|gamma>")
def cmd_glm(session, parsed):
    from forgestat.regression.glm import glm

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: glm <y> ~ <x1> ... family=<name>")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    family = parsed.named.get("family", "gaussian")
    r = glm(X, y, feature_names=parsed.predictors, family=family)
    lines = [f"GLM ({family}): deviance={r.deviance:.2f}, AIC={r.aic:.1f}"]
    for name, coef in r.coefficients.items():
        pv = r.p_values.get(name, 0)
        sig = "*" if pv < 0.05 else ""
        lines.append(f"  {name}: {coef:.4f} (p={pv:.4f}){sig}")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


@register("best_subsets", aliases=["bestsubsets"], category="regression",
          description="Best subsets regression",
          usage="best_subsets <response> ~ <predictors>")
def cmd_best_subsets(session, parsed):
    from forgestat.regression.best_subsets import best_subsets

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: best_subsets <y> ~ <x1> <x2> ...")

    y = session.get_numeric_column(parsed.response)
    X = np.column_stack([session.get_numeric_column(p) for p in parsed.predictors])
    r = best_subsets(X, y, feature_names=parsed.predictors)
    lines = []
    if r.best_bic:
        lines.append(f"Best (BIC): [{', '.join(r.best_bic.features)}] R²={r.best_bic.r_squared:.4f}")
    if r.best_aic:
        lines.append(f"Best (AIC): [{', '.join(r.best_aic.features)}] R²={r.best_aic.r_squared:.4f}")
    return CommandResult(success=True, data={"best_bic": r.best_bic.__dict__ if r.best_bic else {},
                                              "best_aic": r.best_aic.__dict__ if r.best_aic else {}},
                         summary="\n".join(lines))
