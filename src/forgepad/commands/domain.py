"""Domain commands — exploratory, MSA, DOE, SIOP."""

from __future__ import annotations

import numpy as np

from ..registry import register
from ..session import CommandResult


# --- Exploratory ---

@register("pca", category="exploratory",
          description="Principal Component Analysis",
          usage="pca <col1> <col2> <col3> ... [n_components=2]")
def cmd_pca(session, parsed):
    from forgestat.exploratory.multivariate import pca

    cols = parsed.positional
    if len(cols) < 2:
        cols = [c for c in session.columns if len(session.get_numeric_column(c)) > 2]
    if len(cols) < 2:
        return CommandResult(success=False, error="Need at least 2 numeric columns for PCA")

    data = {c: session.get_numeric_column(c) for c in cols}
    n_comp = int(parsed.named.get("n_components", parsed.named.get("k", len(cols))))
    r = pca(data, n_components=n_comp)

    lines = [f"PCA ({r.n_components} components):"]
    for i in range(r.n_components):
        lines.append(f"  PC{i+1}: {r.variance_explained[i]*100:.1f}% variance")
    lines.append(f"  Cumulative: {r.cumulative_variance[-1]*100:.1f}%")
    return CommandResult(success=True, data={"variance_explained": r.variance_explained,
                                              "loadings": r.loadings}, summary="\n".join(lines))


@register("manova", category="exploratory",
          description="One-way MANOVA",
          usage="manova <y1> <y2> ~ <factor>")
def cmd_manova(session, parsed):
    from forgestat.exploratory.multivariate import one_way_manova

    if not parsed.predictors:
        return CommandResult(success=False, error="Usage: manova <y1> <y2> ~ <factor>")

    # Response vars are positional, factor is first predictor
    resp_cols = parsed.positional
    if parsed.response:
        resp_cols = [parsed.response] + resp_cols
    factor = parsed.predictors[0]

    if len(resp_cols) < 2:
        return CommandResult(success=False, error="Need at least 2 response variables")

    data = {c: session.get_numeric_column(c) for c in resp_cols}
    groups = [str(v) for v in session.get_column(factor)]
    r = one_way_manova(data, groups)
    sig = "SIGNIFICANT" if r.p_value < 0.05 else "NOT SIGNIFICANT"
    summary = f"MANOVA: Wilks' Λ={r.wilks_lambda:.4f}, F={r.f_statistic:.2f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("hotelling", aliases=["t2"], category="exploratory",
          description="Hotelling's T² test",
          usage="hotelling <col1> <col2> ...")
def cmd_hotelling(session, parsed):
    from forgestat.exploratory.multivariate import hotelling_t2_one_sample

    cols = parsed.positional
    if len(cols) < 2:
        return CommandResult(success=False, error="Usage: hotelling <col1> <col2> ...")

    data = {c: session.get_numeric_column(c) for c in cols}
    r = hotelling_t2_one_sample(data)
    sig = "SIGNIFICANT" if r.p_value < 0.05 else "NOT SIGNIFICANT"
    summary = f"Hotelling T²={r.t2_statistic:.4f}, F={r.f_statistic:.4f}, p={r.p_value:.4f} — {sig}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("meta", aliases=["meta_analysis"], category="exploratory", requires_data=False,
          description="Meta-analysis",
          usage="meta effects=<col> se=<col> [model=random]")
def cmd_meta(session, parsed):
    from forgestat.exploratory.meta import meta_analysis

    eff_col = parsed.named.get("effects", parsed.named.get("effect"))
    se_col = parsed.named.get("se", parsed.named.get("stderr"))

    if not eff_col or not se_col:
        return CommandResult(success=False, error="Usage: meta effects=<col> se=<col>")

    effects = session.get_numeric_column(eff_col)
    ses = session.get_numeric_column(se_col)
    model = parsed.named.get("model", "random")
    r = meta_analysis(effects, ses, model=model)
    summary = (f"Meta ({model}): pooled={r.pooled_effect:.3f} [{r.pooled_ci_lower:.3f}, {r.pooled_ci_upper:.3f}], "
               f"I²={r.i_squared:.1f}%, k={r.k}")
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("multi_vari", aliases=["multivari"], category="exploratory",
          description="Multi-vari analysis",
          usage="multi_vari <response> ~ <factor1> <factor2>")
def cmd_multi_vari(session, parsed):
    from forgestat.exploratory.multi_vari import multi_vari

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: multi_vari <response> ~ <factor1> <factor2>")

    data = {parsed.response: session.get_column(parsed.response)}
    for f in parsed.predictors:
        data[f] = session.get_column(f)

    r = multi_vari(data, parsed.response, parsed.predictors)
    lines = [f"Multi-vari: dominant source = {r.dominant_source}"]
    for s in r.sources:
        lines.append(f"  {s.factor}: {s.pct_contribution:.1f}%")
    lines.append(f"  Within: {r.within_pct:.1f}%")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


@register("bootstrap", aliases=["boot"], category="exploratory",
          description="Bootstrap confidence interval",
          usage="bootstrap <column> [stat=mean|median|std]")
def cmd_bootstrap(session, parsed):
    from forgestat.exploratory.univariate import bootstrap_ci

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: bootstrap <column>")

    vals = session.get_numeric_column(col)
    stat = parsed.named.get("stat", "mean")
    r = bootstrap_ci(vals, statistic=stat)
    summary = f"Bootstrap {stat}: {r.estimate:.4f}, 95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("tolerance", category="exploratory",
          description="Tolerance interval",
          usage="tolerance <column> [coverage=0.95 confidence=0.95]")
def cmd_tolerance(session, parsed):
    from forgestat.exploratory.univariate import tolerance_interval

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: tolerance <column>")

    vals = session.get_numeric_column(col)
    coverage = float(parsed.named.get("coverage", 0.95))
    confidence = float(parsed.named.get("confidence", 0.95))
    r = tolerance_interval(vals, coverage=coverage, confidence=confidence)
    summary = f"Tolerance ({coverage*100:.0f}/{confidence*100:.0f}): [{r.lower:.4f}, {r.upper:.4f}], k={r.k_factor:.3f}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- MSA extras ---

@register("icc", category="msa",
          description="Intraclass Correlation Coefficient",
          usage="icc <rater1_col> <rater2_col> [<rater3_col> ...]")
def cmd_icc(session, parsed):
    from forgestat.msa.agreement import icc

    cols = parsed.positional
    if len(cols) < 2:
        return CommandResult(success=False, error="Usage: icc <rater1> <rater2> ...")

    ratings = np.column_stack([session.get_numeric_column(c) for c in cols])
    r = icc(ratings)
    summary = f"ICC({r.icc_type}): {r.icc:.3f}, 95% CI [{r.ci_lower:.3f}, {r.ci_upper:.3f}], p={r.p_value:.4f}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("bland_altman", aliases=["ba"], category="msa",
          description="Bland-Altman method comparison",
          usage="bland_altman <method1_col> <method2_col>")
def cmd_bland_altman(session, parsed):
    from forgestat.msa.agreement import bland_altman

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: bland_altman <method1> <method2>")

    m1 = session.get_numeric_column(parsed.positional[0])
    m2 = session.get_numeric_column(parsed.positional[1])
    r = bland_altman(m1, m2)
    summary = (f"Bland-Altman: bias={r.mean_diff:.4f}, LoA=[{r.loa_lower:.4f}, {r.loa_upper:.4f}]"
               f"{' ⚠ proportional bias' if r.proportional_bias else ''}")
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("linearity", aliases=["gage_linearity"], category="msa",
          description="Gage linearity and bias",
          usage="linearity <reference_col> <measured_col>")
def cmd_linearity(session, parsed):
    from forgestat.msa.agreement import linearity_bias

    if len(parsed.positional) < 2:
        return CommandResult(success=False, error="Usage: linearity <reference> <measured>")

    ref = session.get_numeric_column(parsed.positional[0])
    meas = session.get_numeric_column(parsed.positional[1])
    r = linearity_bias(ref, meas)
    summary = (f"Linearity: slope={r.linearity_slope:.4f} (p={r.linearity_p_value:.4f}), "
               f"bias={r.overall_bias:.4f}")
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("krippendorff", aliases=["kalpha"], category="msa",
          description="Krippendorff's alpha",
          usage="krippendorff <rater1> <rater2> ... [level=nominal|ordinal|interval]")
def cmd_krippendorff(session, parsed):
    from forgestat.msa.kappa import krippendorff_alpha

    cols = parsed.positional
    if len(cols) < 2:
        return CommandResult(success=False, error="Usage: krippendorff <rater1> <rater2> ...")

    ratings = [session.get_column(c) for c in cols]
    level = parsed.named.get("level", "nominal")
    r = krippendorff_alpha(ratings, level=level)
    summary = f"Krippendorff α = {r.value:.3f} ({r.interpretation})"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


# --- Quality extras ---

@register("acceptance", aliases=["sampling_plan"], category="quality", requires_data=False,
          description="Acceptance sampling plan",
          usage="acceptance aql=<value> ltpd=<value> [type=attribute|variable]")
def cmd_acceptance(session, parsed):
    from forgestat.quality.acceptance import attribute_plan, variable_plan

    aql = float(parsed.named.get("aql", 0.01))
    ltpd = float(parsed.named.get("ltpd", 0.05))
    plan_type = parsed.named.get("type", "attribute")

    if plan_type == "variable":
        r = variable_plan(aql=aql, ltpd=ltpd)
        summary = f"Variable plan: n={r.sample_size}, k={r.k_value:.3f}"
    else:
        r = attribute_plan(aql=aql, ltpd=ltpd)
        summary = f"Attribute plan: n={r.sample_size}, c={r.acceptance_number}"

    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("variance_components", aliases=["varcomp"], category="quality",
          description="Variance components (random effects)",
          usage="variance_components <response> ~ <factor>")
def cmd_varcomp(session, parsed):
    from forgestat.quality.variance_components import one_way_random

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: variance_components <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    r = one_way_random(groups, factor_name=parsed.predictors[0])
    lines = [f"Variance components (ICC={r.icc:.3f}):"]
    for c in r.components:
        lines.append(f"  {c.source}: {c.pct_contribution:.1f}%")
    return CommandResult(success=True, data=r.__dict__, summary="\n".join(lines))


# --- DOE (forgedoe) ---

@register("factorial", aliases=["doe"], category="doe", requires_data=False,
          description="Generate factorial design",
          usage="factorial factors=<n> [design=full_factorial|fractional|ccd|bbdesign|dsd]")
def cmd_factorial(session, parsed):
    try:
        from forgedoe.core.types import Factor

        n_factors = int(parsed.named.get("factors", parsed.named.get("n", 3)))
        design = parsed.named.get("design", "full_factorial")
        factors = [Factor(name=f"X{i+1}") for i in range(n_factors)]

        if design == "full_factorial":
            from forgedoe.designs.factorial import full_factorial
            d = full_factorial(factors, randomize=False)
        elif design == "ccd":
            from forgedoe.designs.response_surface import central_composite_design
            d = central_composite_design(factors, randomize=False)
        elif design in ("bbdesign", "box_behnken"):
            from forgedoe.designs.response_surface import box_behnken_design
            d = box_behnken_design(factors, randomize=False)
        elif design == "dsd":
            from forgedoe.designs.screening import definitive_screening_design
            d = definitive_screening_design(factors, randomize=False)
        else:
            from forgedoe.designs.factorial import full_factorial
            d = full_factorial(factors, randomize=False)

        summary = f"{design}: {n_factors} factors, {d.n_runs} runs"

        # Load design into session for further analysis
        for i, f in enumerate(factors):
            col_data = [row[i] for row in d.matrix]
            session.data[f.name] = col_data
        session.columns = list(session.data.keys())
        session.n_rows = d.n_runs

        summary += " — loaded into session"
        return CommandResult(success=True, data={"n_runs": d.n_runs, "n_factors": n_factors},
                             summary=summary)
    except ImportError:
        return CommandResult(success=False, error="forgedoe not installed")


# --- SIOP (forgesiop) ---

@register("eoq", category="siop", requires_data=False,
          description="Economic Order Quantity",
          usage="eoq demand=<annual> ordering=<cost> holding=<rate> unit_cost=<cost>")
def cmd_eoq(session, parsed):
    try:
        from forgesiop.inventory.eoq import economic_order_quantity

        demand = float(parsed.named.get("demand", 0))
        ordering = float(parsed.named.get("ordering", 0))
        holding = float(parsed.named.get("holding", 0.25))
        unit_cost = float(parsed.named.get("unit_cost", parsed.named.get("cost", 10)))

        if demand <= 0 or ordering <= 0:
            return CommandResult(success=False, error="Usage: eoq demand=<D> ordering=<S> unit_cost=<C>")

        q = economic_order_quantity(demand, ordering, holding, unit_cost)
        summary = f"EOQ = {q:.1f} units (D={demand}, S={ordering}, H={holding*100:.0f}%)"
        return CommandResult(success=True, data={"eoq": q}, summary=summary)
    except ImportError:
        return CommandResult(success=False, error="forgesiop not installed")


@register("safety_stock", aliases=["ss_calc"], category="siop", requires_data=False,
          description="Safety stock calculation",
          usage="safety_stock demand_mean=<d> demand_std=<s> lead_time=<lt> [service=0.95]")
def cmd_safety_stock(session, parsed):
    try:
        from forgesiop.inventory.safety_stock import dynamic_safety_stock

        d_mean = float(parsed.named.get("demand_mean", parsed.named.get("dm", 0)))
        d_std = float(parsed.named.get("demand_std", parsed.named.get("ds", 0)))
        lt = float(parsed.named.get("lead_time", parsed.named.get("lt", 0)))
        lt_std = float(parsed.named.get("lt_std", 0))
        service = float(parsed.named.get("service", 0.95))

        if d_mean <= 0:
            return CommandResult(success=False, error="Usage: safety_stock demand_mean=<d> demand_std=<s> lead_time=<lt>")

        ss = dynamic_safety_stock(d_mean, d_std, lt, lt_std, service)
        summary = f"Safety stock = {ss:.1f} units (service level {service*100:.0f}%)"
        return CommandResult(success=True, data={"safety_stock": float(ss)}, summary=summary)
    except ImportError:
        return CommandResult(success=False, error="forgesiop not installed")
