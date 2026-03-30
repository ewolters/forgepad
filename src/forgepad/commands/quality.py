"""Quality and SPC commands — capability, control charts, Gage R&R."""

from __future__ import annotations

from ..charts.mapping import charts_for
from ..registry import register
from ..session import CommandResult


@register("capability", aliases=["cpk"], category="quality",
          description="Process capability (Cp, Cpk, Pp, Ppk)",
          usage="capability <column> lsl=<value> usl=<value>")
def cmd_capability(session, parsed):
    from forgespc.capability import calculate_capability

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: capability <column> lsl=<value> usl=<value>")

    vals = session.get_numeric_column(col)
    lsl = float(parsed.named["lsl"]) if "lsl" in parsed.named else None
    usl = float(parsed.named["usl"]) if "usl" in parsed.named else None

    if lsl is None and usl is None:
        return CommandResult(success=False, error="Specify at least one: lsl=<value> or usl=<value>")

    r = calculate_capability(vals, lsl=lsl, usl=usl)
    lines = [f"Cp={r.cp:.3f}, Cpk={r.cpk:.3f}, Pp={r.pp:.3f}, Ppk={r.ppk:.3f}"]
    lines.append(f"σ level={r.sigma_level:.2f}, DPMO={r.dpmo:.0f}, Yield={r.yield_percent:.2f}%")
    return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary="\n".join(lines))


@register("imr", category="spc",
          description="I-MR control chart",
          usage="imr <column>")
def cmd_imr(session, parsed):
    from forgespc.charts import individuals_moving_range_chart

    col = parsed.positional[0] if parsed.positional else None
    if not col:
        return CommandResult(success=False, error="Usage: imr <column>")

    vals = session.get_numeric_column(col)
    r = individuals_moving_range_chart(vals)
    summary = f"I-MR: x̄={r.limits.cl:.4f}, UCL={r.limits.ucl:.4f}, LCL={r.limits.lcl:.4f}"
    if r.out_of_control:
        summary += f" | {len(r.out_of_control)} out-of-control points"
    return CommandResult(success=True, data=r.__dict__, charts=charts_for(r), summary=summary)


@register("anom", category="quality",
          description="Analysis of Means",
          usage="anom <response> ~ <factor>")
def cmd_anom(session, parsed):
    from forgestat.quality.anom import anom

    if not parsed.response or not parsed.predictors:
        return CommandResult(success=False, error="Usage: anom <response> ~ <factor>")

    groups = session.get_groups(parsed.response, parsed.predictors[0])
    names = list(groups.keys())
    arrays = [groups[n] for n in names]
    r = anom(*arrays, labels=names)

    flagged = [g.name for g in r.groups if g.exceeds_upper or g.exceeds_lower]
    summary = f"ANOM: grand mean={r.grand_mean:.4f}, limits=[{r.lower_limit:.4f}, {r.upper_limit:.4f}]"
    if flagged:
        summary += f" | Flagged: {', '.join(flagged)}"
    return CommandResult(success=True, data=r.__dict__, summary=summary)


@register("gagerr", aliases=["grr"], category="quality",
          description="Crossed Gage R&R",
          usage="gagerr <measurement> part=<col> operator=<col>")
def cmd_gagerr(session, parsed):
    from forgestat.msa.gage_rr import crossed_gage_rr

    col = parsed.positional[0] if parsed.positional else None
    part_col = parsed.named.get("part")
    op_col = parsed.named.get("operator", parsed.named.get("op"))

    if not col or not part_col or not op_col:
        return CommandResult(success=False, error="Usage: gagerr <measurement> part=<col> operator=<col>")

    meas = session.get_column(col)
    parts = session.get_column(part_col)
    operators = session.get_column(op_col)

    # Convert measurements to float
    meas_float = [float(m) for m in meas]
    parts_str = [str(p) for p in parts]
    ops_str = [str(o) for o in operators]

    r = crossed_gage_rr(meas_float, parts_str, ops_str)
    summary = f"Gage R&R: {r.pct_gage_rr:.1f}% (NDC={r.ndc})"
    if r.pct_gage_rr < 10:
        summary += " — ACCEPTABLE"
    elif r.pct_gage_rr < 30:
        summary += " — MARGINAL"
    else:
        summary += " — UNACCEPTABLE"
    return CommandResult(success=True, data=r.__dict__, summary=summary)
