"""Tests for ForgePad — session, parser, commands, charts."""

import numpy as np

from forgepad.session import Session
from forgepad.parser import parse


def _make_session():
    """Session with typical manufacturing data."""
    s = Session()
    rng = np.random.default_rng(42)
    n = 50
    s.load_data({
        "yield": (85 + rng.normal(0, 3, n)).tolist(),
        "temp": (200 + rng.normal(0, 10, n)).tolist(),
        "pressure": (30 + rng.normal(0, 2, n)).tolist(),
        "machine": (["M1"] * 25 + ["M2"] * 25),
        "shift": (["Day", "Night"] * 25),
    })
    return s


class TestParser:
    def test_simple_verb(self):
        p = parse("info")
        assert p.verb == "info"
        assert p.positional == []

    def test_positional_args(self):
        p = parse("ttest yield mu=85")
        assert p.verb == "ttest"
        assert p.positional == ["yield"]
        assert p.named == {"mu": "85"}

    def test_formula(self):
        p = parse("anova yield ~ machine")
        assert p.verb == "anova"
        assert p.response == "yield"
        assert p.predictors == ["machine"]

    def test_result_name_arrow(self):
        p = parse("ttest yield mu=80 -> t1")
        assert p.result_name == "t1"
        assert p.verb == "ttest"

    def test_result_name_as(self):
        p = parse("ttest yield mu=80 as t1")
        assert p.result_name == "t1"

    def test_multiple_predictors(self):
        p = parse("regression yield ~ temp pressure")
        assert p.predictors == ["temp", "pressure"]

    def test_quoted_string(self):
        p = parse('unique "my column"')
        assert p.positional == ["my column"]

    def test_empty(self):
        p = parse("")
        assert p.verb == ""

    def test_case_insensitive(self):
        p = parse("TTEST yield mu=80")
        assert p.verb == "ttest"


class TestSession:
    def test_load_data(self):
        s = Session()
        s.load_data({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert s.n_rows == 3
        assert s.columns == ["x", "y"]

    def test_get_column(self):
        s = Session()
        s.load_data({"Temp": [200, 210, 220]})
        assert s.get_column("Temp") == [200, 210, 220]
        # Case-insensitive
        assert s.get_column("temp") == [200, 210, 220]

    def test_get_groups(self):
        s = Session()
        s.load_data({"val": [10, 20, 30, 40], "grp": ["A", "A", "B", "B"]})
        groups = s.get_groups("val", "grp")
        assert set(groups.keys()) == {"A", "B"}
        assert groups["A"] == [10.0, 20.0]

    def test_run_stores_history(self):
        s = _make_session()
        s.run("info")
        s.run("describe yield")
        assert len(s.history) == 2

    def test_named_result(self):
        s = _make_session()
        r = s.run("ttest yield mu=80 -> t1")
        assert r.result_name == "t1"
        assert "t1" in s.named_results


class TestDataCommands:
    def test_info(self):
        s = _make_session()
        r = s.run("info")
        assert r.success
        assert "50 rows" in r.summary

    def test_describe(self):
        s = _make_session()
        r = s.run("describe yield")
        assert r.success
        assert "mean=" in r.summary

    def test_describe_all(self):
        s = _make_session()
        r = s.run("describe")
        assert r.success
        assert "yield" in r.summary

    def test_columns(self):
        s = _make_session()
        r = s.run("columns")
        assert r.success

    def test_head(self):
        s = _make_session()
        r = s.run("head 3")
        assert r.success

    def test_missing(self):
        s = _make_session()
        r = s.run("missing")
        assert r.success

    def test_unique(self):
        s = _make_session()
        r = s.run("unique machine")
        assert r.success
        assert "2 unique" in r.summary

    def test_help(self):
        s = Session()
        r = s.run("help")
        assert r.success
        assert "ttest" in r.summary


class TestStatCommands:
    def test_ttest_one_sample(self):
        s = _make_session()
        r = s.run("ttest yield mu=80")
        assert r.success
        assert "t=" in r.summary
        assert len(r.charts) >= 1

    def test_ttest_two_sample_formula(self):
        s = _make_session()
        r = s.run("ttest yield ~ machine")
        assert r.success
        assert "Welch" in r.summary

    def test_ttest_two_columns(self):
        s = _make_session()
        r = s.run("ttest yield temp")
        assert r.success

    def test_anova(self):
        s = _make_session()
        r = s.run("anova yield ~ machine")
        assert r.success
        assert "F=" in r.summary

    def test_correlation(self):
        s = _make_session()
        r = s.run("correlation yield temp pressure")
        assert r.success
        assert "r=" in r.summary

    def test_normality(self):
        s = _make_session()
        r = s.run("normality yield")
        assert r.success
        assert "NORMAL" in r.summary or "NOT NORMAL" in r.summary

    def test_regression(self):
        s = _make_session()
        r = s.run("regression yield ~ temp pressure")
        assert r.success
        assert "R²=" in r.summary

    def test_equivalence(self):
        s = _make_session()
        r = s.run("equivalence yield temp margin=20")
        assert r.success
        assert "TOST" in r.summary

    def test_proportion(self):
        s = _make_session()
        r = s.run("proportion successes=70 n=100 p0=0.5")
        assert r.success
        assert "p̂=" in r.summary


class TestChartCommands:
    def test_hist(self):
        s = _make_session()
        r = s.run("hist yield")
        assert r.success
        assert len(r.charts) == 1

    def test_scatter(self):
        s = _make_session()
        r = s.run("scatter temp yield")
        assert r.success
        assert len(r.charts) == 1

    def test_bar(self):
        s = _make_session()
        r = s.run("bar machine yield")
        assert r.success
        assert len(r.charts) == 1

    def test_line(self):
        s = _make_session()
        r = s.run("line temp yield")
        assert r.success
        assert len(r.charts) == 1


class TestQualityCommands:
    def test_anom(self):
        s = _make_session()
        r = s.run("anom yield ~ machine")
        assert r.success
        assert "grand mean" in r.summary

    def test_imr(self):
        s = _make_session()
        r = s.run("imr yield")
        assert r.success
        assert "UCL" in r.summary

    def test_capability(self):
        s = _make_session()
        r = s.run("capability yield lsl=75 usl=95")
        assert r.success
        assert "Cpk" in r.summary


class TestErrorHandling:
    def test_unknown_command(self):
        s = _make_session()
        r = s.run("foobar")
        assert not r.success
        assert "Unknown command" in r.error

    def test_missing_column(self):
        s = _make_session()
        r = s.run("ttest nonexistent mu=80")
        assert not r.success
        assert "not found" in r.error.lower() or "Column" in r.error

    def test_no_data(self):
        s = Session()
        r = s.run("ttest yield mu=80")
        assert not r.success
        assert "requires loaded data" in r.error or "data" in r.error.lower()

    def test_bad_args(self):
        s = _make_session()
        r = s.run("ttest")
        assert not r.success
