"""Tests for expanded commands — all new categories."""

import numpy as np
from forgepad.session import Session


def _make_session():
    s = Session()
    rng = np.random.default_rng(42)
    n = 50
    s.load_data({
        "yield": (85 + rng.normal(0, 3, n)).tolist(),
        "temp": (200 + rng.normal(0, 10, n)).tolist(),
        "pressure": (30 + rng.normal(0, 2, n)).tolist(),
        "machine": (["M1"] * 25 + ["M2"] * 25),
        "shift": (["Day", "Night"] * 25),
        "defects": rng.poisson(3, n).tolist(),
        "time_to_fail": (rng.weibull(2, n) * 100).tolist(),
        "event": [1] * n,
        "rater1": rng.choice([1, 2, 3, 4, 5], n).tolist(),
        "rater2": rng.choice([1, 2, 3, 4, 5], n).tolist(),
    })
    return s


class TestNonparametric:
    def test_mann_whitney(self):
        s = _make_session()
        r = s.run("mann_whitney yield ~ machine")
        assert r.success

    def test_kruskal(self):
        s = _make_session()
        r = s.run("kruskal yield ~ machine")
        assert r.success

    def test_wilcoxon(self):
        s = _make_session()
        r = s.run("wilcoxon yield temp")
        assert r.success

    def test_friedman(self):
        s = _make_session()
        r = s.run("friedman yield temp pressure")
        assert r.success

    def test_runs(self):
        s = _make_session()
        r = s.run("runs yield")
        assert r.success


class TestPosthoc:
    def test_tukey(self):
        s = _make_session()
        r = s.run("tukey yield ~ machine")
        assert r.success

    def test_dunnett(self):
        s = _make_session()
        r = s.run("dunnett yield ~ machine control=M1")
        assert r.success

    def test_games_howell(self):
        s = _make_session()
        r = s.run("games_howell yield ~ machine")
        assert r.success

    def test_dunn(self):
        s = _make_session()
        r = s.run("dunn yield ~ machine")
        assert r.success

    def test_scheffe(self):
        s = _make_session()
        r = s.run("scheffe yield ~ machine")
        assert r.success


class TestVariance:
    def test_chi2(self):
        s = _make_session()
        r = s.run("chi2 machine shift")
        assert r.success

    def test_f_test(self):
        s = _make_session()
        r = s.run("f_test yield temp")
        assert r.success

    def test_variance_levene(self):
        s = _make_session()
        r = s.run("variance yield ~ machine")
        assert r.success


class TestRegression:
    def test_logistic(self):
        s = _make_session()
        r = s.run("logistic event ~ yield temp")
        assert r.success or "converge" in r.error.lower()  # may not converge on this data

    def test_robust(self):
        s = _make_session()
        r = s.run("robust yield ~ temp pressure")
        assert r.success

    def test_stepwise(self):
        s = _make_session()
        r = s.run("stepwise yield ~ temp pressure")
        assert r.success

    def test_nonlinear(self):
        s = _make_session()
        r = s.run("nonlinear temp yield model=exponential")
        # May or may not converge
        assert r.success or "converge" in r.error.lower()

    def test_glm(self):
        s = _make_session()
        r = s.run("glm yield ~ temp pressure family=gaussian")
        assert r.success

    def test_best_subsets(self):
        s = _make_session()
        r = s.run("best_subsets yield ~ temp pressure")
        assert r.success


class TestPower:
    def test_power_ttest(self):
        s = Session()
        r = s.run("power test=ttest effect=0.5 n=64")
        assert r.success
        assert "power=" in r.summary

    def test_sample_size(self):
        s = Session()
        r = s.run("sample_size std=5 width=1")
        assert r.success
        assert "n =" in r.summary


class TestReliability:
    def test_weibull(self):
        s = _make_session()
        r = s.run("weibull time_to_fail")
        assert r.success
        assert "β=" in r.summary

    def test_kaplan_meier(self):
        s = _make_session()
        r = s.run("kaplan_meier time_to_fail")
        assert r.success


class TestTimeSeries:
    def test_arima(self):
        s = Session()
        rng = np.random.default_rng(42)
        s.load_data({"y": np.cumsum(rng.normal(0, 1, 100)).tolist()})
        r = s.run("arima y p=1 d=1 q=0 forecast=5")
        assert r.success

    def test_acf(self):
        s = Session()
        rng = np.random.default_rng(42)
        s.load_data({"y": rng.normal(0, 1, 200).tolist()})
        r = s.run("acf y lags=15")
        assert r.success

    def test_changepoint(self):
        s = Session()
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(10, 1, 50), rng.normal(20, 1, 50)]).tolist()
        s.load_data({"y": data})
        r = s.run("changepoint y")
        assert r.success

    def test_granger(self):
        s = Session()
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = 0.7 * x[i - 2] + rng.normal(0, 0.5)
        s.load_data({"x": x.tolist(), "y": y.tolist()})
        r = s.run("granger x y")
        assert r.success


class TestBayesian:
    def test_bayes_ttest(self):
        s = _make_session()
        r = s.run("bayes_ttest yield mu=80")
        assert r.success
        assert "BF" in r.summary

    def test_bayes_proportion(self):
        s = Session()
        r = s.run("bayes_proportion successes=70 n=100")
        assert r.success


class TestExploratory:
    def test_pca(self):
        s = _make_session()
        r = s.run("pca yield temp pressure")
        assert r.success
        assert "PC1" in r.summary

    def test_multi_vari(self):
        s = _make_session()
        r = s.run("multi_vari yield ~ machine shift")
        assert r.success
        assert "dominant" in r.summary

    def test_bootstrap(self):
        s = _make_session()
        r = s.run("bootstrap yield stat=mean")
        assert r.success
        assert "CI" in r.summary

    def test_tolerance(self):
        s = _make_session()
        r = s.run("tolerance yield")
        assert r.success


class TestMSA:
    def test_icc(self):
        s = _make_session()
        r = s.run("icc rater1 rater2")
        assert r.success

    def test_bland_altman(self):
        s = _make_session()
        r = s.run("bland_altman rater1 rater2")
        assert r.success

    def test_linearity(self):
        s = _make_session()
        r = s.run("linearity rater1 rater2")
        assert r.success

    def test_krippendorff(self):
        s = _make_session()
        r = s.run("krippendorff rater1 rater2")
        assert r.success


class TestDomain:
    def test_acceptance(self):
        s = Session()
        r = s.run("acceptance aql=0.01 ltpd=0.05")
        assert r.success
        assert "n=" in r.summary

    def test_variance_components(self):
        s = _make_session()
        r = s.run("variance_components yield ~ machine")
        assert r.success
        assert "ICC" in r.summary

    def test_eoq(self):
        s = Session()
        r = s.run("eoq demand=10000 ordering=50 unit_cost=10")
        assert r.success
        assert "EOQ" in r.summary

    def test_safety_stock(self):
        s = Session()
        r = s.run("safety_stock demand_mean=100 demand_std=20 lead_time=4")
        assert r.success

    def test_factorial(self):
        s = Session()
        r = s.run("factorial factors=3 design=full_factorial")
        assert r.success
        assert "8 runs" in r.summary  # 2^3


class TestCommandCount:
    def test_total_commands(self):
        """Verify we have the expected number of commands."""
        Session()
        from forgepad.registry import REGISTRY
        total = len(REGISTRY._commands)
        assert total >= 60, f"Expected ≥60 commands, got {total}"
