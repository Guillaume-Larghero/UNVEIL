import numpy as np
import pandas as pd
import pytest
from scipy import stats

from unveil.compositional import clr
from unveil.effects import cliffs_delta, cliffs_delta_se, partial_correlation
from unveil.meta import random_effects
from unveil.multiplicity import add_fdr, bh, composite_null_p
from unveil.permutation import freedman_lane


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(0).uniform(size=200) ** 2
    assert np.allclose(bh(p), multipletests(p, method="fdr_bh")[1])


def test_bh_ignores_missing():
    q = bh([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.04])


def test_add_fdr_is_within_family():
    df = pd.DataFrame({"family": ["a", "a", "b"], "p_value": [0.01, 0.02, 0.01]})
    out = add_fdr(df, ["family"])
    assert out["q_value_n_tests"].tolist() == [2, 2, 1]
    assert out.loc[2, "q_value"] == pytest.approx(0.01)


def test_cliffs_delta_matches_brute_force():
    rng = np.random.default_rng(1)
    y1, y0 = rng.integers(0, 6, 40).astype(float), rng.integers(0, 6, 30).astype(float)
    sign = np.sign(y1[:, None] - y0[None, :])
    delta, _ = cliffs_delta(y1, y0)
    assert delta == pytest.approx(sign.mean())

    n1, n0 = len(y1), len(y0)
    d1, d0 = sign.mean(axis=1), sign.mean(axis=0)
    variance = (
        n0**2 * np.sum((d1 - delta) ** 2) + n1**2 * np.sum((d0 - delta) ** 2) - np.sum((sign - delta) ** 2)
    ) / (n1 * n0 * (n1 - 1) * (n0 - 1))
    assert cliffs_delta_se(y1, y0) == pytest.approx(np.sqrt(variance))


def test_partial_correlation_matches_residual_correlation():
    rng = np.random.default_rng(2)
    n = 300
    z = rng.normal(size=n)
    a = z + rng.normal(size=n)
    y = 0.3 * a + z + rng.normal(size=n)
    X = np.column_stack([np.ones(n), a, z])
    r, p, df = partial_correlation(y[:, None], X)
    Z = X[:, [0, 2]]
    resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    assert r[0] == pytest.approx(np.corrcoef(resid(a), resid(y))[0, 1])
    assert df == n - 3


def test_freedman_lane_is_calibrated_under_the_null():
    rng = np.random.default_rng(3)
    p_values = []
    for i in range(200):
        n = 60
        z = rng.normal(size=n)
        a = (z + rng.normal(size=n) > 0).astype(float)
        Y = z[:, None] + rng.normal(size=(n, 3))
        p_values.append(freedman_lane(Y, a, np.column_stack([np.ones(n), z]), n_perm=199, seed=i)[1])
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_freedman_lane_detects_an_effect():
    rng = np.random.default_rng(4)
    n = 120
    a = rng.integers(0, 2, n).astype(float)
    Y = rng.normal(size=(n, 3)) + a[:, None]
    assert freedman_lane(Y, a, np.ones((n, 1)), n_perm=499)[1] < 0.01


def test_clr_rows_sum_to_zero_and_handle_zeros():
    out = clr(np.array([[10, 0, 5], [1, 1, 1]]), n_total=[15, 3])
    assert np.allclose(out.sum(axis=1), 0)
    assert np.allclose(out[1], 0)
    assert np.all(np.isfinite(out))


def test_random_effects_reduces_to_fixed_effect_without_heterogeneity():
    fit = random_effects([0.2, 0.2, 0.2], [0.1, 0.1, 0.1])
    assert fit["tau2"] == 0 and fit["estimate"] == pytest.approx(0.2)
    assert fit["se"] == pytest.approx(0.1 / np.sqrt(3))
    assert fit["se_hk"] >= fit["se"]


def test_composite_null_lies_between_squared_and_max_p():
    rng = np.random.default_rng(5)
    p_a, p_b = rng.uniform(size=500), rng.uniform(size=500)
    p, _ = composite_null_p(p_a, p_b)
    assert np.all(p <= np.maximum(p_a, p_b) + 1e-12)
    assert np.all(p >= np.maximum(p_a, p_b) ** 2 - 1e-12)
