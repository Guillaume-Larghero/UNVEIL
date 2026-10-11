from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unveil.config import load_spec
from unveil.decomposition import EFFECTS, build_design, decompose, fit_once

SPEC = load_spec(Path(__file__).resolve().parents[1] / "configs" / "analysis.yaml")


def simulate(n=2000, a_effect=0.5, m_effect=0.8, direct=0.3, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "technical_cov1": rng.choice(["s1", "s2", "s3", "s4"], n),
        "biological_cov2": rng.normal(size=n),
    })
    site = frame["technical_cov1"].map({"s1": -0.5, "s2": 0.0, "s3": 0.3, "s4": 0.6}).to_numpy()
    a = rng.integers(0, 2, n).astype(float)
    label = rng.integers(0, 2, n).astype(float)
    m = a_effect * a + site + 0.4 * frame["biological_cov2"].to_numpy() + rng.normal(size=n)
    y = direct * a + m_effect * m + 0.2 * label + site + rng.normal(size=n)
    return frame, a, m, y[:, None], label


def test_effects_satisfy_decomposition_identities():
    frame, a, m, Y, label = simulate()
    design = build_design(frame, a, m, Y, label, ["technical_cov1", "biological_cov2"], SPEC, 0.0, 1.0)
    e = dict(zip(EFFECTS, fit_once(design, np.arange(len(a)))[0]))
    assert e["Total"] == pytest.approx(e["IIE"] + e["IDE"])
    assert e["IDE"] == pytest.approx(e["CDE"] + e["INTref"])
    assert e["IIE"] == pytest.approx(e["PIE"] + e["INTmed"])


def test_recovers_planted_indirect_and_direct_effects():
    frame, a, m, Y, label = simulate(n=6000, seed=1)
    design = build_design(frame, a, m, Y, label, ["technical_cov1", "biological_cov2"], SPEC, 0.0, 1.0)
    row = decompose(design, n_boot=200, rng=np.random.default_rng(0))[0]
    assert row["IIE_ci_low"] < 0.5 * 0.8 < row["IIE_ci_high"]
    assert row["IDE_ci_low"] < 0.3 < row["IDE_ci_high"]
    assert row["IIE_diff_ci_low"] < 0 < row["IIE_diff_ci_high"]
    assert row["a_path_p"] < 1e-6 and row["b_path_p"] < 1e-6


def test_random_intercept_is_penalized():
    frame, a, m, Y, label = simulate()
    design = build_design(frame, a, m, Y, label, ["technical_cov1"], SPEC, 0.0, 1.0)
    assert design.n_random_levels == 4
    assert np.isfinite(design.lambda_mediator) and design.lambda_mediator > 0
    assert design.penalty_mediator.shape[0] == 4
