import numpy as np


def _projector_complement(Z):
    Q, _ = np.linalg.qr(np.asarray(Z, dtype=float))
    return lambda M: M - Q @ (Q.T @ M)


def freedman_lane(Y, a, Z, n_perm=9999, seed=0):
    Y = np.asarray(Y, dtype=float).reshape(len(a), -1)
    Z = np.asarray(Z, dtype=float)
    n = len(a)
    residualize = _projector_complement(Z)
    a_resid = residualize(np.asarray(a, dtype=float))
    a_norm2 = a_resid @ a_resid
    df_residual = n - np.linalg.matrix_rank(Z) - 1
    if a_norm2 <= 1e-12 or df_residual < 1:
        return np.nan, np.nan

    def pseudo_f(R):
        explained = np.sum((a_resid @ R) ** 2) / a_norm2
        return explained / ((np.sum(R**2) - explained) / df_residual)

    R = residualize(Y)
    observed = pseudo_f(R)
    rng = np.random.default_rng(seed)
    exceed = sum(pseudo_f(residualize(R[rng.permutation(n)])) >= observed * (1 - 1e-12) for _ in range(n_perm))
    return float(observed), (exceed + 1) / (n_perm + 1)


def dispersion_test(Y, a, Z, n_perm=9999, seed=0):
    Y = np.asarray(Y, dtype=float).reshape(len(a), -1)
    group = np.asarray(a) == 1
    R = _projector_complement(Z)(Y)
    distance = np.empty(len(group))
    for members in (group, ~group):
        distance[members] = np.linalg.norm(R[members] - R[members].mean(axis=0), axis=1)

    def spread_gap(g):
        return abs(distance[g].mean() - distance[~g].mean())

    observed = spread_gap(group)
    rng = np.random.default_rng(seed)
    exceed = sum(spread_gap(rng.permutation(group)) >= observed * (1 - 1e-12) for _ in range(n_perm))
    return float(observed), (exceed + 1) / (n_perm + 1)
