import argparse

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from unveil.data import read_table, write_table
from unveil.multiplicity import add_fdr

GROUP_KEYS = ("cohort", "cancer", "exposure", "contrast", "adjustment")
FDR_FAMILY = ("exposure", "contrast", "adjustment")


def block_permutation_counts(cell_index, group_index, morphometric_index, hits, n_cells, n_groups, n_perm, rng):
    n_morphometrics = morphometric_index.max() + 1
    morphometric_group = np.zeros(n_morphometrics, dtype=int)
    morphometric_group[morphometric_index] = group_index
    size = np.bincount(morphometric_index, minlength=n_morphometrics)
    present = np.unique(morphometric_index)
    strata = [present[size[present] == s] for s in np.unique(size[present])]
    counts = np.empty((n_cells, n_groups, n_perm), dtype=np.int32)
    weights = hits.astype(float)
    for b in range(n_perm):
        permuted = morphometric_group.copy()
        for stratum in strata:
            permuted[stratum] = morphometric_group[rng.permutation(stratum)]
        cells = rng.permutation(n_cells)[cell_index]
        flat = np.bincount(cells * n_groups + permuted[morphometric_index], weights=weights, minlength=n_cells * n_groups)
        counts[:, :, b] = flat.reshape(n_cells, n_groups)
    return counts


def enrichment(results, n_perm, rng):
    codes = {column: pd.factorize(results[column].astype(str), sort=True) for column in ("cell_type", "feature_group", "morphometric")}
    cell_index, cells = codes["cell_type"]
    group_index, groups = codes["feature_group"]
    hits = results["significant"].to_numpy()
    null = block_permutation_counts(cell_index, group_index, codes["morphometric"][0], hits, len(cells), len(groups), n_perm, rng)
    n_background, n_hits = len(results), int(hits.sum())
    rows = []
    for (i, j), members in results.groupby([cell_index, group_index]).groups.items():
        n_category = len(members)
        if n_category < 2:
            continue
        overlap = int(results.loc[members, "significant"].sum())
        rows.append({
            "cell_type": cells[i],
            "feature_group": groups[j],
            "n_background": n_background,
            "n_category": n_category,
            "n_significant": n_hits,
            "n_overlap": overlap,
            "enrichment": (overlap / n_category) / (n_hits / n_background),
            "null_mean_overlap": float(null[i, j].mean()),
            "p_value": float((np.sum(null[i, j] >= overlap) + 1) / (n_perm + 1)),
            "p_value_hypergeometric": float(hypergeom.sf(overlap - 1, n_background, n_category, n_hits)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Concentration of significant features in cell type x feature group categories.")
    parser.add_argument("--results", required=True, help="output of 1_test_associations.py or 2_pool_cohorts.py")
    parser.add_argument("--annotations", required=True, help="feature, cell_type, feature_group, morphometric")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr-family", nargs="+", default=list(FDR_FAMILY))
    args = parser.parse_args()

    annotations = read_table(args.annotations)[["feature", "cell_type", "feature_group", "morphometric"]]
    results = read_table(args.results).merge(annotations, on="feature", how="inner", validate="many_to_one")
    results["significant"] = results["q_value"] < args.alpha
    keys = [k for k in GROUP_KEYS if k in results.columns]
    rng = np.random.default_rng(args.seed)
    rows = []
    for key, group in results.groupby(keys, sort=True):
        if group["significant"].any():
            label = dict(zip(keys, key))
            rows.extend({**label, **row} for row in enrichment(group.reset_index(drop=True), args.n_perm, rng))
    if not rows:
        raise SystemExit("no significant features to test for enrichment")
    table = add_fdr(pd.DataFrame(rows), args.fdr_family)
    smallest = 1 / (args.n_perm + 1)
    if (smallest >= args.alpha / table["q_value_n_tests"]).any():
        print(f"warning: {args.n_perm} permutations cannot reach Bonferroni-level significance in the largest family; increase --n-perm")
    write_table(table, args.output)


if __name__ == "__main__":
    main()
