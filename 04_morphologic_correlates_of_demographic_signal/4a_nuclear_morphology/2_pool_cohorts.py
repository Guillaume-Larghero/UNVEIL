import argparse

import numpy as np

from unveil.data import read_table, write_table
from unveil.meta import pool_table
from unveil.multiplicity import add_fdr

POOL_KEYS = ("cancer", "exposure", "contrast", "adjustment", "effect_type", "feature")
FDR_FAMILY = ("exposure", "contrast", "adjustment")


def standing(table, alpha):
    significant = table["q_value"] < alpha
    robust = table["loo_robust"].astype(bool)
    hk_agrees = table["q_value_hk"] < alpha
    return np.select(
        [~significant, robust & hk_agrees, robust],
        ["not_significant", "robust", "robust_hk_discordant"],
        default="leave_one_out_fragile",
    )


def main():
    parser = argparse.ArgumentParser(description="Random-effects pooling of per-cohort associations on the Fisher z scale.")
    parser.add_argument("--associations", required=True, help="output of 1_test_associations.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohorts", nargs="+", default=None, help="cohorts to pool; defaults to all")
    parser.add_argument("--fdr-family", nargs="+", default=list(FDR_FAMILY))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    associations = read_table(args.associations)
    if args.cohorts:
        associations = associations[associations["cohort"].isin(args.cohorts)]
    pooled = pool_table(associations, POOL_KEYS, "effect", "se_z", unit="cohort", fisher=True, alpha=args.alpha)
    if pooled.empty:
        raise SystemExit("no feature was observed in two or more cohorts")
    pooled = add_fdr(pooled, args.fdr_family)
    pooled = add_fdr(pooled, args.fdr_family, p_column="p_value_hk", q_column="q_value_hk")
    pooled["standing"] = standing(pooled, args.alpha)
    write_table(pooled, args.output)


if __name__ == "__main__":
    main()
