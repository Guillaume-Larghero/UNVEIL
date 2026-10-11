import argparse

import numpy as np
import pandas as pd

from unveil.data import read_table, write_table
from unveil.decomposition import EFFECTS
from unveil.meta import pool_table
from unveil.multiplicity import add_fdr, bh, composite_null_p

FDR_FAMILY = ("cohort", "exposure", "contrast", "outcome")
POOL_KEYS = ("cancer", "task", "exposure", "contrast", "mediator", "outcome")
POOLED_EFFECTS = ("IIE", "IIE_diff", "IDE", "Total")
COMPOSITE_PATHS = {"IIE": ("a_path_p", "b_path_p"), "IIE_diff": ("a_path_p", "b_label_interaction_p")}


def add_composite_fdr(table, family):
    family = [column for column in family if column in table.columns]
    for effect, (p_a, p_b) in COMPOSITE_PATHS.items():
        if p_b not in table.columns:
            continue
        table[f"{effect}_p_composite"] = np.nan
        table[f"{effect}_q_composite"] = np.nan
        table[f"{effect}_pi_null_both"] = np.nan
        for index in table.groupby(family, dropna=False, sort=False).groups.values():
            p, pi_null_both = composite_null_p(table.loc[index, p_a], table.loc[index, p_b])
            table.loc[index, f"{effect}_p_composite"] = p
            table.loc[index, f"{effect}_q_composite"] = bh(p)
            table.loc[index, f"{effect}_pi_null_both"] = pi_null_both
    return table


def main():
    parser = argparse.ArgumentParser(description="Multiplicity control and cross-cohort pooling of the decomposition.")
    parser.add_argument("--decomposition", nargs="+", required=True, help="one or more outputs of 2_decompose_disparity.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pooled-output", default=None)
    parser.add_argument("--fdr-family", nargs="+", default=list(FDR_FAMILY))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    table = pd.concat([read_table(path) for path in args.decomposition], ignore_index=True)
    table = add_composite_fdr(table, args.fdr_family)
    for effect in EFFECTS:
        table = add_fdr(table, args.fdr_family, p_column=f"{effect}_p_value", q_column=f"{effect}_q_value")
    write_table(table, args.output)

    if args.pooled_output:
        pooled = []
        for effect in POOLED_EFFECTS:
            part = pool_table(table, POOL_KEYS, effect, f"{effect}_se", unit="cohort", alpha=args.alpha)
            pooled.append(part.assign(effect=effect))
        pooled = pd.concat(pooled, ignore_index=True)
        if pooled.empty:
            raise SystemExit("no test was estimated in two or more cohorts")
        pooled = add_fdr(pooled, [*args.fdr_family, "effect"])
        pooled = add_fdr(pooled, [*args.fdr_family, "effect"], p_column="p_value_hk", q_column="q_value_hk")
        write_table(pooled, args.pooled_output)


if __name__ == "__main__":
    main()
