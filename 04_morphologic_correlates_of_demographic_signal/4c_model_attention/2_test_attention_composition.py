import argparse

from unveil.cli import add_analysis_arguments, resolve
from unveil.composition_effects import CompositionModel, composition_effects
from unveil.data import attach_clinical, read_table, write_table


def main():
    parser = argparse.ArgumentParser(description="Exposure (or prediction correctness) effects on the attention-weighted cell composition of each task model.")
    parser.add_argument("--attention", required=True, help="output of 1_build_attention_composition.py")
    parser.add_argument("--clinical", required=True, help="patient table with exposures and covariates")
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--stratify-by", nargs="+", default=["task"], help="tested separately within each level; task by default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--omnibus-output", default=None)
    parser.add_argument("--unadjusted", action="store_true")
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr-family", nargs="+", default=None, help="columns defining one Benjamini-Hochberg family; defaults to cohort, the strata, exposure, contrast and adjustment")
    add_analysis_arguments(parser)
    args = parser.parse_args()
    spec, exposures, blocks = resolve(args)
    fdr_family = args.fdr_family or ["cohort", *args.stratify_by, "exposure", "contrast", "adjustment"]

    model = CompositionModel(tuple(args.components), "clr", None, args.n_perm, args.seed)
    data = attach_clinical(read_table(args.attention), read_table(args.clinical), spec)
    effects, omnibus = composition_effects(data, model, spec, exposures, blocks, args.stratify_by, args.unadjusted, fdr_family)
    if effects is None:
        raise SystemExit("no attention effect could be estimated; check group sizes and covariate completeness")
    write_table(effects, args.output)
    if args.omnibus_output and omnibus is not None:
        write_table(omnibus, args.omnibus_output)


if __name__ == "__main__":
    main()
