import argparse

from unveil.cli import add_analysis_arguments, resolve
from unveil.composition_effects import FDR_FAMILY, CompositionModel, composition_effects
from unveil.data import attach_clinical, read_table, write_table


def main():
    parser = argparse.ArgumentParser(description="Exposure effects on cell-type composition and nuclear density.")
    parser.add_argument("--slides", required=True, help="slide table with cohort, cancer, patient and slide ids, component columns and slide-level covariates")
    parser.add_argument("--clinical", required=True, help="patient table with exposures and patient-level covariates")
    parser.add_argument("--components", nargs="+", required=True, help="cell-type count columns, or clr columns with --mode clr")
    parser.add_argument("--mode", choices=["counts", "clr"], default="counts")
    parser.add_argument("--n-tiles-column", default=None, help="tile count per slide; adds log nuclear density as an outcome")
    parser.add_argument("--stratify-by", nargs="+", default=[], help="run within each level of these columns (e.g. a subtype); they are removed from the covariates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--omnibus-output", default=None)
    parser.add_argument("--unadjusted", action="store_true", help="also report the unadjusted patient-level rank effect")
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    add_analysis_arguments(parser, FDR_FAMILY)
    args = parser.parse_args()
    if args.mode == "clr" and args.n_tiles_column:
        parser.error("--n-tiles-column needs --mode counts")
    spec, exposures, blocks = resolve(args)

    model = CompositionModel(tuple(args.components), args.mode, args.n_tiles_column, args.n_perm, args.seed)
    data = attach_clinical(read_table(args.slides), read_table(args.clinical), spec)
    effects, omnibus = composition_effects(data, model, spec, exposures, blocks, args.stratify_by, args.unadjusted, args.fdr_family)
    if effects is None:
        raise SystemExit("no composition effect could be estimated; check group sizes and covariate completeness")
    write_table(effects, args.output)
    if args.omnibus_output and omnibus is not None:
        write_table(omnibus, args.omnibus_output)


if __name__ == "__main__":
    main()
