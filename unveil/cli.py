from pathlib import Path

from .config import COVARIATE_BLOCKS, load_spec

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "configs" / "analysis.yaml"


def add_analysis_arguments(parser, fdr_family=None):
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="analysis spec declaring exposures and covariate blocks")
    parser.add_argument("--exposures", nargs="+", default=None, help="exposures to test; defaults to the demographic covariate block")
    parser.add_argument(
        "--adjust",
        nargs="+",
        default=list(COVARIATE_BLOCKS),
        choices=[*COVARIATE_BLOCKS, "none"],
        help="covariate blocks to adjust for; the exposure itself is always excluded",
    )
    if fdr_family is not None:
        parser.add_argument("--fdr-family", nargs="+", default=list(fdr_family), help="columns defining one Benjamini-Hochberg family")
    return parser


def resolve(args):
    spec = load_spec(args.spec)
    exposures = [spec.exposure(name) for name in (args.exposures or spec.covariates["demographic"])]
    blocks = () if "none" in args.adjust else tuple(args.adjust)
    return spec, exposures, blocks
