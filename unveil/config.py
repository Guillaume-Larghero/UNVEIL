from dataclasses import dataclass

import yaml

COVARIATE_BLOCKS = ("technical", "biological", "demographic")


@dataclass(frozen=True)
class Exposure:
    name: str
    kind: str
    reference: str = None
    levels: tuple = ()

    def __post_init__(self):
        if self.kind not in ("continuous", "categorical"):
            raise ValueError(f"exposure {self.name}: type must be continuous or categorical, got {self.kind}")
        if self.kind == "categorical" and (self.reference is None or not self.levels):
            raise ValueError(f"exposure {self.name}: categorical exposures need a reference and levels")


@dataclass(frozen=True)
class AnalysisSpec:
    patient_id: str
    slide_id: str
    cohort: str
    cancer: str
    exposures: dict
    covariates: dict
    categorical_covariates: frozenset = frozenset()
    random_intercept: tuple = ()
    max_levels: int = 3
    min_level_count: int = 10
    min_group_size: int = 20
    min_residual_df: int = 30
    alpha: float = 0.05

    def exposure(self, name):
        if name not in self.exposures:
            raise KeyError(f"exposure {name} is not declared in the analysis spec")
        return self.exposures[name]

    def covariates_for(self, exposure, blocks=COVARIATE_BLOCKS):
        unknown = set(blocks) - set(self.covariates)
        if unknown:
            raise KeyError(f"unknown covariate blocks: {sorted(unknown)}")
        columns = []
        for block in blocks:
            for column in self.covariates[block]:
                if column != exposure and column not in columns:
                    columns.append(column)
        return columns

    def is_categorical(self, column, series):
        if column in self.categorical_covariates:
            return True
        if column in self.exposures:
            return self.exposures[column].kind == "categorical"
        return series.dtype.kind not in "biuf"


def load_spec(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    columns = raw.get("columns", {})
    exposures = {
        name: Exposure(name, entry["type"], entry.get("reference"), tuple(entry.get("levels", ())))
        for name, entry in raw["exposures"].items()
    }
    covariates = {block: list(raw.get("covariates", {}).get(block) or []) for block in COVARIATE_BLOCKS}
    design = raw.get("design", {})
    return AnalysisSpec(
        patient_id=columns.get("patient_id", "patient_id"),
        slide_id=columns.get("slide_id", "slide_id"),
        cohort=columns.get("cohort", "cohort"),
        cancer=columns.get("cancer", "cancer"),
        exposures=exposures,
        covariates=covariates,
        categorical_covariates=frozenset(raw.get("categorical_covariates") or ()),
        random_intercept=tuple(raw.get("random_intercept") or ()),
        max_levels=int(design.get("max_levels", 3)),
        min_level_count=int(design.get("min_level_count", 10)),
        min_group_size=int(design.get("min_group_size", 20)),
        min_residual_df=int(design.get("min_residual_df", 30)),
        alpha=float(raw.get("alpha", 0.05)),
    )
