from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compositional import clr
from .data import exposure_contrasts, to_patient_level
from .design import adjustment_design, residualize
from .effects import rank_association
from .mixed import random_intercept_effect
from .multiplicity import add_fdr
from .permutation import dispersion_test, freedman_lane

FDR_FAMILY = ("cohort", "exposure", "contrast", "outcome_block", "adjustment")
OMNIBUS_FDR_FAMILY = ("cohort", "exposure", "contrast", "adjustment")


@dataclass(frozen=True)
class CompositionModel:
    components: tuple
    mode: str = "counts"
    n_tiles_column: str = None
    n_perm: int = 9999
    seed: int = 0

    def outcomes(self, table):
        if self.mode == "clr":
            columns = [c if c in table.columns else f"clr_{c}" for c in self.components]
            return table[columns].astype(float).rename(columns=lambda c: c if c.startswith("clr_") else f"clr_{c}")
        counts = table[list(self.components)].to_numpy(dtype=float)
        frame = pd.DataFrame(clr(counts, counts.sum(axis=1)), columns=[f"clr_{c}" for c in self.components], index=table.index)
        if self.n_tiles_column:
            frame["log_density"] = np.log1p(counts.sum(axis=1) / table[self.n_tiles_column].to_numpy(dtype=float))
        return frame

    def patient_table(self, slides, spec):
        summed = list(self.components) + ([self.n_tiles_column] if self.n_tiles_column else []) if self.mode == "counts" else []
        return to_patient_level(slides, [spec.cohort, spec.cancer, spec.patient_id], sum_columns=summed)


def adjusted_effects(model, slides, patients, slide_a, patient_a, covariates, spec):
    """Slide-level mixed model with a patient random intercept, checked by a patient-level permutation test."""
    X_slide, slide_rows = adjustment_design(slides, slide_a, covariates, spec)
    X_patient, patient_rows = adjustment_design(patients, patient_a, covariates, spec)
    if X_slide is None or X_patient is None:
        return [], None
    Y_slide = model.outcomes(slides).loc[slide_rows]
    Y_patient = model.outcomes(patients).loc[patient_rows]
    groups = slides.loc[slide_rows, spec.patient_id].astype(str).to_numpy()
    a_used = patient_a[patient_rows]
    nuisance = X_patient.drop(columns="exposure").to_numpy()
    binary = set(np.unique(a_used)) <= {0.0, 1.0}
    rows = []
    for name in Y_slide.columns:
        fit = random_intercept_effect(Y_slide[name].to_numpy(), X_slide.to_numpy(), groups)
        y_residual = residualize(Y_patient[name].to_numpy(), nuisance)
        a_residual = a_used if binary else residualize(a_used, nuisance)
        _, p_permutation = freedman_lane(Y_patient[name].to_numpy(), a_used, nuisance, model.n_perm, model.seed)
        rows.append({
            "outcome": name,
            "n_patients": len(a_used),
            "n_slides": len(groups),
            "effect_type": "beta",
            "effect": fit["beta"],
            "se": fit["se"],
            "p_value": fit["p_value"],
            "method": fit["method"],
            "fit_status": fit["status"],
            "p_value_permutation": p_permutation,
            "rank_effect_adjusted": rank_association(y_residual, a_residual)["effect"],
            "adjusted_for": ";".join(X_slide.columns[2:]),
        })
    composition = Y_patient.filter(like="clr_").to_numpy()
    pseudo_f, p_omnibus = freedman_lane(composition, a_used, nuisance, model.n_perm, model.seed)
    omnibus = {"n_patients": len(a_used), "pseudo_f": pseudo_f, "p_value": p_omnibus}
    if binary:
        omnibus["p_value_dispersion"] = dispersion_test(composition, a_used, nuisance, model.n_perm, model.seed)[1]
    return rows, omnibus


def unadjusted_effects(model, patients, patient_a):
    Y = model.outcomes(patients)
    return [{"outcome": name, "n_patients": len(patient_a), **rank_association(Y[name].to_numpy(), patient_a)} for name in Y.columns]


def composition_effects(data, model, spec, exposures, blocks, stratify_by=(), unadjusted=False, fdr_family=FDR_FAMILY):
    """Exposure effects on every composition coordinate, per cohort, cancer and stratum.

    Adjusted for the covariate blocks by default; the unadjusted rank effect is added on request
    or when no block is given. Returns (effects, omnibus tests), each with BH q-values.
    """
    data = data[np.isfinite(model.outcomes(data)).all(axis=1)]
    strata = [spec.cohort, spec.cancer] + list(stratify_by)
    results, omnibus = [], []
    for key, cell in data.groupby(strata, sort=True):
        label = dict(zip(["cohort", "cancer"] + list(stratify_by), key))
        patients = model.patient_table(cell, spec)
        for exposure in exposures:
            covariates = [c for c in spec.covariates_for(exposure.name, blocks) if c not in stratify_by]
            for contrast, patient_frame, patient_a in exposure_contrasts(patients, exposure, spec.min_group_size):
                kept = patient_frame[spec.patient_id].astype(str)
                slide_frame = cell[cell[spec.patient_id].astype(str).isin(kept)]
                slide_a = slide_frame[spec.patient_id].astype(str).map(dict(zip(kept, patient_a))).to_numpy(dtype=float)
                tag = {**label, "exposure": exposure.name, "contrast": contrast}
                if blocks:
                    rows, test = adjusted_effects(model, slide_frame, patient_frame, slide_a, patient_a, covariates, spec)
                    results.extend({**tag, "adjustment": "adjusted", **row} for row in rows)
                    if test:
                        omnibus.append({**tag, "adjustment": "adjusted", **test})
                if unadjusted or not blocks:
                    results.extend({**tag, "adjustment": "unadjusted", **row} for row in unadjusted_effects(model, patient_frame, patient_a))
    if not results:
        return None, None
    table = pd.DataFrame(results)
    table["outcome_block"] = np.where(table["outcome"].str.startswith("clr_"), "composition", "density")
    table = add_fdr(table, fdr_family)
    tests = None
    if omnibus:
        tests = add_fdr(pd.DataFrame(omnibus), OMNIBUS_FDR_FAMILY)
        if "p_value_dispersion" in tests:
            tests = add_fdr(tests, OMNIBUS_FDR_FAMILY, p_column="p_value_dispersion", q_column="q_value_dispersion")
    return table, tests
