from __future__ import annotations

import numpy as np

from algorithm.biomarker_discovery.statistics import (
    adjust_bh,
    adjust_holm,
    feature_association,
    fit_ordered_logit,
)


def test_multiplicity_adjustments_match_known_values() -> None:
    p_values = [0.01, 0.04, 0.03, 0.20]
    np.testing.assert_allclose(
        adjust_bh(p_values),
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.20],
    )
    np.testing.assert_allclose(adjust_holm(p_values), [0.04, 0.09, 0.09, 0.20])


def test_ordered_logit_recovers_positive_higher_grade_effect() -> None:
    rng = np.random.default_rng(4)
    feature = rng.normal(size=300)
    latent = 0.9 * feature + rng.logistic(size=300)
    grade = np.digitize(latent, [-1.0, -0.25, 0.45, 1.2])
    covariates = np.empty((grade.size, 0), dtype=np.float64)

    baseline = fit_ordered_logit(grade, covariates)
    result = feature_association(
        grade,
        covariates,
        feature,
        reduced_fit=baseline,
    )

    assert baseline.converged
    assert result["converged"] is True
    assert result["beta_per_sd"] > 0
    assert result["odds_ratio_per_sd"] > 1
    assert result["p_value"] < 0.001
