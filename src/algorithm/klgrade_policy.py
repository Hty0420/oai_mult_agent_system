from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabeledKLGradeRows:
    frame: pd.DataFrame
    input_rows: int
    labeled_rows: int
    excluded_unlabeled_rows: int


def select_labeled_klgrade_rows(
    frame: pd.DataFrame,
    *,
    dataset_role: str,
) -> LabeledKLGradeRows:
    if "KLGrade" not in frame.columns:
        raise ValueError(f"{dataset_role} is missing required column: KLGrade")

    numeric = pd.to_numeric(frame["KLGrade"], errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=frame.index,
    )
    integral = numeric.notna() & numeric.eq(np.floor(numeric))
    in_domain = numeric.between(-1, 4, inclusive="both")
    invalid = ~(finite & integral & in_domain)
    if invalid.any():
        raise ValueError(
            f"{dataset_role} KLGrade contains {int(invalid.sum())} invalid value(s); "
            "expected -1 for unlabeled or integers 0 through 4."
        )

    unlabeled = numeric.eq(-1)
    selected = frame.loc[~unlabeled].copy()
    selected["KLGrade"] = numeric.loc[~unlabeled].astype("int64")
    if selected.empty:
        raise ValueError(f"{dataset_role} has no labeled KLGrade rows after excluding -1.")

    return LabeledKLGradeRows(
        frame=selected,
        input_rows=int(len(frame)),
        labeled_rows=int(len(selected)),
        excluded_unlabeled_rows=int(unlabeled.sum()),
    )
