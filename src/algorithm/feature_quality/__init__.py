"""Feature quality assessment module.

Provides:
- FeatureQualityAssessor: Assess feature CSV files for low variance,
  collinearity, and stability issues.
- QualityReport: Structured report with quality score and recommendations.
"""

from .quality_assessor import (
    FeatureQualityAssessor,
    QualityReport,
    get_feature_columns,
    validate_feature_dataframe,
)

__all__ = [
    "FeatureQualityAssessor",
    "QualityReport",
    "get_feature_columns",
    "validate_feature_dataframe",
]
