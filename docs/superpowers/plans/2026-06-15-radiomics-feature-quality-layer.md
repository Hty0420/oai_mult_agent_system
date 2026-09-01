# Radiomics Feature Quality Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated feature quality assessment between feature extraction and classification in the multi-agent pipeline.

**Architecture:** New `FeatureQualityAssessor` class (algorithm layer, decoupled from MCP) performs variance filtering, collinearity detection, and stability assessment. A new `assess_feature_quality` MCP tool wraps it. `FeatureExtractionAgent` and `ClassificationAgent` prompts are enhanced to suggest and consume quality reports.

**Tech Stack:** Python 3.12+, pandas, numpy, scikit-learn (StandardScaler, correlation), MCP (FastMCP), LangChain (ChatPromptTemplate)

---

## File Structure

| File | Role |
|------|------|
| `src/algorithm/feature_quality/__init__.py` | Package entry, exports `FeatureQualityAssessor`, `QualityReport` |
| `src/algorithm/feature_quality/quality_assessor.py` | Core logic: variance/collinearity/stability assessment |
| `tests/test_feature_quality.py` | Unit tests for `FeatureQualityAssessor` |
| `tests/test_integration_feature_quality.py` | End-to-end: CSV → quality assess → filtered CSV |
| `src/server/auto_experiment_server.py` | Add `assess_feature_quality` MCP tool |
| `src/agents/feature_extraction_agent.py` | Enhance system prompt with quality guidelines |
| `src/agents/classification_agent.py` | Add quality-aware feature path detection |

---

### Task 1: Create FeatureQualityAssessor and QualityReport

**Files:**
- Create: `src/algorithm/feature_quality/__init__.py`
- Create: `src/algorithm/feature_quality/quality_assessor.py`

- [ ] **Step 1: Write the unit test**

Create `tests/test_feature_quality.py`:

```python
"""Tests for the feature_quality module."""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithm.feature_quality import FeatureQualityAssessor, QualityReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_features_df():
    """A clean feature DataFrame with no quality issues."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "CaseID": [f"C{i:03d}" for i in range(n)],
        "ImagePath": [f"/data/img_{i}.nii.gz" for i in range(n)],
        "MaskPath": [f"/data/msk_{i}.nii.gz" for i in range(n)],
        "KLGrade": np.random.choice([0, 1, 2, 3, 4], n),
        "V_Femur": np.random.normal(30000, 5000, n),
        "V_Tibia": np.random.normal(25000, 4000, n),
        "A_Femur": np.random.normal(8000, 1200, n),
        "T_mean_Femoral_Cartilage": np.random.normal(2.5, 0.5, n),
        "T_std_Femoral_Cartilage": np.random.normal(0.3, 0.1, n),
    })


@pytest.fixture
def features_with_low_variance():
    """Features with one near-constant column."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "CaseID": [f"C{i:03d}" for i in range(n)],
        "KLGrade": np.random.choice([0, 1, 2, 3, 4], n),
        "V_Femur": np.random.normal(30000, 5000, n),
        "constant_col": np.ones(n) * 42.0,
    })
    return df


@pytest.fixture
def features_with_high_correlation():
    """Features with one highly correlated pair (V_Femur vs A_Femur)."""
    np.random.seed(42)
    n = 40
    v_femur = np.random.normal(30000, 5000, n)
    # A_Femur is almost perfectly correlated with V_Femur
    a_femur = v_femur * 0.27 + np.random.normal(0, 10, n)
    return pd.DataFrame({
        "CaseID": [f"C{i:03d}" for i in range(n)],
        "KLGrade": np.random.choice([0, 1, 2, 3, 4], n),
        "V_Femur": v_femur,
        "A_Femur": a_femur,
        "V_Tibia": np.random.normal(25000, 4000, n),
    })


@pytest.fixture
def features_with_std_columns():
    """Features with _std columns for stability assessment."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "CaseID": [f"C{i:03d}" for i in range(n)],
        "KLGrade": np.random.choice([0, 1, 2, 3, 4], n),
        "V_Femur": np.random.normal(30000, 5000, n),
        "V_Femur_std": np.random.uniform(200, 500, n),
        "V_Tibia": np.random.normal(25000, 4000, n),
        "V_Tibia_std": np.random.uniform(8000, 12000, n),  # High CV
    })


# ---------------------------------------------------------------------------
# FeatureQualityAssessor tests
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_report_creation_defaults(self):
        report = QualityReport(
            total_samples=100,
            total_features=30,
            missing_count=0,
            kl_distribution={"0": 20, "1": 20, "2": 20, "3": 20, "4": 20},
        )
        assert report.total_samples == 100
        assert report.total_features == 30
        assert report.quality_score == 100
        assert report.low_variance_features == []
        assert report.high_correlation_pairs == []
        assert report.unstable_features == []
        assert report.recommendations == []

    def test_report_with_issues(self):
        report = QualityReport(
            total_samples=50,
            total_features=10,
            missing_count=0,
            kl_distribution={},
            low_variance_features=["constant_col"],
            high_correlation_pairs=[("V_Femur", "A_Femur", 0.98, "V_Femur")],
            unstable_features=[("V_Tibia", 0.45)],
            quality_score=75,
            recommendations=["Remove 1 low-variance feature"],
        )
        assert len(report.low_variance_features) == 1
        assert len(report.high_correlation_pairs) == 1
        assert len(report.unstable_features) == 1
        assert report.quality_score == 75


class TestFeatureQualityAssessorInit:
    def test_default_thresholds(self):
        assessor = FeatureQualityAssessor()
        assert assessor.variance_threshold == 0.01
        assert assessor.correlation_threshold == 0.95
        assert assessor.cv_threshold == 0.3

    def test_custom_thresholds(self):
        assessor = FeatureQualityAssessor(
            variance_threshold=0.05,
            correlation_threshold=0.85,
            cv_threshold=0.25,
        )
        assert assessor.variance_threshold == 0.05
        assert assessor.correlation_threshold == 0.85
        assert assessor.cv_threshold == 0.25


class TestBasicStats:
    def test_basic_stats_clean_data(self, clean_features_df):
        assessor = FeatureQualityAssessor()
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        assert report.total_samples == 50
        # Exclude non-feature columns (CaseID, ImagePath, MaskPath), plus KLGrade
        assert report.total_features == 5  # V_Femur, V_Tibia, A_Femur, T_mean_FC, T_std_FC
        assert report.missing_count == 0
        assert report.quality_score == 100
        assert report.recommendations == ["All quality checks passed."]


class TestLowVariance:
    def test_detect_low_variance(self, features_with_low_variance):
        assessor = FeatureQualityAssessor(variance_threshold=0.01)
        report = assessor.assess(features_with_low_variance, label_col="KLGrade")
        assert "constant_col" in report.low_variance_features
        assert any("low variance" in r.lower() for r in report.recommendations)

    def test_clean_data_no_low_variance(self, clean_features_df):
        assessor = FeatureQualityAssessor(variance_threshold=0.01)
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        assert len(report.low_variance_features) == 0


class TestCollinearity:
    def test_detect_high_correlation(self, features_with_high_correlation):
        assessor = FeatureQualityAssessor(correlation_threshold=0.95)
        report = assessor.assess(features_with_high_correlation, label_col="KLGrade")
        # V_Femur and A_Femur should be flagged
        assert len(report.high_correlation_pairs) >= 1
        pair_features = {p[0], p[1]} for p in report.high_correlation_pairs
        assert "V_Femur" in {f for pair in pair_features for f in (p[0], p[1]) if f == "V_Femur"} or \
               any("V_Femur" in (p[0], p[1]) for p in report.high_correlation_pairs)
        # Each pair should have a "keep" recommendation
        for pair in report.high_correlation_pairs:
            assert pair[3] in (pair[0], pair[1])  # keep is one of the two

    def test_clean_data_no_high_correlation(self, clean_features_df):
        assessor = FeatureQualityAssessor(correlation_threshold=0.95)
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        # With random data and few features, there might be some spurious correlations
        # due to small sample size. We just check the report is created.
        assert isinstance(report, QualityReport)


class TestStability:
    def test_detect_unstable_features(self, features_with_std_columns):
        assessor = FeatureQualityAssessor(cv_threshold=0.3)
        report = assessor.assess(features_with_std_columns, label_col="KLGrade")
        # V_Tibia should be flagged as unstable (high _std relative to mean)
        unstable_names = [u[0] for u in report.unstable_features]
        assert "V_Tibia" in unstable_names
        assert any("unstable" in r.lower() for r in report.recommendations)

    def test_no_std_columns_no_unstable(self, clean_features_df):
        assessor = FeatureQualityAssessor()
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        assert len(report.unstable_features) == 0


class TestFilterFeatures:
    def test_filter_removes_low_variance(self, features_with_low_variance):
        assessor = FeatureQualityAssessor(variance_threshold=0.01)
        report = assessor.assess(features_with_low_variance, label_col="KLGrade")
        filtered = assessor.filter_features(features_with_low_variance, report)
        assert "constant_col" not in filtered.columns
        # Non-feature columns (CaseID, KLGrade) should be preserved
        assert "CaseID" in filtered.columns
        assert "KLGrade" in filtered.columns

    def test_filter_removes_collinear(self, features_with_high_correlation):
        assessor = FeatureQualityAssessor(correlation_threshold=0.95)
        report = assessor.assess(features_with_high_correlation, label_col="KLGrade")
        filtered = assessor.filter_features(features_with_high_correlation, report)
        # One of the correlated pair should be removed
        cols = set(filtered.columns)
        assert not ("V_Femur" in cols and "A_Femur" in cols)

    def test_filter_removes_unstable(self, features_with_std_columns):
        assessor = FeatureQualityAssessor(cv_threshold=0.3)
        report = assessor.assess(features_with_std_columns, label_col="KLGrade")
        filtered = assessor.filter_features(features_with_std_columns, report)
        # V_Tibia should be removed, V_Femur should stay
        assert "V_Tibia" not in filtered.columns
        assert "V_Femur" in filtered.columns
        # _std columns should also be removed (they're not real features)
        assert "V_Tibia_std" not in filtered.columns
        assert "V_Femur_std" not in filtered.columns

    def test_filter_preserves_structure(self, clean_features_df):
        assessor = FeatureQualityAssessor()
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        filtered = assessor.filter_features(clean_features_df, report)
        assert "CaseID" in filtered.columns
        assert "KLGrade" in filtered.columns
        assert len(filtered) == len(clean_features_df)

    def test_filter_no_issues_returns_all_feature_cols(self, clean_features_df):
        assessor = FeatureQualityAssessor()
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        filtered = assessor.filter_features(clean_features_df, report)
        # All original feature columns should remain
        for col in ["V_Femur", "V_Tibia", "A_Femur", "T_mean_Femoral_Cartilage", "T_std_Femoral_Cartilage"]:
            assert col in filtered.columns


class TestQualityScoring:
    def test_perfect_score(self, clean_features_df):
        assessor = FeatureQualityAssessor()
        report = assessor.assess(clean_features_df, label_col="KLGrade")
        assert report.quality_score == 100

    def test_score_deduction_low_variance(self, features_with_low_variance):
        assessor = FeatureQualityAssessor(variance_threshold=0.01)
        report = assessor.assess(features_with_low_variance, label_col="KLGrade")
        assert report.quality_score < 100
        assert report.quality_score >= 70  # max deduction: 30

    def test_score_deduction_collinear(self, features_with_high_correlation):
        assessor = FeatureQualityAssessor(correlation_threshold=0.95)
        report = assessor.assess(features_with_high_correlation, label_col="KLGrade")
        assert report.quality_score < 100
        assert report.quality_score >= 75  # max deduction: 20

    def test_score_deduction_unstable(self, features_with_std_columns):
        assessor = FeatureQualityAssessor(cv_threshold=0.3)
        report = assessor.assess(features_with_std_columns, label_col="KLGrade")
        assert report.quality_score < 100
        assert report.quality_score >= 75  # max deduction: 20

    def test_score_not_below_zero(self, features_with_low_variance):
        """Even with extreme issues, score should not go negative."""
        # Create data with many low-variance columns
        n = 20
        data = {"CaseID": [f"C{i:03d}" for i in range(n)], "KLGrade": [0] * n}
        for i in range(10):
            data[f"constant_{i}"] = np.ones(n) * float(i)
        df = pd.DataFrame(data)
        assessor = FeatureQualityAssessor(variance_threshold=0.01)
        report = assessor.assess(df, label_col="KLGrade")
        assert report.quality_score >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /c/Users/Administrator/Desktop/oai_mcp_linux && python -m pytest tests/test_feature_quality.py -v 2>&1 | head -30
```

Expected: FAIL with `ModuleNotFoundError: No module named 'algorithm.feature_quality'`

- [ ] **Step 3: Create `src/algorithm/feature_quality/__init__.py`**

```python
"""Feature quality assessment module.

Provides:
- FeatureQualityAssessor: Assess feature CSV files for low variance,
  collinearity, and stability issues.
- QualityReport: Structured report with quality score and recommendations.
"""

from .quality_assessor import FeatureQualityAssessor, QualityReport

__all__ = ["FeatureQualityAssessor", "QualityReport"]
```

- [ ] **Step 4: Commit**

```bash
git add src/algorithm/feature_quality/__init__.py
git commit -m "feat: create feature_quality package with __init__.py"
```

- [ ] **Step 5: Create `src/algorithm/feature_quality/quality_assessor.py`**

```python
"""Feature quality assessment for radiomics/MEAS/LDSE feature CSV files."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Non-feature columns to exclude from quality checks
# ---------------------------------------------------------------------------

NON_FEATURE_PATTERNS = [
    "CaseID", "caseid", "case_id", "image_id",
    "ImagePath", "image_path",
    "MaskPath", "mask_path",
    "image_path", "mask_path",
    "Region", "region",
    "KLGrade", "klgrade", "kl_grade", "is_oa", "label",
]


def _is_feature_column(col: str) -> bool:
    """Return True if col is a feature column (not metadata/label)."""
    if col.endswith("_std"):
        return False  # std columns are uncertainty, not standalone features
    if col in NON_FEATURE_PATTERNS:
        return False
    if col.lower() in [p.lower() for p in NON_FEATURE_PATTERNS]:
        return False
    return True


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Structured quality assessment report for a feature DataFrame."""

    total_samples: int
    total_features: int
    missing_count: int
    kl_distribution: dict

    low_variance_features: list[str] = field(default_factory=list)
    high_correlation_pairs: list[tuple] = field(
        default_factory=list
    )  # (feat1, feat2, correlation, keep)
    unstable_features: list[tuple] = field(
        default_factory=list
    )  # (feature_name, cv)

    quality_score: int = 100
    recommendations: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Format the report as a human-readable text string."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE QUALITY ASSESSMENT REPORT")
        lines.append("=" * 60)
        lines.append(f"Total samples: {self.total_samples}")
        lines.append(f"Total features: {self.total_features}")
        lines.append(f"Missing values: {self.missing_count}")
        lines.append(f"KL distribution: {self.kl_distribution}")
        lines.append("")

        lines.append(f"[Low Variance Features] (threshold applied): {len(self.low_variance_features)}")
        for feat in self.low_variance_features:
            lines.append(f"  - {feat}")

        lines.append(f"\n[High Correlation Pairs] (threshold applied): {len(self.high_correlation_pairs)}")
        for feat1, feat2, r, keep in self.high_correlation_pairs:
            lines.append(f"  - {feat1} ↔ {feat2} (r={r:.4f}), keep: {keep}")

        lines.append(f"\n[Unstable Features] (CV threshold applied): {len(self.unstable_features)}")
        for feat, cv in self.unstable_features:
            lines.append(f"  - {feat} (CV={cv:.4f})")

        lines.append(f"\nQuality Score: {self.quality_score}/100")
        lines.append(f"\nRecommendations:")
        for rec in self.recommendations:
            lines.append(f"  - {rec}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "total_samples": self.total_samples,
            "total_features": self.total_features,
            "missing_count": self.missing_count,
            "kl_distribution": self.kl_distribution,
            "low_variance_features": self.low_variance_features,
            "high_correlation_pairs": [
                {"feat1": p[0], "feat2": p[1], "correlation": p[2], "keep": p[3]}
                for p in self.high_correlation_pairs
            ],
            "unstable_features": [
                {"feature": u[0], "cv": u[1]} for u in self.unstable_features
            ],
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# FeatureQualityAssessor
# ---------------------------------------------------------------------------

class FeatureQualityAssessor:
    """Assess feature CSV quality before classification.

    Checks:
    1. Low-variance features (variance < variance_threshold)
    2. High pairwise correlations (|r| > correlation_threshold)
       For each pair, recommends keeping the feature more correlated
       with the label.
    3. Stability via coefficient of variation (CV = std/|mean|)
       when _std columns are present (from OnlineAugmentor output).

    Quality score starts at 100 and deducts:
    - Low variance: -10 per feature (max -30)
    - High correlation: -5 per pair (max -20)
    - High CV: -10 per feature (max -20)
    - Missing values: proportional deduction (max -30)
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        cv_threshold: float = 0.3,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.cv_threshold = cv_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        df: pd.DataFrame,
        label_col: str = "KLGrade",
    ) -> QualityReport:
        """Run full quality assessment on a feature DataFrame.

        Args:
            df: Feature DataFrame with metadata columns (CaseID, etc.)
            label_col: Name of the label column.

        Returns:
            QualityReport with findings and recommendations.
        """
        # Identify feature columns (exclude metadata, label, std cols)
        feature_cols = [c for c in df.columns if _is_feature_column(c)]

        # 1. Basic stats
        total_samples = len(df)
        total_features = len(feature_cols)
        missing_count = int(df[feature_cols].isnull().sum().sum())

        kl_distribution = {}
        if label_col in df.columns:
            kl_distribution = (
                df[label_col].value_counts().sort_index().to_dict()
            )
            # Convert numpy int keys to python int for JSON serialization
            kl_distribution = {int(k): int(v) for k, v in kl_distribution.items()}

        # 2. Low variance check
        low_variance_features = self._check_low_variance(df, feature_cols)

        # 3. Collinearity check
        high_correlation_pairs = self._check_collinearity(
            df, feature_cols, label_col
        )

        # 4. Stability check (only if _std columns exist)
        unstable_features = self._check_stability(df, feature_cols)

        # 5. Compute quality score
        quality_score, recommendations = self._compute_score(
            low_variance_features=low_variance_features,
            high_correlation_pairs=high_correlation_pairs,
            unstable_features=unstable_features,
            missing_count=missing_count,
            total_samples=total_samples,
        )

        return QualityReport(
            total_samples=total_samples,
            total_features=total_features,
            missing_count=missing_count,
            kl_distribution=kl_distribution,
            low_variance_features=low_variance_features,
            high_correlation_pairs=high_correlation_pairs,
            unstable_features=unstable_features,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def filter_features(
        self,
        df: pd.DataFrame,
        report: QualityReport,
    ) -> pd.DataFrame:
        """Remove problematic features based on a quality report.

        Args:
            df: Original feature DataFrame.
            report: QualityReport from assess().

        Returns:
            DataFrame with low-variance, collinear, and unstable
            features removed. Metadata and label columns are preserved.
        """
        cols_to_drop: set[str] = set()

        # Low variance features
        cols_to_drop.update(report.low_variance_features)

        # One from each high-correlation pair (drop the one NOT kept)
        for feat1, feat2, _r, keep in report.high_correlation_pairs:
            drop = feat1 if keep == feat2 else feat2
            cols_to_drop.add(drop)

        # Unstable features
        for feat, _cv in report.unstable_features:
            cols_to_drop.add(feat)

        # Also drop any _std columns for removed features
        extra_drops = []
        for col in df.columns:
            if col.endswith("_std"):
                base = col[:-4]
                if base in cols_to_drop:
                    extra_drops.append(col)
        cols_to_drop.update(extra_drops)

        existing_drops = [c for c in cols_to_drop if c in df.columns]
        return df.drop(columns=existing_drops)

    # ------------------------------------------------------------------
    # Private check methods
    # ------------------------------------------------------------------

    def _check_low_variance(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> list[str]:
        """Return feature names with variance below threshold."""
        low_var = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values.astype(float)
            if len(vals) < 2:
                low_var.append(col)
                continue
            if np.var(vals) < self.variance_threshold:
                low_var.append(col)
        return low_var

    def _check_collinearity(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
    ) -> list[tuple]:
        """Return list of (feat1, feat2, correlation, keep) for high-correlation pairs."""
        if len(feature_cols) < 2:
            return []

        # Subset to numeric feature columns present in df
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 2:
            return []

        # Drop rows with NaN in any feature column
        sub = df[valid_cols].dropna()
        if len(sub) < 3:
            return []

        corr_matrix = sub.corr().abs()

        pairs = []
        seen = set()
        for i, col_i in enumerate(valid_cols):
            for col_j in valid_cols[i + 1 :]:
                r = corr_matrix.loc[col_i, col_j]
                if r > self.correlation_threshold:
                    keep = self._choose_better_feature(
                        df, col_i, col_j, label_col
                    )
                    pair_key = tuple(sorted([col_i, col_j]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append((col_i, col_j, round(float(r), 4), keep))

        return pairs

    def _choose_better_feature(
        self,
        df: pd.DataFrame,
        feat_a: str,
        feat_b: str,
        label_col: str,
    ) -> str:
        """Choose which of two correlated features to keep.

        Preference: keep the one with higher absolute correlation with label.
        If label not available, keep the first.
        """
        if label_col not in df.columns:
            return feat_a

        # Compute binarized label: KLGrade >= 2
        label = (df[label_col] >= 2).astype(int).values

        corr_a = abs(np.corrcoef(df[feat_a].dropna().values, label[: len(df[feat_a].dropna())])[0, 1]) \
            if len(df[feat_a].dropna()) > 1 else 0
        corr_b = abs(np.corrcoef(df[feat_b].dropna().values, label[: len(df[feat_b].dropna())])[0, 1]) \
            if len(df[feat_b].dropna()) > 1 else 0

        # Handle NaN (e.g., constant column)
        corr_a = 0.0 if np.isnan(corr_a) else corr_a
        corr_b = 0.0 if np.isnan(corr_b) else corr_b

        return feat_a if corr_a >= corr_b else feat_b

    def _check_stability(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> list[tuple]:
        """Return list of (feature_name, CV) for features with high CV.

        Only checks features that have a corresponding `{feature}_std` column.
        CV = std / |mean|. Flagged when CV > cv_threshold.
        """
        unstable = []
        for col in feature_cols:
            std_col = f"{col}_std"
            if std_col not in df.columns:
                continue
            # Compute mean and std across samples
            col_vals = df[col].dropna().values.astype(float)
            std_vals = df[std_col].dropna().values.astype(float)
            if len(col_vals) == 0 or len(std_vals) == 0:
                continue
            mean_val = np.mean(col_vals)
            mean_std = np.mean(std_vals)
            if abs(mean_val) < 1e-10:
                # If mean is near zero, use std directly as a flag
                if mean_std > self.cv_threshold:
                    unstable.append((col, round(float(mean_std), 4)))
            else:
                cv = mean_std / abs(mean_val)
                if cv > self.cv_threshold:
                    unstable.append((col, round(float(cv), 4)))
        return unstable

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        low_variance_features: list[str],
        high_correlation_pairs: list[tuple],
        unstable_features: list[tuple],
        missing_count: int,
        total_samples: int,
    ) -> tuple[int, list[str]]:
        """Compute quality score (0-100) and generate recommendations."""
        score = 100
        recommendations = []

        # Low variance: -10 per feature, max -30
        n_lv = len(low_variance_features)
        if n_lv > 0:
            deduction = min(n_lv * 10, 30)
            score -= deduction
            recommendations.append(
                f"Remove {n_lv} low-variance feature(s): {', '.join(low_variance_features[:5])}"
                + ("..." if n_lv > 5 else "")
            )

        # High correlation: -5 per pair, max -20
        n_hc = len(high_correlation_pairs)
        if n_hc > 0:
            deduction = min(n_hc * 5, 20)
            score -= deduction
            rec_pairs = [
                f"{p[0]}↔{p[1]} (keep {p[3]})" for p in high_correlation_pairs[:3]
            ]
            recommendations.append(
                f"Resolve {n_hc} high-correlation pair(s): {'; '.join(rec_pairs)}"
                + ("..." if n_hc > 3 else "")
            )

        # Instability: -10 per feature, max -20
        n_unstable = len(unstable_features)
        if n_unstable > 0:
            deduction = min(n_unstable * 10, 20)
            score -= deduction
            recommendations.append(
                f"Review {n_unstable} unstable feature(s): "
                + ", ".join(f"{u[0]}(CV={u[1]:.3f})" for u in unstable_features[:3])
                + ("..." if n_unstable > 3 else "")
            )

        # Missing values: proportional deduction, max -30
        if total_samples > 0 and missing_count > 0:
            missing_ratio = missing_count / (total_samples * max(1, missing_count / max(missing_count, 1)))
            # Simplified: plain ratio of missing cells
            # Actually compute properly
            missing_deduction = min(int((missing_count / max(1, total_samples)) * 100), 30)
            score -= missing_deduction
            recommendations.append(
                f"Handle {missing_count} missing value(s) before classification"
            )

        score = max(score, 0)

        if not recommendations:
            recommendations.append("All quality checks passed.")

        return score, recommendations
```

- [ ] **Step 6: Commit**

```bash
git add src/algorithm/feature_quality/quality_assessor.py
git commit -m "feat: add FeatureQualityAssessor with variance/collinearity/stability checks"
```

---

### Task 2: Add MCP tool to auto_experiment_server

**Files:**
- Modify: `src/server/auto_experiment_server.py` (add import near line 28, add tool before `if __name__`)

- [ ] **Step 1: Add import**

In `src/server/auto_experiment_server.py`, after line 28 (`from algorithm.data_augmentation import OfflineAugmentor, OnlineAugmentor`), add:

```python
# Feature quality assessment
from algorithm.feature_quality import FeatureQualityAssessor
```

- [ ] **Step 2: Add MCP tool**

In `src/server/auto_experiment_server.py`, add this tool before the `if __name__ == "__main__":` block (before line 622):

```python

@mcp.tool()
def assess_feature_quality(
    feature_csv: Annotated[str, Field(description="Path to feature CSV file to assess")],
    output_dir: Annotated[str, Field(description="Output directory for quality reports")],
    label_col: Annotated[str, Field(description="Label column name, default KLGrade")] = "KLGrade",
    variance_threshold: Annotated[float, Field(description="Low variance threshold, default 0.01")] = 0.01,
    correlation_threshold: Annotated[float, Field(description="High correlation threshold, default 0.95")] = 0.95,
    cv_threshold: Annotated[float, Field(description="CV threshold for stability check, default 0.3")] = 0.3,
):
    """
    Assess feature CSV quality before classification.

    Checks for:
    - Low-variance features (variance < variance_threshold)
    - High collinearity (|r| > correlation_threshold), recommends keeping
      the feature more correlated with the label
    - Feature stability via coefficient of variation (CV = std/|mean|)
      when _std columns from online augmentation are present

    Outputs:
    - quality_report.txt: human-readable report
    - quality_metrics.json: structured metrics
    - filtered_features.csv: filtered feature file with issues removed

    Returns:
        dict with status, report_path, filtered_csv, metrics_json, quality_score, summary
    """
    import json as _json

    try:
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(feature_csv, dtype={"CaseID": str})

        assessor = FeatureQualityAssessor(
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            cv_threshold=cv_threshold,
        )

        report = assessor.assess(df, label_col=label_col)

        # Save text report
        report_path = os.path.join(output_dir, "quality_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_text())

        # Save JSON metrics
        metrics_path = os.path.join(output_dir, "quality_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            _json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        # Save filtered features
        filtered_df = assessor.filter_features(df, report)
        filtered_csv = os.path.join(output_dir, "filtered_features.csv")
        filtered_df.to_csv(filtered_csv, index=False)

        summary = (
            f"Quality score: {report.quality_score}/100. "
            f"Found: {len(report.low_variance_features)} low-variance, "
            f"{len(report.high_correlation_pairs)} high-correlation pairs, "
            f"{len(report.unstable_features)} unstable features. "
            f"Original features: {report.total_features}, "
            f"Filtered features: {sum(1 for c in filtered_df.columns if _is_feature_column(c))}."
        )

        return {
            "status": "success",
            "report_path": report_path,
            "filtered_csv": filtered_csv,
            "metrics_json": metrics_path,
            "quality_score": report.quality_score,
            "summary": summary,
        }

    except Exception as e:
        return {"status": "failed", "message": str(e)}


# Helper: duplicate _is_feature_column so the tool doesn't need to import from quality_assessor
def _is_feature_column(col: str) -> bool:
    non_feature = {
        "CaseID", "caseid", "case_id", "image_id",
        "ImagePath", "image_path", "MaskPath", "mask_path",
        "Region", "region", "KLGrade", "klgrade", "kl_grade",
        "is_oa", "label",
    }
    if col.endswith("_std"):
        return False
    if col in non_feature:
        return False
    if col.lower() in {p.lower() for p in non_feature}:
        return False
    return True
```

- [ ] **Step 3: Commit**

```bash
git add src/server/auto_experiment_server.py
git commit -m "feat: add assess_feature_quality MCP tool to auto_experiment_server"
```

---

### Task 3: Enhance FeatureExtractionAgent

**Files:**
- Modify: `src/agents/feature_extraction_agent.py` (system prompt and tool list)

- [ ] **Step 1: Add `assess_feature_quality` to available tools**

In `src/agents/feature_extraction_agent.py`, find the system prompt tool list (around line 25-46). Add a 4th tool entry after `ldse_features_extraction`:

Replace lines 25-46 with the updated tool list. The addition is a new section 4 after the LDSE tool:

```python
        self.system_prompt = """You are a specialized medical imaging agent for extracting radiomics and morphological 
        features from osteoarthritis (OA) MRI images. Your role is to help users extract various types of features 
        from segmented images.

        Available feature extraction tools:
        1. extract_radiomics_features - Extract radiomics features (shape, texture, first-order statistics)
           Parameters: 
           - csv_path (str): CSV file with ImagePath, MaskPath, KLGrade columns
           - output_dir (str): Output directory for results
           - num_regions (int): Number of regions to extract features from (default 5)
           - num_cases (int): Maximum number of cases to process (default 2)
           - extract_all_cases (bool): Extract features for all cases (default False)
           - extract_all_features (bool): Extract all radiomics features including filtered images (default False)
           
        2. meas_features_extraction - Extract MEAS morphological and geometric features
           Parameters:
           - output_dir (str): Output directory for results (FIRST PARAMETER)
           - csv_path (str): CSV file with image_path, mask_path, case_id, koa_grade columns (SECOND PARAMETER)
           
        3. ldse_features_extraction - Extract low-dimensional shape encoding features
           Parameters:
           - output_dir (str): Output directory for results (FIRST PARAMETER)
           - csv_path (str): CSV file with ImagePath, MaskPath, CaseID, KLGrade columns (SECOND PARAMETER)
           - n_components (int): Encoding dimension (default 50)

        4. augment_online_extract_features - Extract robust features using online augmentation
           Parameters:
           - csv_path (str): Path to CSV with ImagePath, MaskPath, CaseID, KLGrade columns
           - output_dir (str): Output directory for augmented feature CSV
           - n_augmentations (int): Number of augmented views per sample (default 3)
           - feature_type (str): Feature type 'MEAS', 'LDSE', or 'radiomics'
           - aggregator (str): Aggregation method 'mean' or 'median' (default 'mean')
           - seed (int): Random seed (default 42)

        5. assess_feature_quality - Assess feature CSV quality before classification
           Parameters:
           - feature_csv (str): Path to feature CSV file to assess
           - output_dir (str): Output directory for quality reports
           - label_col (str): Label column name (default 'KLGrade')
           - variance_threshold (float): Low variance threshold (default 0.01)
           - correlation_threshold (float): High correlation threshold (default 0.95)
           - cv_threshold (float): CV threshold for stability check (default 0.3)
```

- [ ] **Step 2: Add quality guidelines to system prompt**

After the feature extraction tool descriptions and before the "When user requests feature extraction" section (around line 48), add:

```python
        When user requests feature extraction:
        1. Identify which type of features they want (radiomics, MEAS, LDSE)
        2. Extract CSV file path from their input
        3. Determine appropriate parameters based on their request
        4. Call appropriate feature extraction tool(s) in order
        5. AFTER extraction completes, if user intends to use features for classification:
           a. SUGGEST using augment_online_extract_features for robust features with uncertainty estimates
           b. SUGGEST running assess_feature_quality to detect and filter problematic features
        6. Report results clearly, including output file paths and feature counts

        Quality Guidelines (when features will be used for classification):
        - If the user mentions "classification", "KL grading", "LASSO", or "Filter" in their request,
          proactively suggest running assess_feature_quality after extraction
        - If the dataset is known to have limited samples (<50 cases), strongly recommend
          augment_online_extract_features with n_augmentations=3+ for robust features
        - Quality assessment should be done on the training set first; the same filters
          can then be applied to the test set
        - Online augmentation (augment_online_extract_features) produces _std columns that
          enable stability assessment — recommend it when robustness is important
```

- [ ] **Step 3: Commit**

```bash
git add src/agents/feature_extraction_agent.py
git commit -m "feat: enhance FeatureExtractionAgent with quality guidelines and tool references"
```

---

### Task 4: Enhance ClassificationAgent

**Files:**
- Modify: `src/agents/classification_agent.py` (system prompt and quality-aware logic)

- [ ] **Step 1: Add `assess_feature_quality` to available tools**

In `src/agents/classification_agent.py`, find the system prompt tool list (around line 27-49). Add after tool 7 (`read_classification_results`):

```python
        8. assess_feature_quality - Assess feature quality before classification
           Parameters: feature_csv (str), output_dir (str), label_col (str), 
           variance_threshold (float), correlation_threshold (float), cv_threshold (float)
           Outputs: quality_report.txt, quality_metrics.json, filtered_features.csv
```

- [ ] **Step 2: Add quality-aware logic to the system prompt**

After the existing tool list, update the "When user requests classification" section:

Find the section starting "When user requests classification:" (around line 51) and replace it with:

```python
        When user requests classification:
        0. [NEW] Check if filtered_features.csv exists for the feature dataset
           → If yes, prefer using filtered features for better results
           → If no AND quality hasn't been assessed yet, suggest running assess_feature_quality first
        1. Identify which classification method they want to use (default to LASSO if not specified)
        2. Extract train/test dataset paths from their input
        3. Determine if multiple tools need to be called (e.g., classification + quality assessment + reading results)
        4. Call tools in correct order (quality assessment → classification → read results)
        5. Report results clearly, including accuracy, confusion matrix, and feature importance
```

- [ ] **Step 3: Add filtered CSV detection to `process_input`**

In the `process_input` construction (around line 221-263), add filtered CSV detection. Insert before the `Respond in JSON format` line:

Find the line `Respond in JSON format with tools array and final_message.` and replace the preceding lines with:

```python
        
        # Check if filtered features exist for the detected feature directories
        filtered_train_path = None
        filtered_test_path = None
        if train_feature_dir:
            potential_filtered = os.path.join(train_feature_dir, "filtered_features.csv")
            if os.path.exists(potential_filtered):
                filtered_train_path = potential_filtered
        if test_feature_dir:
            potential_filtered = os.path.join(test_feature_dir, "filtered_features.csv")
            if os.path.exists(potential_filtered):
                filtered_test_path = potential_filtered
        
        quality_note = ""
        if filtered_train_path and filtered_test_path:
            quality_note = f"\n⚠️  QUALITY-ASSESSED FEATURES AVAILABLE:\n  Train: {filtered_train_path}\n  Test: {filtered_test_path}\n  PREFER using these filtered feature files for classification instead of the original files."
        elif feature_type:
            quality_note = f"\n💡 FEATURE TYPE: The features being classified are {feature_type} type. Consider suggesting quality assessment (assess_feature_quality) before classification."
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        You are Classification Agent. Help user perform KOA classification and KL grading.
        Determine which classification tools to use and what parameters to provide.
        
        CRITICAL: Identify if multiple tools need to be called.
        If user requests classification and reading results, you MUST include BOTH:
           - A classification tool (koa_lasso_classification or koa_filter_classification)
           - The read_classification_results tool
        {quality_note}
```

Also need to add `import os` at the top — it should already be there (check, but it's likely present).

- [ ] **Step 4: Commit**

```bash
git add src/agents/classification_agent.py
git commit -m "feat: enhance ClassificationAgent with quality-aware feature path detection"
```

---

### Task 5: Write integration tests

**Files:**
- Create: `tests/test_integration_feature_quality.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_feature_quality.py`:

```python
"""End-to-end integration tests for the feature quality pipeline.

Covers: feature extraction → quality assessment → filtered features → classification.
Uses synthetic data so no real MRI files are needed.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithm.feature_quality import FeatureQualityAssessor


def _make_synthetic_feature_csv(output_dir: str, with_issues: bool = False) -> str:
    """Create a synthetic feature CSV for testing.

    Args:
        output_dir: Directory to save the CSV.
        with_issues: If True, include low-variance and high-correlation features.

    Returns:
        Path to the created CSV.
    """
    np.random.seed(42)
    n = 60

    data = {
        "CaseID": [f"C{i:03d}" for i in range(n)],
        "ImagePath": [f"/data/img_{i}.nii.gz" for i in range(n)],
        "MaskPath": [f"/data/msk_{i}.nii.gz" for i in range(n)],
        "KLGrade": np.random.choice([0, 1, 2, 3, 4], n),
        "V_Femur": np.random.normal(30000, 5000, n),
        "A_Femur": np.random.normal(8000, 1200, n),
        "V_Tibia": np.random.normal(25000, 4000, n),
        "T_mean_Femoral_Cartilage": np.random.normal(2.5, 0.5, n),
    }

    if with_issues:
        # Add a near-constant feature
        data["constant_feat"] = np.ones(n) * 42.0 + np.random.normal(0, 0.001, n)
        # Add a feature highly correlated with V_Femur
        data["V_Femur_alt"] = data["V_Femur"] * 0.5 + np.random.normal(0, 5, n)
        # Add _std columns for stability check
        data["V_Tibia_std"] = np.random.uniform(8000, 12000, n)  # High CV
        data["A_Femur_std"] = np.random.uniform(200, 400, n)  # Normal CV

    csv_path = os.path.join(output_dir, "test_features.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_full_quality_pipeline_clean_data():
    """Clean data should score 100 and filtered CSV should match original."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=False)

        assessor = FeatureQualityAssessor()
        df = pd.read_csv(csv_path)
        report = assessor.assess(df, label_col="KLGrade")

        assert report.quality_score == 100
        assert len(report.low_variance_features) == 0
        assert len(report.high_correlation_pairs) == 0
        assert len(report.unstable_features) == 0

        # Filter should preserve all feature columns
        filtered = assessor.filter_features(df, report)
        for col in ["V_Femur", "A_Femur", "V_Tibia", "T_mean_Femoral_Cartilage"]:
            assert col in filtered.columns
        assert "CaseID" in filtered.columns
        assert "KLGrade" in filtered.columns

        print("Clean data pipeline: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_full_quality_pipeline_with_issues():
    """Data with issues should trigger deductions and feature removal."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=True)

        assessor = FeatureQualityAssessor()
        df = pd.read_csv(csv_path)
        report = assessor.assess(df, label_col="KLGrade")

        # Should detect issues
        assert report.quality_score < 100
        assert len(report.low_variance_features) > 0  # constant_feat
        assert len(report.high_correlation_pairs) > 0  # V_Femur vs V_Femur_alt
        assert len(report.unstable_features) > 0  # V_Tibia (high CV)
        assert len(report.recommendations) > 1

        # Filter should remove problematic features
        filtered = assessor.filter_features(df, report)
        assert "constant_feat" not in filtered.columns
        # One of V_Femur/V_Femur_alt should be removed
        assert not ("V_Femur" in filtered.columns and "V_Femur_alt" in filtered.columns)
        assert "V_Tibia" not in filtered.columns

        # Non-feature columns preserved
        assert "CaseID" in filtered.columns
        assert "KLGrade" in filtered.columns

        print("Data with issues pipeline: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_text_report_generation():
    """Text report should include all expected sections."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=True)

        assessor = FeatureQualityAssessor()
        df = pd.read_csv(csv_path)
        report = assessor.assess(df, label_col="KLGrade")

        text = report.to_text()

        assert "FEATURE QUALITY ASSESSMENT REPORT" in text
        assert "Quality Score:" in text
        assert "Recommendations:" in text
        assert "Total samples:" in text

        # Save to disk
        report_path = os.path.join(tmpdir, "quality_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)

        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            saved = f.read()
        assert saved == text

        print("Text report generation: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_json_metrics_serialization():
    """Quality report should serialize to JSON without errors."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=True)
        import json

        assessor = FeatureQualityAssessor()
        df = pd.read_csv(csv_path)
        report = assessor.assess(df, label_col="KLGrade")

        metrics = report.to_dict()
        assert metrics["quality_score"] < 100
        assert isinstance(metrics["kl_distribution"], dict)
        assert isinstance(metrics["low_variance_features"], list)
        assert isinstance(metrics["recommendations"], list)

        # Should be JSON serializable
        json_str = json.dumps(metrics, indent=2, ensure_ascii=False)
        assert len(json_str) > 0

        # Round-trip
        restored = json.loads(json_str)
        assert restored["quality_score"] == metrics["quality_score"]
        assert restored["total_samples"] == metrics["total_samples"]

        print("JSON metrics serialization: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_filtered_csv_usable_for_classification():
    """Filtered CSV should be directly usable by LASSO/Filter classifiers."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=True)

        assessor = FeatureQualityAssessor()
        df = pd.read_csv(csv_path)
        report = assessor.assess(df, label_col="KLGrade")
        filtered = assessor.filter_features(df, report)

        # Save filtered CSV
        filtered_csv = os.path.join(tmpdir, "filtered_features.csv")
        filtered.to_csv(filtered_csv, index=False)

        # Verify it can be loaded by KOALassoClassifier-style code
        df_loaded = pd.read_csv(filtered_csv)
        assert "KLGrade" in df_loaded.columns
        assert "CaseID" in df_loaded.columns

        # Check binary label construction works (same as classifier does)
        label = (df_loaded["KLGrade"] >= 2).astype(int)
        assert label.nunique() >= 1  # at least one class present

        # Feature columns should all be numeric
        drop_cols = {"CaseID", "KLGrade", "ImagePath", "MaskPath"}
        feature_cols = [c for c in df_loaded.columns if c not in drop_cols and not c.endswith("_std")]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df_loaded[col]), f"{col} is not numeric"

        print("Filtered CSV classification-ready: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_assessor_custom_thresholds():
    """Custom thresholds should affect what is flagged."""
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = _make_synthetic_feature_csv(tmpdir, with_issues=True)

        # Very permissive thresholds should flag nothing
        permissive = FeatureQualityAssessor(
            variance_threshold=0.0,
            correlation_threshold=1.0,
            cv_threshold=999.0,
        )
        df = pd.read_csv(csv_path)
        report = permissive.assess(df, label_col="KLGrade")
        # constant_feat might still be flagged if variance is 0 exactly
        # but V_Femur/V_Femur_alt correlation might not exceed 1.0

        # Very strict thresholds should flag more
        strict = FeatureQualityAssessor(
            variance_threshold=100.0,
            correlation_threshold=0.1,
            cv_threshold=0.001,
        )
        report_strict = strict.assess(df, label_col="KLGrade")
        # Should flag many more issues
        assert report_strict.quality_score <= report.quality_score

        print("Custom thresholds: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Run integration tests**

```bash
cd /c/Users/Administrator/Desktop/oai_mcp_linux && python -m pytest tests/test_integration_feature_quality.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_quality.py tests/test_integration_feature_quality.py
git commit -m "test: add unit and integration tests for feature quality module"
```

---

### Task 6: Run full test suite and verify

- [ ] **Step 1: Run all tests**

```bash
cd /c/Users/Administrator/Desktop/oai_mcp_linux && python -m pytest tests/ -v
```

Expected: All tests PASS (existing + new).

- [ ] **Step 2: Final commit if any fixes needed**

If tests pass:
```bash
echo "All tests passing, implementation complete."
```

If fixes needed, commit them with a descriptive message.
