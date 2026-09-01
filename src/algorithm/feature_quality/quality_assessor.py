"""Feature quality assessment for radiomics/MEAS/LDSE feature CSV files."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Non-feature columns to exclude from quality checks
# ---------------------------------------------------------------------------

NON_FEATURE_PATTERNS = [
    "CaseID", "ImagePath", "MaskPath", "Region", "KLGrade",
    "image_path", "mask_path",
    "KneeSide", "Gender", "Age", "BMI", "Augmented", "SourceCaseID",
    "image_id",
]


def _is_feature_column(col: str) -> bool:
    """Return True if col is a feature column (not metadata/label)."""
    if col.endswith("_std"):
        return False
    if col in NON_FEATURE_PATTERNS:
        return False
    if col.lower() in [p.lower() for p in NON_FEATURE_PATTERNS]:
        return False
    return True


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be treated as model features."""
    return [c for c in df.columns if _is_feature_column(c)]


def validate_feature_dataframe(
    df: pd.DataFrame,
    label_col: str = "KLGrade",
) -> tuple[bool, str, list[str]]:
    """Validate that a DataFrame contains real feature columns."""
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        return (
            False,
            "No feature columns found. The input looks like a label/metadata CSV, "
            "not an extracted feature CSV. Run feature extraction first and pass "
            "the generated *_features_summary.csv or filtered_features.csv file.",
            [],
        )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return (
            False,
            f"Feature columns must be numeric. Non-numeric columns: {non_numeric[:10]}",
            feature_cols,
        )

    if label_col not in df.columns:
        return (
            False,
            f"Label column '{label_col}' was not found in the feature CSV.",
            feature_cols,
        )

    return True, "", feature_cols


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

    low_variance_features: list = field(default_factory=list)
    high_correlation_pairs: list = field(default_factory=list)
    unstable_features: list = field(default_factory=list)

    quality_score: int = 100
    recommendations: list = field(default_factory=list)

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
            lines.append(f"  - {feat1} <-> {feat2} (r={r:.4f}), keep: {keep}")

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
    3. Stability via coefficient of variation (CV = std/|mean|)
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        cv_threshold: float = 3.0,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.cv_threshold = cv_threshold

    def assess(
        self,
        df: pd.DataFrame,
        label_col: str,
    ) -> QualityReport:
        """Run full quality assessment on a feature DataFrame."""
        feature_cols = get_feature_columns(df)

        # 1. Basic stats
        total_samples = len(df)
        total_features = len(feature_cols)
        missing_count = int(df[feature_cols].isnull().sum().sum())

        kl_distribution = {}
        if label_col in df.columns:
            kl_distribution = (
                df[label_col].value_counts().sort_index().to_dict()
            )
            kl_distribution = {int(k): int(v) for k, v in kl_distribution.items()}

        # 2. Low variance check
        low_variance_features = self._check_low_variance(df, feature_cols)

        # 3. Collinearity check
        high_correlation_pairs = self._check_collinearity(
            df, feature_cols, label_col
        )

        # 4. Stability check
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
        filter_stability: bool = False,
    ) -> pd.DataFrame:
        """Remove problematic features based on a quality report.

        Args:
            df: Input feature DataFrame.
            report: QualityReport from assess().
            filter_stability: If True, also remove unstable features.
                Default False — stability issues are reported but NOT
                auto-filtered because downstream classifiers (e.g. LASSO)
                can handle moderate noise through regularization.
        """
        cols_to_drop = set()

        cols_to_drop.update(report.low_variance_features)

        for feat1, feat2, _r, keep in report.high_correlation_pairs:
            drop = feat1 if keep == feat2 else feat2
            cols_to_drop.add(drop)

        if filter_stability:
            for feat, _cv in report.unstable_features:
                cols_to_drop.add(feat)

        # Also drop associated _std companion columns for removed features
        extra_drops = [
            col for col in df.columns
            if col.endswith("_std") and col[:-4] in cols_to_drop
        ]
        cols_to_drop.update(extra_drops)

        existing_drops = [c for c in cols_to_drop if c in df.columns]
        return df.drop(columns=existing_drops)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _check_low_variance(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> list:
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
        feature_cols: list,
        label_col: str,
    ) -> list:
        """Return list of (feat1, feat2, correlation, keep) for high-correlation pairs."""
        if len(feature_cols) < 2:
            return []

        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 2:
            return []

        sub = df[valid_cols].dropna()
        if len(sub) < 3:
            return []

        corr_matrix = sub.corr().abs()

        pairs = []
        for i, col_i in enumerate(valid_cols):
            for col_j in valid_cols[i + 1:]:
                r = corr_matrix.loc[col_i, col_j]
                if r > self.correlation_threshold:
                    keep = self._choose_better_feature(
                        df, col_i, col_j, label_col
                    )
                    pairs.append((col_i, col_j, round(float(r), 4), keep))

        return pairs

    def _choose_better_feature(
        self,
        df: pd.DataFrame,
        feat_a: str,
        feat_b: str,
        label_col: str,
    ) -> str:
        """Choose which of two correlated features to keep."""
        if label_col not in df.columns:
            return feat_a

        label = (df[label_col] >= 2).astype(int)

        def _label_corr(feat: str) -> float:
            """Pearson correlation between feature and label, with index-aligned NaN handling."""
            paired = df[[feat, label_col]].dropna()
            if len(paired) < 2:
                return 0.0
            paired_label = (paired[label_col] >= 2).astype(int).values
            c = np.corrcoef(paired[feat].values, paired_label)[0, 1]
            return 0.0 if np.isnan(c) else abs(float(c))

        corr_a = _label_corr(feat_a)
        corr_b = _label_corr(feat_b)

        return feat_a if corr_a >= corr_b else feat_b

    def _check_stability(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> list:
        """Return list of (feature_name, CV) for features with high CV.

        Stability is measured as: mean(_std) / std(feature_values)
        This compares augmentation noise (within-sample variation) to
        biological signal (cross-sample variation).  Using cross-sample
        std as denominator avoids the near-zero-mean problem that PCA
        scores exhibit (where mean ≈ 0 makes traditional CV explode).
        """
        unstable = []
        for col in feature_cols:
            std_col = f"{col}_std"
            if std_col not in df.columns:
                continue
            col_vals = df[col].dropna().values.astype(float)
            std_vals = df[std_col].dropna().values.astype(float)
            if len(col_vals) < 2 or len(std_vals) < 2:
                continue
            # Use cross-sample std as the denominator — captures signal range
            signal_std = np.std(col_vals)
            if signal_std < 1e-10:
                # Feature is essentially constant across samples → low signal
                # Fall back to mean-based check: does it vary at all?
                mean_abs = np.mean(np.abs(col_vals))
                if mean_abs < 1e-10:
                    continue  # truly zero feature, not meaningful
                cv = np.mean(std_vals) / mean_abs
            else:
                # Primary metric: noise-to-signal ratio
                cv = np.mean(std_vals) / signal_std
            if cv > self.cv_threshold:
                unstable.append((col, round(float(cv), 4)))
        return unstable

    def _compute_score(
        self,
        low_variance_features: list,
        high_correlation_pairs: list,
        unstable_features: list,
        missing_count: int,
        total_samples: int,
    ) -> tuple:
        """Compute quality score (0-100) and generate recommendations."""
        score = 100
        recommendations = []

        n_lv = len(low_variance_features)
        if n_lv > 0:
            deduction = min(n_lv * 10, 30)
            score -= deduction
            recommendations.append(
                f"Remove {n_lv} low variance feature(s): "
                + ", ".join(low_variance_features[:5])
                + ("..." if n_lv > 5 else "")
            )

        n_hc = len(high_correlation_pairs)
        if n_hc > 0:
            deduction = min(n_hc * 5, 20)
            score -= deduction
            rec_pairs = [
                f"{p[0]}<->{p[1]} (keep {p[3]})"
                for p in high_correlation_pairs[:3]
            ]
            recommendations.append(
                f"Resolve {n_hc} high-correlation pair(s): "
                + "; ".join(rec_pairs)
                + ("..." if n_hc > 3 else "")
            )

        n_unstable = len(unstable_features)
        if n_unstable > 0:
            deduction = min(n_unstable * 10, 20)
            score -= deduction
            recommendations.append(
                f"Review {n_unstable} unstable feature(s): "
                + ", ".join(
                    f"{u[0]}(CV={u[1]:.3f})" for u in unstable_features[:3]
                )
                + ("..." if n_unstable > 3 else "")
            )

        if total_samples > 0 and missing_count > 0:
            missing_deduction = min(
                int((missing_count / max(1, total_samples)) * 100), 30
            )
            score -= missing_deduction
            recommendations.append(
                f"Handle {missing_count} missing value(s) before classification"
            )

        score = max(score, 0)

        if not recommendations:
            recommendations.append("All quality checks passed.")

        return score, recommendations
