"""Final execution for the atomic ANOVA_TOPK_L2 pipeline recipe."""

from __future__ import annotations

import json
import os

import matplotlib

# Classification runs inside headless MCP subprocesses.  Selecting the backend
# before importing pyplot avoids accidental Tk initialization on Windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from algorithm.klgrade_policy import select_labeled_klgrade_rows

from .classification_artifacts import compute_binary_metrics
from .grouped_validation import (
    aggregate_group_probabilities,
    assert_disjoint_groups,
)
from .patient_feature_table import PatientFeatureTableBuilder


plt.rcParams["font.family"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


class UniversalFilterSelector:
    """Run the same ANOVA+TopK+L2 recipe used during candidate validation."""

    MODEL_RECIPE = "ANOVA_TOPK_L2"

    def __init__(
        self,
        train_feat_path,
        test_feat_path,
        output_dir,
        feature_type,
        id_col,
        label_col,
        exclude_cols=None,
    ):
        self.train_feat_path = train_feat_path
        self.test_feat_path = test_feat_path
        self.feature_type = feature_type
        self.output_dir = output_dir
        self.id_col = id_col
        self.label_col = label_col
        self.default_exclude = [id_col, label_col, "is_oa", "Region"]
        self.exclude_cols = list(
            set(self.default_exclude + (exclude_cols if exclude_cols else []))
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_prepare_data(self):
        try:
            train_df = pd.read_csv(
                self.train_feat_path, encoding="utf-8", dtype={self.id_col: str}
            )
            test_df = pd.read_csv(
                self.test_feat_path, encoding="utf-8", dtype={self.id_col: str}
            )
            for column in (self.id_col, self.label_col):
                if column not in train_df.columns or column not in test_df.columns:
                    raise ValueError(f"训练集或测试集缺少必要列 '{column}'")
            if self.id_col != "CaseID" or self.label_col != "KLGrade":
                train_df = train_df.rename(
                    columns={self.id_col: "CaseID", self.label_col: "KLGrade"}
                )
                test_df = test_df.rename(
                    columns={self.id_col: "CaseID", self.label_col: "KLGrade"}
                )
            train_selection = select_labeled_klgrade_rows(
                train_df, dataset_role="Filter training feature CSV"
            )
            test_selection = select_labeled_klgrade_rows(
                test_df, dataset_role="Filter test feature CSV"
            )
            builder = PatientFeatureTableBuilder()
            train_table = builder.build(train_selection.frame)
            test_table = builder.build(
                test_selection.frame, allow_repeated_groups=True
            )
            assert_disjoint_groups(
                train_table.frame[train_table.group_column],
                test_table.frame[test_table.group_column],
            )
            feature_cols = list(train_table.feature_columns)
            missing_test = [
                column for column in feature_cols if column not in test_table.frame
            ]
            if missing_test:
                raise ValueError(f"测试集缺少训练特征: {missing_test[0]}")
            X_train = train_table.frame[feature_cols].astype(float).to_numpy()
            X_test = test_table.frame[feature_cols].astype(float).to_numpy()
            y_train = train_table.frame["label"].to_numpy(dtype=int)
            y_test = test_table.frame["label"].to_numpy(dtype=int)
            if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
                raise ValueError("训练集和测试集均必须包含两个二分类类别")
            test_groups = test_table.frame[test_table.group_column].astype(str).to_numpy()
            return (
                X_train,
                y_train,
                X_test,
                y_test,
                test_groups,
                feature_cols,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"特征文件未找到: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"数据加载和准备失败: {exc}") from exc

    @staticmethod
    def _bounded_k(k, feature_count: int):
        if isinstance(k, str) and k.strip().lower() == "all":
            return "all"
        try:
            value = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError("k 必须是正整数或 'all'") from exc
        if value < 1:
            raise ValueError("k 必须至少为 1")
        return min(value, feature_count)

    def run_selection_pipeline(self, k=10, model_c=1.0):
        """Fit ANOVA selection on train only, then evaluate one frozen recipe."""
        if float(model_c) <= 0:
            raise ValueError("model_c 必须大于 0")
        X_train, y_train, X_test, y_test, test_groups, feature_names = (
            self.load_and_prepare_data()
        )
        bounded_k = self._bounded_k(k, len(feature_names))

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        selector = SelectKBest(score_func=f_classif, k=bounded_k)
        X_train_selected = selector.fit_transform(X_train_imputed, y_train)
        X_test_selected = selector.transform(X_test_imputed)
        support = selector.get_support()
        selected_features = [
            feature_names[index] for index in np.flatnonzero(support)
        ]
        selected_scores = selector.scores_[support]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=float(model_c),
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)
        y_probability = model.predict_proba(X_test_scaled)[:, 1]
        patient_predictions = aggregate_group_probabilities(
            test_groups, y_test, y_probability
        )
        patient_metrics = compute_binary_metrics(patient_predictions)
        metrics = {
            "auc": float(patient_metrics["AUC"]),
            "precision": float(patient_metrics["Precision"]),
            "recall": float(patient_metrics["Sensitivity"]),
        }

        feature_path = os.path.join(
            self.output_dir, f"SelectedFeatures_Top{bounded_k}.txt"
        )
        with open(feature_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("Method\tFeature\tScore\n")
            for feature, score in zip(selected_features, selected_scores):
                stream.write(f"ANOVA F-value (F值)\t{feature}\t{score:.8g}\n")

        metrics_path = os.path.join(self.output_dir, "metrics_summary.txt")
        with open(metrics_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("=====  特征选择性能总结 =====\n\n")
            stream.write(
                "ANOVA F-value (F值)        : "
                f"AUC={metrics['auc']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}\n"
            )

        manifest = {
            "schema_version": "1.0",
            "model_recipe": self.MODEL_RECIPE,
            "feature_type": self.feature_type,
            "test_data_used_for_model_selection": False,
            "parameters": {"selector__k": bounded_k, "model__C": float(model_c)},
            "selected_feature_count": len(selected_features),
            "patient_count": int(patient_metrics["NumPatients"]),
            "metrics": metrics,
            "random_seed": 42,
        }
        with open(
            os.path.join(self.output_dir, "filter_recipe_manifest.json"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")

        self._plot_scores(selected_features, selected_scores, bounded_k)
        return self.output_dir

    def _plot_scores(self, features, scores, k):
        plt.figure(figsize=(15, 6))
        plt.bar(features, scores, color="#3498db")
        plt.title(f"特征 - ANOVA F-value Top {k} 特征得分", fontsize=16)
        plt.ylabel("得分值", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout(pad=3.0)
        plt.savefig(
            os.path.join(self.output_dir, "filter_scores.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
