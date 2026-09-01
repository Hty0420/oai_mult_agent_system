from auto_experiment_server import koa_lasso_classification
import os
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("comb_radiomics_features_lasso")
import warnings
warnings.filterwarnings("ignore")
from pydantic import Field
from typing import Annotated, List
import pandas as pd
import itertools

def _detect_split_from_filename(path: str) -> str:
    """Detect train/test from filename; return 'unknown' if not detected."""
    fname = os.path.basename(path).lower()
    if "train" in fname:
        return "train"
    if "test" in fname:
        return "test"
    return "unknown"


def _infer_feature_name(path: str) -> str:
    """Infer feature name from filename: train_shape_features.csv / test_shape_features.csv → shape"""
    fname = os.path.basename(path)
    fname = fname.replace("train_", "").replace("test_", "")
    fname = fname.replace("_features.csv", "")
    return fname


@mcp.tool()
def combine_radiomics_features_by_region(
    feature_files: Annotated[List[str],Field(description="需要组合的 radiomics 特征 CSV 文件路径列表。")],
    output_root: Annotated[str,Field(description="组合结果输出根目录；程序会在该目录下按 train/test 和 region 创建子目录。")]="comb_datasets",
    regions: Annotated[int,Field(description="需要处理的区域数量，表示处理 region 1 到 region N，默认 5。")] = 5,
    key_cols: Annotated[list,Field(description='合并特征文件时使用的非特征对齐列，默认 ["CaseID", "Region", "KLGrade"]。')] = ["CaseID", "Region", "KLGrade"]
):
    """
    按 Region 分区组合 radiomics 特征文件。

    对输入的多个 radiomics 特征 CSV 执行两两组合、三元组合以及全部组合，
    并把每个 Region 的组合结果保存到 output_root 下。
    输入文件名需要包含 train 或 test，且一次只能处理一个 split。

    Returns:
        dict: 包含 status、split、output_root、regions 等字段。
    """
    if not feature_files:
        raise ValueError("feature_files is empty: Please provide at least one CSV file path.")

        # ===== 1) Determine current split (train/test) =====
    splits = {_detect_split_from_filename(p) for p in feature_files}

    # Not allowed: all unknown
    if "unknown" in splits and len(splits) == 1:
        raise ValueError(
            "Cannot determine train/test from filenames: Please ensure filenames contain 'train' or 'test', e.g., train_shape_features.csv"
        )

    # Not allowed: mixed (e.g., both train and test)
    if len(splits - {"unknown"}) != 1:
        raise ValueError(f"Detected possibly mixed splits: {splits}. This function requires uploading only train or test at a time.")

    split = list(splits - {"unknown"})[0]

    # ===== 2) Final output root directory: output_root/<split>_by_region/ =====
    split_out_root = os.path.join(output_root,f"{split}_by_region")
    os.makedirs(split_out_root, exist_ok=True)

    # ===== 3) Read all feature files =====
    data = {}
    feature_names = []

    for path in feature_files:
        feat_name = _infer_feature_name(path)
        df = pd.read_csv(path)
        data[feat_name] = df
        feature_names.append(feat_name)

    # ===== 4) Combination list =====
    all_combinations = []
    all_combinations.extend(itertools.combinations(feature_names, 2))  # Pairwise
    all_combinations.extend(itertools.combinations(feature_names, 3))  # Triple
    all_combinations.append(tuple(feature_names))  # All (quadruple)

    results = {}

    # ===== 5) Process by region =====
    for region in range(1, regions + 1):
        region_key = f"region{region}"

        # Output directory for this region: output_root/<split>/regionX
        region_dir = os.path.join(split_out_root, region_key)
        os.makedirs(region_dir, exist_ok=True)

        # Extract data for this region
        region_data = {
            name: df[df["Region"] == region]
            for name, df in data.items()
        }

        results[region_key] = []

        for combo in all_combinations:
            merged = region_data[combo[0]]
            for name in combo[1:]:
                merged = pd.merge(merged, region_data[name], on=key_cols, how="inner")

            combo_name = "_".join(combo)
            output_path = os.path.join(region_dir, f"{combo_name}.csv")
            merged.to_csv(output_path, index=False)

            results[region_key].append({"features": combo, "path": output_path})
            print(f"[{split}][{region_key}] Saved: {output_path}")

    return {
        "status": "completed",
        "split": split,
        "output_root": split_out_root,
        "regions": regions,
        # "results": results
    }

@mcp.tool()
def combine_radiomics_features_all(
    feature_files: Annotated[List[str],Field(description='需要组合的 radiomics 特征 CSV 文件路径列表。')],
    output_root: Annotated[str, Field(description="组合结果输出根目录；程序会在该目录下按 train/test 创建子目录。")] = "comb_datasets",
    key_cols: Annotated[list, Field(description='合并特征文件时使用的非特征对齐列，默认 ["CaseID", "Region", "KLGrade"]。')] = ["CaseID", "Region", "KLGrade"]
):
    """
        不按 Region 分区，直接在整份数据上组合 radiomics 特征文件。

        对输入的多个 radiomics 特征 CSV 执行两两组合、三元组合以及全部组合，
        并把组合结果保存到 output_root 下。输入文件名需要包含 train 或 test，
        且一次只能处理一个 split。

        Returns:
            dict: 包含 status、split、output_root、num_inputs、num_outputs 等字段。
        """
    if not feature_files:
        raise ValueError("feature_files is empty: Please provide at least one CSV file path.")

        # ===== 1) Determine current split (train or test) =====
    splits = {_detect_split_from_filename(p) for p in feature_files}

    # Safety check: No mixing allowed, all unknown not allowed
    if "unknown" in splits and len(splits) == 1:
        raise ValueError(
            "Cannot determine train/test from filenames: Please ensure filenames contain 'train' or 'test', e.g., train_shape_features.csv"
        )
    if len(splits - {"unknown"}) != 1:
        # For example, both train and test, or mixed train/test/unknown
        raise ValueError(f"Detected possibly mixed splits: {splits}. This function requires uploading only train or test at a time.")

    # Extract the only valid split
    split = list(splits - {"unknown"})[0]

    # ===== 2) Generate final output directory output_root/<split>/ =====
    split_out_dir = os.path.join(output_root, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # ===== 3) Read all feature files =====
    data = {}  # {feature_name: df}
    feature_names = []  # ["shape","firstorder",...]

    for path in feature_files:
        feat_name = _infer_feature_name(path)
        df = pd.read_csv(path)
        data[feat_name] = df
        feature_names.append(feat_name)

    # Safety check: At least 2 feature files for pairwise combination
    if len(feature_names) < 2:
        return {
            "status": "completed",
            "split": split,
            "output_root": split_out_dir,
            "warning": f"Only {len(feature_names)} feature files detected, cannot perform pairwise/triple combinations (at least 2 required).",
            "outputs": []
        }

    # ===== 4) Generate combination list: 2-combinations, 3-combinations, full combination =====
    all_combinations = []
    all_combinations.extend(itertools.combinations(feature_names, 2))
    all_combinations.extend(itertools.combinations(feature_names, 3))
    all_combinations.append(tuple(feature_names))

    # ===== 5) Merge each combination and save =====
    outputs = []

    for combo in all_combinations:
        merged = data[combo[0]]
        for name in combo[1:]:
            merged = pd.merge(merged, data[name], on=key_cols, how="inner")

        combo_name = "_".join(combo)
        output_path = os.path.join(split_out_dir, f"{combo_name}.csv")
        merged.to_csv(output_path, index=False)

        outputs.append({"features": combo, "path": output_path})
        print(f"[{split}] Saved: {output_path}")

    return {
        "status": "completed",
        "split": split,
        "output_root": split_out_dir,
        "num_inputs": len(feature_files),
        "num_outputs": len(outputs),
    }


@mcp.tool()
def lasso_comb_radiomics_features_by_region(train_root:Annotated[str,Field(description='训练集组合 radiomics 特征根目录，目录下应包含各 Region 子目录，每个子目录内是组合后的特征 CSV。')]=r"../comb_datasets/train_by_region",
                                            test_root:Annotated[str,Field(description='测试集组合 radiomics 特征根目录，目录下应包含各 Region 子目录，每个子目录内是组合后的特征 CSV。')]=r"../comb_datasets/test_by_region",
                                            output_root:Annotated[str,Field(description='组合 radiomics LASSO 分类结果输出根目录。')]=r"../comb_lasso_radiomics_results"):
    """
    对按 Region 组合后的 radiomics 特征批量执行 LASSO 二分类。

    适用于比较不同 Region 的组合特征分类效果。函数会遍历 train_root 下的
    Region 子目录及其中的 .csv 特征文件，寻找 test_root 中对应测试集文件，
    为每个组合特征创建输出目录，并调用 koa_lasso_classification 执行分类。

    Returns:
        dict: 包含 status 和 output_root，output_root 为结果保存根目录。
    """
    for region_name in os.listdir(train_root):
        train_region_dir = os.path.join(train_root, region_name)
        test_region_dir = os.path.join(test_root, region_name)

        if not os.path.isdir(train_region_dir):
            continue

        region_output_dir = os.path.join(output_root, region_name)
        os.makedirs(region_output_dir, exist_ok=True)

        for filename in os.listdir(train_region_dir):
            if not filename.endswith(".csv"):
                continue

            train_path = os.path.join(train_region_dir, filename)
            test_path = os.path.join(test_region_dir, filename.replace("_train", "_test"))

            if not os.path.exists(test_path):
                continue

            feature_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(region_output_dir, feature_name)
            os.makedirs(output_dir, exist_ok=True)

            koa_lasso_classification(train_path, test_path, output_dir)

    return {"status": "completed", "output_root": output_root}

@mcp.tool()
def lasso_comb_radiomics_features_all(train_root:Annotated[str,Field(description='训练集组合 radiomics 特征目录，目录内应包含组合后的特征 CSV。')]=r"../comb_datasets/train",
                            test_root:Annotated[str,Field(description='测试集组合 radiomics 特征目录，目录内应包含组合后的特征 CSV。')]=r"../comb_datasets/test",
                            output_root:Annotated[str,Field(description='不按 Region 分区的组合 radiomics LASSO 分类结果输出目录。')]=r"../comb_lasso_radiomics_results/all"):
    """
    对不按 Region 分区的组合 radiomics 特征批量执行 LASSO 二分类。

    适用于全局合并分析场景。函数会遍历 train_root 下所有 .csv 特征文件，
    在 test_root 中寻找对应测试集文件，为每个组合特征创建输出目录，
    并调用 koa_lasso_classification 执行分类。

    Returns:
        dict: 包含 status 和 output_root，output_root 为结果保存目录。
    """
    os.makedirs(output_root, exist_ok=True)

    # Traverse all csv files under train_root
    for filename in os.listdir(train_root):
        if not filename.endswith(".csv"):
            continue

        train_path = os.path.join(train_root, filename)
        # Corresponding test file path (naming rule: replace train with test)
        test_path = os.path.join(test_root, filename.replace("train", "test"))

        # Skip if corresponding test file doesn't exist
        if not os.path.exists(test_path):
            print(f"Skipped: No corresponding test file found {test_path}")
            continue

        # Output directory naming: remove .csv suffix
        feature_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_root, feature_name)
        os.makedirs(output_dir, exist_ok=True)

        # Call classification
        koa_lasso_classification(train_path, test_path, output_dir)
        print(f"Completed: {feature_name}")

    return {"status": "completed", "output_root": output_root}

if __name__ == "__main__":
    mcp.run(transport='stdio')
    # mcp.run(transport="sse")
    # feature_files = [
    #     "../datasets/radiomics_features_all_summary/split_train_radiomics_features/train_shape_features.csv",
    #     "../datasets/radiomics_features_all_summary/split_train_radiomics_features/train_firstorder_features.csv",
    #     "../datasets/radiomics_features_all_summary/split_train_radiomics_features/train_texture_features.csv",
    #     "../datasets/radiomics_features_all_summary/split_train_radiomics_features/train_filtered_features.csv",
    # ]

    # res = combine_radiomics_features_all(
    #     feature_files=feature_files,
    # )

    # res = combine_radiomics_features_by_region(
    #     feature_files=feature_files,
    # )
    #
    # print(res)
    # lasso_comb_radiomics_features_by_region()
    # lasso_comb_radiomics_features_all()
