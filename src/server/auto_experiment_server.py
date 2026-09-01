# Radiomics feature extraction packages
import os
import asyncio
import numpy as np
import pandas as pd
from radiomics import featureextractor
from tqdm import tqdm

# Import MCP
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from typing import Annotated

# Image segmentation packages (network requests)
import requests

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Lasso classification packages
from algorithm.feature_class_algorithm.koa_lasso_classifier import KOALassoClassifier
from algorithm.feature_class_algorithm.classification_output_contract import validate_and_collect_classification_outputs
from algorithm.feature_class_algorithm.training_only_validation import validate_training_candidate
# Filter selection packages
from algorithm.feature_class_algorithm.universal_filter_classifier import UniversalFilterSelector
# MEAS feature extraction
from algorithm.KOA_feature_extraction.meas_feature_extraction import OAIZIB_MEAS_FeatureExtractor
# LDSE feature extraction
from server.ldse_tool_handlers import (
    execute_ldse_single_extraction,
    execute_ldse_train_test_extraction,
    prepare_ldse_online_augmentation_extractor,
    stable_mask_array_sample_key,
)
# Data augmentation
from algorithm.data_augmentation import OfflineAugmentor, OnlineAugmentor

# Feature quality assessment
from algorithm.feature_quality import FeatureQualityAssessor, validate_feature_dataframe

# UTF-8 encoding format to support emoji
import sys
import io

# Plotting charts
import matplotlib

# Set font (for Windows)
matplotlib.rcParams['font.family'] = 'Arial'
# Avoid negative sign '-' showing as square
matplotlib.rcParams['axes.unicode_minus'] = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
mcp = FastMCP("auto_experiment")


def _consume_progress_future(completed):
    try:
        completed.result()
    except Exception:
        pass


def _make_thread_phase_progress(ctx, loop, phase_ranges, phase_labels):
    """Bridge synchronous per-case callbacks into MCP progress notifications."""
    def forward(event):
        phase = str(event.get("phase") or "process")
        processed = int(event.get("processed_cases", 0))
        total = int(event.get("total_cases", 0))
        successful = int(event.get("successful_cases", 0))
        failed = int(event.get("failed_cases", 0))
        start, span = phase_ranges.get(phase, (0.0, 100.0))
        fraction = (processed / total) if total else 0.0
        overall = max(0.0, min(100.0, start + span * fraction))
        label = phase_labels.get(phase, phase)
        detail = (
            f"{label}：已处理 {processed}/{total}；"
            f"成功 {successful}；跳过或失败 {failed}"
        )
        case_id = event.get("case_id")
        if case_id:
            detail += f"；当前 CaseID={case_id}"
        future = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(overall, 100.0, detail),
            loop,
        )
        future.add_done_callback(_consume_progress_future)

    return forward
# Disable all non-critical error log output in terminal
import contextlib
import logging
# Silent output context manager
@contextlib.contextmanager
def suppress_output():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)

@mcp.tool()
def umamba_segment_mri(file_paths:Annotated[list,Field(description="需要上传到 U-Mamba 分割服务的本地 MRI 文件路径列表。")],
                       save_dir:Annotated[str,Field(description="保存 U-Mamba 分割结果的本地目录。")]=r"umamba_seg_results",):
    """
    使用 U-Mamba 模型对 MRI 图像进行分割，并把结果保存到 save_dir。

    支持单个或多个 .nii.gz MRI 文件。若服务端返回多个分割文件，会保存为 zip 包；
    若返回单个文件，会按服务端文件名保存。

    Returns:
        dict: 包含 status、files、message 等字段。
    """
    server_url="http://localhost:8000/predict/"
    try:
        os.makedirs(save_dir, exist_ok=True)
        # Construct upload file list
        files = [("files", (os.path.basename(f), open(f, "rb"), "application/gzip")) for f in file_paths]

        # Send POST request
        response = requests.post(server_url, files=files)

        if response.status_code != 200:
            return {"status": "failed", "files": [], "message": f"Request failed, status code: {response.status_code}"}

        content_type = response.headers.get("content-type")
        saved_files = []

        if content_type == "application/zip":
            # Multiple files case, save as zip package
            zip_path = os.path.join(save_dir, "umamba_seg_results.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            saved_files.append(zip_path)
        else:
            # Single file case, use server returned filename
            seg_name = response.headers.get("content-disposition").split("filename=")[-1].strip('"')
            out_path = os.path.join(save_dir, seg_name)
            with open(out_path, "wb") as f:
                f.write(response.content)
            saved_files.append(out_path)

        return {"status": "success", "files": saved_files, "message": "Segmentation files saved successfully. You can choose whether to download the segmented MRI images."}

    except Exception as e:
        return {"status": "failed", "files": [], "message": str(e)}

@mcp.tool()
def lkm_unet_segment_mri(file_paths:Annotated[list,Field(description="需要上传到 LKM-UNet 分割服务的本地 MRI 文件路径列表。")],
                       save_dir:Annotated[str,Field(description="保存 LKM-UNet 分割结果的本地目录。")]=r"lkm_unet_seg_results",):
    """
    使用 LKM-UNet 模型对 MRI 图像进行分割，并把结果保存到 save_dir。

    支持单个或多个 .nii.gz MRI 文件。若服务端返回多个分割文件，会保存为 zip 包；
    若返回单个文件，会按服务端文件名保存。

    Returns:
        dict: 包含 status、files、message 等字段。
    """
    server_url="http://localhost:8000/predict/"
    try:
        os.makedirs(save_dir, exist_ok=True)
        # Construct upload file list
        files = [("files", (os.path.basename(f), open(f, "rb"), "application/gzip")) for f in file_paths]

        # Send POST request
        response = requests.post(server_url, files=files)

        if response.status_code != 200:
            return {"status": "failed", "files": [], "message": f"Request failed, status code: {response.status_code}"}

        content_type = response.headers.get("content-type")
        saved_files = []

        if content_type == "application/zip":
            # Multiple files case, save as zip package
            zip_path = os.path.join(save_dir, "lkm_unet_seg_results.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            saved_files.append(zip_path)
        else:
            # Single file case, use server returned filename
            seg_name = response.headers.get("content-disposition").split("filename=")[-1].strip('"')
            out_path = os.path.join(save_dir, seg_name)
            with open(out_path, "wb") as f:
                f.write(response.content)
            saved_files.append(out_path)

        return {"status": "success", "files": saved_files, "message": "Segmentation files saved successfully. You can choose whether to download the segmented MRI images."}

    except Exception as e:
        return {"status": "failed", "files": [], "message": str(e)}

@mcp.tool()
def lightm_unet_segment_mri(file_paths:Annotated[list,Field(description="需要上传到 LightM-UNet 分割服务的本地 MRI 文件路径列表。")],
                       save_dir:Annotated[str,Field(description="保存 LightM-UNet 分割结果的本地目录。")]=r"lightm_unet_seg_results",):
    """
    使用 LightM-UNet 模型对 MRI 图像进行分割，并把结果保存到 save_dir。

    支持单个或多个 .nii.gz MRI 文件。若服务端返回多个分割文件，会保存为 zip 包；
    若返回单个文件，会按服务端文件名保存。

    Returns:
        dict: 包含 status、files、message 等字段。
    """
    server_url="http://localhost:8000/predict/"
    try:
        os.makedirs(save_dir, exist_ok=True)
        # Construct upload file list
        files = [("files", (os.path.basename(f), open(f, "rb"), "application/gzip")) for f in file_paths]

        # Send POST request
        response = requests.post(server_url, files=files)

        if response.status_code != 200:
            return {"status": "failed", "files": [], "message": f"Request failed, status code: {response.status_code}"}

        content_type = response.headers.get("content-type")
        saved_files = []

        if content_type == "application/zip":
            # Multiple files case, save as zip package
            zip_path = os.path.join(save_dir, "lightm_unet_seg_results.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            saved_files.append(zip_path)
        else:
            # Single file case, use server returned filename
            seg_name = response.headers.get("content-disposition").split("filename=")[-1].strip('"')
            out_path = os.path.join(save_dir, seg_name)
            with open(out_path, "wb") as f:
                f.write(response.content)
            saved_files.append(out_path)

        return {"status": "success", "files": saved_files, "message": "Segmentation files saved successfully. You can choose whether to download the segmented MRI images."}

    except Exception as e:
        return {"status": "failed", "files": [], "message": str(e)}


logging.getLogger('radiomics').setLevel(logging.CRITICAL)
@mcp.tool()
async def extract_radiomics_features(
    ctx: Context,
    csv_path:Annotated[str,Field(description="输入 CSV 文件路径，需包含 ImagePath、MaskPath、CaseID、KLGrade 等列。")],
    output_dir: Annotated[str,Field(description="radiomics 特征输出目录。")] = r"radiomics_results/test",
    num_regions: Annotated[int,Field(description="需要提取特征的分割区域数量，表示 region 1 到 region N，默认 5。")] = 5,
    num_cases: Annotated[int,Field(description="最多处理的病例数量；extract_all_cases=False 时生效。")] = 2,
    extract_all_cases: Annotated[bool,Field(description="是否处理 CSV 中全部病例，默认 False。")] = False,
    extract_all_features: Annotated[bool,Field(description="是否提取完整 radiomics 特征；True 时启用所有 image types 和 feature classes，默认 False。")] = False,
):
    """
    从 MRI 图像和 mask 中提取 radiomics 特征。

    该工具读取 csv_path 中的 ImagePath、MaskPath、CaseID、KLGrade，
    按 Region 提取影像组学特征并保留 KLGrade 标签。
    extract_all_features=False 时主要提取原始图像上的基础 radiomics 特征；
    extract_all_features=True 时会启用更多滤波/变换图像和完整特征类。

    Returns:
        str: 汇总后的 radiomics_features_summary.csv 路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    if extract_all_features:
        # Enable all feature types
        extractor.enableAllImageTypes()  # All image filters (LoG, Wavelet, etc)
        extractor.enableAllFeatures()  # All feature types (GLCM, GLRLM, etc)
    df = pd.read_csv(csv_path,dtype={"CaseID": str})
    if not extract_all_cases:
        df = df.head(num_cases)  # ✅ Limit number of cases
    all_features_list = []
    total_cases = int(len(df))
    await ctx.report_progress(0.0, float(total_cases) if total_cases else None, "正在读取 Radiomics 病例 CSV")
    for processed_cases, (_, row) in enumerate(df.iterrows(), start=1):
        CaseID = row.get("CaseID")
        ImagePath = row["ImagePath"]
        MaskPath = row["MaskPath"]
        KLGrade = row["KLGrade"]

        if not os.path.exists(ImagePath):
            print(f"❌ Image file does not exist: {ImagePath}")
            await ctx.report_progress(
                float(processed_cases), float(total_cases) if total_cases else None,
                f"已处理 {processed_cases}/{total_cases}；当前 CaseID={CaseID}；影像缺失",
            )
            continue
        if not os.path.exists(MaskPath):
            print(f"❌ Mask file does not exist: {MaskPath}")
            await ctx.report_progress(
                float(processed_cases), float(total_cases) if total_cases else None,
                f"已处理 {processed_cases}/{total_cases}；当前 CaseID={CaseID}；掩膜缺失",
            )
            continue

        print(f"\n📂 Processing case: {CaseID}")

        case_features = []

        for region in range(1, num_regions + 1):
            try:
                features = extractor.execute(ImagePath, MaskPath, label=region)
                feature_dict = {
                    'CaseID': CaseID,
                    'Region': region,
                    'KLGrade': KLGrade,  # ✅ Preserve KOA grade
                    **{k: v for k, v in features.items() if not k.startswith('diagnostics')}
                }
                case_features.append(feature_dict)
                print(f"  ✅ Region {region} extraction successful")
            except Exception as e:
                print(f"  ⚠️ Region {region} extraction failed: {e}")

        if case_features:
            case_df = pd.DataFrame(case_features)
            case_df.to_csv(os.path.join(output_dir, f'features_{CaseID}.csv'), index=False)
            all_features_list.extend(case_features)
            print(f"  💾 Features saved: features_{CaseID}.csv")
        await ctx.report_progress(
            float(processed_cases), float(total_cases) if total_cases else None,
            f"已处理 {processed_cases}/{total_cases}；当前 CaseID={CaseID}；提取区域 {len(case_features)}/{num_regions}",
        )

    if all_features_list:
        all_df = pd.DataFrame(all_features_list)
        summary_path = os.path.join(output_dir, "radiomics_features_summary.csv")
        all_df.to_csv(summary_path, index=False)
        print(f"\n✅ All features extracted successfully, total {len(all_df)} records")
        print(f"📄 Summary feature file path: {summary_path}")
        return summary_path
    else:
        print("❌ No features extracted successfully")
        return None

# 设置日志级别为最高，防止 logging 输出
logging.getLogger().setLevel(logging.CRITICAL)
@mcp.tool()
async def meas_features_extraction(
    ctx: Context,
    output_dir: Annotated[str,Field(description="MEAS 特征输出目录。")],
    csv_path: Annotated[str,Field(description="输入 CSV 文件路径，需包含 ImagePath/MaskPath/CaseID/KLGrade 或兼容列。")],
):
    """
    提取 MEAS 形态学和几何测量特征。

    该工具面向 OAI-ZIB 膝关节数据，从不同膝关节结构中提取体积、
    表面积、厚度等形态学/几何特征，并保存为 meas_features_summary.csv。

    Returns:
        str: meas_features_summary.csv 路径。
    """
    try:
        loop = asyncio.get_running_loop()

        def forward_progress(event):
            processed = int(event.get("processed_cases", 0))
            total = int(event.get("total_cases", 0))
            successful = int(event.get("successful_cases", 0))
            failed = int(event.get("failed_cases", 0))
            case_id = event.get("case_id")
            detail = f"已处理 {processed}/{total}；成功 {successful}；跳过或失败 {failed}"
            if case_id:
                detail += f"；当前 CaseID={case_id}"
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(
                    float(processed),
                    float(total) if total else None,
                    detail,
                ),
                loop,
            )
            future.add_done_callback(_consume_progress_future)

        await ctx.report_progress(0.0, None, "正在读取 MEAS 病例 CSV")
        extractor = OAIZIB_MEAS_FeatureExtractor(output_dir)
        print("=== Starting MEAS feature extraction ===")
        await asyncio.to_thread(
            extractor.process_from_csv,
            csv_path,
            True,
            forward_progress,
        )
        counts = extractor.last_run_counts
        summary_path = os.path.join(output_dir, "meas_features_summary.csv")
        if counts.get("output_rows", 0) <= 0 or not os.path.isfile(summary_path):
            return {
                "status": "failed",
                "message": "MEAS extraction produced no valid feature rows.",
                "feature_type": "MEAS",
                "input_csv": csv_path,
                "output_csv": None,
                "counts": counts,
            }
        print(
            "MEAS rows: "
            f"input={counts['input_rows']}, labeled={counts['labeled_rows']}, "
            f"excluded_unlabeled={counts['excluded_unlabeled_rows']}, "
            f"output={counts['output_rows']}, other_skipped={counts['other_skipped_rows']}"
        )
        return {
            "status": "success",
            "message": "MEAS feature extraction completed.",
            "feature_type": "MEAS",
            "input_csv": csv_path,
            "output_csv": summary_path,
            "counts": counts,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": str(exc),
            "feature_type": "MEAS",
            "input_csv": csv_path,
            "output_csv": None,
            "counts": {},
        }

@mcp.tool()
async def ldse_train_test_features_extraction(
    ctx: Context,
    output_dir: Annotated[str, Field(description="Output directory. PCA/scaler fitting occurs only on the training CSV; test data uses the frozen training artifact.")],
    train_csv_path: Annotated[str, Field(description="Training CSV used exclusively to fit PCA/scalers before feature extraction.")],
    test_csv_path: Annotated[str, Field(description="Held-out test CSV transformed only with the frozen artifact fitted from training data.")],
    n_components: Annotated[int, Field(description="Requested LDSE PCA components per structure; PCA/scalers are fit only on training data.")] = 50,
    random_seed: Annotated[int, Field(description="Random seed for the training-fitted LDSE boundary sampling and PCA workflow.")] = 42,
) -> dict[str, str]:
    """Fit PCA/scalers only on train data, then transform test data with the frozen training artifact.

    Returns structured paths for both feature CSVs, the verified training model, and
    the manifest that records train-only provenance.
    """
    loop = asyncio.get_running_loop()
    forward_progress = _make_thread_phase_progress(
        ctx,
        loop,
        {
            "fit": (0.0, 40.0),
            "train_transform": (40.0, 30.0),
            "test_transform": (70.0, 30.0),
        },
        {
            "fit": "训练集形状模型拟合",
            "train_transform": "训练集特征变换",
            "test_transform": "测试集特征变换",
        },
    )
    await ctx.report_progress(0.0, 100.0, "正在读取 LDSE 训练/测试 CSV")
    result = await asyncio.to_thread(
        execute_ldse_train_test_extraction,
        output_dir,
        train_csv_path,
        test_csv_path,
        n_components,
        random_seed,
        forward_progress,
    )
    await ctx.report_progress(100.0, 100.0, "LDSE 训练/测试特征提取完成")
    return result


@mcp.tool()
async def ldse_features_extraction(
    ctx: Context,
    output_dir: Annotated[str, Field(description="Output directory for an explicit LDSE run. PCA/scalers may be fit only for dataset_role='train'; test transformations use a frozen training artifact.")],
    csv_path: Annotated[str, Field(description="Input CSV. Training input may fit PCA/scalers; test input must use the verified frozen training artifact.")],
    mode: Annotated[str, Field(description="Required safe mode: 'fit_transform' only for training, or 'transform' only with an explicit frozen training model artifact.")],
    dataset_role: Annotated[str | None, Field(description="Dataset role. 'train' is required to fit PCA/scalers; held-out test data is transformed with the frozen training artifact.")] = None,
    model_path: Annotated[str | None, Field(description="Verified training model artifact required for mode='transform'; it freezes the train-fitted PCA/scalers for test data.")] = None,
    n_components: Annotated[int, Field(description="Requested PCA components when fitting training data; ignored by a loaded frozen artifact.")] = 50,
    random_seed: Annotated[int, Field(description="Random seed used only when fitting the training data.")] = 42,
) -> str:
    """Run an explicit safe LDSE lifecycle without implicit fitting or model lookup.

    PCA/scalers are fit only when mode='fit_transform' and dataset_role='train'.
    Test data must use mode='transform' with a verified frozen training artifact.
    """
    loop = asyncio.get_running_loop()
    if mode == "fit_transform":
        phase_ranges = {"fit": (0.0, 50.0), "train_transform": (50.0, 50.0)}
    else:
        phase_ranges = {"test_transform": (0.0, 100.0), "transform": (0.0, 100.0)}
    forward_progress = _make_thread_phase_progress(
        ctx,
        loop,
        phase_ranges,
        {
            "fit": "训练集形状模型拟合",
            "train_transform": "训练集特征变换",
            "test_transform": "测试集特征变换",
            "transform": "特征变换",
        },
    )
    await ctx.report_progress(0.0, 100.0, "正在读取 LDSE 病例 CSV")
    result = await asyncio.to_thread(
        execute_ldse_single_extraction,
        output_dir,
        csv_path,
        mode,
        dataset_role,
        model_path,
        n_components,
        random_seed,
        forward_progress,
    )
    await ctx.report_progress(100.0, 100.0, "LDSE 特征提取完成")
    return result

from contextlib import contextmanager
@contextmanager
def suppress_output():
    """屏蔽第三方库的打印输出"""
    import sys
    import io
    sys_stdout = sys.stdout
    sys.stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys.stderr

@mcp.tool()
def koa_lasso_classification(train_path:Annotated[str,Field(description="训练集特征 CSV 文件路径，需包含 KLGrade 和数值型特征列，可包含 image_id、CaseID、Region 等非特征列。")],
                             test_path:Annotated[str,Field(description="测试集特征 CSV 文件路径，需包含 KLGrade 和数值型特征列，可包含 image_id、CaseID、Region 等非特征列。")],
                             output_dir:Annotated[str,Field(description="LASSO 分类图表、指标和报告的输出目录。")]=r"KOA_LASSO_Results",
                             by_region:Annotated[bool,Field(description="是否按 Region 分区分别做 LASSO 特征选择和二分类，默认 False。")]=False):
    """
    使用患者/来源患者分组的 L1 Logistic 执行 KOA 二分类。

    KLGrade < 2 为 0，KLGrade >= 2 为 1。所有缺失值填补、标准化和
    超参数选择均在训练数据的嵌套分组交叉验证内完成；测试数据仅用于
    来源患者级概率聚合与最终评估。输出包括可复现的 JSON/CSV 工件、
    患者级 metrics、ROC/PR 曲线、可选 SHAP 图以及被选中特征文件。

    Returns:
        dict: 包含 status、message、output_dir、charts、txt_files、diagnostics 等字段。
    """
    import pandas as pd

    # --- Input validation ---
    diagnostics = []

    # Check files exist
    for label, path in [("train_path", train_path), ("test_path", test_path)]:
        if not os.path.exists(path):
            return {
                "status": "failed",
                "message": f"{label} does not exist: {path}",
                "output_dir": output_dir,
                "charts": [],
                "txt_files": [],
                "diagnostics": diagnostics,
            }

    # Check that CSVs have feature columns (not just metadata)
    for label, path in [("train", train_path), ("test", test_path)]:
        try:
            df_check = pd.read_csv(path)
            is_valid, validation_message, feature_cols = validate_feature_dataframe(df_check)
            if not is_valid:
                return {
                    "status": "failed",
                    "message": f"{label} CSV is not a valid numeric feature file.",
                    "output_dir": output_dir,
                    "charts": [],
                    "txt_files": [],
                    "diagnostics": diagnostics,
                }
            missing_count = int(df_check[feature_cols].isna().sum().sum())
            if missing_count:
                diagnostics.append(
                    f"{label}: {missing_count} missing feature values; "
                    "imputation will be fit inside training folds"
                )
            diagnostics.append(
                f"{label}: {len(df_check)} rows, {len(feature_cols)} numeric feature columns"
            )
        except Exception:
            return {
                "status": "failed",
                "message": f"Failed to read {label} feature CSV.",
                "output_dir": output_dir,
                "charts": [],
                "txt_files": [],
                "diagnostics": diagnostics,
            }

    # Initialize
    clf = KOALassoClassifier(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        by_region=by_region  # Set to False to not split by region
    )
    try:
        clf.run()
    except ValueError as exc:
        diagnostics.extend(clf.diagnostics)
        return {
            "status": "failed",
            "message": str(exc),
            "output_dir": output_dir,
            "charts": [],
            "txt_files": [],
            "diagnostics": diagnostics,
        }
    diagnostics.extend(clf.diagnostics)

    try:
        image_files, txt_or_csv_files = validate_and_collect_classification_outputs(
            output_dir, clf.required_relative_outputs
        )
    except ValueError as exc:
        return {"status": "failed", "message": str(exc), "output_dir": output_dir,
                "charts": [], "txt_files": [], "diagnostics": diagnostics}
    status = "success"
    message = f"LASSO classification completed, generated {len(image_files)} charts and {len(txt_or_csv_files)} data files"

    return {
        "status": status,
        "message": message,
        "output_dir": output_dir,
        "charts": [{"file_path": path} for path in image_files],
        "txt_files": [{"file_path": path} for path in txt_or_csv_files],
        "diagnostics": diagnostics,
    }

@mcp.tool()
def validate_kl_pipeline_candidate(
    train_path: Annotated[
        str,
        Field(
            description=(
                "训练特征 CSV。该工具只接受训练数据并执行患者级嵌套交叉验证，"
                "不读取测试集。"
            )
        ),
    ],
    classification_method: Annotated[
        str, Field(description="候选分类方法，只允许 LASSO 或 Filter。")
    ],
    model_recipe: Annotated[
        str | None,
        Field(
            description=(
                "原子模型配方。LASSO 对应 L1_LOGISTIC，Filter 对应 "
                "ANOVA_TOPK_L2；省略时根据 classification_method 推断。"
            )
        ),
    ]=None,
    outer_splits: Annotated[int, Field(description="外层患者级交叉验证折数。")]=3,
    inner_splits: Annotated[int, Field(description="内层调参交叉验证折数。")]=3,
    bootstrap_samples: Annotated[
        int, Field(description="患者级分层 bootstrap 次数，至少 100。")
    ]=500,
) -> dict:
    """Evaluate one candidate on training data only for pipeline selection."""
    try:
        return validate_training_candidate(
            train_path,
            classification_method,
            model_recipe=model_recipe,
            requested_outer_splits=outer_splits,
            requested_inner_splits=inner_splits,
            bootstrap_samples=bootstrap_samples,
        )
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "status": "failed",
            "validation_scope": "training_only_nested_grouped_cv",
            "test_data_used": False,
            "classification_method": classification_method,
            "model_recipe": model_recipe,
            "error": f"{type(exc).__name__}: {exc}",
        }


@mcp.tool()
def koa_filter_classification(
        train_path:Annotated[str,Field(description="训练集特征 CSV 文件路径，需包含 label_col 和数值型特征列。")],
        test_path:Annotated[str,Field(description="测试集特征 CSV 文件路径，需包含 label_col 和数值型特征列。")],
        output_dir:Annotated[str,Field(description="Filter 分类图表、特征评分和指标报告的输出目录。")]=r"KOA_Filter_Results",
        feature_type:Annotated[str,Field(description="特征类型名称，用于报告和图表标识，例如 MEAS、LDSE、radiomics。")]="MEAS",
        id_col:Annotated[str,Field(description="样本 ID 列名，用于区分病例，默认 CaseID。")]="CaseID",
        label_col:Annotated[str,Field(description="标签列名，默认 KLGrade。")]="KLGrade",
        model_recipe:Annotated[
            str, Field(description="固定原子配方，只允许 ANOVA_TOPK_L2。")
        ]="ANOVA_TOPK_L2",
        k:Annotated[int | str,Field(description="ANOVA 选择的特征数，正整数或 all。")] = 10,
        model_c:Annotated[float,Field(description="L2 LogisticRegression 的 C 参数。")] = 1.0,
):
    """
    使用固定的 ANOVA_TOPK_L2 配方做特征选择并执行 KOA 二分类。

    该工具会将 label_col（默认 KLGrade）转换为二分类标签：
    KLGrade < 2 为 0，KLGrade >= 2 为 1。缺失值填补、ANOVA Top-K、
    标准化和 L2 LogisticRegression 均只在训练数据上拟合；测试集仅作一次
    最终评估。k 与 model_c 应来自训练集内部候选验证的推荐参数。

    Returns:
        dict: 包含 status、message、output_dir、charts 等字段。
        输出目录中通常包含 SelectedFeatures_Top{k}.txt、metrics_summary.txt、
        filter_scores.png。
    """
    if str(model_recipe).strip().upper() != "ANOVA_TOPK_L2":
        return {
            "status": "failed",
            "message": "koa_filter_classification only supports ANOVA_TOPK_L2",
            "output_dir": output_dir,
            "charts": [],
            "txt_files": [],
        }
    universalFilterSelector = UniversalFilterSelector(
        train_feat_path=f"{train_path}",
        test_feat_path=f"{test_path}",
        output_dir=output_dir,
        feature_type=feature_type,
        id_col=id_col,
        label_col=label_col
    )
    try:
        output_sub_dir = universalFilterSelector.run_selection_pipeline(
            k=k, model_c=model_c
        )
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "output_dir": output_dir,
            "charts": [],
            "txt_files": [],
        }
    # Collect generated image files
    image_files = []
    txt_or_csv_files = []
    if os.path.exists(output_sub_dir):
        for file in os.listdir(output_sub_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(output_sub_dir, file))
            if file.lower().endswith(('.txt', '.csv', '.json')):
                txt_or_csv_files.append(os.path.join(output_sub_dir, file))

    return {
            "status":"success",
            "message": (
                "ANOVA_TOPK_L2 classification completed, generated "
                f"{len(image_files)} charts"
            ),
            "output_dir": output_sub_dir,
            "charts": [{"file_path": path} for path in image_files],
            "txt_files": [{"file_path": path} for path in txt_or_csv_files]
    }

@mcp.tool()
def read_classification_results(classification_result_dir: Annotated[str,Field(description="分类结果目录路径，目录中应包含 LASSO 或 Filter 生成的 .txt 报告文件。")]):
    """
    读取分类结果目录中的 .txt 报告文件。

    可用于读取 LASSO 分类或 Filter 分类生成的 metrics、特征选择结果等文本报告。

    Returns:
        str: 拼接后的分类报告文本。
    """
    if not os.path.exists(classification_result_dir):
        return f"Error: Directory does not exist - {classification_result_dir}"

    result = []
    # Iterate through all txt files in the directory
    for filename in os.listdir(classification_result_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(classification_result_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    result.append(f"=== File: {filename} ===\n{content}\n")
            except Exception as e:
                result.append(f"Error reading file {filename}: {str(e)}\n")

    if not result:
        return f"No txt files found in directory {classification_result_dir}"

    return  '\n'.join(result)

@mcp.tool()
async def augment_dataset_offline(
    ctx: Context,
    csv_path: Annotated[str, Field(description="输入标签 CSV 路径，需包含 ImagePath、MaskPath、KLGrade 等列。")],
    output_dir: Annotated[str, Field(description="离线增强后图像、mask 和标签 CSV 的输出目录。")],
    multiplier: Annotated[int, Field(description="每个原始样本生成的增强副本数量。")] = 5,
    seed: Annotated[int, Field(description="随机种子，用于保证增强结果可复现。")] = 42,
):
    """
    离线数据增强：把增强后的膝关节 MRI 图像和 mask 保存到磁盘。

    该工具读取 csv_path 中的图像和 mask 路径，对每个 image-mask pair 应用
    MONAI 几何增强和强度增强，保存增强后的 .nii.gz 文件，并生成保留 KLGrade
    的增强标签 CSV。

    增强包括：
    - 几何增强：随机翻转、旋转、缩放、平移、弹性形变
    - 强度增强：高斯噪声、MRI bias field、对比度调整、高斯平滑

    Returns:
        dict: 包含 status、output_csv、total_samples、message 等字段。
    """
    try:
        loop = asyncio.get_running_loop()
        forward_progress = _make_thread_phase_progress(
            ctx,
            loop,
            {"augment": (0.0, 100.0)},
            {"augment": "病例增强"},
        )
        await ctx.report_progress(0.0, 100.0, "正在读取待增强病例 CSV")
        augmentor = OfflineAugmentor(multiplier=multiplier, seed=seed)
        out_csv = await asyncio.to_thread(
            augmentor.augment_dataset,
            csv_path,
            output_dir,
            "imagesTr_aug",
            "labelsTr_aug",
            forward_progress,
        )
        await ctx.report_progress(100.0, 100.0, "离线数据增强完成")
        import pandas as pd
        df = pd.read_csv(out_csv)
        return {
            "status": "success",
            "output_csv": out_csv,
            "total_samples": len(df),
            "message": f"Augmentation complete: {len(df)} total samples saved to {output_dir}",
        }
    except Exception as e:
        return {"status": "failed", "message": str(e)}


@mcp.tool()
async def augment_online_extract_features(
    ctx: Context,
    csv_path: Annotated[str, Field(description="输入 CSV 路径，需包含 ImagePath、MaskPath、CaseID、KLGrade 等列。")],
    output_dir: Annotated[str, Field(description="在线增强后聚合特征 CSV 的输出目录。")],
    n_augmentations: Annotated[int, Field(description="每个样本生成的增强视图数量；总视图数为 1 个原始视图 + N 个增强视图。")] = 3,
    feature_type: Annotated[str, Field(description="要提取的特征类型，可选 'MEAS'、'LDSE' 或 'radiomics'。")] = "MEAS",
    aggregator: Annotated[str, Field(description="多个增强视图特征的聚合方式，可选 'mean' 或 'median'。")] = "mean",
    seed: Annotated[int, Field(description="随机种子。")] = 42,
    dataset_role: Annotated[str | None, Field(description="LDSE 在线增强的数据集角色，必须显式为 'train' 或 'test'。")] = None,
    ldse_model_path: Annotated[str | None, Field(description="测试集 LDSE 在线增强所需的已验证训练模型路径。")] = None,
):
    """
    在线数据增强并提取特征。

    该工具不会把所有增强图像长期保存到磁盘，而是对每个样本生成多个增强视图，
    分别提取 MEAS、LDSE 或 radiomics 特征，再按 aggregator 聚合成一个特征表。
    可用于提升特征对随机扰动的鲁棒性。

    Returns:
        dict: 包含 status、output_csv、total_samples、message 等字段。
    """
    try:
        await ctx.report_progress(0.0, None, "正在准备在线增强与特征提取")
        if feature_type.upper() != "LDSE":
            os.makedirs(output_dir, exist_ok=True)

        if feature_type.upper() == "MEAS":
            from algorithm.KOA_feature_extraction.meas_feature_extraction import OAIZIB_MEAS_FeatureExtractor
            extractor_obj = OAIZIB_MEAS_FeatureExtractor(output_dir)

            def extractor_fn(image_array, mask_array):
                """Extract MEAS features from in-memory arrays."""
                import tempfile
                import SimpleITK as sitk
                with tempfile.TemporaryDirectory() as tmpdir:
                    img_path = os.path.join(tmpdir, "_tmp_img.nii.gz")
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(image_array), img_path)
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    feats = extractor_obj.extract_features("_online", img_path, msk_path)
                    if feats:
                        return {k: v for k, v in feats.items()
                                if isinstance(v, (int, float, np.floating, np.integer))}
                    return {}

        elif feature_type.upper() == "LDSE":
            ldse = prepare_ldse_online_augmentation_extractor(
                output_dir,
                csv_path,
                dataset_role,
                model_path=ldse_model_path,
                n_components=50,
                random_seed=seed,
            )

            def extractor_fn(image_array, mask_array):
                import tempfile
                import SimpleITK as sitk
                sample_key = stable_mask_array_sample_key(mask_array)
                with tempfile.TemporaryDirectory() as tmpdir:
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    feats = ldse.extract_ldse_features(msk_path, sample_key=sample_key)
                    if feats:
                        return {k: v for k, v in feats.items()
                                if isinstance(v, (int, float, np.floating, np.integer))}
                    return {}

        elif feature_type.lower() == "radiomics":
            from radiomics import featureextractor as rx_extractor
            rx = rx_extractor.RadiomicsFeatureExtractor()

            def extractor_fn(image_array, mask_array):
                import tempfile
                import SimpleITK as sitk
                with tempfile.TemporaryDirectory() as tmpdir:
                    img_path = os.path.join(tmpdir, "_tmp_img.nii.gz")
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(image_array), img_path)
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    try:
                        with suppress_output():
                            result = rx.execute(img_path, msk_path, label=1)
                        return {k: v for k, v in result.items()
                                if not k.startswith('diagnostics')
                                and isinstance(v, (int, float, np.floating, np.integer))}
                    except Exception:
                        return {}
        else:
            return {"status": "failed", "message": f"Unknown feature_type: {feature_type}. Choose MEAS, LDSE, or radiomics."}

        online_aug = OnlineAugmentor(n_augmentations=n_augmentations, seed=seed)

        df = pd.read_csv(csv_path, dtype={"CaseID": str})
        all_aggregated = []
        total_cases = int(len(df))

        for processed_cases, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=f"Online augment + {feature_type} extraction"),
            start=1,
        ):
            case_id = row["CaseID"]
            image_path = row["ImagePath"]
            mask_path = row["MaskPath"]
            kl_grade = row.get("KLGrade", None)
            knee_side = int(row.get("KneeSide", 1))

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"  Skipping {case_id}: missing file")
                await ctx.report_progress(
                    float(processed_cases), float(total_cases) if total_cases else None,
                    f"已处理 {processed_cases}/{total_cases}；成功 {len(all_aggregated)}；当前 CaseID={case_id}；输入缺失",
                )
                continue

            try:
                agg_features = online_aug.extract_with_augmentation(
                    image_path, mask_path,
                    extractor_fn=extractor_fn,
                    knee_side=knee_side,
                    aggregator=aggregator,
                )
                agg_features["CaseID"] = str(case_id)
                if kl_grade is not None:
                    agg_features["KLGrade"] = kl_grade
                all_aggregated.append(agg_features)
            except Exception as e:
                print(f"  Error processing {case_id}: {e}")
                await ctx.report_progress(
                    float(processed_cases), float(total_cases) if total_cases else None,
                    f"已处理 {processed_cases}/{total_cases}；成功 {len(all_aggregated)}；当前 CaseID={case_id}；处理失败",
                )
                continue
            await ctx.report_progress(
                float(processed_cases), float(total_cases) if total_cases else None,
                f"已处理 {processed_cases}/{total_cases}；成功 {len(all_aggregated)}；当前 CaseID={case_id}",
            )

        if not all_aggregated:
            return {"status": "failed", "message": "No features extracted successfully"}

        result_df = pd.DataFrame(all_aggregated)
        feature_csv = os.path.join(output_dir, f"{feature_type.lower()}_augmented_features.csv")
        result_df.to_csv(feature_csv, index=False)

        result = {
            "status": "success",
            "output_csv": feature_csv,
            "total_samples": len(result_df),
            "message": f"Online augmentation complete: {len(result_df)} samples x {1 + n_augmentations} views, features saved to {feature_csv}",
        }
        if feature_type.upper() == "LDSE":
            result["model_path"] = ldse.model_path
        return result

    except Exception as e:
        return {"status": "failed", "message": str(e)}


@mcp.tool()
def assess_feature_quality(
    feature_csv: Annotated[str, Field(description="需要评估质量的特征 CSV 路径，通常是 *_features_summary.csv 或 filtered_features.csv。")],
    output_dir: Annotated[str, Field(description="质量评估报告、指标 JSON 和过滤后特征 CSV 的输出目录。")],
    label_col: Annotated[str, Field(description="标签列名，默认 KLGrade。")] = "KLGrade",
    variance_threshold: Annotated[float, Field(description="低方差阈值，默认 0.01；低于该阈值的特征会被视为低方差特征。")] = 0.01,
    correlation_threshold: Annotated[float, Field(description="高相关阈值，默认 0.95；绝对相关系数高于该阈值的特征对会被报告。")] = 0.95,
    cv_threshold: Annotated[float, Field(description="稳定性检查的 CV 阈值，默认 3.0，用于评估噪声/信号比例。")] = 3.0,
    filter_stability: Annotated[bool, Field(description="是否从 filtered_features.csv 中移除不稳定特征。默认 False：只报告不稳定特征，不自动过滤，因为 LASSO 等下游分类器可处理部分噪声。")] = False,
):
    """
    在分类前评估特征 CSV 的质量。

    该工具检查：
    - 低方差特征：variance < variance_threshold
    - 高共线特征对：|r| > correlation_threshold，并建议保留与标签更相关的特征
    - 特征稳定性：当存在在线增强产生的 _std 列时，使用 CV = std / |mean| 评估

    输出文件：
    - quality_report.txt：可读的质量分析报告
    - quality_metrics.json：结构化质量指标
    - filtered_features.csv：过滤明显问题特征后的特征文件

    Returns:
        dict: 包含 status、report_path、filtered_csv、metrics_json、quality_score、summary 等字段。
    """
    import json as _json

    try:
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(feature_csv, dtype={"CaseID": str})
        is_valid, validation_message, feature_cols = validate_feature_dataframe(
            df,
            label_col=label_col,
        )
        if not is_valid:
            return {
                "status": "failed",
                "message": (
                    f"{validation_message} Input path: {feature_csv}. "
                    "For LDSE, use ldse_features_summary.csv from the "
                    "LDSE_Features_* output directory."
                ),
                "feature_csv": feature_csv,
                "output_dir": output_dir,
            }

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

        # Save filtered features (stability issues reported but NOT auto-filtered)
        filtered_df = assessor.filter_features(df, report, filter_stability=filter_stability)
        filtered_csv = os.path.join(output_dir, "filtered_features.csv")
        filtered_df.to_csv(filtered_csv, index=False)

        # Count filtered features
        filtered_feature_count = sum(
            1 for c in filtered_df.columns
            if c not in {"CaseID", "ImagePath", "MaskPath", "Region", "KLGrade"}
            and not c.endswith("_std")
        )

        stability_note = (
            " (reported but NOT filtered — set filter_stability=true to remove)"
            if len(report.unstable_features) > 0 and not filter_stability else ""
        )
        stability_action = (
            f"Removed {len(report.unstable_features)} unstable features. "
            if filter_stability else ""
        )
        summary = (
            f"Quality score: {report.quality_score}/100. "
            f"Found: {len(report.low_variance_features)} low-variance, "
            f"{len(report.high_correlation_pairs)} high-correlation pairs, "
            f"{len(report.unstable_features)} unstable features{stability_note}. "
            f"{stability_action}"
            f"Original features: {report.total_features}, "
            f"Filtered features: {filtered_feature_count}."
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


if __name__ == "__main__":
    mcp.run(transport='stdio')
    # res = koa_filter_classification(train_path="LDSE_Features_Train_20260125_144158/ldse_features_summary.csv",
    #                            test_path="LDSE_Features_Test_20260125_144158/ldse_features_summary.csv",
    #                            output_dir="Filter_Classification_MEAS_20260125_141325",
    #                            feature_type="MEAS")
    # res = koa_lasso_classification(train_path="../LDSE_Features_Train_20260125_144158/ldse_features_summary.csv",
    #                            test_path="../LDSE_Features_Test_20260125_144158/ldse_features_summary.csv",
    #                            output_dir= "LASSO_Classification_20260125_151010")
    # print(res)
