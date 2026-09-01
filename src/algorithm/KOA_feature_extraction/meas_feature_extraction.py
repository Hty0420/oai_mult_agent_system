import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
import pickle

from ..klgrade_policy import select_labeled_klgrade_rows


class OAIZIB_MEAS_FeatureExtractor:
    """
    针对OAI-ZIB膝关节数据集的MEAS特征提取器
    功能：提取膝关节不同结构（股骨、软骨等）的形态学与几何学特征（体积、表面积、厚度等），支持批量处理数据集并保存结果
    适用场景：医学影像分析中，对膝关节结构进行量化特征提取，为后续机器学习建模、医学研究提供数据支撑
    """

    def __init__(self, output_dir):
        """
        初始化特征提取器，配置数据集路径与输出目录，定义标签映射、结构对及特征名称映射规则
        参数：
            output_dir (str)：OAI-ZIB数据集根目录，需包含imagesTr、labelsTr等子目录
            output_dir (str, 可选)：特征结果保存目录，默认在output_dir下创建meas_extracted_features目录
        返回：
            无显式返回，初始化类实例属性
        """
        self.output_dir = output_dir
        self.output_dir = output_dir or os.path.join(output_dir, "meas_extracted_features")
        os.makedirs(self.output_dir, exist_ok=True)
        self.last_run_counts = {}
        # 标签映射：掩码数值 -> 膝关节解剖结构名称（如1对应股骨Femur）
        self.labels_map = {
            1: "Femur",
            2: "Femoral Cartilage",
            3: "Tibia",
            4: "Medial Tibial Cartilage",
            5: "Lateral Tibial Cartilage"
        }
        # 软骨-骨结构对：用于计算软骨特征时关联对应的骨结构（如股骨软骨关联股骨）
        self.structure_pairs = {
            "Femoral Cartilage": "Femur",
            "Medial Tibial Cartilage": "Tibia",
            "Lateral Tibial Cartilage": "Tibia"
        }
        # 特征名称映射：长特征名 -> 简写（如volume简化为V，方便输出列名简洁）
        self.feature_name_map = {
            "volume": "V",
            "surface_area": "A",
            "thickness_mean": "T_mean",
            "thickness_std": "T_std",
            "surface_to_volume_ratio": "SVR",
            "cartilage_bone_ratio": "CBR"
        }

    def load_image(self, image_path):
        """Load a NIfTI image or mask, returning None without logging on failure."""
        try:
            image = sitk.ReadImage(image_path)
            return image
        except Exception:
            return None

    def get_voxel_volume(self, image):
        """
        计算单个体素的物理体积（mm³），基于图像的体素间距
        参数：
            image (sitk.Image对象)：需包含Spacing信息（x、y、z方向的物理间距）
        返回：
            float，单个体素的体积（mm³）
        """
        spacing = image.GetSpacing()
        return spacing[0] * spacing[1] * spacing[2]

    def calculate_volume(self, mask_image, label_value):
        """
        计算掩码中指定标签对应的结构体积（mm³）
        参数：
            mask_image (sitk.Image对象)：掩码图像
            label_value (int)：目标结构的标签值（如1对应股骨）
        返回：
            float，结构的体积（mm³）
        """
        mask_array = sitk.GetArrayFromImage(mask_image)
        voxel_volume = self.get_voxel_volume(mask_image)
        return np.sum(mask_array == label_value) * voxel_volume

    def calculate_surface_area(self, mask_image, label_value):
        """
        计算掩码中指定标签对应的结构表面积（mm²），通过Sobel梯度估计表面体素，结合体素间距转换
        参数：
            mask_image (sitk.Image对象)：掩码图像
            label_value (int)：目标结构的标签值
        返回：
            float，结构的表面积（mm²）
        """
        mask_array = sitk.GetArrayFromImage(mask_image)
        label_mask = (mask_array == label_value).astype(np.uint8)

        grad_x = ndimage.sobel(label_mask, axis=0)
        grad_y = ndimage.sobel(label_mask, axis=1)
        grad_z = ndimage.sobel(label_mask, axis=2)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        surface_voxels = np.sum(gradient_magnitude > 0)

        spacing = mask_image.GetSpacing()
        avg_spacing = (spacing[0] + spacing[1] + spacing[2]) / 3
        return surface_voxels * (avg_spacing ** 2)

    def calculate_cartilage_thickness(self, cartilage_mask, bone_mask):
        """
        计算软骨结构的平均厚度和厚度标准差（mm），需传入软骨与对应骨的掩码，基于距离变换实现
        参数：
            cartilage_mask (sitk.Image对象)：软骨的二值掩码
            bone_mask (sitk.Image对象)：对应骨的二值掩码
        返回：
            tuple(float, float)，(平均厚度, 厚度标准差)（mm）
        """
        cartilage_array = sitk.GetArrayFromImage(cartilage_mask)
        bone_array = sitk.GetArrayFromImage(bone_mask)

        bone_boundary = ndimage.binary_dilation(bone_array) & (~bone_array)
        distance_transform = ndimage.distance_transform_edt(~bone_boundary)

        cartilage_distances = distance_transform[cartilage_array > 0]
        if len(cartilage_distances) == 0:
            return 0, 0

        mean_thickness = np.mean(cartilage_distances)
        std_thickness = np.std(cartilage_distances)

        spacing = cartilage_mask.GetSpacing()
        avg_spacing = (spacing[0] + spacing[1] + spacing[2]) / 3
        return mean_thickness * avg_spacing, std_thickness * avg_spacing

    def calculate_surface_to_volume_ratio(self, volume, surface_area):
        """
        计算表面积与体积的比值（SVR）
        参数：
            volume (float)：结构体积（mm³）
            surface_area (float)：结构表面积（mm²）
        返回：
            float，表面积体积比（SVR），体积为0时返回0避免除零错误
        """
        if volume == 0:
            return 0
        return surface_area / volume

    def calculate_cartilage_bone_ratio(self, cartilage_volume, bone_volume):
        """
        计算软骨体积与对应骨体积的比值（CBR）
        参数：
            cartilage_volume (float)：软骨体积（mm³）
            bone_volume (float)：对应骨的体积（mm³）
        返回：
            float，软骨骨体积比（CBR），骨体积为0时返回0避免除零错误
        """
        if bone_volume == 0:
            return 0
        return cartilage_volume / bone_volume

    def extract_features(self, CaseID, image_path, mask_path):
        """
        提取单个样本的MEAS特征，涵盖体积、表面积、厚度等，返回特征字典
        参数：
            CaseID (str)：样本ID（如001）
            image_path (str)：图像文件路径
            mask_path (str)：掩码文件路径
        返回：
            dict（特征完整时）或None（图像/掩码加载失败时）
        """
        image = self.load_image(image_path)
        mask = self.load_image(mask_path)
        if image is None or mask is None:
            return None

        features = {"CaseID": CaseID}

        for label_value, structure_name in self.labels_map.items():
            volume = self.calculate_volume(mask, label_value)
            surface_area = self.calculate_surface_area(mask, label_value)
            svr = self.calculate_surface_to_volume_ratio(volume, surface_area)

            features[f"{self.feature_name_map['volume']}_{structure_name}"] = volume
            features[f"{self.feature_name_map['surface_area']}_{structure_name}"] = surface_area
            features[f"{self.feature_name_map['surface_to_volume_ratio']}_{structure_name}"] = svr

            if structure_name in self.structure_pairs:
                bone_name = self.structure_pairs[structure_name]
                bone_label = [k for k, v in self.labels_map.items() if v == bone_name][0]

                cartilage_mask = sitk.BinaryThreshold(mask, lowerThreshold=label_value, upperThreshold=label_value)
                bone_mask = sitk.BinaryThreshold(mask, lowerThreshold=bone_label, upperThreshold=bone_label)

                mean_thickness, std_thickness = self.calculate_cartilage_thickness(cartilage_mask, bone_mask)
                features[f"{self.feature_name_map['thickness_mean']}_{structure_name}"] = mean_thickness
                features[f"{self.feature_name_map['thickness_std']}_{structure_name}"] = std_thickness

                bone_volume = self.calculate_volume(mask, bone_label)
                cbr = self.calculate_cartilage_bone_ratio(volume, bone_volume)
                features[f"{self.feature_name_map['cartilage_bone_ratio']}_{structure_name}"] = cbr

        return features

    def process_from_csv(self, csv_path, save=True, progress_callback=None):
        """
        从CSV文件中读取image_path和mask_path，提取每个样本的MEAS特征，并附加KOA分级标签。
        参数：
            csv_path (str): ImagePath、MaskPath、CaseID、KLGrade列的CSV路径
            save (bool): 是否保存为CSV和PKL文件（默认True）
        返回：
            pd.DataFrame: 包含提取特征与KOA分级标签的DataFrame
        """
        df = pd.read_csv(csv_path)
        df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]
        aliases = {
            "caseid": "CaseID",
            "case_id": "CaseID",
            "imagepath": "ImagePath",
            "image_path": "ImagePath",
            "maskpath": "MaskPath",
            "mask_path": "MaskPath",
            "klgrade": "KLGrade",
            "kl_grade": "KLGrade",
            "koa_grade": "KLGrade",
        }
        rename_map = {
            column: aliases[column.lower()]
            for column in df.columns
            if column.lower() in aliases and aliases[column.lower()] not in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        required_columns = ["CaseID", "ImagePath", "MaskPath", "KLGrade"]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(
                "MEAS input CSV is missing required columns "
                f"{missing_columns}; available columns: {list(df.columns)}"
            )
        df["CaseID"] = df["CaseID"].astype(str)
        selection = select_labeled_klgrade_rows(df, dataset_role="MEAS input")
        df = selection.frame
        all_features = []
        successful_rows = 0

        def report(processed, case_id=None, status=None):
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "phase": "extract",
                        "processed_cases": int(processed),
                        "total_cases": int(len(df)),
                        "successful_cases": int(successful_rows),
                        "failed_cases": int(processed - successful_rows),
                        "case_id": str(case_id) if case_id is not None else None,
                        "case_status": status,
                    }
                )
            except Exception:
                # Progress reporting is observational and must not alter results.
                pass

        report(0)
        for processed, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Processing CSV entries"),
            start=1,
        ):
            CaseID = row["CaseID"]
            ImagePath = row["ImagePath"]
            MaskPath = row["MaskPath"]
            KLGrade = row["KLGrade"]

            if not os.path.exists(ImagePath) or not os.path.exists(MaskPath):
                report(processed, CaseID, "missing_input")
                continue

            features = self.extract_features(CaseID, ImagePath, MaskPath)
            if features:
                features["KLGrade"] = KLGrade
                all_features.append(features)
                successful_rows += 1
                report(processed, CaseID, "success")
            else:
                report(processed, CaseID, "no_features")

        features_df = pd.DataFrame(all_features)
        self.last_run_counts = {
            "input_rows": selection.input_rows,
            "labeled_rows": selection.labeled_rows,
            "excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
            "output_rows": int(len(features_df)),
            "other_skipped_rows": int(selection.labeled_rows - len(features_df)),
        }

        if save and len(features_df) > 0:
            os.makedirs(self.output_dir, exist_ok=True)
            csv_out = os.path.join(self.output_dir, "meas_features_summary.csv")
            features_df.to_csv(csv_out, index=False)
            pkl_out = os.path.join(self.output_dir, "meas_features_summary.pkl")
            with open(pkl_out, "wb") as f:
                pickle.dump(features_df, f)

        return features_df

# if __name__ == "__main__":
#     # 1. 配置数据集根目录（根据实际路径修改）
#     output_dir = r"D:\datasets\OAIZIB"
#     # 2. 初始化特征提取器实例
#     extractor = OAIZIB_MEAS_FeatureExtractor(output_dir)
#
#     # 3. 处理训练集，提取特征并保存
#     print("=== 开始处理训练集 ===")
#     train_features = extractor.process_dataset(subset="train")
#     # 4. 处理测试集，提取特征并保存
#     print("\n=== 开始处理测试集 ===")
#     test_features = extractor.process_dataset(subset="test")
#
#     # 5. 打印训练集特征提取结果统计
#     if not train_features.empty:
#         feature_count = len(train_features.columns) - 3  # 减去CaseID、image_path、mask_path三列
#         sample_count = len(train_features)
#         print(f"\n训练集特征提取结果：共提取 {feature_count} 个特征，涵盖 {sample_count} 个样本")
#         print("训练集特征示例（前10个特征）：")
#         print(train_features.iloc[0][:10])
#     else:
#         print("\n训练集未提取到有效特征")
#
#     # 6. 打印测试集特征提取结果统计
#     if not test_features.empty:
#         feature_count = len(test_features.columns) - 3
#         sample_count = len(test_features)
#         print(f"\n测试集特征提取结果：共提取 {feature_count} 个特征，涵盖 {sample_count} 个样本")
#     else:
#         print("\n测试集未提取到有效特征")
if __name__ == "__main__":
    output_dir = r"D:\oai_mcp\datasets"  # 可任意指定，仅用于保存输出
    extractor = OAIZIB_MEAS_FeatureExtractor(output_dir)

    # 从CSV中提取
    csv_path = r"D:\oai_mcp\datasets\labels_test.csv"
    print("=== 从CSV提取特征 ===")
    feature_df = extractor.process_from_csv(csv_path)

    if not feature_df.empty:
        print(f"\n提取完成：共 {len(feature_df)} 个样本，{len(feature_df.columns)-4} 个特征")
        print(feature_df.head())
    else:
        print("未提取到任何特征。")
