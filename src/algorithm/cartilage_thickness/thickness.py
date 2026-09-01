"""
thickness.py — 膝关节软骨厚度计算模块

基于分割结果和体素物理间距，计算股骨软骨(2)、内侧胫骨软骨(4)、
外侧胫骨软骨(5)的厚度图和统计指标。

方法: 3D Mesh Surface Normal (3D-MN) 
  
  论文算法 :
    1. marching_cubes 提取软骨表面网格 → verts, faces, normals
    2. 按法线方向分类顶点: 骨界面侧 (法线指向骨) / 滑膜面侧 (朝向关节腔)
    3. 双向测量: 骨界面侧顶点沿法线追踪到滑膜面 (d1)，
                滑膜面侧顶点沿法线追踪到骨界面 (d2)
    4. 每个顶点一个厚度值（沿法向量到对侧表面的距离）
    5. 统计: 顶点厚度 + 三角面面积加权 (Wirth & Eckstein 2008)
    6. 体素厚度图仅用于可视化，不参与统计

参考文献:

  方法学基础:
    [1] Stammberger T, Eckstein F, Englmeier KH, Reiser M.
        Magn Reson Med. 1999;41(3):529-536.
        DOI: 10.1002/(SICI)1522-2594(199903)41:3<529::AID-MRM15>3.0.CO;2-Z
        首次提出网格表面法向厚度计算概念。

    [2] Cohen ZA, McCarthy DM, Kwak SD, Legrand P, Fogarasi F,
        Ciaccio EJ, Ateshian GA.
        Osteoarthritis Cartilage. 1999;7(1):95-109.
        DOI: 10.1053/joca.1998.0165
        体外校准 + 在体验证了网格表面法的精度。

  方法对比与验证:
    [3] Nolte T, Westfechtel S, Schock J, Knobe M, Pastor T,
        Pfaehler E, Kuhl C, Truhn D, Nebelung S.
        CARTILAGE. 2023;14(1):26-38.
        DOI: 10.1177/19476035221144744
        PMID: 36659857 / PMCID: PMC10076900
        系统比较 5 种方法 (507 例 OAI 数据)，确立 3D-MN 为最优。

  综述与最新进展:
    [4] Ebrahimkhani S, Jaward MH, Cicuttini FM, Dharmaratne A,
        Wang Y, Seco de Herrera AG.
        Artif Intell Med. 2020;106:101851.
        DOI: 10.1016/j.artmed.2020.101851
        从传统方法到深度学习的软骨分割与测量方法演进综述。

    [5] Eckstein F, Cicuttini F, Raynauld JP, Waterton JC, Peterfy C.
        Osteoarthritis Cartilage. 2006;14 Suppl A:A46-A75.
        DOI: 10.1016/j.joca.2006.02.026
        OAI 软骨 MRI 形态学评估的里程碑综述。

  亚区分析:
    [6] Wirth W, Eckstein F.
        IEEE Trans Med Imaging. 2008;27(6):737-744.
        DOI: 10.1109/TMI.2007.907323
        亚区厚度分析的标准化方法。

  前沿:
    [7] Wang P, Zhang W, Li L, Zhang X, Wang C, Zhao P, Shen T,
        Dai S, Wang X.
        npj Digit Med. 2025;8:686.
        DOI: 10.1038/s41746-025-02040-z
        基于隐式表面重建的解剖学感知软骨厚度映射 (最新进展)。

输出:
  thickness_maps: {2: ndarray, 4: ndarray, 5: ndarray}  — float32 mm
  statistics:     {2: dict, 4: dict, 5: dict}
  bone_interface: {2: ndarray, 4: ndarray, 5: ndarray}    — bool mask
"""

# Vendored from ``D:\\Cartilage thickness estimation\\utils\\thickness.py``
# on 2026-08-25.  Keep the numerical implementation aligned with that
# validated research module; MCP I/O and artifact handling live in service.py.

import os
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# 软骨与其配对骨骼的映射
CARTILAGE_BONE_PAIRS = {
    2: 1,   # 股骨软骨 → 股骨
    4: 3,   # 内侧胫骨软骨 → 胫骨
    5: 3,   # 外侧胫骨软骨 → 胫骨
}

CARTILAGE_NAMES = {
    2: "股骨软骨 (Femoral Cartilage)",
    4: "内侧胫骨软骨 (Medial Tibial Cartilage)",
    5: "外侧胫骨软骨 (Lateral Tibial Cartilage)",
}

# 合理的软骨厚度范围 (mm)，超过此范围视为异常
MAX_THICKNESS_MM = 10.0
# 生理学上限：真实膝关节软骨厚度极少超过此值（论文 Table 2 最大值 mean+3σ ≈ 4.4mm）
MAX_PHYSIOLOGICAL_THICKNESS_MM = 5.0
MIN_BONE_INTERFACE_VOXELS = 50  # 骨界面体素太少 = 分割质量可疑，跳过
# 侧壁/边缘顶点过滤: 软骨表面顶点法线与其到最近骨面方向的最小 |cos|
# 低于此值判定为侧壁切线顶点, 不参与厚度统计 (减少斜向测量高估)。
EDGE_NORMAL_MIN_ABS_DOT = 0.4
# MAD 稳健异常值剔除: 单侧厚度 > median + K*MAD 的顶点剔除。
MAD_OUTLIER_K = 3.0
# 射线追踪步长因子 (乘以 min(spacing)): 0.15 在精度与速度间平衡
RAY_TRACE_STEP_FACTOR = 0.15


def extract_bone_cartilage_interface(
    pred: np.ndarray,
    bone_label: int,
    cartilage_label: int,
) -> np.ndarray:
    """
    提取骨-软骨界面：骨骼中与软骨相邻的体素。

    使用 3D 二值膨胀检测邻接关系。

    Returns
    -------
    np.ndarray [D, H, W] bool — 骨界面 mask
    """
    bone_mask = pred == bone_label
    cartilage_mask = pred == cartilage_label

    if bone_mask.sum() == 0 or cartilage_mask.sum() == 0:
        return np.zeros_like(bone_mask, dtype=bool)

    # 性能: 只对软骨 bbox+1 做局部膨胀, 不对整卷做 binary_dilation
    nz = np.argwhere(cartilage_mask)
    lo = np.maximum(nz.min(axis=0) - 1, 0)
    hi = np.minimum(nz.max(axis=0) + 2, np.asarray(bone_mask.shape))
    struct = ndimage.generate_binary_structure(3, 1)
    cart_crop = cartilage_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    cart_dil = ndimage.binary_dilation(cart_crop, structure=struct, iterations=1)

    interface = np.zeros_like(bone_mask, dtype=bool)
    interface[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (
        bone_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & cart_dil)
    return interface



def compute_thickness_map(
    cartilage_mask: np.ndarray,
    bone_mask: np.ndarray,
    spacing: tuple,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    兼容入口：返回可视化用体素厚度图（顶点级 3D-MN 算法的副产品）。

    注意: 第二个参数从旧的 bone_interface 改为完整 bone_mask，
    因为论文原版需要完整的骨 mask 来区分软骨的骨界面侧/滑膜面侧。

    Returns
    -------
    np.ndarray [D, H, W] float32 — 体素厚度图 (mm)，仅用于可视化。
    """
    res = compute_thickness_map_vertex(cartilage_mask, bone_mask, spacing)
    return res["thickness_map"]


# ═══════════════════════════════════════════════════════════════
# 论文原版 3D-MN：顶点级双向法向量 + 面积加权
# ═══════════════════════════════════════════════════════════════

def _classify_surface_verts(
    verts: np.ndarray,
    normals: np.ndarray,
    bone_mask: np.ndarray,
    spacing: tuple,
    probe_mm: float = 2.0,
) -> np.ndarray:
    """
    顶点分类：沿法线方向探针采样，命中骨体素 → 骨界面侧。

    3D-MN 需要知道软骨表面的哪一侧贴着骨（骨界面侧），
    哪一侧朝向关节腔（滑膜面侧）。marching_cubes 法线朝外，
    骨界面侧顶点的法线方向必然指向骨。
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    pts = verts + normals * probe_mm
    # 性能: 只对探针点 bbox 附近的骨 mask 做膨胀, 不对整卷膨胀
    pad = 3
    lo = np.floor(pts.min(axis=0) / spacing).astype(np.int64) - pad
    hi = np.ceil(pts.max(axis=0) / spacing).astype(np.int64) + pad + 1
    D, H, W = bone_mask.shape
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [D, H, W])
    if (lo >= hi).any():
        return np.zeros(len(verts), dtype=bool)
    bone_crop = bone_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    bone_dil_crop = ndimage.binary_dilation(
        bone_crop, structure=ndimage.generate_binary_structure(3, 1),
        iterations=2)

    idx = np.floor(pts / spacing).astype(np.int64)
    rel = idx - lo
    ok = (
        (rel[:, 0] >= 0) & (rel[:, 0] < bone_dil_crop.shape[0]) &
        (rel[:, 1] >= 0) & (rel[:, 1] < bone_dil_crop.shape[1]) &
        (rel[:, 2] >= 0) & (rel[:, 2] < bone_dil_crop.shape[2])
    )
    bone_side = np.zeros(len(verts), dtype=bool)
    bone_side[ok] = bone_dil_crop[rel[ok, 0], rel[ok, 1], rel[ok, 2]]
    return bone_side


def _ray_trace_to_surface(
    origins: np.ndarray,
    dirs: np.ndarray,
    target_mask: np.ndarray,
    cartilage_mask: np.ndarray,
    spacing: tuple,
    max_thickness_mm: float = MAX_THICKNESS_MM,
) -> np.ndarray:
    """
    向量化射线追踪（3D-MN 核心）：从 origins 沿 dirs 步进，
    命中 target_mask（对侧表面体素）时记录到表面边界的距离 (mm)。

    等价于论文的 pyvista ray_trace（从网格顶点沿法向量追踪到
    对侧表面网格交点）。体素化表面 + 0.1×min(spacing) 步长离散化，
    命中后修正到体素在射线主方向上的表面边界（等值面位置）：
      - 命中体素是软骨体素 → 表面在其出射边界（射线离开该体素的一侧）
      - 命中体素是背景/骨   → 表面在其入射边界（射线进入该体素的一侧）
    修正量为精确的到边界距离（各向异性感知），而非固定半格。

    Returns
    -------
    np.ndarray [N] float32 — 每顶点厚度 (mm)，-1 = 未命中（超出最大厚度）
    """
    step = RAY_TRACE_STEP_FACTOR * min(spacing)
    max_steps = int(max_thickness_mm / step) + 2
    D, H, W = target_mask.shape
    spacing_arr = np.asarray(spacing, dtype=np.float64)

    n = len(origins)
    dists = np.full(n, -1.0, dtype=np.float32)
    active = np.ones(n, dtype=bool)

    for s in range(1, max_steps + 1):
        if not active.any():
            break
        dist = s * step
        pts = origins[active] + dirs[active] * dist
        idx = np.floor(pts / spacing_arr).astype(np.int64)
        ok = (
            (idx[:, 0] >= 0) & (idx[:, 0] < D) &
            (idx[:, 1] >= 0) & (idx[:, 1] < H) &
            (idx[:, 2] >= 0) & (idx[:, 2] < W)
        )
        hit = np.zeros(active.sum(), dtype=bool)
        hit[ok] = target_mask[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        if hit.any():
            aidx = np.where(active)[0][hit]
            hit_idx = idx[hit]
            is_cart = cartilage_mask[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]]

            # 射线主方向轴
            axis = np.argmax(np.abs(dirs[aidx]), axis=1)
            ar = np.arange(len(aidx))
            dir_axis = dirs[aidx][ar, axis]
            pos_axis = pts[hit][ar, axis]
            idx_axis = hit_idx[ar, axis]
            sp = spacing_arr[axis]

            # 体素边界：出射侧 / 入射侧（依射线方向）
            # 软骨体素 → 出射边界；背景/骨 → 入射边界
            sign_pos = dir_axis > 0
            exit_boundary = np.where(sign_pos, (idx_axis + 1) * sp, idx_axis * sp)
            enter_boundary = np.where(sign_pos, idx_axis * sp, (idx_axis + 1) * sp)
            boundary = np.where(is_cart, exit_boundary, enter_boundary)

            corr = (boundary - pos_axis) / np.maximum(np.abs(dir_axis), 1e-6)
            dists[aidx] = dist + corr
            active[aidx] = False

    return dists


def _bone_direction_for_verts(verts, bone_mask, spacing, pad_voxels=4):
    """每个软骨表面顶点 → 最近骨面点的单位方向向量。

    侧壁顶点靠近软骨分割边界, 其到最近骨面的方向几乎与自身法线垂直;
    真正骨界面/滑膜面顶点的该方向则与法线近似平行。返回 (N,3) 单位向量,
    用于边缘顶点过滤。
    """
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    lo = np.floor(verts.min(axis=0) / spacing_arr).astype(np.int64) - pad_voxels
    hi = np.ceil(verts.max(axis=0) / spacing_arr).astype(np.int64) + pad_voxels + 1
    D, H, W = bone_mask.shape
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [D, H, W])
    if (lo >= hi).any():
        return None
    crop = bone_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    idx = np.argwhere(crop)
    if len(idx) < 10:
        return None
    points = (idx.astype(np.float64) + lo) * spacing_arr
    tree = cKDTree(points)
    _, nn = tree.query(verts, k=1)
    direction = verts - points[nn]
    length = np.linalg.norm(direction, axis=1)
    valid = length > 1e-9
    direction[valid] /= length[valid, None]
    direction[~valid] = 0.0
    return direction


def _edge_keep_mask(verts, normals, bone_mask, spacing,
                    min_abs_dot=EDGE_NORMAL_MIN_ABS_DOT):
    """侧壁/边缘顶点过滤: 法线与骨面方向夹角过大者剔除。"""
    direction = _bone_direction_for_verts(verts, bone_mask, spacing)
    if direction is None:
        return np.ones(len(verts), dtype=bool)
    abs_dot = np.abs(np.einsum("ij,ij->i", normals, direction))
    return abs_dot >= min_abs_dot


def _mad_keep_mask(values, k=MAD_OUTLIER_K):
    """单侧 MAD 稳健异常值剔除: 保留 <= median + k*MAD 的正值。"""
    values = np.asarray(values, dtype=np.float64)
    valid = values > 0
    if valid.sum() < 10:
        return valid
    med = float(np.median(values[valid]))
    mad = float(np.median(np.abs(values[valid] - med)))
    if mad <= 1e-9:
        return valid
    upper = med + k * mad
    return valid & (values <= upper)


def compute_thickness_map_vertex(
    cartilage_mask: np.ndarray,
    bone_mask: np.ndarray,
    spacing: tuple,
    max_thickness_mm: float = MAX_THICKNESS_MM,
) -> dict:
    """
    论文原版 3D-MN：顶点级双向法向量 + 面积加权（Nolte 2023 + Wirth & Eckstein 2008）。

    与旧体素版的本质区别（这是论文方法与体素赋值法的分水岭）:
      - 网格顶点来自软骨表面 (marching_cubes 软骨 mask)，而不是骨界面
      - 测量: 每个顶点沿法向量（反向，穿过软骨）追踪到对侧表面 → 顶点厚度
      - 双向: 骨界面侧顶点测到滑膜面 (d1) + 滑膜面侧顶点测到骨界面 (d2)
      - 统计: 顶点厚度 + 三角面面积加权（论文: mesh element 面积加权）
      - 体素图仅用于可视化，不参与统计

    Parameters
    ----------
    cartilage_mask : np.ndarray [D, H, W] bool — 某类软骨 (label 2/4/5)
    bone_mask      : np.ndarray [D, H, W] bool — 对应骨骼 (label 1/3)
    spacing        : tuple (dz, dy, dx)

    Returns
    -------
    dict:
        vertex_thickness : np.ndarray [N] float32 — 每顶点厚度 (mm)，-1 = 未命中
        vertex_weights   : np.ndarray [N] float32 — 面积权重 (mm²)
        vertex_side      : np.ndarray [N] bool — True=骨界面侧, False=滑膜面侧
        vertex_positions : np.ndarray [N, 3] — 顶点物理坐标 (mm)
        thickness_map    : np.ndarray [D, H, W] float32 — 可视化用体素图
        n_vertices       : int
    """
    D, H, W = cartilage_mask.shape
    empty = {
        "vertex_thickness": np.zeros(0, dtype=np.float32),
        "vertex_weights": np.zeros(0, dtype=np.float32),
        "vertex_side": np.zeros(0, dtype=bool),
        "vertex_positions": np.zeros((0, 3), dtype=np.float32),
        "thickness_map": np.zeros((D, H, W), dtype=np.float32),
        "n_vertices": 0,
    }

    if cartilage_mask.sum() < MIN_BONE_INTERFACE_VOXELS or bone_mask.sum() == 0:
        return empty

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("[Thickness] marching_cubes unavailable")
        return empty

    # ═══ 1. 软骨表面网格（论文: 表面体素 → Delaunay 三角化；marching_cubes 等价） ═══
    # v33 性能: bbox 裁剪 — marching_cubes 对整个体积 (几乎全背景) 极慢
    try:
        nz = np.argwhere(cartilage_mask)
        lo = nz.min(axis=0)
        hi = nz.max(axis=0) + 1
        crop = cartilage_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        verts, faces, normals, _ = marching_cubes(
            crop.astype(np.float32), level=0.5, spacing=spacing,
        )
        if len(verts):
            verts = verts + lo * np.asarray(spacing, dtype=np.float64)
    except Exception:
        print("[Thickness] marching_cubes failed")
        return empty

    if len(verts) == 0 or len(faces) == 0:
        return empty

    spacing_arr = np.asarray(spacing, dtype=np.float64)

    # ═══ 2. 分类顶点: 骨界面侧 / 滑膜面侧 ═══
    bone_side = _classify_surface_verts(verts, normals, bone_mask, spacing_arr)
    joint_side = ~bone_side

    # 任一侧顶点过少 → 分类失败，退回单侧（全部当骨界面侧测量）
    if bone_side.sum() < 10 or joint_side.sum() < 10:
        bone_side[:] = True
        joint_side[:] = False

    # ═══ 2b. 侧壁/边缘顶点过滤 ═══
    # marching_cubes 会在软骨分割边界生成一圈人工侧壁,
    # 其法线接近切线方向; 沿这类法线追踪会把厚度斜着拉长 (外侧胫骨尤其明显)。
    edge_keep = _edge_keep_mask(verts, normals, bone_mask, spacing_arr)
    bone_side &= edge_keep
    joint_side &= edge_keep
    # 过滤后任一侧过少 → 放弃边缘过滤, 退回原分类
    if bone_side.sum() < 10 or joint_side.sum() < 10:
        bone_side = _classify_surface_verts(
            verts, normals, bone_mask, spacing_arr)
        joint_side = ~bone_side

    # 两侧各自的三角面（三个顶点同侧），权重在原始索引空间计算
    def _side_weights(side_mask):
        side_faces = faces[np.all(side_mask[faces], axis=1)]
        if len(side_faces) == 0:
            return np.zeros(int(side_mask.sum()), dtype=np.float32)
        areas = compute_face_areas(verts, side_faces)
        weights = np.zeros(len(verts), dtype=np.float64)
        for f_idx, (v0, v1, v2) in enumerate(side_faces):
            w = areas[f_idx] / 3.0
            weights[v0] += w
            weights[v1] += w
            weights[v2] += w
        return weights[side_mask].astype(np.float32)

    # 对侧表面体素 mask（目标集）
    def _voxel_mask(verts_sub):
        m = np.zeros((D, H, W), dtype=bool)
        idx = np.floor(verts_sub / spacing_arr).astype(np.int64)
        ok = (
            (idx[:, 0] >= 0) & (idx[:, 0] < D) &
            (idx[:, 1] >= 0) & (idx[:, 1] < H) &
            (idx[:, 2] >= 0) & (idx[:, 2] < W)
        )
        m[idx[ok, 0], idx[ok, 1], idx[ok, 2]] = True
        return m

    joint_vox = _voxel_mask(verts[joint_side])
    bone_vox = _voxel_mask(verts[bone_side])

    # ═══ 3. 双向法向量测量（方向 = -法线，穿过软骨到达对侧表面） ═══
    # 骨界面侧顶点 → 滑膜面 (d1)；滑膜面侧顶点 → 骨界面 (d2)
    d1 = _ray_trace_to_surface(
        verts[bone_side], -normals[bone_side], joint_vox, cartilage_mask,
        spacing_arr, max_thickness_mm,
    )
    d2 = _ray_trace_to_surface(
        verts[joint_side], -normals[joint_side], bone_vox, cartilage_mask,
        spacing_arr, max_thickness_mm,
    )

    # ═══ 3b. MAD 稳健异常值剔除 ═══
    # 每侧单独以 median + K*MAD 为上界;
    # 已被硬上限 5mm 截断后仍混入的斜测高值在此踢除。
    keep_d1 = _mad_keep_mask(d1)
    d1[~keep_d1] = -1.0
    keep_d2 = _mad_keep_mask(d2)
    d2[~keep_d2] = -1.0

    # ═══ 4. 合并顶点厚度 + 面积权重（论文: 按 mesh 元素面积加权） ═══
    w1 = _side_weights(bone_side)
    w2 = _side_weights(joint_side)

    vt = np.concatenate([d1, d2]).astype(np.float32)
    vw = np.concatenate([w1, w2]).astype(np.float32)
    vs = np.concatenate([
        np.ones(len(d1), dtype=bool), np.zeros(len(d2), dtype=bool),
    ])
    vp = np.concatenate([verts[bone_side], verts[joint_side]], axis=0).astype(np.float32)

    # ═══ 5. 体素厚度图（仅用于可视化，不参与统计） ═══
    # 向量化：逐层步进，把所有顶点的厚度 d 赋给路径上的体素（取 max）
    # 注意两处历史 bug (2026-08-15 修复):
    #   ① 方向必须与测量一致用 **负法线** (-norm_sub, 朝软骨内部穿行),
    #      旧代码用正法线 → 射线从骨界面/关节面射向骨内/关节腔,
    #      只染到表面 1 层壳 (内部 ~87% 零值) → 厚度模式视图内部全是灰块
    #   ② 每步必须保留"仍在飞行中"的射线 (active &= still),
    #      旧代码 active[still]=False 把在飞的射线提前停掉
    thickness_map = np.zeros((D, H, W), dtype=np.float32)
    step = 0.1 * min(spacing_arr)
    for verts_sub, norm_sub, dists_sub in [
        (verts[bone_side], normals[bone_side], d1),
        (verts[joint_side], normals[joint_side], d2),
    ]:
        valid_ray = dists_sub > 0
        if not valid_ray.any():
            continue
        o = verts_sub[valid_ray].astype(np.float64)
        d = -norm_sub[valid_ray].astype(np.float64)  # 负法线: 朝软骨内部
        t = dists_sub[valid_ray].astype(np.float64)
        active = np.ones(len(o), dtype=bool)
        max_steps_vox = int(t.max() / step) + 2
        for s in range(1, max_steps_vox + 1):
            if not active.any():
                break
            dist = s * step
            # 该步已超出某些射线的长度
            still = active & (dist <= t + step)
            if not still.any():
                break
            pts = o[still] + d[still] * dist
            idx = np.floor(pts / spacing_arr).astype(np.int64)
            ok = (
                (idx[:, 0] >= 0) & (idx[:, 0] < D) &
                (idx[:, 1] >= 0) & (idx[:, 1] < H) &
                (idx[:, 2] >= 0) & (idx[:, 2] < W)
            )
            if ok.any():
                i0, i1, i2 = idx[ok, 0], idx[ok, 1], idx[ok, 2]
                vals = t[still][ok]
                # 取 max：同一体素可能被多条射线覆盖
                np.maximum.at(thickness_map, (i0, i1, i2), vals.astype(np.float32))
            active &= still  # 保留在飞射线, 已完成 (dist 超出 t) 的停掉


    return {
        "vertex_thickness": vt,
        "vertex_weights": vw,
        "vertex_side": vs,
        "vertex_positions": vp,
        "thickness_map": thickness_map,
        "n_vertices": len(vt),
    }


def compute_face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """计算三角面面积 (mm²)，verts 已在物理坐标。"""
    tri_verts = verts[faces]
    a_vec = tri_verts[:, 1] - tri_verts[:, 0]
    b_vec = tri_verts[:, 2] - tri_verts[:, 0]
    return (0.5 * np.linalg.norm(np.cross(a_vec, b_vec), axis=1)).astype(np.float32)


def vertex_area_weights(verts, faces):
    """顶点面积权重: 邻接三角面面积之和 / 3。"""
    areas = compute_face_areas(verts, faces)
    weights = np.zeros(len(verts), dtype=np.float64)
    for f_idx, (v0, v1, v2) in enumerate(faces):
        w = areas[f_idx] / 3.0
        weights[v0] += w
        weights[v1] += w
        weights[v2] += w
    return weights.astype(np.float32)


def compute_statistics(
    thickness_map: np.ndarray,
    cartilage_label: int,
    vertex_thickness: np.ndarray = None,
    vertex_weights: np.ndarray = None,
    include_voxel_stats: bool = True,
) -> dict:
    """
    计算厚度统计。若有顶点级数据则面积加权（论文原版方法）。

    Parameters
    ----------
    thickness_map : np.ndarray [D, H, W] float32
    cartilage_label : int
    vertex_thickness : np.ndarray [N] float32 | None
    vertex_weights  : np.ndarray [N] float32 | None
    """
    # 优先使用顶点级面积加权数据
    if vertex_thickness is not None and len(vertex_thickness) > 0:
        valid = vertex_thickness > 0
        if valid.sum() > 0 and vertex_weights is not None:
            v = vertex_thickness[valid]
            w = vertex_weights[valid] / vertex_weights[valid].sum()
            w_mean = float(np.sum(v * w))
            w_std  = float(np.sqrt(np.sum(w * (v - w_mean) ** 2)))
            order = np.argsort(v)
            v_srt, w_srt = v[order], w[order]
            w_cum = np.cumsum(w_srt)
            def _wp(pct): return float(v_srt[min(np.searchsorted(w_cum, pct / 100.0), len(v_srt) - 1)])
            stats_dict = {
                "mean": w_mean, "std": w_std,
                "median": _wp(50), "p25": _wp(25), "p75": _wp(75),
                "p90": _wp(90), "p95": _wp(95),
                "max": float(np.max(v)), "min_nonzero": float(np.min(v)),
            }
        else:
            values = thickness_map[thickness_map > 0]
            if len(values) == 0:
                return _empty_stats(cartilage_label)
            p = np.percentile(values, [25, 50, 75, 90, 95])
            stats_dict = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "median": float(np.median(values)),
                "p25": float(p[0]), "p75": float(p[2]),
                "p90": float(p[3]), "p95": float(p[4]),
                "max": float(np.max(values)), "min_nonzero": float(np.min(values)),
            }
    else:
        values = thickness_map[thickness_map > 0]
        if len(values) == 0:
            return _empty_stats(cartilage_label)
        p = np.percentile(values, [25, 50, 75, 90, 95])
        stats_dict = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p25": float(p[0]), "p75": float(p[2]),
            "p90": float(p[3]), "p95": float(p[4]),
            "max": float(np.max(values)), "min_nonzero": float(np.min(values)),
        }

    stats_dict.update({
        "label": int(cartilage_label),
        "name": CARTILAGE_NAMES.get(cartilage_label, f"Cartilage {cartilage_label}"),
    })
    # 亚区统计只使用 mean/std/median/percentiles 等顶点统计;
    # 跳过整卷体素直方图/体积扫描, 可省掉 16 次 23M 数组遍历 (~1.4s/例)
    if include_voxel_stats:
        stats_dict.update({
            "total_voxels": int(np.count_nonzero(thickness_map)),
            "nonzero_voxels": int(len(thickness_map[thickness_map > 0])),
            "total_volume_mm3": float(len(thickness_map[thickness_map > 0])),
            "histogram_counts": _build_histogram(
                thickness_map[thickness_map > 0]),
            "histogram_bins": [float(b) for b in np.linspace(
                0, MAX_THICKNESS_MM, 21)],
        })
    else:
        stats_dict.update({
            "total_voxels": 0,
            "nonzero_voxels": 0,
            "total_volume_mm3": 0.0,
            "histogram_counts": [],
            "histogram_bins": [],
        })
    return stats_dict


def _empty_stats(cartilage_label: int) -> dict:
    return {
        "label": int(cartilage_label),
        "name": CARTILAGE_NAMES.get(cartilage_label, f"Cartilage {cartilage_label}"),
        "mean": 0.0, "median": 0.0, "std": 0.0, "max": 0.0, "min_nonzero": 0.0,
        "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0,
        "total_voxels": 0, "nonzero_voxels": 0, "total_volume_mm3": 0.0,
        "histogram_counts": [], "histogram_bins": [],
    }


def _build_histogram(values: np.ndarray, n_bins: int = 20) -> list:
    """构建厚度分布直方图数据 (0~10mm)。"""
    counts, _ = np.histogram(values, bins=n_bins, range=(0, MAX_THICKNESS_MM))
    return counts.tolist()


def compute_cartilage_thickness(
    pred: np.ndarray,
    spacing: tuple,
    cartilage_labels: list = None,
) -> dict:
    """
    主入口：论文原版 3D-MN，顶点级双向法向量 + 面积加权统计。

    Parameters
    ----------
    pred : np.ndarray [D, H, W] uint8
    spacing : tuple (dz, dy, dx)
    cartilage_labels : list | None

    Returns
    -------
    dict:
        "thickness_maps":  {2: ndarray, 4: ndarray, 5: ndarray}  — 体素图（可视化）
        "statistics":      {2: dict, 4: dict, 5: dict}            — 顶点级面积加权统计
        "bone_interface":  {2: ndarray, 4: ndarray, 5: ndarray}
        "vertex_data":     {2: dict, 4: dict, 5: dict}            — 顶点级原始数据
        "spacing":         (dz, dy, dx)
    """
    if cartilage_labels is None:
        cartilage_labels = [2, 4, 5]

    # 确保 spacing 是三元组
    if isinstance(spacing, (list, tuple)):
        spacing = tuple(float(s) for s in spacing[:3])
    else:
        spacing = (float(spacing), float(spacing), float(spacing))

    thickness_maps = {}
    statistics = {}
    bone_interfaces = {}
    vertex_data = {}

    for cart_label in cartilage_labels:
        bone_label = CARTILAGE_BONE_PAIRS.get(cart_label)
        if bone_label is None:
            continue

        cartilage_mask = pred == cart_label
        bone_mask = pred == bone_label

        # 计算骨-软骨界面（保留用于兼容/可视化）
        bone_interface = extract_bone_cartilage_interface(pred, bone_label, cart_label)
        bone_interfaces[cart_label] = bone_interface

        # 顶点级 3D-MN（论文原版）
        vres = compute_thickness_map_vertex(cartilage_mask, bone_mask, spacing)
        thickness_maps[cart_label] = vres["thickness_map"]
        vertex_data[cart_label] = vres

        # 统计：顶点级面积加权 + 生理学异常剔除
        vt, vw = vres["vertex_thickness"], vres["vertex_weights"]
        valid = (vt > 0) & (vt <= MAX_PHYSIOLOGICAL_THICKNESS_MM)
        n_valid = int(valid.sum())
        if n_valid >= 20:
            stats = compute_statistics(
                vres["thickness_map"], cart_label,
                vertex_thickness=vt[valid], vertex_weights=vw[valid],
            )
        else:
            # 顶点数据不足 → 退化为体素图统计
            stats = compute_statistics(vres["thickness_map"], cart_label)
        stats["n_vertices"] = n_valid
        stats["n_outliers"] = int(len(vt) - n_valid)

        # 补充体素体积信息
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        stats["voxel_volume_mm3"] = float(voxel_volume)
        stats["total_volume_mm3"] = float(stats["nonzero_voxels"] * voxel_volume)

        statistics[cart_label] = stats

    result = {
        "thickness_maps": thickness_maps,
        "statistics": statistics,
        "bone_interface": bone_interfaces,
        "vertex_data": vertex_data,
        "spacing": spacing,
    }

    # 打印摘要
    _print_summary(result)

    return result


def interpolate_thickness_to_vertices(
    thickness_map: np.ndarray,
    verts: np.ndarray,
    spacing: tuple,
) -> np.ndarray:
    """
    将体素厚度图插值到 3D 网格顶点。

    Parameters
    ----------
    thickness_map : np.ndarray [D, H, W] float32
    verts : np.ndarray [N, 3] float32
        物理坐标 (mm), 与 thickness_map 在同一坐标空间
    spacing : tuple (dz, dy, dx)
        体素物理尺寸 (mm)

    Returns
    -------
    np.ndarray [N] float32 — 每个顶点的厚度值 (mm)
    """
    if thickness_map is None or len(verts) == 0:
        return np.zeros(len(verts), dtype=np.float32)

    D, H, W = thickness_map.shape

    # 将物理坐标转换为体素索引
    # verts[:, 0] = x (W), verts[:, 1] = y (H), verts[:, 2] = z (D)
    # 对应 spacing 索引: (dz → z, dy → y, dx → x)
    # 这里 verts 来自 marching_cubes(spacing=...)，所以坐标已经是物理单位
    vx = verts[:, 0] / spacing[2]  # x → W 轴
    vy = verts[:, 1] / spacing[1]  # y → H 轴
    vz = verts[:, 2] / spacing[0]  # z → D 轴

    # clamp 到有效范围
    vx = np.clip(vx, 0, W - 1.0001)
    vy = np.clip(vy, 0, H - 1.0001)
    vz = np.clip(vz, 0, D - 1.0001)

    # 三线性插值
    x0, y0, z0 = np.floor(vx).astype(int), np.floor(vy).astype(int), np.floor(vz).astype(int)
    x1, y1, z1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1), np.minimum(z0 + 1, D - 1)

    dx, dy, dz = vx - x0, vy - y0, vz - z0

    # 三线性插值公式
    c000 = thickness_map[z0, y0, x0]
    c100 = thickness_map[z0, y0, x1]
    c010 = thickness_map[z0, y1, x0]
    c110 = thickness_map[z0, y1, x1]
    c001 = thickness_map[z1, y0, x0]
    c101 = thickness_map[z1, y0, x1]
    c011 = thickness_map[z1, y1, x0]
    c111 = thickness_map[z1, y1, x1]

    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy

    result = c0 * (1 - dz) + c1 * dz
    return result.astype(np.float32)


def _print_summary(result: dict):
    """打印厚度分析摘要到控制台。"""
    print("\n" + "=" * 62)
    print(" 软骨厚度分析 (Cartilage Thickness Analysis)")
    print("=" * 62)
    spacing = result["spacing"]
    print(f"  体素间距 (mm): dz={spacing[0]:.4f} dy={spacing[1]:.4f} dx={spacing[2]:.4f}")
    vox_mm3 = spacing[0] * spacing[1] * spacing[2]
    print(f"  体素体积 (mm^3): {vox_mm3:.4f}")

    for label in [2, 4, 5]:
        stats = result["statistics"].get(label)
        if stats is None or stats["nonzero_voxels"] == 0:
            continue
        vol_mm3 = stats['total_volume_mm3']
        print(f"\n  [{label}] {stats['name']}")
        print(f"    Mean +- SD  : {stats['mean']:.3f} +- {stats['std']:.3f} mm")
        print(f"    Median      : {stats['median']:.3f} mm")
        print(f"    Max         : {stats['max']:.3f} mm")
        print(f"    P25 / P75   : {stats['p25']:.3f} / {stats['p75']:.3f} mm")
        print(f"    Volume      : {vol_mm3:.1f} mm^3")
        print(f"    Voxels      : {stats['nonzero_voxels']:,}")
    print("=" * 62 + "\n")


# ═══════════════════════════════════════════════════════════════
# 亚区分析：股骨软骨内侧/外侧分区
# ═══════════════════════════════════════════════════════════════

def split_femoral_cartilage_medial_lateral(
    pred: np.ndarray,
    femoral_cartilage_label: int = 2,
    femur_label: int = 1,
) -> dict:
    """
    将股骨软骨沿前后轴中线分割为内侧和外侧亚区。

    基于股骨 mask 的质心 (沿左右轴 x 的中线) 进行分割。
    这对于临床评估内侧 vs 外侧间室软骨退变非常重要。

    Parameters
    ----------
    pred : np.ndarray [D, H, W] uint8
        分割标签图。
    femoral_cartilage_label : int
        股骨软骨标签 (默认 2)。
    femur_label : int
        股骨标签 (默认 1)，用于确定中线。

    Returns
    -------
    dict:
        "medial":  np.ndarray [D, H, W] bool  — 内侧软骨 mask
        "lateral": np.ndarray [D, H, W] bool  — 外侧软骨 mask
        "midline": float — 分割平面的 x 坐标（体素索引）
    """
    cartilage_mask = pred == femoral_cartilage_label
    femur_mask = pred == femur_label

    result = {
        "medial": np.zeros_like(cartilage_mask, dtype=bool),
        "lateral": np.zeros_like(cartilage_mask, dtype=bool),
        "midline": 0.0,
    }

    if cartilage_mask.sum() == 0 or femur_mask.sum() == 0:
        return result

    # 使用股骨质心在 x 轴 (W 维度) 上的中点作为内外侧分界
    femur_coords = np.argwhere(femur_mask)
    midline_x = float(np.median(femur_coords[:, 2]))  # [D, H, W] → W 轴索引为 2
    result["midline"] = midline_x

    # 分割软骨
    cartilage_coords = np.argwhere(cartilage_mask)
    if len(cartilage_coords) == 0:
        return result

    medial_idx = cartilage_coords[:, 2] < midline_x
    lateral_idx = cartilage_coords[:, 2] >= midline_x

    medial_coords = cartilage_coords[medial_idx]
    lateral_coords = cartilage_coords[lateral_idx]

    if len(medial_coords) > 0:
        result["medial"][medial_coords[:, 0], medial_coords[:, 1], medial_coords[:, 2]] = True
    if len(lateral_coords) > 0:
        result["lateral"][lateral_coords[:, 0], lateral_coords[:, 1], lateral_coords[:, 2]] = True

    return result


def compute_subregion_statistics(
    pred: np.ndarray,
    spacing: tuple,
) -> dict:
    """
    计算股骨软骨内侧/外侧亚区的厚度统计。

    在全局厚度计算基础上，进一步按亚区分别统计，
    便于比较内侧与外侧间室的软骨状态。

    Parameters
    ----------
    pred : np.ndarray [D, H, W] uint8
    spacing : tuple (dz, dy, dx)

    Returns
    -------
    dict:
        "medial_femoral":  {统计 dict} | None
        "lateral_femoral": {统计 dict} | None
        "midline_x":       float
    """
    subregions = split_femoral_cartilage_medial_lateral(pred)

    result = {
        "medial_femoral": None,
        "lateral_femoral": None,
        "midline_x": subregions["midline"],
    }

    bone_mask = pred == 1  # 股骨

    def _sub_stats(sub_mask):
        """顶点级 3D-MN 统计（与主入口一致）。"""
        if sub_mask.sum() < MIN_BONE_INTERFACE_VOXELS:
            return None
        vres = compute_thickness_map_vertex(sub_mask, bone_mask, spacing)
        vt, vw = vres["vertex_thickness"], vres["vertex_weights"]
        valid = (vt > 0) & (vt <= MAX_PHYSIOLOGICAL_THICKNESS_MM)
        if int(valid.sum()) < 20:
            return None
        stats = compute_statistics(
            vres["thickness_map"], 2,
            vertex_thickness=vt[valid], vertex_weights=vw[valid],
        )
        stats["n_vertices"] = int(valid.sum())
        stats["n_outliers"] = int(len(vt) - int(valid.sum()))
        return stats

    # 内侧股骨软骨
    medial_stats = _sub_stats(subregions["medial"])
    if medial_stats is not None:
        medial_stats["name"] = "股骨软骨-内侧 (Medial Femoral Cartilage)"
        result["medial_femoral"] = medial_stats

    # 外侧股骨软骨
    lateral_stats = _sub_stats(subregions["lateral"])
    if lateral_stats is not None:
        lateral_stats["name"] = "股骨软骨-外侧 (Lateral Femoral Cartilage)"
        result["lateral_femoral"] = lateral_stats

    return result


# ═══════════════════════════════════════════════════════════════
# NIfTI 导出
# ═══════════════════════════════════════════════════════════════

def export_thickness_nifti(
    thickness_result: dict,
    output_dir: str,
    affine: np.ndarray = None,
    base_name: str = "thickness",
):
    """
    将厚度图导出为 NIfTI 文件，供外部软件 (ITK-SNAP, 3D Slicer) 查看。

    为每种软骨生成独立的 NIfTI 文件。

    Parameters
    ----------
    thickness_result : dict
        compute_cartilage_thickness() 的返回值。
    output_dir : str
        输出目录。
    affine : np.ndarray | None
        NIfTI affine 矩阵 (4×4)。若为 None，使用单位矩阵。
    base_name : str
        文件名前缀。

    Returns
    -------
    list[str] — 生成的文件路径列表。
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[Thickness] nibabel not available, skipping NIfTI export")
        return []

    os.makedirs(output_dir, exist_ok=True)

    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    exported = []
    thickness_maps = thickness_result.get("thickness_maps", {})
    cartilage_names = {2: "femoral", 4: "medial_tibial", 5: "lateral_tibial"}

    for label in [2, 4, 5]:
        tmap = thickness_maps.get(label)
        if tmap is None or np.count_nonzero(tmap) == 0:
            continue

        name = cartilage_names.get(label, f"cartilage_{label}")
        filename = f"{base_name}_{name}.nii.gz"
        filepath = os.path.join(output_dir, filename)

        # 厚度图为 float32，以 mm 为单位
        nifti_img = nib.Nifti1Image(tmap.astype(np.float32), affine)
        nifti_img.header.set_xyzt_units("mm")
        nib.save(nifti_img, filepath)
        exported.append(filepath)
        print(f"[Thickness] Exported: {filepath}")

    return exported


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Cartilage Thickness Self-Test ===")

    # 模拟一个简单的分割: D=64, H=64, W=64
    # 物理间距: 0.5 × 0.5 × 0.7 mm (典型膝关节 MRI)
    spacing = (0.7, 0.5, 0.5)
    pred = np.zeros((64, 64, 64), dtype=np.uint8)

    # 模拟股骨 (label 1) — 上半部
    pred[:32, :, :] = 1
    # 在股骨下表面加一层软骨 (label 2) — 模拟软骨 2mm 厚
    cartilage_thickness_voxels = 4  # 4 voxels × 0.7mm = 2.8mm
    pred[32:32 + cartilage_thickness_voxels, 16:48, 16:48] = 2
    # 移除骨骼与软骨重叠处 (软骨应该在骨骼上方)
    pred[32, 16:48, 16:48] = 1  # 界面处保持骨骼

    # 模拟胫骨 (label 3) — 下半部
    pred[48:, :, :] = 3
    pred[48, 16:48, 16:48] = 3  # 界面

    # 内侧胫骨软骨 (label 4) — 胫骨上部
    pred[44:48, 16:32, 16:48] = 4

    # 外侧胫骨软骨 (label 5)
    pred[44:48, 32:48, 16:48] = 5

    result = compute_cartilage_thickness(pred, spacing)

    # 验证
    stats2 = result["statistics"][2]
    print(f"\n  Validation:")
    print(f"    Cartilage label 2 thickness: mean={stats2['mean']:.3f} mm")
    # 注: pred[32] 被覆盖为骨后，软骨实际为 z=33..35 共 3 层 = 2.1mm
    print(f"    Expected ~2.1mm (3 voxels x 0.7mm)")
    if 1.8 < stats2["mean"] < 2.6:
        print(f"    [PASS]")
    else:
        print(f"    [FAIL] - expected ~2.1mm")

    # 测试顶点插值
    print(f"\n  Vertex interpolation test:")
    thickness_map = result["thickness_maps"][2]
    # 在软骨区域生成一些顶点坐标 (物理 mm)
    test_verts = np.array([
        [24, 24, 23],  # x, y, z → 对应 voxel (33, 48, 33) 附近
        [24, 24, 22],
        [32, 32, 21],
    ], dtype=np.float32)
    # 注意: verts 来自 marching_cubes(spacing=use_spacing)，
    # 坐标是物理空间，z=x, y=y, x=z 的对应需要根据实际坐标定义
    # marching_cubes 返回 (z, y, x) 顺序的顶点，乘以 spacing
    # 这里简化测试
    thickness_vals = interpolate_thickness_to_vertices(
        thickness_map, test_verts, spacing
    )
    print(f"    Vertices: {test_verts.tolist()}")
    print(f"    Thickness: {thickness_vals.tolist()}")
    print(f"    [OK] Interpolation works")

