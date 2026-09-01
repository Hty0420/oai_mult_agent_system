"""
subregion.py — 膝关节软骨 16 亚区划分（照 Wirth & Eckstein 2008 方法学）

划分方案（Eckstein 2015 / Wirth & Eckstein 2008）:
  - 胫骨每侧 5 亚区: 中央(cMT/cLT) + 内(iMT/iLT) + 外(eMT/eLT) + 前(aMT/aLT) + 后(pMT/pLT)
  - 负重股骨每侧 3 亚区: 中央(ccMF/ccLF) + 内(icMF/icLF) + 外(ecMF/ecLF)
  - 合计 16 亚区/膝

方法学（Wirth & Eckstein 2008, IEEE TMI 27(6):737-744）:
  1. tAB（软骨下骨面）: 骨-软骨界面，用于确定亚区边界与面积比例
  2. 胫骨: 质心 + 椭圆柱(占 tAB 面积 20%) 划分中央区
     → 剩余外周按解剖轴切分为内/外/前/后
  3. 负重股骨: 胫骨平台沿上下轴投影限定负重区（方案B）→ 33% 等分内/中/外
  4. 顶点归属: 软骨表面顶点坐标 → 亚区体积 mask（完整软骨柱）

坐标约定（OAI DESS 矢状位，pred[D,H,W]）:
  - D 轴: 左右方向（内/外），spacing[0]=0.7mm 层厚
  - H 轴: 前后方向（前/后），spacing[1]
  - W 轴: 上下方向（头-脚），spacing[2]

依赖: 本包 thickness 模块 (compute_thickness_map_vertex, compute_statistics 等)
"""
import numpy as np
from scipy import ndimage

from .thickness import (
    compute_thickness_map_vertex,
    compute_face_areas,
    compute_statistics,
)

# 亚区命名
ALL_SUBREGIONS = [
    "cMT", "iMT", "eMT", "aMT", "pMT",
    "cLT", "iLT", "eLT", "aLT", "pLT",
    "ccMF", "icMF", "ecMF", "ccLF", "icLF", "ecLF",
]

# 中央区表面积比例（论文标准）
TIBIAL_CENTRAL_FRACTION = 0.20
FEMORAL_SUBREGION_FRACTION = 1.0 / 3.0

# 解剖轴索引: D=0(内外), H=1(前后), W=2(上下)
AXIS_LR, AXIS_AP, AXIS_SI = 0, 1, 2


# ═══════════════════════════════════════════════════════════════
# P1: tAB 重建
# ═══════════════════════════════════════════════════════════════

def extract_bone_cartilage_interface_mask(
    bone_mask: np.ndarray, cartilage_mask: np.ndarray
) -> np.ndarray:
    """骨-软骨界面 mask（骨体素中与软骨相邻者）。

    性能: 只对软骨 bbox+1 的小块做二值膨胀, 不再对整卷做
    binary_dilation (旧实现 4 次调用合计约 2.7s)。
    """
    if bone_mask.sum() == 0 or cartilage_mask.sum() == 0:
        return np.zeros_like(bone_mask, dtype=bool)
    nz = np.argwhere(cartilage_mask)
    lo = np.maximum(nz.min(axis=0) - 1, 0)
    hi = np.minimum(nz.max(axis=0) + 2, np.asarray(bone_mask.shape))
    struct = ndimage.generate_binary_structure(3, 1)
    cart_crop = cartilage_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    cart_dil = ndimage.binary_dilation(
        cart_crop, structure=struct, iterations=1)
    out = np.zeros_like(bone_mask, dtype=bool)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (
        bone_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & cart_dil)
    return out


def reconstruct_tAB(
    bone_mask: np.ndarray,
    cartilage_mask: np.ndarray,
    spacing: tuple,
) -> tuple:
    """重建软骨下骨面 (tAB) 三角网格。返回 (verts 物理坐标, faces)。

    性能: 直接裁剪软骨 bbox 后做局部膨胀+marching_cubes,
    避免生成/扫描整卷 interface mask。
    """
    if bone_mask.sum() == 0 or cartilage_mask.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    try:
        nz = np.argwhere(cartilage_mask)
        lo = np.maximum(nz.min(axis=0) - 1, 0)
        hi = np.minimum(nz.max(axis=0) + 2, np.asarray(bone_mask.shape))
        struct = ndimage.generate_binary_structure(3, 1)
        cart_crop = cartilage_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        cart_dil = ndimage.binary_dilation(
            cart_crop, structure=struct, iterations=1)
        interface_crop = (
            bone_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & cart_dil)
        if int(interface_crop.sum()) < 50:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        verts, faces, _, _ = marching_cubes(
            interface_crop.astype(np.float32), level=0.5, spacing=spacing)
        if len(verts):
            verts = verts + lo * np.asarray(spacing, dtype=np.float64)
    except Exception:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return verts.astype(np.float64), faces.astype(np.int64)


def _tAB_vertex_weights(verts, faces) -> np.ndarray:
    """tAB 顶点面积权重（邻接三角形面积/3）。"""
    if len(verts) == 0 or len(faces) == 0:
        return np.zeros(len(verts), dtype=np.float64)
    areas = compute_face_areas(verts, faces)
    weights = np.zeros(len(verts), dtype=np.float64)
    for f_idx, (v0, v1, v2) in enumerate(faces):
        w = areas[f_idx] / 3.0
        weights[v0] += w
        weights[v1] += w
        weights[v2] += w
    return weights


# ═══════════════════════════════════════════════════════════════
# P2: 胫骨 5 亚区
# ═══════════════════════════════════════════════════════════════

def _ellipsoid_central_threshold(
    verts: np.ndarray, weights: np.ndarray, fraction: float,
    ap_idx: int, lr_idx: int,
) -> float:
    """
    用 tAB 顶点在 (前后, 内外) 平面上的椭圆距离，迭代求中央区阈值。

    归一化椭圆距离: d = sqrt((x_ap/s_ap)^2 + (x_lr/s_lr)^2)，s=各自标准差。
    返回 d_thr 使得 d<=d_thr 的顶点面积累计 = fraction*total。
    """
    if len(verts) == 0 or weights.sum() <= 0:
        return 1.0
    x_ap = verts[:, ap_idx] - verts[:, ap_idx].mean()
    x_lr = verts[:, lr_idx] - verts[:, lr_idx].mean()
    s_ap = max(x_ap.std(), 1e-6)
    s_lr = max(x_lr.std(), 1e-6)
    d = np.sqrt((x_ap / s_ap) ** 2 + (x_lr / s_lr) ** 2)

    total = weights.sum()
    lo, hi = 0.0, float(d.max())
    for _ in range(40):
        mid = (lo + hi) / 2.0
        frac = weights[d <= mid].sum() / total
        if frac < fraction:
            lo = mid
        else:
            hi = mid
    return float(hi)


def tibial_subregions(
    cartilage_mask: np.ndarray,
    bone_mask: np.ndarray,
    spacing: tuple,
    inward_direction: int = 1,
) -> dict:
    """
    胫骨 5 亚区（单侧）: c + i/e + a/p。

    解剖轴: D=内外, H=前后。中央区 = 椭圆柱（占 tAB 面积 20%），
    外周按 H 分前后(a/p)、D 分内外(i/e)。返回完整软骨体积 mask。

    inward_direction: 本侧"内侧(i)"在 D 轴上的符号方向 (+1 或 -1)。
      Wirth2008 定义 i = 朝向对侧胫骨平台 (关节中心) 的亚区, e = 远离。
      内侧胫骨 (MT) 与外侧胫骨 (LT) 的 i 方向相反 → 必须传入各自方向,
      否则两侧 i/e 同向, 视觉上"两块划分一样"而非镜像对称。
      默认 +1 = D 增大方向为内侧 (兼容旧调用, 但 compute_subregion_thickness
      会按两侧质心显式传参)。

    Returns: {"c","i","e","a","p"} bool mask [D,H,W]
    """
    D, H, W = cartilage_mask.shape
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    empty = {k: np.zeros((D, H, W), dtype=bool) for k in ["c", "i", "e", "a", "p"]}

    tAB_verts, tAB_faces = reconstruct_tAB(bone_mask, cartilage_mask, spacing_arr)
    if len(tAB_verts) == 0:
        return empty
    weights = _tAB_vertex_weights(tAB_verts, tAB_faces)

    d_thr = _ellipsoid_central_threshold(tAB_verts, weights, TIBIAL_CENTRAL_FRACTION,
                                         AXIS_AP, AXIS_LR)

    coords = np.argwhere(cartilage_mask)
    if len(coords) == 0:
        return empty
    c_ap = coords[:, AXIS_AP].mean()
    c_lr = coords[:, AXIS_LR].mean()

    # 软骨体素 → 亚区（矢量化）
    # 中央区: 椭圆柱内 (d <= d_thr)
    # 外周: 4 个互斥区域, 由两个正交平面切分 (Wirth2008 原文方法)
    # v33 性能: 坐标式 — 只处理软骨体素 (~2 万), 不用全图 meshgrid/广播
    # (160×384×384 ≈ 7000 万体素, 是之前 subregion 慢的主因)
    ap = (coords[:, AXIS_AP] - c_ap) * spacing_arr[AXIS_AP]   # 前后 (H 轴)
    lr = (coords[:, AXIS_LR] - c_lr) * spacing_arr[AXIS_LR]   # 内外 (D 轴)
    s_ap = max(np.std(tAB_verts[:, AXIS_AP]), 1e-6)
    s_lr = max(np.std(tAB_verts[:, AXIS_LR]), 1e-6)
    ap_n = ap / s_ap
    lr_n = lr / s_lr
    d = np.sqrt(ap_n ** 2 + lr_n ** 2)

    is_central = d <= d_thr
    is_outer = ~is_central
    # 外周四带: 前/后/内/外 (Wirth 两平面方向带, 每体素归主方向)
    # v27 缝隙修复: p 含等号 + e 取补集 → 四带互斥且全覆盖 (任何体素必归一带)
    sel_a = is_outer & (ap_n > 0) & (np.abs(ap_n) >= np.abs(lr_n))
    sel_p = is_outer & (ap_n <= 0) & (np.abs(ap_n) >= np.abs(lr_n))
    sel_i = is_outer & (lr_n > 0) & (np.abs(lr_n) > np.abs(ap_n))
    sel_e = is_outer & ~(sel_a | sel_p | sel_i)  # 补集: 全覆盖
    # i/e 方向: i 朝向对侧胫骨平台 (关节中心), e 背离。
    if inward_direction < 0:
        sel_i, sel_e = sel_e, sel_i

    def _to_mask(sel):
        m = np.zeros(cartilage_mask.shape, dtype=bool)
        c = coords[sel]
        if len(c):
            m[c[:, 0], c[:, 1], c[:, 2]] = True
        return m

    return {"c": _to_mask(is_central), "i": _to_mask(sel_i),
            "e": _to_mask(sel_e), "a": _to_mask(sel_a), "p": _to_mask(sel_p)}


# ═══════════════════════════════════════════════════════════════
# P3: 负重股骨 3 亚区（60% 距离准则, Chondrometrics OAI 标准）
# ═══════════════════════════════════════════════════════════════

def _femoral_condyle_cutoff_h(
    side_mask: np.ndarray,
) -> tuple:
    """
    单个股骨髁(内或外)的负重区前后界 [H 轴]。

    60% 准则 (Eckstein 综述 PMC3200067; Framingham PMC2982217):
      "The weight-bearing region of the femoral condyles was analyzed
       between the intercondylar notch and 60% of the distance to the
       posterior end of the femoral condyles"

    实现 (2026-08-14 修正, 对照 Wirth & Eckstein 2008 解剖):
      1. 下表面轮廓: 每个 H 位置软骨下表面 (最小 W, 关节面) — 旧版用均值,
         且切迹检测用"下降段曲率峰" (≈后髁面最陡处, H≈105) 而非髁间切迹
      2. 髁间切迹 = 下表面轮廓谷底 (最小 W 处) — 解剖上切迹是髁面最凹陷处
      3. 后缘 = 轮廓最后方 (最小 H = 股骨髁后缘) —
         旧版误用最大 H (滑车前缘) → 负重区错误延伸到滑车/中间连接
      4. 负重区 = [切迹, 切迹 + 0.6*(后缘-切迹)] (向后方延伸)

    Returns: (notch_h, posterior_h) — H 轴体素坐标 (posterior_h < notch_h)。
    """
    import collections
    from scipy.ndimage import gaussian_filter1d

    idx = np.argwhere(side_mask)
    if len(idx) < 50:
        return None
    h_bottom = collections.defaultdict(list)
    for d, h, w in idx:
        h_bottom[h].append(w)
    hs = sorted(h_bottom.keys())
    if len(hs) < 10:
        return None
    hs_arr = np.array(hs, dtype=float)
    # 下表面轮廓: 每个 H 位置软骨的最小 W (关节面, 最凹陷处)
    prof = np.array([float(np.min(h_bottom[h])) for h in hs])
    prof_s = gaussian_filter1d(prof, sigma=3.0)

    # 髁间切迹 = 下表面轮廓谷底 (最小 W)
    # (髁面从后缘向切迹下降, 过切迹后向滑车回升 → 谷底即切迹)
    notch_idx = int(np.argmin(prof_s))
    # 髁后缘 = 最小 H (最后方), 不是最大 H!
    return float(hs_arr[notch_idx]), float(hs_arr[0])


def femoral_weight_bearing_mask(
    femoral_cartilage_mask: np.ndarray,
    tibial_cartilage_mask: np.ndarray = None,
) -> np.ndarray:
    """
    负重股骨 mask: 60% 距离准则 (Chondrometrics OAI 标准)。

    文献依据 (Eckstein 2011 综述 / Framingham 2010):
      负重区 = 髁间切迹 → 股骨髁后缘距离的 60%;
      内侧/外侧髁分别计算; cMF 约占股骨 tAB 36%±1.8%。

    参数:
      femoral_cartilage_mask: label 2 (完整股骨软骨) [D,H,W] bool
      tibial_cartilage_mask: 保留参数(向后兼容), 不再使用。
        旧方案 B (胫骨投影) 已被 60% 距离准则取代。

    Returns: bool mask [D,H,W] — 负重区软骨。
    """
    fem = np.asarray(femoral_cartilage_mask, dtype=bool)
    wb = np.zeros_like(fem)

    # 内侧/外侧分别计算 (文献标准): 沿 D 轴(内外)中位切分
    coords = np.argwhere(fem)
    if len(coords) == 0:
        return wb
    mid_d = np.median(coords[:, AXIS_LR])
    # v33 性能: 坐标式 (只处理软骨体素, 不用全图 d[:,None,None]/h[None,:,None] 广播)
    d_idx = coords[:, AXIS_LR]
    h_idx = coords[:, AXIS_AP]

    def _to_mask(sel):
        m = np.zeros_like(fem)
        c = coords[sel]
        if len(c):
            m[c[:, 0], c[:, 1], c[:, 2]] = True
        return m

    for side_sel in (d_idx <= mid_d, d_idx > mid_d):
        side = _to_mask(side_sel)
        bounds = _femoral_condyle_cutoff_h(side)
        if bounds is None:
            continue
        notch_h, post_h = bounds
        # 负重区 = [切迹, 切迹 + 0.6*(后缘-切迹)] — 向后方 (H 减小方向) 延伸
        # post_h < notch_h (后缘在切迹后方) → cutoff < notch_h
        cutoff = notch_h + 0.6 * (post_h - notch_h)
        wb |= _to_mask(side_sel & (h_idx <= notch_h) & (h_idx >= cutoff))

    # 剔除中间连接 (滑车沟底): 负重区应为左右两个分离的髁。
    # 60% 区间可能把 D 中位附近的滑车沟底软骨 (H≈切迹、W偏高) 也算入,
    # 但解剖上负重区不跨过髁间切迹 → 用连通域保留最大两块。
    if wb.any():
        lbl, n_lbl = ndimage.label(wb)
        if n_lbl > 2:
            sizes = ndimage.sum(wb, lbl, index=np.arange(1, n_lbl + 1))
            keep = np.argsort(sizes)[-2:] + 1  # 两个最大连通域
            wb = np.isin(lbl, keep)
    return wb


def femoral_subregions(
    weight_bearing_mask: np.ndarray,
    bone_mask: np.ndarray,
    spacing: tuple,
    flip_ic_ec: bool = False,
) -> dict:
    """
    负重股骨 3 亚区: 沿左右轴(D)按 tAB 面积 33% 等分 → ic/cc/ec。

    方法学 (Wirth & Eckstein 2008, IEEE TMI 27(6):737):
      - 三角条带沿内→外(D轴)排列, 条带前后边序列分 3 等份
      - 内/中/外分类依据到对侧股骨区域的距离 (ic=靠近对侧/髁间窝,
        ec=远离对侧/髁外侧缘)
      - 各亚区占负重区 tAB 面积 33%

    flip_ic_ec: 内侧髁 (MF) 传 True。内侧髁位于 D 小侧, 其对侧 (外侧髁/
      髁间窝) 在 D 增大方向 → ic 应朝 D 大端, 与外侧髁镜像。
      旧实现两侧同向 → MF 的 ic/ec 反了 (蓝绿贴错边)。

    返回: {"cc","ic","ec"} bool mask [D,H,W]（完整软骨体积）
    """
    D, H, W = weight_bearing_mask.shape
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    empty = {k: np.zeros((D, H, W), dtype=bool) for k in ["cc", "ic", "ec"]}

    tAB_verts, tAB_faces = reconstruct_tAB(bone_mask, weight_bearing_mask, spacing_arr)
    if len(tAB_verts) == 0:
        return empty
    weights = _tAB_vertex_weights(tAB_verts, tAB_faces)
    total = weights.sum()
    if total <= 0:
        return empty

    # 沿左右轴(D) 按 tAB 面积 33% 等分 (Wirth2008: 条带沿内→外)
    x = tAB_verts[:, AXIS_LR]
    order = np.argsort(x)
    cum = np.cumsum(weights[order])
    q1 = np.searchsorted(cum, FEMORAL_SUBREGION_FRACTION * total)
    q2 = np.searchsorted(cum, 2 * FEMORAL_SUBREGION_FRACTION * total)
    q1 = min(max(q1, 0), len(order) - 1)
    q2 = min(max(q2, q1 + 1), len(order) - 1)
    b1 = x[order[q1]]
    b2 = x[order[q2]]

    # v33 性能: 坐标式 (负重区体素 ~1 万, 不用全图 lr_phys[:,None,None] 广播)
    wb_coords = np.argwhere(weight_bearing_mask)
    if len(wb_coords) == 0:
        return empty
    lr_phys = (wb_coords[:, AXIS_LR] + 0.5) * spacing_arr[AXIS_LR]

    def _to_mask(sel):
        m = np.zeros(weight_bearing_mask.shape, dtype=bool)
        c = wb_coords[sel]
        if len(c):
            m[c[:, 0], c[:, 1], c[:, 2]] = True
        return m

    inner = _to_mask(lr_phys <= b1)
    central = _to_mask((lr_phys > b1) & (lr_phys <= b2))
    outer = _to_mask(lr_phys > b2)
    if flip_ic_ec:
        # 内侧髁: 对侧 (髁间窝) 在 D 大端 → ic 在 D 大端, ec 在 D 小端
        inner, outer = outer, inner
    return {"cc": central, "ic": inner, "ec": outer}


# ═══════════════════════════════════════════════════════════════
# P4: 顶点归属 + 16 亚区统计
# ═══════════════════════════════════════════════════════════════

def _assign_verts_to_mask(verts, mask, spacing):
    """顶点(物理坐标)是否落在体积 mask 内。"""
    if len(verts) == 0:
        return np.zeros(0, dtype=bool)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    idx = np.floor(verts / spacing_arr).astype(np.int64)
    D, H, W = mask.shape
    ok = (
        (idx[:, 0] >= 0) & (idx[:, 0] < D) &
        (idx[:, 1] >= 0) & (idx[:, 1] < H) &
        (idx[:, 2] >= 0) & (idx[:, 2] < W)
    )
    inside = np.zeros(len(verts), dtype=bool)
    inside[ok] = mask[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return inside


def _split_femoral_medial_lateral(wb: np.ndarray) -> tuple:
    """负重股骨沿左右轴(D)分内外侧（按中位 D）。"""
    D = wb.shape[0]
    coords = np.argwhere(wb)
    if len(coords) == 0:
        return np.zeros_like(wb), np.zeros_like(wb)
    mid = float(np.median(coords[:, AXIS_LR]))
    medial = wb & (np.arange(D)[:, None, None] <= mid)
    lateral = wb & (np.arange(D)[:, None, None] > mid)
    return medial, lateral


def compute_subregion_thickness(
    pred: np.ndarray,
    spacing: tuple,
    vertex_data: dict = None,
) -> dict:
    """
    主入口: 计算 16 亚区的顶点级 3D-MN 厚度统计。

    vertex_data: 可选 {label: vres} — `compute_cartilage_thickness` 返回的
        vertex_data。提供时复用其顶点级 3D-MN 结果 (跳过重复的
        compute_thickness_map_vertex): 每例 7 次 3D-MN → 3 次 (2.3×提速)。
        胫骨 (label 4/5) 复用完全相同; 股骨 (label 2) 复用整板顶点后按
        负重区 mask 归属 (数值与子集网格略异但更准确, 边界法线来自完整表面)。

    Returns:
        "subregions": {name: stats_dict}
        "vertex_counts": {name: int}
        "masks": {name: bool mask}
    """
    D, H, W = pred.shape
    stats_out = {}
    counts = {}
    masks_out = {}
    plate_means = {}
    vertex_data = vertex_data or {}

    # ── 胫骨: label 4=MT, 5=LT ──
    # 内侧方向 (Wirth2008): i 亚区朝向对侧胫骨平台 (关节中心)。
    # MT 与 LT 的 i 方向在 D 轴上相反 → 按两侧软骨质心确定方向:
    #   i 始终指向对侧胫骨 (关节中心):
    #     若 MT 质心 D < LT 质心 D → MT 的 i 朝 D 增方向 (+1), LT 朝 D 减方向 (-1)
    mt_coords = np.argwhere(pred == 4)
    lt_coords = np.argwhere(pred == 5)
    mt_centroid_d = float(mt_coords[:, 0].mean()) if len(mt_coords) else None
    lt_centroid_d = float(lt_coords[:, 0].mean()) if len(lt_coords) else None
    inward = {}
    if mt_centroid_d is not None and lt_centroid_d is not None:
        if mt_centroid_d < lt_centroid_d:
            inward["MT"], inward["LT"] = +1, -1
        else:
            inward["MT"], inward["LT"] = -1, +1
    else:
        inward["MT"], inward["LT"] = 1, 1

    for cart_label, plate in [(4, "MT"), (5, "LT")]:
        cart = pred == cart_label
        bone = pred == 3
        sub = tibial_subregions(cart, bone, spacing, inward_direction=inward[plate])
        if cart_label in vertex_data:
            vres = vertex_data[cart_label]  # 复用 (与 compute_thickness_map_vertex 完全相同)
        else:
            vres = compute_thickness_map_vertex(cart, bone, spacing)
        vt, vw, vp = (vres["vertex_thickness"], vres["vertex_weights"],
                      vres["vertex_positions"])
        valid = (vt > 0) & (vt <= 5.0)
        if valid.sum() and vw[valid].sum() > 0:
            plate_means[plate] = float(
                np.sum(vt[valid] * vw[valid]) / np.sum(vw[valid]))
        for short, full in [("c", "c" + plate), ("i", "i" + plate),
                            ("e", "e" + plate), ("a", "a" + plate),
                            ("p", "p" + plate)]:
            m = sub[short]
            masks_out[full] = m
            sel = valid & _assign_verts_to_mask(vp, m, spacing)
            if sel.sum() >= 10:
                st = compute_statistics(vres["thickness_map"], cart_label,
                                        vertex_thickness=vt[sel],
                                        vertex_weights=vw[sel],
                                        include_voxel_stats=False)
            else:
                st = compute_statistics(vres["thickness_map"], cart_label,
                                          include_voxel_stats=False)
            st["name"] = full
            stats_out[full] = st
            counts[full] = int(sel.sum())

    # ── 股骨: label 2, 负重区(60%准则) + 内外侧 ──
    # Wirth2008: 股骨亚区只在负重区内划分 (非负重区不分析)
    fem_cart = pred == 2
    tib_all = (pred == 4) | (pred == 5)
    wb = femoral_weight_bearing_mask(fem_cart, tib_all)
    bone_femur = pred == 1
    wb_medial, wb_lateral = _split_femoral_medial_lateral(wb)

    # ic/ec 方向: ic 朝向对侧股骨 (髁间窝)。medial 块若在 D 小侧,
    # 其对侧在 D 大端 → 需翻转 ic/ec; 方向由两侧质心动态判断 (同胫骨逻辑)。
    md_coords = np.argwhere(wb_medial)
    ld_coords = np.argwhere(wb_lateral)
    md = float(md_coords[:, 0].mean()) if len(md_coords) else None
    ld = float(ld_coords[:, 0].mean()) if len(ld_coords) else None
    flip_medial = (md is not None and ld is not None and md < ld)

    for wb_mask, plate, flip in [(wb_medial, "MF", flip_medial),
                                 (wb_lateral, "LF", not flip_medial)]:
        if wb_mask.sum() < 50:
            for short in ["cc", "ic", "ec"]:
                full = short + plate
                masks_out[full] = np.zeros_like(wb)
                stats_out[full] = compute_statistics(np.zeros_like(wb), 2,
                                                  include_voxel_stats=False)
                counts[full] = 0
            continue
        sub = femoral_subregions(wb_mask, bone_femur, spacing, flip_ic_ec=flip)
        if 2 in vertex_data:
            vres = vertex_data[2]  # 复用整板 label 2 顶点, 按负重区 mask 归属
        else:
            vres = compute_thickness_map_vertex(wb_mask, bone_femur, spacing)
        vt, vw, vp = (vres["vertex_thickness"], vres["vertex_weights"],
                      vres["vertex_positions"])
        valid = (vt > 0) & (vt <= 5.0)
        plate_union = np.zeros(len(vt), dtype=bool)
        for short, full in [("cc", "cc" + plate), ("ic", "ic" + plate),
                            ("ec", "ec" + plate)]:
            m = sub[short]
            masks_out[full] = m
            sel = valid & _assign_verts_to_mask(vp, m, spacing)
            plate_union |= sel
            if sel.sum() >= 10:
                st = compute_statistics(vres["thickness_map"], 2,
                                        vertex_thickness=vt[sel],
                                        vertex_weights=vw[sel],
                                        include_voxel_stats=False)
            else:
                st = compute_statistics(vres["thickness_map"], 2,
                                          include_voxel_stats=False)
            st["name"] = full
            stats_out[full] = st
            counts[full] = int(sel.sum())
        if plate_union.sum() and vw[plate_union].sum() > 0:
            plate_means[plate] = float(
                np.sum(vt[plate_union] * vw[plate_union]) /
                np.sum(vw[plate_union]))

    return {
        "subregions": stats_out,
        "vertex_counts": counts,
        "masks": masks_out,
        "plate_means": plate_means,
    }


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Subregion Self-Test ===")
    # OAI 坐标约定: D=左右(内外,0.7mm层厚), H=前后, W=上下
    spacing = (0.7, 0.5, 0.5)
    pred = np.zeros((64, 64, 64), dtype=np.uint8)
    # 股骨（W 上半）
    pred[:, :, :32] = 1
    # 股骨软骨（W 32-36, D/H 16-48）
    pred[16:48, 16:48, 32:36] = 2
    pred[16:48, 16:48, 32] = 1  # 界面保持骨
    # 胫骨（W 下半）
    pred[:, :, 48:] = 3
    pred[16:48, 16:48, 48] = 3
    # 内侧胫骨软骨 label 4（D 16-32 内侧, W 44-48）
    pred[16:32, 16:48, 44:48] = 4
    # 外侧胫骨软骨 label 5（D 32-48 外侧, W 44-48）
    pred[32:48, 16:48, 44:48] = 5

    result = compute_subregion_thickness(pred, spacing)
    print("亚区数:", len(result["subregions"]))
    for name in ALL_SUBREGIONS:
        st = result["subregions"].get(name)
        cnt = result["vertex_counts"].get(name, 0)
        if st and st.get("nonzero_voxels", 0) > 0:
            print(f"  {name}: mean={st['mean']:.3f} n_verts={cnt}")

