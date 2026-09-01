"""Dependency-light 3D surface mesh generation for OAI-ZIB labels.

This is the MCP-facing snapshot of the validated pipeline in
``Cartilage thickness estimation/utils/mesh.py``.  It intentionally has no
VTK/PyVista dependency; visualization applications can consume the exported
PLY/STL files instead.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes


MAX_SMOOTH_VERTS = 30000


def build_mesh(
    mask: np.ndarray,
    label_id: int,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    max_mc_verts: int = 120000,
    smooth: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a triangular surface mesh for one anatomical label.

    Vertices are returned in local physical coordinates (index axes multiplied
    by spacing).  The service boundary is responsible for applying the image
    direction and origin to obtain physical LPS coordinates.
    """
    if int(np.count_nonzero(mask)) < 100:
        return None, None

    use_spacing = tuple(float(value) for value in spacing)
    if smooth:
        mc_input = ndimage.gaussian_filter(
            mask.astype(np.float32),
            sigma=0.55,
            mode="nearest",
            truncate=3.0,
        )
    else:
        mc_input = mask.astype(np.float32)

    target = max(int(max_mc_verts), 3)
    dimensions = np.asarray(mc_input.shape, dtype=np.float64)
    voxel_count = float(np.sum(mc_input > 0.01))
    step_size = max(
        1,
        int(round(max(dimensions) / (target ** (1.0 / 3.0)))),
    )
    step_size = min(
        step_size,
        max(1, int(round(voxel_count ** (1.0 / 3.0) / 30))),
    )

    try:
        vertices, faces, _, _ = marching_cubes(
            mc_input,
            level=0.5,
            spacing=use_spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
    except (RuntimeError, ValueError):
        return None, None

    if len(vertices) < 3 or len(faces) < 1:
        return None, None

    if label_id == 3:
        vertices, faces = _open_distal_end(
            vertices,
            faces,
            mask,
            use_spacing,
            pred=pred,
        )
        if len(vertices) < 3 or len(faces) < 1:
            return None, None

    vertices, faces = _decimate_mesh(vertices, faces, target_ratio=0.55)
    if len(vertices) < 3 or len(faces) < 1:
        return None, None

    if smooth and len(vertices) <= MAX_SMOOTH_VERTS:
        vertices, faces = _laplacian_smooth(vertices, faces, iterations=5)
        if len(vertices) < 3 or len(faces) < 1:
            return None, None

    vertices, faces = _decimate_mesh(vertices, faces, target_ratio=0.25)
    if len(vertices) < 3 or len(faces) < 1:
        return None, None
    return vertices.astype(np.float32), faces.astype(np.int32)


def _open_distal_end(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    pred: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the artificial distal tibial end cap on the Z-min side."""
    del mask, pred
    if len(faces) < 10:
        return vertices, faces

    margin = float(spacing[2])
    mesh_tip = float(vertices[:, 2].min())
    face_extreme = vertices[faces, 2].astype(np.float64).max(axis=1)
    keep = face_extreme > (mesh_tip + margin)
    if int(keep.sum()) < 4:
        return vertices, faces
    return vertices, faces[keep]


def _decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    number_of_faces = len(faces)
    target_count = max(int(number_of_faces * (1.0 - target_ratio)), 800)
    if number_of_faces <= target_count:
        return vertices, faces

    points = vertices.astype(np.float64)
    triangles = faces.astype(np.int32)
    edge0, edge1, edge2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    all_edges = np.empty((number_of_faces * 3, 2), dtype=np.int32)
    all_edges[0::3, 0] = np.minimum(edge0, edge1)
    all_edges[0::3, 1] = np.maximum(edge0, edge1)
    all_edges[1::3, 0] = np.minimum(edge1, edge2)
    all_edges[1::3, 1] = np.maximum(edge1, edge2)
    all_edges[2::3, 0] = np.minimum(edge2, edge0)
    all_edges[2::3, 1] = np.maximum(edge2, edge0)

    deltas = points[all_edges[:, 1]] - points[all_edges[:, 0]]
    weights = 1.0 / (np.linalg.norm(deltas, axis=1) + 1e-8)
    edge_keys = (
        all_edges[:, 0].astype(np.int64) * len(points)
        + all_edges[:, 1].astype(np.int64)
    )
    sort_indices = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_indices]
    sorted_weights = weights[sort_indices]
    unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
    aggregate_weights = np.add.reduceat(sorted_weights, start_indices)
    collapse_order = np.argsort(-aggregate_weights)
    unique_edges = np.column_stack(
        [
            unique_keys[collapse_order] // len(points),
            unique_keys[collapse_order] % len(points),
        ]
    ).astype(np.int32)

    merged = np.full(len(points), -1, dtype=np.int64)
    active_mask = np.ones(len(points), dtype=bool)
    current_faces = triangles.copy()
    for first, second in unique_edges:
        if target_count >= len(current_faces):
            break
        if not active_mask[first] or not active_mask[second]:
            continue
        active_mask[second] = False
        merged[second] = first
        columns = [
            np.where(
                merged[current_faces[:, index]] >= 0,
                merged[current_faces[:, index]],
                current_faces[:, index],
            )
            for index in range(3)
        ]
        keep = (
            (columns[0] != columns[1])
            & (columns[1] != columns[2])
            & (columns[0] != columns[2])
        )
        current_faces = np.column_stack([column[keep] for column in columns])

    active = np.where(active_mask)[0]
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[active] = np.arange(len(active))
    return (
        points[active].astype(np.float32),
        remap[current_faces].astype(np.int32),
    )


def _laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    number_of_vertices = len(vertices)
    if number_of_vertices < 4 or len(faces) < 1:
        return vertices, faces

    adjacency: list[list[int]] = [[] for _ in range(number_of_vertices)]
    for triangle in faces:
        for index in range(3):
            first = int(triangle[index])
            second = int(triangle[(index + 1) % 3])
            if second not in adjacency[first]:
                adjacency[first].append(second)
            if first not in adjacency[second]:
                adjacency[second].append(first)

    points = vertices.astype(np.float64).copy()
    for iteration in range(iterations):
        neighbour_means = np.zeros_like(points)
        for index, neighbours in enumerate(adjacency):
            neighbour_means[index] = (
                points[neighbours].mean(axis=0) if neighbours else points[index]
            )
        delta = neighbour_means - points
        points = points + 0.60 * delta
        if iteration < iterations // 2:
            points = points - 0.50 * delta
    return points.astype(np.float32), faces
