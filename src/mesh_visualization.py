"""Safe, dependency-light helpers for interactive cartilage mesh previews."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
from typing import Any, Iterable

import numpy as np


MESH_LABEL_NAMES_ZH = {
    1: "股骨",
    2: "股骨软骨",
    3: "胫骨",
    4: "内侧胫骨软骨",
    5: "外侧胫骨软骨",
}

_OUTPUT_EXTENSIONS = (
    "nii.gz", "nii", "csv", "txt", "json", "png", "jpg", "jpeg",
    "gif", "md", "zip", "npz", "ply", "stl",
)
_OUTPUT_EXTENSION_PATTERN = "|".join(
    re.escape(extension) for extension in _OUTPUT_EXTENSIONS
)
_DIRECT_OUTPUT_KEYS = (
    "file_path", "output_csv", "filtered_csv", "report_path", "model_path",
    "manifest_path", "global_csv", "subregion_csv", "analysis_json",
    "subregion_mask",
)


class UIRefreshTracker:
    """Claim changing UI identities once per browser page."""

    def __init__(self, max_clients: int = 512):
        self._max_clients = max(1, int(max_clients))
        self._identities: dict[str, Any] = {}
        self._lock = threading.Lock()

    def claim(self, client_id: str, identity: Any) -> bool:
        key = str(client_id or "anonymous")
        with self._lock:
            if key in self._identities and self._identities[key] == identity:
                return False
            self._identities.pop(key, None)
            self._identities[key] = identity
            while len(self._identities) > self._max_clients:
                oldest = next(iter(self._identities))
                self._identities.pop(oldest, None)
        return True


class MeshRefreshTracker:
    """Suppress duplicate viewer refreshes for each browser page.

    Gradio events can overlap and may carry a stale component-state value.  A
    server-side page/session identity therefore provides a second guard against
    repeatedly sending the same 3D file back to the browser.
    """

    def __init__(self, max_clients: int = 512):
        self._max_clients = max(1, int(max_clients))
        self._identities: dict[str, tuple[str, tuple[Any, ...] | None]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _manifest_identity(manifest_path: str | None) -> tuple[Any, ...] | None:
        if not manifest_path:
            return None
        path = Path(str(manifest_path)).expanduser()
        try:
            resolved = path.resolve()
            stat = resolved.stat()
            return (
                os.path.normcase(str(resolved)),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except OSError:
            return (os.path.normcase(os.path.abspath(str(path))),)

    def _identity(
        self,
        session_id: str | None,
        manifest_path: str | None,
    ) -> tuple[str, tuple[Any, ...] | None]:
        return str(session_id or ""), self._manifest_identity(manifest_path)

    def claim(
        self,
        client_id: str,
        session_id: str | None,
        manifest_path: str | None,
    ) -> bool:
        """Return true exactly when this page needs a viewer-state update."""
        key = str(client_id or "anonymous")
        identity = self._identity(session_id, manifest_path)
        with self._lock:
            if self._identities.get(key) == identity:
                return False
            self._identities.pop(key, None)
            self._identities[key] = identity
            while len(self._identities) > self._max_clients:
                oldest = next(iter(self._identities))
                self._identities.pop(oldest, None)
        return True

    def rollback(
        self,
        client_id: str,
        session_id: str | None,
        manifest_path: str | None,
    ) -> None:
        """Allow a failed refresh to be retried by the next polling cycle."""
        key = str(client_id or "anonymous")
        identity = self._identity(session_id, manifest_path)
        with self._lock:
            if self._identities.get(key) == identity:
                self._identities.pop(key, None)


def _normalized_existing_file(value: str) -> str | None:
    candidate = str(value or "").strip().strip("\"'` ").rstrip(",;，。)")
    if not candidate:
        return None
    candidate = candidate.replace("\\\\", "\\")
    path = Path(candidate).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return str(resolved) if resolved.is_file() else None


def collect_structured_output_files(response: str) -> list[str]:
    """Collect existing output artifacts without treating input paths as outputs."""
    text = str(response or "")
    candidates: list[str] = []
    quoted_path_pattern = re.compile(
        rf"[\"']([^\"'\r\n]+?\.(?:{_OUTPUT_EXTENSION_PATTERN}))[\"']",
        re.IGNORECASE,
    )

    files_list_pattern = re.compile(
        r"[\"']files[\"']\s*:\s*\[(.*?)\]",
        re.IGNORECASE | re.DOTALL,
    )
    for match in files_list_pattern.finditer(text):
        candidates.extend(quoted_path_pattern.findall(match.group(1)))

    keys = "|".join(re.escape(key) for key in _DIRECT_OUTPUT_KEYS)
    direct_pattern = re.compile(
        rf"[\"']?(?:{keys})[\"']?\s*[:=]\s*"
        rf"[\"']([^\"'\r\n]+?\.(?:{_OUTPUT_EXTENSION_PATTERN}))[\"']",
        re.IGNORECASE,
    )
    candidates.extend(direct_pattern.findall(text))

    results: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        existing = _normalized_existing_file(candidate)
        if not existing:
            continue
        identity = os.path.normcase(existing)
        if identity not in seen:
            seen.add(identity)
            results.append(existing)
    return results


def find_mesh_manifest(
    output_files: Iterable[str],
    response: str = "",
) -> str | None:
    candidates = list(output_files or [])
    candidates.extend(collect_structured_output_files(response))
    manifests: list[Path] = []
    seen: set[str] = set()
    for value in candidates:
        existing = _normalized_existing_file(str(value))
        if not existing:
            continue
        path = Path(existing)
        if path.name.lower() != "cartilage_3d_mesh_manifest.json":
            continue
        identity = os.path.normcase(str(path))
        if identity not in seen:
            seen.add(identity)
            manifests.append(path)
    if not manifests:
        return None
    return str(max(manifests, key=lambda path: path.stat().st_mtime_ns))


def load_mesh_manifest(manifest_path: str) -> dict[str, Any]:
    path = Path(str(manifest_path)).expanduser().resolve()
    if not path.is_file() or path.name.lower() != "cartilage_3d_mesh_manifest.json":
        raise ValueError("三维网格清单不存在或文件名不正确。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取三维网格清单: {exc}") from exc
    meshes = payload.get("meshes")
    if not isinstance(meshes, dict) or not meshes:
        raise ValueError("三维网格清单中没有可显示的结构。")
    payload["_manifest_path"] = str(path)
    return payload


def mesh_structure_choices(manifest_path: str) -> list[tuple[str, str]]:
    payload = load_mesh_manifest(manifest_path)
    choices: list[tuple[str, str]] = []
    for raw_label in sorted(payload["meshes"], key=lambda value: int(value)):
        label_id = int(raw_label)
        if label_id not in MESH_LABEL_NAMES_ZH:
            continue
        choices.append((MESH_LABEL_NAMES_ZH[label_id], str(label_id)))
    return choices


def _validated_npz_path(manifest: dict[str, Any], mesh_record: dict[str, Any]) -> Path:
    manifest_path = Path(manifest["_manifest_path"])
    files = mesh_record.get("files")
    npz_value = files.get("npz") if isinstance(files, dict) else None
    if not isinstance(npz_value, str):
        raise ValueError("网格缺少 NPZ 预览数据；请重新生成并保留 NPZ 格式。")
    npz_path = Path(npz_value).expanduser().resolve()
    if npz_path.parent != manifest_path.parent or not npz_path.is_file():
        raise ValueError("网格 NPZ 路径超出当前结果目录或文件不存在。")
    return npz_path


def _load_mesh_arrays(
    manifest: dict[str, Any],
    mesh_record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = _validated_npz_path(manifest, mesh_record)
    try:
        with np.load(path, allow_pickle=False) as data:
            vertices = np.asarray(
                data["vertices_lps_mm"] if "vertices_lps_mm" in data else data["vertices"],
                dtype=np.float32,
            )
            faces = np.asarray(data["faces"], dtype=np.int32)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"无法读取三维网格数据 {path.name}: {exc}") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{path.name} 的顶点数组无效。")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError(f"{path.name} 的三角面数组无效。")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path.name} 包含非有限顶点。")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"{path.name} 的三角面索引越界。")
    return vertices, faces


def _smooth_preview_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    """Approximate the reference VTK surface smoothing without VTK.

    Only the cached browser preview is smoothed. Scientific PLY/STL/NPZ
    outputs remain byte-for-byte independent from this presentation layer.
    """
    if iterations <= 0 or len(vertices) < 4 or len(faces) < 1:
        return np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    directed_edges = np.concatenate(
        [
            triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
            triangles[:, [1, 0]], triangles[:, [2, 1]], triangles[:, [0, 2]],
        ],
        axis=0,
    )
    sources = directed_edges[:, 0]
    targets = directed_edges[:, 1]
    counts = np.bincount(sources, minlength=len(vertices)).astype(np.float64)
    counts[counts == 0] = 1.0
    points = np.asarray(vertices, dtype=np.float64).copy()

    def laplacian(values: np.ndarray) -> np.ndarray:
        neighbour_sums = np.zeros_like(values)
        np.add.at(neighbour_sums, sources, values[targets])
        return neighbour_sums / counts[:, None] - values

    # Taubin lambda/mu pairs smooth stair-step artifacts while limiting shrink.
    for _ in range(int(iterations)):
        points += 0.33 * laplacian(points)
        points -= 0.34 * laplacian(points)
    return points.astype(np.float32)


def _oriented_faces_and_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward-oriented triangles and area-weighted point normals."""
    oriented = np.asarray(faces, dtype=np.int32).copy()
    triangles = vertices[oriented]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_centres = triangles.mean(axis=1)
    centre = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    orientation_score = float(
        np.sum(face_normals * (face_centres - centre), dtype=np.float64)
    )
    if orientation_score < 0:
        oriented = oriented[:, [0, 2, 1]]
        face_normals = -face_normals

    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, oriented[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 1.0, 0.0)
    return oriented.astype(np.int32), normals.astype(np.float32)


def _write_model3d_glb(path: Path, meshes: list[dict[str, Any]]) -> None:
    """Write a self-contained GLB with smooth normals and PBR materials."""
    binary = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    gltf_meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def append_array(values: np.ndarray, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        byte_offset = len(binary)
        payload = np.ascontiguousarray(values).tobytes(order="C")
        binary.extend(payload)
        view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": len(payload),
                "target": target,
            }
        )
        return view_index

    for mesh_index, mesh in enumerate(meshes):
        vertices = _smooth_preview_vertices(mesh["vertices"], mesh["faces"])
        faces, normals = _oriented_faces_and_normals(vertices, mesh["faces"])
        positions = np.asarray(vertices, dtype="<f4")
        normal_values = np.asarray(normals, dtype="<f4")
        indices = np.asarray(faces.reshape(-1), dtype="<u4")

        position_view = append_array(positions, 34962)
        normal_view = append_array(normal_values, 34962)
        index_view = append_array(indices, 34963)
        position_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": int(len(positions)),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            }
        )
        normal_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": normal_view,
                "componentType": 5126,
                "count": int(len(normal_values)),
                "type": "VEC3",
            }
        )
        index_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": int(len(indices)),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            }
        )

        color = np.clip(np.asarray(mesh["color"], dtype=np.float64), 0, 255) / 255.0
        material_index = len(materials)
        materials.append(
            {
                "name": f"{mesh['name']}实体材质",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*color.astype(float).tolist(), 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                },
                "doubleSided": True,
            }
        )
        gltf_meshes.append(
            {
                "name": mesh["name"],
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                        },
                        "indices": index_accessor,
                        "material": material_index,
                        "mode": 4,
                    }
                ],
            }
        )
        nodes.append({"name": mesh["name"], "mesh": mesh_index})

    document = {
        "asset": {
            "version": "2.0",
            "generator": "OAI MCP cartilage reference-style PBR preview",
        },
        "scene": 0,
        "scenes": [{"name": "膝关节三维结构", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    payload = b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN\x00"),
            bytes(binary),
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build_mesh_preview(
    manifest_path: str,
    selected_labels: Iterable[str | int] | None,
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    manifest = load_mesh_manifest(manifest_path)
    choices = mesh_structure_choices(manifest_path)
    available = [value for _, value in choices]
    if selected_labels is None:
        selected = available
    else:
        selected_set = {str(value) for value in selected_labels}
        selected = [value for value in available if value in selected_set]
    if not selected:
        return {
            "preview_path": None,
            "choices": choices,
            "selected_labels": [],
            "structure_count": 0,
            "vertex_count": 0,
            "triangle_count": 0,
        }

    preview_meshes: list[dict[str, Any]] = []
    vertex_count = 0
    triangle_count = 0
    meshes = manifest["meshes"]
    for label in selected:
        record = meshes.get(label)
        if not isinstance(record, dict):
            continue
        vertices, faces = _load_mesh_arrays(manifest, record)
        color = np.asarray(record.get("color_rgb", [180, 180, 180]), dtype=np.int64)
        if color.shape != (3,):
            color = np.asarray([180, 180, 180], dtype=np.int64)
        color = np.clip(color, 0, 255).astype(np.uint8)
        preview_meshes.append(
            {
                "name": MESH_LABEL_NAMES_ZH[int(label)],
                "vertices": vertices,
                "faces": faces,
                "color": color,
            }
        )
        vertex_count += len(vertices)
        triangle_count += len(faces)

    if not preview_meshes:
        raise ValueError("所选结构没有可用于预览的网格数据。")

    manifest_file = Path(manifest["_manifest_path"])
    stat = manifest_file.stat()
    identity = "|".join(
        [
            "reference-pbr-glb-v1",
            str(manifest_file),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            *selected,
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    preview_root = Path(cache_root or Path(tempfile.gettempdir()) / "oai_mcp_mesh_previews")
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_path = preview_root / f"cartilage_mesh_preview_{digest}.glb"
    if not preview_path.is_file():
        _write_model3d_glb(preview_path, preview_meshes)

    return {
        "preview_path": str(preview_path),
        "choices": choices,
        "selected_labels": selected,
        "structure_count": len(selected),
        "vertex_count": int(vertex_count),
        "triangle_count": int(triangle_count),
        "coordinate_system": str(manifest.get("coordinate_system") or "ITK_LPS_mm"),
    }
