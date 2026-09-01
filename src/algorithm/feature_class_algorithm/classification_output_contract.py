"""Dependency-light validation and collection of completed classifier runs."""

from pathlib import Path


def validate_and_collect_classification_outputs(output_dir, required_relative_outputs):
    root = Path(output_dir).resolve()
    if not root.is_dir() or not required_relative_outputs:
        raise ValueError("Classification output is incomplete.")
    for relative in required_relative_outputs:
        candidate_relative = Path(relative)
        if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
            raise ValueError("Required output paths must be safe and contained.")
        candidate = (root / candidate_relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Required output paths must be safe and contained.") from exc
        if not candidate.is_file():
            raise ValueError("Classification output is incomplete.")
    charts, data = [], []
    for candidate in sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink()):
        if candidate.is_symlink():
            raise ValueError("Classification outputs must be safe and contained regular files.")
        resolved_path = candidate.resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Classification outputs must be safe and contained regular files.") from exc
        if not resolved_path.is_file():
            raise ValueError("Classification output is incomplete.")
        resolved = str(resolved_path)
        if candidate.suffix.lower() in (".png", ".jpg", ".jpeg"):
            charts.append(resolved)
        elif candidate.suffix.lower() in (".txt", ".csv", ".json"):
            data.append(resolved)
    return charts, data
