"""Shared locations and result grouping, independent of the working directory."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "meshing.ini"


def resolve_path(value, base_dir=PROJECT_ROOT):
    """Resolve a configured path, retaining support for absolute paths."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def method_tag(method):
    return re.sub(r"[^a-z0-9_]+", "_", method.lower()).strip("_")


def result_directory(source, method, output_root=None, input_root=None):
    """Save in a folder named after the point cloud, alongside the source.

    An optional output_root mirrors dataset/source folders under another base.
    Single-file and batch runs of local datasets share the same destination.
    """
    source = resolve_path(source)
    method_folder = Path("meshes") if method == "auto" else Path("meshes") / method_tag(method)
    if output_root is None:
        return source.parent / source.stem / method_folder
    root = resolve_path(output_root)
    try:
        relative = source.relative_to(DATASETS_DIR)
    except ValueError:
        batch_root = resolve_path(input_root) if input_root is not None else source.parent
        relative = source.relative_to(batch_root)
        dataset = batch_root.name
        subfolder = relative.parent
    else:
        if len(relative.parts) > 1:
            dataset = relative.parts[0]
            subfolder = Path(*relative.parts[1:]).parent
        else:
            dataset = "unassigned"
            subfolder = Path()
    return root / dataset / subfolder / source.stem / method_folder
