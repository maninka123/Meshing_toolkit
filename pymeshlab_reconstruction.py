#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run mesh reconstruction using PyMeshLab.
"""

import contextlib
import io
import os
import re
import sys
from datetime import datetime


# ============================
# I/O
# ============================
INPUT_PATH = "Dataset_01/PC_20260112_191613691.pcd"  # Source point cloud (PCD/PLY).
OUTPUT_DIR = "meshes"  # Folder for generated meshes and filter dumps.
OUTPUT_PATH = ""  # Optional explicit output path; leave empty for auto naming.

# ============================
# Method Selection
# ============================
RECONSTRUCTION_METHOD = "ball_pivoting"  # "ball_pivoting", "alpha_shape", "poisson".

# ============================
# Shared (Normals)
# ============================
NORMALS_ENABLED = True  # Estimate normals before reconstruction.
NORMAL_K = 30  # Number of neighbors used for normal estimation.
NORMAL_SMOOTH_ITER = 0  # Smoothing iterations for normals (0 = none).
NORMAL_FLIP = False  # Flip normals if they are oriented inward.

# ============================
# Ball Pivoting (Open Boundaries)
# ============================
BALL_RADIUS_AUTO = True  # Auto-estimate ball radius from point spacing (Open3D).
BALL_RADIUS_SCALE = 2.5  # Multiplier for average spacing if auto-estimate is used.
BALL_RADIUS = 0.02  # Fallback ball radius if auto-estimate is disabled/unavailable.

# ============================
# Alpha Shapes (Open Boundaries)
# ============================
ALPHA_VALUE = 0.05  # Alpha shape radius; smaller keeps more open boundaries.

# ============================
# Poisson (Watertight)
# ============================
POISSON_DEPTH = 10  # Octree depth (higher = more detail, more memory).
POISSON_CG_DEPTH = 8  # Solver depth for conjugate gradient iterations.
POISSON_ITERS = 8  # Solver iterations (higher can improve accuracy).
POISSON_SCALE = 1.1  # Bounding box scale (slightly >1 keeps margins).
POISSON_SAMPLES_PER_NODE = 1.5  # Samples per octree node (controls smoothness).
POISSON_POINT_WEIGHT = 4.0  # Fidelity to points (higher = closer to input).
POISSON_CONFIDENCE = False  # Use confidence (requires per-point quality).
POISSON_PRE_CLEAN = False  # Pre-clean input before reconstruction.


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]


def normalize_name(name):
    return name.replace("_", "").lower()


def try_apply_filter(meshset, filter_name, params):
    try:
        meshset.apply_filter(filter_name, **params)
        return True
    except Exception:
        if params:
            try:
                meshset.apply_filter(filter_name)
                return True
            except Exception:
                return False
        return False


def extract_filter_names(text):
    names = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("filter name:"):
            names.append(stripped.split(":", 1)[1].strip())
            continue
        if lower.startswith("filter:"):
            names.append(stripped.split(":", 1)[1].strip())
            continue
        if re.match(r"^[a-z0-9_]+$", stripped):
            names.append(stripped)
    return names


def get_filter_list_text(meshset):
    if hasattr(meshset, "filter_list"):
        try:
            return "\n".join(meshset.filter_list())
        except Exception:
            pass
    if hasattr(meshset, "print_filter_list"):
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                meshset.print_filter_list()
            return buf.getvalue()
        except Exception:
            pass
    try:
        import pymeshlab

        if hasattr(pymeshlab, "filter_list"):
            return "\n".join(pymeshlab.filter_list())
        if hasattr(pymeshlab, "print_filter_list"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pymeshlab.print_filter_list()
            return buf.getvalue()
    except Exception:
        pass
    return ""


def get_filter_names(meshset):
    text = get_filter_list_text(meshset)
    if not text:
        return []
    names = extract_filter_names(text)
    if names:
        return names
    return [line.strip() for line in text.splitlines() if line.strip()]


def dump_filter_list(meshset, output_dir):
    text = get_filter_list_text(meshset)
    if not text:
        return ""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pymeshlab_filters.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return output_path


def apply_filter_with_fallback(meshset, candidates, desired_params, keyword_groups):
    available = get_filter_names(meshset)
    to_try = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            to_try.append(candidate)
            seen.add(candidate)
    if available:
        for group in keyword_groups:
            for name in available:
                norm = normalize_name(name)
                if all(word in norm for word in group):
                    if name not in seen:
                        to_try.append(name)
                        seen.add(name)

    last_exc = None
    for candidate in to_try:
        params = resolve_params(meshset, candidate, desired_params)
        try:
            if try_apply_filter(meshset, candidate, params):
                return candidate
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise RuntimeError(
            f"Failed to apply any filter from: {', '.join(candidates)}"
        ) from last_exc
    raise RuntimeError(f"Failed to apply any filter from: {', '.join(candidates)}")


def resolve_params(meshset, filter_name, desired_params):
    available = {}
    if hasattr(meshset, "filter_parameter_list"):
        try:
            for item in meshset.filter_parameter_list(filter_name):
                name = item["name"] if isinstance(item, dict) else str(item)
                available[normalize_name(name)] = name
        except Exception:
            available = {}

    resolved = {}
    for key, value in desired_params.items():
        if value is None:
            continue
        norm = normalize_name(key)
        if available:
            target = available.get(norm)
            if not target:
                print(
                    f"Warning: parameter '{key}' not supported by '{filter_name}', skipping."
                )
                continue
            resolved[target] = value
        else:
            resolved[key] = value
    return resolved


def build_output_path(input_path, output_dir, method_tag):
    base = os.path.splitext(os.path.basename(input_path))[0]
    safe_tag = re.sub(r"[^a-z0-9_]+", "_", method_tag.lower()).strip("_")
    filename = f"{base}_{safe_tag}_{get_timestamp()}.ply"
    return os.path.join(output_dir, filename)


def convert_pcd_to_ply(input_path, output_dir):
    try:
        import open3d as o3d
    except ImportError:
        raise RuntimeError(
            "PCD input requires open3d. Install with `pip install open3d`."
        )

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    converted_path = os.path.join(output_dir, f"{base}_pymeshlab_input.ply")
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"PCD input is empty: {input_path}")
    o3d.io.write_point_cloud(converted_path, pcd)
    return converted_path


def estimate_ball_radius(input_path):
    try:
        import open3d as o3d
    except ImportError:
        return None

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        return None
    distances = pcd.compute_nearest_neighbor_distance()
    if not distances:
        return None
    avg_distance = sum(distances) / len(distances)
    return avg_distance * BALL_RADIUS_SCALE


def main():
    input_path = INPUT_PATH
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}")
        return 1

    try:
        import pymeshlab
    except ImportError:
        print("pymeshlab is not installed. Install with `pip install pymeshlab`.")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_PATH or build_output_path(
        input_path, OUTPUT_DIR, RECONSTRUCTION_METHOD
    )

    ms = pymeshlab.MeshSet()
    load_path = input_path
    _, ext = os.path.splitext(input_path)
    if ext.lower() == ".pcd":
        try:
            load_path = convert_pcd_to_ply(input_path, OUTPUT_DIR)
            print(f"Converted PCD to PLY for PyMeshLab: {load_path}")
        except Exception as exc:
            print(f"Failed to convert PCD input: {exc}")
            return 1

    try:
        ms.load_new_mesh(load_path)
    except Exception as exc:
        print(f"Failed to load mesh '{load_path}': {exc}")
        return 1

    if NORMALS_ENABLED:
        try:
            normal_filter = apply_filter_with_fallback(
                ms,
                [
                    "compute_normals_for_point_sets",
                    "compute_normal_for_point_clouds",
                    "compute_normals_for_point_clouds",
                ],
                {
                    "k": NORMAL_K,
                    "smoothiter": NORMAL_SMOOTH_ITER,
                    "flipflag": NORMAL_FLIP,
                },
                [["normal"]],
            )
        except Exception as exc:
            print(f"Normal estimation skipped due to error: {exc}")

    method = RECONSTRUCTION_METHOD.strip().lower()
    if method in {"ball_pivoting", "bpa", "ballpivoting"}:
        ball_radius = BALL_RADIUS
        if BALL_RADIUS_AUTO:
            auto_radius = estimate_ball_radius(input_path)
            if auto_radius:
                ball_radius = auto_radius
                print(f"Auto-estimated ball radius: {ball_radius}")
        try:
            apply_filter_with_fallback(
                ms,
                [
                    "surface_reconstruction_ball_pivoting",
                    "surface_reconstruction_ballpivoting",
                    "ball_pivoting",
                    "ballpivoting",
                    "generate_surface_reconstruction_ball_pivoting",
                ],
                {"ballradius": ball_radius},
                [["ball", "pivot"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, OUTPUT_DIR)
            if dumped:
                print(f"Available PyMeshLab filters written to {dumped}")
            print(f"Ball Pivoting reconstruction failed: {exc}")
            return 1
    elif method in {"alpha_shape", "alpha", "alphashape"}:
        try:
            apply_filter_with_fallback(
                ms,
                [
                    "surface_reconstruction_alpha_shapes",
                    "alpha_shapes",
                    "alpha_shape",
                    "generate_surface_reconstruction_alpha_shapes",
                ],
                {"alpha": ALPHA_VALUE},
                [["alpha", "shape"], ["alpha"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, OUTPUT_DIR)
            if dumped:
                print(f"Available PyMeshLab filters written to {dumped}")
            print(f"Alpha Shape reconstruction failed: {exc}")
            return 1
    elif method in {"poisson", "screened_poisson"}:
        try:
            apply_filter_with_fallback(
                ms,
                [
                    "surface_reconstruction_screened_poisson",
                    "surface_reconstruction_poisson",
                    "screened_poisson",
                    "generate_surface_reconstruction_screened_poisson",
                    "generate_surface_reconstruction_poisson",
                ],
                {
                    "depth": POISSON_DEPTH,
                    "cgdepth": POISSON_CG_DEPTH,
                    "iters": POISSON_ITERS,
                    "scale": POISSON_SCALE,
                    "samplespernode": POISSON_SAMPLES_PER_NODE,
                    "pointweight": POISSON_POINT_WEIGHT,
                    "confidence": POISSON_CONFIDENCE,
                    "preclean": POISSON_PRE_CLEAN,
                },
                [["screened", "poisson"], ["poisson"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, OUTPUT_DIR)
            if dumped:
                print(f"Available PyMeshLab filters written to {dumped}")
            print(f"Poisson reconstruction failed: {exc}")
            return 1
    else:
        print(f"Unknown RECONSTRUCTION_METHOD: {RECONSTRUCTION_METHOD}")
        return 1

    ms.save_current_mesh(output_path)
    print(f"Mesh saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
