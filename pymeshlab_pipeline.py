#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyMeshLab reconstruction helpers for CLI usage.
"""

import contextlib
import io
import os
import re
from datetime import datetime


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
                continue
            resolved[target] = value
        else:
            resolved[key] = value
    return resolved


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


def estimate_ball_radius(input_path, radius_scale):
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
    return avg_distance * radius_scale


def reconstruct_mesh(input_path, output_path, config):
    try:
        import pymeshlab
    except ImportError:
        return False, "pymeshlab is not installed. Install with `pip install pymeshlab`."

    output_dir = config["output_dir"]
    method = config["method"]

    load_path = input_path
    _, ext = os.path.splitext(input_path)
    if ext.lower() == ".pcd":
        try:
            load_path = convert_pcd_to_ply(input_path, output_dir)
        except Exception as exc:
            return False, f"Failed to convert PCD input: {exc}"

    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(load_path)
    except Exception as exc:
        return False, f"Failed to load mesh '{load_path}': {exc}"

    normals = config["normals"]
    if normals["enabled"]:
        try:
            apply_filter_with_fallback(
                ms,
                [
                    "compute_normals_for_point_sets",
                    "compute_normal_for_point_clouds",
                    "compute_normals_for_point_clouds",
                ],
                {
                    "k": normals["k"],
                    "smoothiter": normals["smooth_iter"],
                    "flipflag": normals["flip"],
                },
                [["normal"]],
            )
        except Exception as exc:
            return False, f"Normal estimation failed: {exc}"

    if method == "ball_pivoting":
        ball = config["ball_pivoting"]
        radius = ball["radius"]
        if ball["radius_auto"]:
            auto_radius = estimate_ball_radius(input_path, ball["radius_scale"])
            if auto_radius:
                radius = auto_radius
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
                {"ballradius": radius},
                [["ball", "pivot"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, output_dir)
            msg = f"Ball Pivoting reconstruction failed: {exc}"
            if dumped:
                msg += f" (filters dumped to {dumped})"
            return False, msg
    elif method == "alpha_shape":
        alpha = config["alpha_shape"]
        try:
            apply_filter_with_fallback(
                ms,
                [
                    "surface_reconstruction_alpha_shapes",
                    "alpha_shapes",
                    "alpha_shape",
                    "generate_surface_reconstruction_alpha_shapes",
                ],
                {"alpha": alpha["alpha"]},
                [["alpha", "shape"], ["alpha"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, output_dir)
            msg = f"Alpha Shape reconstruction failed: {exc}"
            if dumped:
                msg += f" (filters dumped to {dumped})"
            return False, msg
    elif method == "poisson":
        poisson = config["poisson"]
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
                    "depth": poisson["depth"],
                    "cgdepth": poisson["cg_depth"],
                    "iters": poisson["iters"],
                    "scale": poisson["scale"],
                    "samplespernode": poisson["samples_per_node"],
                    "pointweight": poisson["point_weight"],
                    "confidence": poisson["confidence"],
                    "preclean": poisson["pre_clean"],
                },
                [["screened", "poisson"], ["poisson"]],
            )
        except Exception as exc:
            dumped = dump_filter_list(ms, output_dir)
            msg = f"Poisson reconstruction failed: {exc}"
            if dumped:
                msg += f" (filters dumped to {dumped})"
            return False, msg
    else:
        return False, f"Unknown reconstruction method: {method}"

    ms.save_current_mesh(output_path)
    return True, f"Saved mesh to {output_path}"
