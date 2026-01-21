#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config-driven CLI for PyMeshLab reconstruction.
"""

import configparser
import os
import sys
import time

from pymeshlab_pipeline import build_output_path, reconstruct_mesh


CONFIG_PATH = "pymeshlab_config.ini"
BAR_WIDTH = 36
SPINNER = ["|", "/", "-", "\\"]
PROGRESS_COLOR = "\x1b[36m"
RESET_COLOR = "\x1b[0m"
LAST_PROGRESS_LEN = 0
IS_INTERACTIVE = sys.stdout.isatty()


def print_banner():
    line = "=" * 72
    print(line)
    print("PyMeshLab Reconstruction CLI")
    print("Open-boundary meshing for PCD/PLY point clouds")
    print(line)


def format_duration(seconds):
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def supports_color():
    if not IS_INTERACTIVE:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


def colorize(text, color_code):
    if not supports_color():
        return text
    return f"{color_code}{text}{RESET_COLOR}"


def clear_progress_line():
    global LAST_PROGRESS_LEN
    if LAST_PROGRESS_LEN <= 0:
        return
    sys.stdout.write("\r" + (" " * LAST_PROGRESS_LEN) + "\r")
    sys.stdout.flush()
    LAST_PROGRESS_LEN = 0


def write_progress_line(text):
    global LAST_PROGRESS_LEN
    padded = text.ljust(LAST_PROGRESS_LEN)
    output = colorize(padded, PROGRESS_COLOR)
    sys.stdout.write("\r" + output)
    sys.stdout.flush()
    LAST_PROGRESS_LEN = max(LAST_PROGRESS_LEN, len(text))


def render_progress(current, total, elapsed):
    if total <= 0:
        return ""
    pct = current / total
    filled = int(BAR_WIDTH * pct)
    bar = "=" * filled + "-" * (BAR_WIDTH - filled)
    avg = elapsed / current if current > 0 else None
    remaining = avg * (total - current) if avg is not None else None
    spinner = SPINNER[current % len(SPINNER)]
    return (
        f"{spinner} [{bar}] {current}/{total} {pct*100:5.1f}% "
        f"ETA {format_duration(remaining)}"
    )


def load_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def get_str(config, section, key, fallback=""):
    if config.has_option(section, key):
        value = config.get(section, key)
        if value is None:
            return fallback
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        return value
    return fallback


def get_bool(config, section, key, fallback=False):
    if config.has_option(section, key):
        try:
            return config.getboolean(section, key)
        except ValueError:
            return fallback
    return fallback


def get_int(config, section, key, fallback=0):
    if config.has_option(section, key):
        try:
            return config.getint(section, key)
        except ValueError:
            return fallback
    return fallback


def get_float(config, section, key, fallback=0.0):
    if config.has_option(section, key):
        try:
            return config.getfloat(section, key)
        except ValueError:
            return fallback
    return fallback


def parse_extensions(raw_extensions):
    if not raw_extensions:
        return [".pcd"]
    parts = []
    for item in raw_extensions.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        parts.append(ext)
    return parts or [".pcd"]


def normalize_method(raw):
    if not raw:
        return ""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def collect_files(input_path, extensions, recursive):
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    if recursive:
        for root, _, names in os.walk(input_path):
            for name in names:
                if os.path.splitext(name)[1].lower() in extensions:
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_path):
            if os.path.splitext(name)[1].lower() in extensions:
                files.append(os.path.join(input_path, name))
    return sorted(files)

def existing_mesh_for_input(output_dir, file_path, method):
    base = os.path.splitext(os.path.basename(file_path))[0]
    prefix = f"{base}_{method}_"
    if not os.path.isdir(output_dir):
        return ""
    for name in os.listdir(output_dir):
        if name.startswith(prefix) and name.lower().endswith(".ply"):
            return os.path.join(output_dir, name)
    return ""


def build_config(config, method, output_dir):
    return {
        "method": method,
        "output_dir": output_dir,
        "skip_existing": get_bool(config, "general", "skip_existing", True),
        "normals": {
            "enabled": get_bool(config, "normals", "enabled", True),
            "k": get_int(config, "normals", "k", 30),
            "smooth_iter": get_int(config, "normals", "smooth_iter", 0),
            "flip": get_bool(config, "normals", "flip", False),
        },
        "ball_pivoting": {
            "radius_auto": get_bool(config, "ball_pivoting", "radius_auto", True),
            "radius_scale": get_float(config, "ball_pivoting", "radius_scale", 2.5),
            "radius": get_float(config, "ball_pivoting", "radius", 0.02),
        },
        "alpha_shape": {
            "alpha": get_float(config, "alpha_shape", "alpha", 0.05),
        },
        "poisson": {
            "depth": get_int(config, "poisson", "depth", 10),
            "cg_depth": get_int(config, "poisson", "cg_depth", 8),
            "iters": get_int(config, "poisson", "iters", 8),
            "scale": get_float(config, "poisson", "scale", 1.1),
            "samples_per_node": get_float(config, "poisson", "samples_per_node", 1.5),
            "point_weight": get_float(config, "poisson", "point_weight", 4.0),
            "confidence": get_bool(config, "poisson", "confidence", False),
            "pre_clean": get_bool(config, "poisson", "pre_clean", False),
        },
    }

def print_config_summary(input_path, method, output_dir, extensions, recursive, config):
    print("")
    print("Config Summary")
    print("-" * 72)
    print(f"Input: {input_path}")
    print(f"Mode: {'folder' if os.path.isdir(input_path) else 'single file'}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Recursive: {recursive}")
    print(f"Skip existing: {get_bool(config, 'general', 'skip_existing', True)}")
    print("")
    print("Normals")
    print(
        f"  enabled={get_bool(config, 'normals', 'enabled', True)} "
        f"k={get_int(config, 'normals', 'k', 30)} "
        f"smooth_iter={get_int(config, 'normals', 'smooth_iter', 0)} "
        f"flip={get_bool(config, 'normals', 'flip', False)}"
    )
    if method == "ball_pivoting":
        print("Ball Pivoting")
        print(
            f"  radius_auto={get_bool(config, 'ball_pivoting', 'radius_auto', True)} "
            f"radius_scale={get_float(config, 'ball_pivoting', 'radius_scale', 2.5)} "
            f"radius={get_float(config, 'ball_pivoting', 'radius', 0.02)}"
        )
    elif method == "alpha_shape":
        print("Alpha Shape")
        print(
            f"  alpha={get_float(config, 'alpha_shape', 'alpha', 0.05)}"
        )
    elif method == "poisson":
        print("Poisson")
        print(
            f"  depth={get_int(config, 'poisson', 'depth', 10)} "
            f"cg_depth={get_int(config, 'poisson', 'cg_depth', 8)} "
            f"iters={get_int(config, 'poisson', 'iters', 8)}"
        )
        print(
            f"  scale={get_float(config, 'poisson', 'scale', 1.1)} "
            f"samples_per_node={get_float(config, 'poisson', 'samples_per_node', 1.5)} "
            f"point_weight={get_float(config, 'poisson', 'point_weight', 4.0)}"
        )
        print(
            f"  confidence={get_bool(config, 'poisson', 'confidence', False)} "
            f"pre_clean={get_bool(config, 'poisson', 'pre_clean', False)}"
        )
    print("-" * 72)


def main():
    print_banner()

    if not os.path.isfile(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        print("Create it or copy the provided template.")
        return 1

    config = load_config(CONFIG_PATH)
    default_input = get_str(config, "general", "input_path", "")
    default_method = get_str(config, "general", "reconstruction_method", "ball_pivoting")
    default_output = get_str(config, "general", "output_dir", "meshes")
    extensions = parse_extensions(get_str(config, "general", "extensions", ".pcd"))
    recursive = get_bool(config, "general", "recursive", False)

    input_path = default_input
    if not input_path:
        print("No input_path set in config.")
        print("Set [general] input_path in pymeshlab_config.ini.")
        return 1
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return 1

    method = normalize_method(default_method) or "ball_pivoting"
    valid_methods = {"ball_pivoting", "alpha_shape", "poisson"}
    if method not in valid_methods:
        print(f"Unknown reconstruction_method in config: {default_method}")
        print("Valid options: ball_pivoting, alpha_shape, poisson")
        return 1

    if os.path.isdir(input_path):
        output_dir = os.path.join(input_path, "meshes")
    else:
        output_dir = default_output

    files = collect_files(input_path, extensions, recursive)
    if not files:
        print("No input files found.")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    config_block = build_config(config, method, output_dir)

    print_config_summary(
        input_path, method, output_dir, extensions, recursive, config
    )
    print(f"Files found: {len(files)}")

    start = time.perf_counter()
    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, file_path in enumerate(files, 1):
        name = os.path.basename(file_path)
        if IS_INTERACTIVE:
            clear_progress_line()
        print(f"Processing {idx}/{len(files)}: {name}")
        output_path = build_output_path(file_path, output_dir, method)
        if config_block["skip_existing"]:
            existing = existing_mesh_for_input(output_dir, file_path, method)
            if existing:
                print(f"⏭️  Skipping (already exists): {existing}")
                skip_count += 1
                elapsed = time.perf_counter() - start
                progress = render_progress(idx, len(files), elapsed)
                if IS_INTERACTIVE:
                    write_progress_line(progress)
                else:
                    print(progress)
                continue
        ok, message = reconstruct_mesh(file_path, output_path, config_block)
        if ok:
            success_count += 1
            print(f"✅ {message}")
        else:
            fail_count += 1
            print(f"❌ {message}")
        elapsed = time.perf_counter() - start
        progress = render_progress(idx, len(files), elapsed)
        if IS_INTERACTIVE:
            write_progress_line(progress)
        else:
            print(progress)

    total_elapsed = time.perf_counter() - start
    if IS_INTERACTIVE:
        clear_progress_line()
    print("")
    print("=" * 72)
    print(
        f"Done. Success: {success_count}/{len(files)} | "
        f"Skipped: {skip_count} | "
        f"Failed: {fail_count} | "
        f"Elapsed: {format_duration(total_elapsed)}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
