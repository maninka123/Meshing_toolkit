"""Batch production meshing: select only improved Ball Pivoting or Poisson."""

import argparse
import configparser
import json
import os
from pathlib import Path

from .paths import DEFAULT_CONFIG, resolve_path, result_directory

PRODUCTION_METHODS = ("improved_ball_pivoting", "improved_poisson")


def collect_files(input_path, extensions, recursive, excluded_dirs=()):
    source = Path(input_path)
    if source.is_file():
        return [str(source)]
    excluded = [resolve_path(path) for path in excluded_dirs]
    files = []
    for root, dirs, names in os.walk(source):
        stems = {Path(name).stem.casefold() for name in names if Path(name).suffix.lower() in extensions}
        dirs[:] = [name for name in dirs if not name.startswith(".")
                   and name.casefold() not in stems
                   and name not in {"meshes", "benchmark", "adaptive", "pymeshlab", "open3d", "legacy", "results", "__pycache__"}
                   and not any(resolve_path(Path(root) / name) == path or path in resolve_path(Path(root) / name).parents for path in excluded)]
        files.extend(str(Path(root) / name) for name in names
                     if Path(name).suffix.lower() in extensions and (Path(root) / name).is_file())
        if not recursive:
            break
    return sorted(files)


def existing_mesh_for_input(output_dir, file_path, method):
    """Auto skips only successful auto runs, not a manually forced alternative."""
    base = Path(file_path).stem
    folders = [Path(output_dir) / name for name in PRODUCTION_METHODS] if method == "auto" else [Path(output_dir)]
    for folder in folders:
        for mesh in sorted(folder.glob(f"{base}_improved_*.ply"), reverse=True):
            report_path = mesh.with_suffix(".json")
            if not report_path.is_file():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if method != "auto" or report.get("requested_method") == "auto":
                return str(mesh)
    return ""


def build_config(config):
    defaults = {"max_points": 120000, "neighbors": 24, "denoise_iterations": 2,
                "voxel_fraction": 0.8, "support_multiplier": 2.5, "max_depth": 10, "threads": 8}
    return {name: (config.getfloat if isinstance(default, float) else config.getint)("adaptive", name, fallback=default)
            for name, default in defaults.items()}


def reconstruct_mesh(file_path, output_root, config):
    from .adaptive import reconstruct_file
    return reconstruct_file(file_path, output_root, config["parameters"], config["method"], config.get("input_root"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        return 1
    config = configparser.ConfigParser()
    try:
        config.read(config_path, encoding="utf-8")
        raw_input = config.get("general", "input_path", fallback="").strip()
        if not raw_input:
            raise ValueError("Set [general] input_path.")
        source = resolve_path(raw_input, config_path.parent)
        if not source.exists():
            raise ValueError(f"Input not found: {source}")
        method = config.get("general", "reconstruction_method", fallback="auto").strip().lower()
        if method not in ("auto", *PRODUCTION_METHODS):
            raise ValueError("Use auto, improved_ball_pivoting, or improved_poisson. Baselines are available only through python -m benchmark.run.")
        raw_output = config.get("general", "output_dir", fallback="").strip()
        output_root = resolve_path(raw_output, config_path.parent) if raw_output else None
        if output_root is not None and (output_root == source or output_root in source.parents):
            raise ValueError("output_dir must not contain the input path.")
        recursive = config.getboolean("general", "recursive", fallback=True)
        skip_existing = config.getboolean("general", "skip_existing", fallback=True)
        extensions = ["." + item.strip().lower().lstrip(".") for item in config.get("general", "extensions", fallback=".pcd").split(",") if item.strip()]
        files = collect_files(source, extensions, recursive, [output_root] if output_root else [])
        if not files:
            raise ValueError("No input point clouds found.")
        parameters = build_config(config)
        destinations = {file: result_directory(file, method, output_root, source if source.is_dir() else None) for file in files}
        identities = [(str(folder).casefold(), Path(file).stem.casefold()) for file, folder in destinations.items()]
        if len(identities) != len(set(identities)):
            raise ValueError("Matching input stems in one folder are ambiguous; separate them into different folders.")
    except (ValueError, configparser.Error) as exc:
        print(f"Configuration error: {exc}")
        return 1
    print(f"Production method: {method} | inputs: {len(files)} | colours retained when present")
    successes = failures = skipped = 0
    for file, folder in destinations.items():
        existing = existing_mesh_for_input(folder, file, method) if skip_existing else ""
        if args.dry_run:
            print(f"{'SKIP' if existing else 'PROCESS'}: {file} -> {folder}")
        elif existing:
            skipped += 1
            print(f"[SKIP] {existing}")
        else:
            print(f"Processing {Path(file).name}", flush=True)
            ok, message = reconstruct_mesh(file, output_root, {"method": method, "parameters": parameters, "input_root": source if source.is_dir() else None})
            successes += int(ok)
            failures += int(not ok)
            print(f"[{'OK' if ok else 'FAIL'}] {message}", flush=True)
    if not args.dry_run:
        print(f"Done. Success: {successes} | skipped: {skipped} | failed: {failures}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
