"""Benchmark all five methods on a shared held-out split of the real captures."""

import argparse
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import open3d as o3d

from reconstruction.adaptive import (Settings, CANDIDATE_METHODS, PRODUCTION_METHODS,
    unique_finite, validation_split, prepare, build_candidate, evaluate,
    select_candidate, transfer_colors, save_result)
from reconstruction.mesh_preview import save_gallery, save_thumbnails
from reconstruction.paths import DATASETS_DIR, PROJECT_ROOT, resolve_path
from .baselines import BASELINE_METHODS, prepare_baseline, build_baseline


def display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)

METHODS = (*BASELINE_METHODS, *PRODUCTION_METHODS)
LABELS = {"baseline_ball_pivoting": "Baseline\nBall Pivoting", "baseline_poisson": "Baseline\nPoisson",
          "baseline_alpha_shape": "Baseline\nAlpha Shape", "improved_ball_pivoting": "Improved\nBall Pivoting", "improved_poisson": "Improved\nPoisson"}


def world_mesh(mesh, origin, scale):
    result = o3d.geometry.TriangleMesh(mesh)
    result.vertices = o3d.utility.Vector3dVector(np.asarray(result.vertices) * scale + origin)
    result.compute_vertex_normals()
    return result


def write_mesh(path, mesh):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True):
        raise RuntimeError(f"Could not save {path}")
    reloaded = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_vertex_colors() and not reloaded.has_vertex_colors():
        raise RuntimeError(f"Lost vertex colours in {path}")


def benchmark_capture(source, settings, output, sample_capture):
    started = time.perf_counter()
    original = o3d.io.read_point_cloud(str(source))
    clean = unique_finite(original)
    cloud, training, validation, origin, scale = validation_split(clean, settings)
    report = {"capture": source.stem, "input": display_path(source),
              "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "raw_input_points": len(original.points), "finite_unique_points": len(clean.points),
              "training_points": len(training.points), "validation_points": len(validation.points),
              "normalization_origin": origin.tolist(), "normalization_scale": scale,
              "methods": {}, "settings": asdict(settings)}
    print(f"{source.name}: {len(training.points)} training, {len(validation.points)} held-out", flush=True)
    start = time.perf_counter()
    basic = prepare_baseline(training, settings)
    baseline_prep_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prepared, preparation = prepare(training, settings)
    improved_prep_seconds = time.perf_counter() - start
    report["improved_preparation"] = preparation
    report["baseline_prepared_points"] = len(basic.points)
    validation_reports, validation_meshes = {}, {}
    for kind in CANDIDATE_METHODS:
        print(f"  {kind}", flush=True)
        start = time.perf_counter()
        try:
            mesh, params = build_candidate(prepared, kind, settings)
            transfer_colors(mesh, training)
            metrics = evaluate(mesh, validation, training, settings)
            validation_reports[kind] = dict(metrics, parameters=params, reconstruction_and_evaluation_seconds=time.perf_counter() - start)
            validation_meshes[kind] = mesh
        except (ValueError, RuntimeError) as exc:
            validation_reports[kind] = {"error": str(exc)}
    report["improved_variants"] = validation_reports
    selected, margin = select_candidate(validation_reports)
    report["recommended_method"] = CANDIDATE_METHODS[selected]
    report["selected_variant"] = selected
    report["selection_near_tie_margin"] = margin
    meshes = {}
    for method in METHODS:
        print(f"  benchmark {method}", flush=True)
        start = time.perf_counter()
        try:
            if method in BASELINE_METHODS:
                mesh, params = build_baseline(basic, method, settings)
                metrics = evaluate(mesh, validation, training, settings)
                metrics.update(parameters=params, seconds=baseline_prep_seconds + time.perf_counter() - start)
            else:
                variants = {key: value for key, value in validation_reports.items() if CANDIDATE_METHODS[key] == method}
                variant, _ = select_candidate(variants)
                mesh = validation_meshes[variant]
                metrics = dict(validation_reports[variant])
                metrics["selected_variant"] = variant
                metrics["seconds"] = improved_prep_seconds + sum(value.get("reconstruction_and_evaluation_seconds", 0) for value in variants.values())
            metrics["point_to_surface_p95_input_units"] = metrics["point_to_surface_p95"] * scale
            metrics["vertex_colors_saved"] = mesh.has_vertex_colors()
            world = world_mesh(mesh, origin, scale)
            full_path = source.parent / source.stem / "benchmark" / method / "validation_mesh.ply"
            write_mesh(full_path, world)
            metrics["local_mesh"] = display_path(full_path)
            report["methods"][method] = metrics
            meshes[method] = world
            if source.stem == sample_capture:
                # These are presentation samples only; metrics always use the full mesh.
                sample = world.simplify_quadric_decimation(60000) if len(world.triangles) > 60000 else o3d.geometry.TriangleMesh(world)
                transfer_colors(sample, original)
                sample.compute_vertex_normals()
                sample_path = output / "samples" / f"{source.stem}_{method}.ply"
                write_mesh(sample_path, sample)
                metrics["sample"] = display_path(sample_path)
                metrics["sample_triangles"] = len(sample.triangles)
            print(f"    score {metrics['selection_score']:.3f}, coverage {metrics['coverage_within_2_local_spacings']:.1%}", flush=True)
        except (ValueError, RuntimeError) as exc:
            report["methods"][method] = {"error": str(exc)}
            print(f"    FAILED: {exc}", flush=True)
    ordered = [meshes.get(method) for method in METHODS]
    if source.stem == sample_capture:
        save_thumbnails(original, meshes, output / "figures", source.stem)
    for colored, suffix in [(True, "colours"), (False, "geometry")]:
        figure = output / "figures" / f"{source.stem}_{suffix}.png"
        save_gallery(original, ordered, [LABELS[method] for method in METHODS], figure, colored)
    # Refit the chosen improved variant on all points; no baselines enter this decision.
    prepared_full, full_preparation = prepare(cloud, settings)
    final, params = build_candidate(prepared_full, selected, settings)
    transfer_colors(final, cloud)
    final_metrics = evaluate(final, cloud, cloud, settings)
    final = world_mesh(final, origin, scale)
    final_report = dict(report, selected_method=CANDIDATE_METHODS[selected], requested_method="auto",
                        final_preparation=full_preparation, final_parameters=params,
                        final_input_consistency=final_metrics, seconds=time.perf_counter() - started,
                        metric_units="normalized by normalization_scale; multiply distances by scale for input units",
                        open3d_version=o3d.__version__, numpy_version=np.__version__)
    final_path = save_result(source, final, final_report)
    report["production_mesh"] = display_path(final_path)
    if source.stem == sample_capture:
        sample = final.simplify_quadric_decimation(60000) if len(final.triangles) > 60000 else o3d.geometry.TriangleMesh(final)
        transfer_colors(sample, original)
        write_mesh(output / "samples" / f"{source.stem}_final_recommended.ply", sample)
        shutil.copyfile(final_path.with_name(final_path.stem + "_preview.png"), output / "figures" / "recommended_capture_1.png")
    report["seconds"] = time.perf_counter() - started
    return report


def save_tables(output, reports):
    columns = ["capture", "method", "coverage_pct", "p95_distance_input_units", "p95_distance_local_spacings", "unsupported_surface_pct", "selection_score", "triangles", "boundary_edges", "nonmanifold_edges", "nonmanifold_vertices", "seconds", "vertex_colors_saved"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            for method, metrics in report["methods"].items():
                if "error" in metrics:
                    continue
                writer.writerow(dict(capture=report["capture"], method=method,
                    coverage_pct=100 * metrics["coverage_within_2_local_spacings"],
                    p95_distance_input_units=metrics["point_to_surface_p95_input_units"],
                    p95_distance_local_spacings=metrics["point_to_surface_p95_spacing"],
                    unsupported_surface_pct=100 * metrics["unsupported_surface_fraction"],
                    **{key: metrics[key] for key in columns[6:]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASETS_DIR / "captures")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmark")
    parser.add_argument("--sample-capture", default="capture_1")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    source, output = resolve_path(args.input), resolve_path(args.output)
    files = [source] if source.is_file() else sorted(source.glob("*.pcd")) + sorted(source.glob("*.ply"))
    if not files:
        parser.error("No input point clouds found.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    settings = Settings(threads=args.threads)
    reports = []
    summary = {"created": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
               "open3d": o3d.__version__, "numpy": np.__version__, "settings": asdict(settings),
               "protocol": "Identical seed-42 80/20 split; metrics on training meshes, not full-input refits. Poisson improved resolution uses this same holdout (selection consistency, not independent test accuracy).", "captures": reports}
    for file in files:
        reports.append(benchmark_capture(file, settings, output, args.sample_capture))
        (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        save_tables(output, reports)
    from .report import write_report
    write_report(output, reports)
    return int(any("error" in metrics for report in reports for metrics in report["methods"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
