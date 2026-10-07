"""Scale-aware open-surface reconstruction and per-input model selection.

No scene coordinates, camera poses, or physical units are assumed. Candidates
are compared on held-out input points, surface support, and mesh topology.
These are consistency checks, not ground-truth geometric accuracy.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .paths import DATASETS_DIR, resolve_path, result_directory

PRODUCTION_METHODS = ("improved_ball_pivoting", "improved_poisson")
CANDIDATE_METHODS = {
    "multi_radius_bpa": "improved_ball_pivoting",
    "trimmed_poisson_coarse": "improved_poisson",
    "trimmed_poisson": "improved_poisson",
}


@dataclass(frozen=True)
class Settings:
    max_points: int = 120000
    neighbors: int = 24
    denoise_iterations: int = 2
    voxel_fraction: float = 0.8
    support_multiplier: float = 2.5
    validation_fraction: float = 0.2
    max_depth: int = 10
    threads: int = 8
    seed: int = 42
    metric_samples: int = 20000

    def validate(self):
        if self.max_points < 100 or self.neighbors < 6 or self.metric_samples < 100:
            raise ValueError("Point and neighbor limits are too small.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5.")
        if not 0 < self.voxel_fraction <= 3 or not 0 < self.support_multiplier <= 10:
            raise ValueError("Invalid spacing multipliers.")
        if not 5 <= self.max_depth <= 12 or self.threads < 1 or self.denoise_iterations < 0:
            raise ValueError("Invalid reconstruction limits.")


def cloud_from_arrays(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def unique_finite(cloud):
    points = np.asarray(cloud.points)
    valid = np.flatnonzero(np.isfinite(points).all(axis=1))
    _, unique = np.unique(points[valid], axis=0, return_index=True)
    indices = valid[np.sort(unique)]
    return cloud.select_by_index(indices.tolist())


def spacing(points):
    """Robust spacing and local density from neighbors, in input units."""
    if len(points) < 7:
        raise ValueError("At least seven distinct finite points are required.")
    distances, indices = cKDTree(points).query(points, k=7, workers=-1)
    positive = distances[:, 1][distances[:, 1] > 0]
    if not len(positive):
        raise ValueError("Point cloud has no positive point spacing.")
    median = float(np.median(positive))
    local = np.maximum(distances[:, -1] / np.sqrt(6), median * 0.35)
    return median, local, indices


def orient_normals(points, normals, local, neighbors=12):
    """Orient a local minimum-spanning forest without connecting distant sheets.

    Unlike a camera-based flip, this does not assume the scanner is at (0,0,0).
    Disconnected patches are retained rather than keeping only one component.
    """
    n = len(points)
    distance, index = cKDTree(points).query(points, k=min(neighbors + 1, n), workers=-1)
    rows = np.repeat(np.arange(n), index.shape[1] - 1)
    cols = index[:, 1:].ravel()
    length = distance[:, 1:].ravel()
    support = local[rows] + local[cols]
    keep = length < 3 * support
    rows, cols, length, support = rows[keep], cols[keep], length[keep], support[keep]
    dot = np.einsum("ij,ij->i", normals[rows], normals[cols])
    weight = 1 - np.abs(dot) + 0.01 * length / support + 1e-8
    graph = coo_matrix((weight, (rows, cols)), shape=(n, n)).tocsr()
    tree = minimum_spanning_tree(graph)
    tree = (tree + tree.T).tocsr()
    result = normals.copy()
    visited = np.zeros(n, dtype=bool)
    labels = np.empty(n, dtype=np.int32)
    component = 0
    roots = []
    for root in range(n):
        if visited[root]:
            continue
        roots.append(root)
        visited[root] = True
        stack = [root]
        while stack:
            parent = stack.pop()
            labels[parent] = component
            for child in tree.indices[tree.indptr[parent]:tree.indptr[parent + 1]]:
                if not visited[child]:
                    visited[child] = True
                    if np.dot(result[parent], result[child]) < 0:
                        result[child] *= -1
                    stack.append(int(child))
        component += 1
    outward = np.einsum("ij,ij->i", result, points - np.median(points, axis=0))
    scores = np.bincount(labels, weights=outward, minlength=component)
    flip = scores < 0
    ambiguous = np.abs(scores) < 1e-9 * np.bincount(labels, minlength=component)
    anchors = result[roots]
    anchor_sign = anchors[np.arange(component), np.argmax(np.abs(anchors), axis=1)]
    flip[ambiguous] = anchor_sign[ambiguous] < 0
    result[flip[labels]] *= -1
    return result, component


def local_planes(points, neighbors):
    """Gaussian weighted local PCA, evaluated in bounded batches."""
    count = min(neighbors, len(points))
    tree = cKDTree(points)
    normals = np.empty_like(points)
    centers = np.empty_like(points)
    ratio = np.empty(len(points))
    radius = np.empty(len(points))
    for start in range(0, len(points), 5000):
        query = points[start:start + 5000]
        distances, indices = tree.query(query, k=count, workers=-1)
        neighbor = points[indices]
        bandwidth = np.maximum(distances[:, -1], np.finfo(float).eps)
        weights = np.exp(-2 * (distances / bandwidth[:, None]) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        center = np.einsum("nk,nkj->nj", weights, neighbor)
        delta = neighbor - center[:, None]
        covariance = np.einsum("nk,nki,nkj->nij", weights, delta, delta)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        end = start + len(query)
        normals[start:end] = eigenvectors[:, :, 0]
        centers[start:end] = center
        ratio[start:end] = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-20)
        radius[start:end] = bandwidth
    return normals, centers, ratio, radius


def prepare(cloud, settings):
    """Remove isolated noise, regularize sampling, and denoise flat neighborhoods.

    Projection is limited to the normal direction and capped by local spacing;
    high-curvature/edge neighborhoods are protected from plane projection.
    """
    cloud = unique_finite(cloud)
    points = np.asarray(cloud.points)
    median, local, indices = spacing(points)
    neighbor_scale = np.median(local[indices[:, 1:]], axis=1)
    keep = local <= 3.5 * np.maximum(neighbor_scale, median * 0.35)
    retained = cloud.select_by_index(np.flatnonzero(keep).tolist())
    voxel = median * settings.voxel_fraction
    cloud = retained.voxel_down_sample(voxel)
    while len(cloud.points) > settings.max_points:
        voxel *= 1.2
        # Resample the original retained cloud, not progressively averaged voxels.
        cloud = retained.voxel_down_sample(voxel)
    colors = np.asarray(cloud.colors).copy() if cloud.has_colors() else None
    points = np.asarray(cloud.points).copy()
    _, local, _ = spacing(points)
    moved = np.zeros(len(points))
    for _ in range(settings.denoise_iterations):
        normals, centers, ratio, _ = local_planes(points, settings.neighbors)
        displacement = np.einsum("ij,ij->i", centers - points, normals)
        # Fade out denoising near edges or neighborhoods that do not fit a plane.
        confidence = np.clip((0.20 - ratio) / 0.15, 0, 1)
        shift = np.clip(displacement * confidence * 0.7, -0.6 * local, 0.6 * local)
        points += shift[:, None] * normals
        moved += np.abs(shift)
    normals, _, _, _ = local_planes(points, settings.neighbors)
    median, local, _ = spacing(points)
    normals, components = orient_normals(points, normals, local)
    return cloud_from_arrays(points, colors, normals), {
        "input_points": len(keep), "isolated_removed": int((~keep).sum()),
        "prepared_points": len(points), "voxel_size": voxel,
        "median_spacing": median, "normal_components": components,
        "median_denoise_displacement": float(np.median(moved)),
        "p95_denoise_displacement": float(np.quantile(moved, 0.95)),
    }


def split_vertex_fans(mesh):
    """Separate surface patches touching only at a vertex, without moving faces.

    Support trimming can create pinched boundaries. Duplicating their shared
    vertex repairs topology while keeping geometry, gaps, and colors intact.
    """
    bad = list(mesh.get_non_manifold_vertices())
    if not bad:
        return mesh
    vertices = np.asarray(mesh.vertices).tolist()
    faces = np.asarray(mesh.triangles).copy()
    colors = np.asarray(mesh.vertex_colors).tolist() if mesh.has_vertex_colors() else None
    face_ids = np.repeat(np.arange(len(faces)), 3)
    incidence = coo_matrix((np.ones(len(face_ids)), (faces.ravel(), face_ids)), shape=(len(vertices), len(faces))).tocsr()
    for vertex in bad:
        incident = incidence.indices[incidence.indptr[vertex]:incidence.indptr[vertex + 1]]
        parents = list(range(len(incident)))

        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        edge_faces = {}
        for index, face_id in enumerate(incident):
            for neighbor in faces[face_id]:
                if neighbor == vertex:
                    continue
                if int(neighbor) in edge_faces:
                    parents[root(index)] = root(edge_faces[int(neighbor)])
                else:
                    edge_faces[int(neighbor)] = index
        groups = {}
        for index, face_id in enumerate(incident):
            groups.setdefault(root(index), []).append(int(face_id))
        for group in list(groups.values())[1:]:
            replacement = len(vertices)
            vertices.append(vertices[vertex].copy())
            if colors is not None:
                colors.append(colors[vertex].copy())
            for face_id in group:
                faces[face_id, faces[face_id] == vertex] = replacement
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = split_vertex_fans(mesh)
    mesh.compute_vertex_normals()
    return mesh


def supported_mesh(mesh, reference, multiplier):
    """Trim unsupported triangle interiors as well as vertices.

    Checking vertices alone would keep triangles bridging empty scan regions.
    Density thresholds scale locally so sparse valid patches can survive.
    """
    points = np.asarray(reference.points)
    _, local, _ = spacing(points)
    local = np.minimum(local, np.quantile(local, 0.95) * 2)
    tree = cKDTree(points)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if not len(faces):
        return mesh
    distance, index = tree.query(vertices, workers=-1)
    vertex_ok = distance <= multiplier * local[index]
    keep = vertex_ok[faces].all(axis=1)
    # Interior and midpoint checks prevent long triangles across missing data.
    for start in range(0, len(faces), 20000):
        triangle = vertices[faces[start:start + 20000]]
        probes = np.stack([
            triangle.mean(axis=1), (triangle[:, 0] + triangle[:, 1]) / 2,
            (triangle[:, 1] + triangle[:, 2]) / 2,
            (triangle[:, 2] + triangle[:, 0]) / 2,
        ], axis=1)
        distance, index = tree.query(probes.reshape(-1, 3), workers=-1)
        ok = (distance <= multiplier * local[index]).reshape(-1, 4).all(axis=1)
        keep[start:start + len(triangle)] &= ok
    mesh.remove_triangles_by_mask(~keep)
    return clean_mesh(mesh)


def build_candidate(prepared, kind, settings):
    points = np.asarray(prepared.points)
    median, local, _ = spacing(points)
    if kind == "multi_radius_bpa":
        base = float(np.median(local))
        radii = [base * value for value in (1.2, 2.0, 3.2)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            prepared, o3d.utility.DoubleVector(radii)
        )
        detail = {"radii": radii}
    elif kind.startswith("trimmed_poisson"):
        extent = float(np.ptp(points, axis=0).max())
        depth = int(np.clip(np.ceil(np.log2(extent / (2 * median))), 6, settings.max_depth))
        if kind.endswith("coarse"):
            depth = max(6, depth - 1)
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            prepared, depth=depth, scale=1.05, linear_fit=True, n_threads=settings.threads
        )
        detail = {"depth": depth, "low_density_p01": float(np.quantile(density, 0.01))}
    else:
        raise ValueError(f"Unknown candidate: {kind}")
    before = len(mesh.triangles)
    mesh = supported_mesh(mesh, prepared, settings.support_multiplier)
    detail.update({"triangles_before_support_trim": before, "triangles_after_support_trim": len(mesh.triangles)})
    return mesh, detail


def transfer_colors(mesh, reference):
    """Retain original RGB in every candidate and final mesh, after topology edits."""
    if reference.has_colors() and len(mesh.vertices):
        _, nearest = cKDTree(np.asarray(reference.points)).query(np.asarray(mesh.vertices), workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(reference.colors)[nearest], 0, 1))
    return mesh


def validation_split(clean, settings):
    """Shared normalization and split for production selection and benchmarking."""
    points = np.asarray(clean.points)
    if len(points) < 100:
        raise ValueError("Reconstruction requires at least 100 distinct finite points.")
    origin = np.median(points, axis=0)
    scale = float(np.ptp(np.quantile(points, [0.005, 0.995], axis=0), axis=0).max())
    if scale <= 0:
        raise ValueError("Point cloud has zero spatial extent.")
    normalized = cloud_from_arrays((points - origin) / scale, np.asarray(clean.colors) if clean.has_colors() else None)
    indices = np.random.default_rng(settings.seed).permutation(len(points))
    count = max(20, int(len(points) * settings.validation_fraction))
    return normalized, normalized.select_by_index(indices[count:].tolist()), normalized.select_by_index(indices[:count].tolist()), origin, scale


def select_candidate(reports):
    """Only improved methods can win; near ties favor fewer triangles."""
    valid = {key: value for key, value in reports.items() if key in CANDIDATE_METHODS and "selection_score" in value}
    if not valid:
        raise RuntimeError(f"All improved reconstruction candidates failed: {reports}")
    best_score = min(value["selection_score"] for value in valid.values())
    margin = max(0.01, abs(best_score) * 0.03)
    near = {key: value for key, value in valid.items() if value["selection_score"] <= best_score + margin}
    winner = min(near, key=lambda key: (near[key]["triangles"], near[key]["selection_score"]))
    return winner, margin


def sample_surface(mesh, count, seed):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    triangle = vertices[faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1) / 2
    if not len(area) or area.sum() <= 0:
        raise ValueError("Reconstruction produced no nonzero-area triangles.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(faces), count, p=area / area.sum())
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    selected = triangle[indices]
    samples = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    return samples, area


def evaluate(mesh, validation, support, settings):
    """Unsigned point-to-triangle distance, area support, and topology checks."""
    if not len(mesh.triangles):
        raise ValueError("Reconstruction produced an empty mesh.")
    scene = o3d.t.geometry.RaycastingScene(nthreads=settings.threads)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points = np.asarray(validation.points)
    if len(points) > settings.metric_samples:
        points = points[np.random.default_rng(settings.seed).choice(len(points), settings.metric_samples, replace=False)]
    distance = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy().astype(float)
    support_points = np.asarray(support.points)
    median, local, _ = spacing(support_points)
    tree = cKDTree(support_points)
    _, idx = tree.query(points, workers=-1)
    relative = distance / local[idx]
    sampled, areas = sample_surface(mesh, settings.metric_samples, settings.seed)
    surface_distance, idx = tree.query(sampled, workers=-1)
    unsupported = float(np.mean(surface_distance > settings.support_multiplier * local[idx]))
    triangles = np.asarray(mesh.triangles)
    vertex = np.asarray(mesh.vertices)
    xyz = vertex[triangles]
    length = np.stack([np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1), np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1), np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1)], axis=1)
    quality = 4 * np.sqrt(3) * areas / np.maximum((length ** 2).sum(axis=1), 1e-20)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    labels, component_counts, component_areas = mesh.cluster_connected_triangles()
    component_counts, component_areas = np.asarray(component_counts), np.asarray(component_areas)
    tiny = component_counts < 20
    tiny_fraction = float(component_areas[tiny].sum() / areas.sum())
    sliver_fraction = float(areas[quality < 0.1].sum() / areas.sum())
    coverage = float(np.mean(relative <= 2))
    # Boundary count is reported, not minimized: real openings must survive.
    score = float(np.quantile(relative, 0.95) + 3 * np.mean(relative > 2) + 4 * unsupported + 2 * tiny_fraction + 0.5 * sliver_fraction)
    return {
        "vertices": len(vertex), "triangles": len(triangles),
        "point_to_surface_median": float(np.median(distance)),
        "point_to_surface_p95": float(np.quantile(distance, 0.95)),
        "point_to_surface_p95_spacing": float(np.quantile(relative, 0.95)),
        "coverage_within_2_local_spacings": coverage,
        "unsupported_surface_fraction": unsupported,
        "components": len(component_counts), "tiny_component_area_fraction": tiny_fraction,
        "sliver_area_fraction": sliver_fraction,
        "boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
        "nonmanifold_vertices": len(mesh.get_non_manifold_vertices()),
        "surface_area": float(areas.sum()), "median_support_spacing": median,
        "selection_score": score,
    }


def reconstruct(cloud, settings=None, progress=print, keep_candidates=False, selection="auto"):
    """Select per input on validation points, then refit on every valid point."""
    settings = settings or Settings()
    settings.validate()
    if selection not in ("auto", *PRODUCTION_METHODS):
        raise ValueError(f"Production method must be auto or one of {PRODUCTION_METHODS}.")
    start = time.perf_counter()
    raw_count = len(cloud.points)
    clean = unique_finite(cloud)
    cloud, training, validation, origin, scale = validation_split(clean, settings)
    progress(f"Preparing {len(training.points)} training points; holding out {len(validation.points)} points.")
    prepared, preparation = prepare(training, settings)
    reports = {}
    meshes = {}
    for kind, method in CANDIDATE_METHODS.items():
        if selection != "auto" and method != selection:
            continue
        progress(f"Testing {kind}...")
        candidate_start = time.perf_counter()
        try:
            mesh, detail = build_candidate(prepared, kind, settings)
            transfer_colors(mesh, training)
            metrics = evaluate(mesh, validation, training, settings)
            metrics.update(detail)
            metrics["seconds"] = time.perf_counter() - candidate_start
            reports[kind] = metrics
            if keep_candidates:
                world = o3d.geometry.TriangleMesh(mesh)
                world.vertices = o3d.utility.Vector3dVector(np.asarray(world.vertices) * scale + origin)
                if world.has_vertex_colors():
                    world.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(world.vertex_colors), 0, 1))
                meshes[kind] = world
            progress(f"  score={metrics['selection_score']:.3f}, coverage={metrics['coverage_within_2_local_spacings']:.1%}, unsupported={metrics['unsupported_surface_fraction']:.1%}")
        except (RuntimeError, ValueError) as exc:
            reports[kind] = {"error": str(exc)}
            progress(f"  rejected: {exc}")
    winner, near_tie_margin = select_candidate(reports)
    progress(f"Selected {winner}; refitting on all {len(cloud.points)} valid points.")
    prepared, final_preparation = prepare(cloud, settings)
    mesh, detail = build_candidate(prepared, winner, settings)
    if not len(mesh.triangles):
        raise RuntimeError("Final full-input reconstruction is empty.")
    # Transfer colors from original input after reconstruction and before restoring units.
    transfer_colors(mesh, cloud)
    final_metrics = evaluate(mesh, cloud, cloud, settings)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale + origin)
    mesh.compute_vertex_normals()
    report = {
        "selected_method": CANDIDATE_METHODS[winner], "selected_variant": winner,
        "requested_method": selection, "settings": asdict(settings),
        "open3d_version": o3d.__version__, "numpy_version": np.__version__,
        "raw_input_points": raw_count, "finite_unique_points": len(cloud.points),
        "normalization_origin": origin.tolist(), "normalization_scale": scale,
        "training_points": len(training.points), "validation_points": len(validation.points),
        "validation_preparation": preparation, "validation_candidates": reports,
        "selection_near_tie_margin": near_tie_margin,
        "final_preparation": final_preparation, "final_parameters": detail,
        "final_input_consistency": final_metrics, "seconds": time.perf_counter() - start,
        "metric_units": "normalized by normalization_scale; multiply distance/spacing by scale and area by scale**2 for input units",
        "limitations": [
            "Held-out scan consistency is not ground-truth accuracy.",
            "Unobserved geometry and gaps below the local sample spacing cannot be recovered reliably.",
            "Automatic normal orientation does not substitute for known scanner poses.",
            "Small components and open boundaries are preserved; output need not be watertight.",
        ],
    }
    return mesh, report, meshes


def save_result(source, mesh, report, output_root=None, input_root=None):
    """Save a complete coloured production result under its selected method."""
    source = Path(source)
    method = report["selected_method"]
    output = result_directory(source, method, output_root, input_root)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mesh_path = output / f"{source.stem}_{method}_{timestamp}.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True):
        raise RuntimeError(f"Failed to save {mesh_path}")
    reloaded = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.has_vertex_colors() and not reloaded.has_vertex_colors():
        raise RuntimeError(f"Vertex colours were lost saving {mesh_path}")
    report.update({"input": str(source.resolve()), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "output": str(mesh_path), "vertex_colors_saved": reloaded.has_vertex_colors()})
    mesh_path.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    from .mesh_preview import save_preview
    save_preview(o3d.io.read_point_cloud(str(source)), None, mesh, report, mesh_path.with_name(mesh_path.stem + "_preview.png"))
    return mesh_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASETS_DIR / "captures", help="Input cloud or folder")
    parser.add_argument("--method", choices=("auto", *PRODUCTION_METHODS), default="auto")
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    input_path = resolve_path(args.input)
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pcd")) + sorted(input_path.glob("*.ply"))
    if not files:
        parser.error(f"No point clouds found: {input_path}")
    settings = Settings(max_points=args.max_points, max_depth=args.max_depth, threads=args.threads)
    settings.validate()
    failures = 0
    for source in files:
        print(f"\n{source.name}", flush=True)
        try:
            cloud = o3d.io.read_point_cloud(str(source))
            mesh, report, _ = reconstruct(cloud, settings, progress=lambda text: print(text, flush=True), selection=args.method)
            mesh_path = save_result(source, mesh, report)
            print(f"Saved {mesh_path}", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"FAILED: {source.name}: {exc}", flush=True)
            failures += 1
    return 1 if failures else 0


def reconstruct_file(input_path, output_root=None, parameters=None, selection="auto", input_root=None):
    """Entry point for the existing config-driven CLI."""
    try:
        source = Path(input_path)
        settings = Settings(**(parameters or {}))
        cloud = o3d.io.read_point_cloud(str(source))
        mesh, report, _ = reconstruct(cloud, settings, progress=lambda text: print(text, flush=True), selection=selection)
        output = save_result(source, mesh, report, output_root, input_root)
        return True, f"Saved {report['selected_method']} mesh to {output}"
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return False, f"Adaptive reconstruction failed: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
