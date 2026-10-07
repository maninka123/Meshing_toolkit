"""Three standard baselines. Never imported by the production selector.

Common baseline normals use KNN=30 and tangent-plane orientation. No denoising,
support trimming, multi-radius pivoting, or held-out parameter tuning is applied.
"""

import numpy as np
import open3d as o3d

from reconstruction.adaptive import cloud_from_arrays, spacing, transfer_colors

BASELINE_METHODS = ("baseline_ball_pivoting", "baseline_poisson", "baseline_alpha_shape")


def prepare_baseline(cloud, settings):
    sampled = cloud_from_arrays(np.asarray(cloud.points), np.asarray(cloud.colors) if cloud.has_colors() else None)
    if len(sampled.points) > settings.max_points:
        median, _, _ = spacing(np.asarray(sampled.points))
        voxel = median * settings.voxel_fraction
        while len(sampled.points) > settings.max_points:
            sampled = cloud.voxel_down_sample(voxel)
            voxel *= 1.2
    sampled.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    sampled.orient_normals_consistent_tangent_plane(30)
    return sampled


def build_baseline(cloud, method, settings):
    median, _, _ = spacing(np.asarray(cloud.points))
    if method == "baseline_ball_pivoting":
        radius = float(np.mean(cloud.compute_nearest_neighbor_distance())) * 2.5
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector([radius]))
        params = {"radius": radius}
    elif method == "baseline_poisson":
        depth = min(9, settings.max_depth)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth, scale=1.1, linear_fit=False, n_threads=settings.threads)
        params = {"depth": depth, "scale": 1.1, "linear_fit": False}
    elif method == "baseline_alpha_shape":
        alpha = 5 * median
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
        except (IndexError, RuntimeError) as exc:
            raise RuntimeError(f"Alpha Shape failed on this input (possibly degenerate tetrahedra): {exc}") from exc
        params = {"alpha": alpha, "alpha_spacing_multiplier": 5}
    else:
        raise ValueError(f"Unknown benchmark baseline: {method}")
    # Preserve baseline geometry/topology so problems remain visible in metrics.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    transfer_colors(mesh, cloud)
    return mesh, params
