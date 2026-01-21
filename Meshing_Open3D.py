#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:47:04 2025

@author: Pasindu
"""

import os
from datetime import datetime

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage import measure

# ============================
# Configuration
# ============================

DATASET_PATH = "final_point_cloud_colored.pcd"
SAVE_DIR = "meshes"

OUTLIER_NB_NEIGHBORS = 4
OUTLIER_STD_RATIO = 5.0

MARCHING_CUBES_VOXEL_SCALE = 1.5
MARCHING_CUBES_ISO_PERCENTILE_SCALE = 6.0

BPA_RADIUS_MULTIPLIER = 3.0
BPA_DECIMATION_TARGET = 50000

POISSON_DEPTH = 10
POISSON_SCALE = 1.0

ALPHA_VALUE = 0.02
ALPHA_DECIMATION_TARGET = 50000

FINE_TUNE_SUBDIVIDE_ITERATIONS = 0  # 0 disables loop subdivision
SAVE_MESH = True

# ============================
# Helper Functions
# ============================


def get_timestamp():
    """Generate a timestamped filename including milliseconds."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]


def remove_statistical_outliers_with_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove outlier points based on statistical analysis of neighbors,
    and return both the inlier and outlier point clouds.
    """
    inlier_cloud, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(f"Removed {len(pcd.points) - len(inlier_indices)} outlier points")
    all_indices = np.arange(len(pcd.points))
    outlier_indices = np.setdiff1d(all_indices, inlier_indices)
    outlier_cloud = pcd.select_by_index(outlier_indices)
    return inlier_cloud, outlier_cloud


def estimate_parameters(pcd):
    """Estimate voxel size and iso-level percentile based on point density."""
    points = np.asarray(pcd.points)
    if len(points) < 2:
        raise ValueError("Point cloud must contain at least 2 points.")

    k = min(30, len(points) - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    avg_distance = np.mean(distances[:, 1:])
    voxel_size = avg_distance * 2

    all_distances, _ = tree.query(points, k=4)
    nearest_distances = all_distances[:, 1]
    cv = np.std(nearest_distances) / np.mean(nearest_distances)
    iso_level_percentile = 50 * (1 - cv)
    iso_level_percentile = np.clip(iso_level_percentile, 10, 90)

    return voxel_size, iso_level_percentile


# ============================
# Meshing Algorithms
# ============================


def mesh_marching_cubes(pcd, voxel_size=0.05, iso_level_percentile=50):
    """
    Convert a point cloud to a mesh using marching cubes,
    then transfer color from the original point cloud to mesh vertices.
    """
    points = np.asarray(pcd.points)
    if len(points) < 2:
        raise ValueError("Point cloud must contain at least 2 points.")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive.")

    has_colors = pcd.has_colors()
    orig_colors = np.asarray(pcd.colors) if has_colors else None

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    if np.any((maxs - mins) <= 0):
        raise ValueError("Point cloud bounds are degenerate.")

    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    tree = cKDTree(points)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(x.shape)

    iso_level = np.percentile(distances, iso_level_percentile)
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)
    verts = verts * voxel_size + mins

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if has_colors:
        tree_colors = cKDTree(points)
        _, idx = tree_colors.query(verts)
        vertex_colors = orig_colors[idx]
    else:
        vertex_colors = np.zeros((verts.shape[0], 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh


def mesh_bpa(pcd, radius_multiplier=3.0, decimation_target=50000):
    """Generate a mesh using the Ball-Pivoting Algorithm (BPA)."""
    distances = pcd.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        raise ValueError("Point cloud has no neighbor distances for BPA.")
    avg_dist = np.mean(distances)
    radius = radius_multiplier * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    return bpa_mesh.simplify_quadric_decimation(decimation_target)


def mesh_poisson(pcd, depth=10, scale=1.1):
    """Generate a mesh using Poisson Reconstruction."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=20)
    )
    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=scale, linear_fit=False
    )
    poisson_mesh.compute_vertex_normals()
    return poisson_mesh


def mesh_alpha(pcd, alpha=0.05, decimation_target=80000):
    """Generate a mesh using the Alpha Shape algorithm."""
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    return alpha_mesh.simplify_quadric_decimation(decimation_target)


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def fine_tune_mesh(mesh, iterations):
    if iterations <= 0:
        return mesh
    print("Fine tuning mesh...")
    mesh = mesh.subdivide_loop(number_of_iterations=iterations)
    return clean_mesh(mesh)


# ============================
# Main
# ============================


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    pcd = o3d.io.read_point_cloud(DATASET_PATH)
    if len(pcd.points) == 0:
        print(f"Point cloud is empty: {DATASET_PATH}")
        return
    print("Loaded point cloud with", len(pcd.points), "points.")

    print("Removing outliers...")
    inlier_cloud, outlier_cloud = remove_statistical_outliers_with_outliers(
        pcd, nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
    )
    if len(inlier_cloud.points) == 0:
        print("All points were removed as outliers.")
        return
    outlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        window_name="Preprocessed Point Cloud\n(Inliers: original colors; Outliers: red)",
    )

    print("\nSelect a meshing algorithm:")
    print("1: Marching Cubes with Color Transfer (auto-estimated parameters)")
    print("2: Ball-Pivoting Algorithm (BPA)")
    print("3: Poisson Reconstruction")
    print("4: Alpha Shape")
    choice = input("Enter choice number (1-4): ").strip()

    mesh = None
    if choice == "1":
        voxel_size, iso_level_percentile = estimate_parameters(inlier_cloud)
        voxel_size = voxel_size / MARCHING_CUBES_VOXEL_SCALE
        iso_level_percentile = iso_level_percentile / MARCHING_CUBES_ISO_PERCENTILE_SCALE
        iso_level_percentile = float(np.clip(iso_level_percentile, 1, 99))

        print(
            "Auto-Estimated Parameters for Marching Cubes:",
            f"Voxel Size = {voxel_size},",
            f"Iso-Level Percentile = {iso_level_percentile}",
        )
        mesh = mesh_marching_cubes(
            inlier_cloud,
            voxel_size=voxel_size,
            iso_level_percentile=iso_level_percentile,
        )
    elif choice == "2":
        print("Running Ball-Pivoting Algorithm (BPA)...")
        mesh = mesh_bpa(
            inlier_cloud,
            radius_multiplier=BPA_RADIUS_MULTIPLIER,
            decimation_target=BPA_DECIMATION_TARGET,
        )
    elif choice == "3":
        print("Running Poisson Reconstruction...")
        mesh = mesh_poisson(inlier_cloud, depth=POISSON_DEPTH, scale=POISSON_SCALE)
    elif choice == "4":
        print("Running Alpha Shape Algorithm...")
        mesh = mesh_alpha(
            inlier_cloud,
            alpha=ALPHA_VALUE,
            decimation_target=ALPHA_DECIMATION_TARGET,
        )
    else:
        print("Invalid choice. Exiting.")
        return

    mesh = clean_mesh(mesh)
    mesh = fine_tune_mesh(mesh, FINE_TUNE_SUBDIVIDE_ITERATIONS)

    o3d.visualization.draw_geometries([mesh], window_name="Generated Mesh")
    print("Meshing complete.")

    if SAVE_MESH:
        mesh_filename = f"mesh_{get_timestamp()}.ply"
        mesh_path = os.path.join(SAVE_DIR, mesh_filename)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Mesh saved as {mesh_path}")
    else:
        print("Mesh not saved.")


if __name__ == "__main__":
    main()
