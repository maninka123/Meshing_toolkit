#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:47:04 2025

@author: Pasindu
"""

import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree
import os
from datetime import datetime

# =========================== #
# 1. Load and View Point Cloud #
# =========================== #

def remove_statistical_outliers_with_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove outlier points based on statistical analysis of neighbors,
    and return both the inlier and outlier point clouds.
    
    :param pcd: Input open3d.geometry.PointCloud
    :param nb_neighbors: Number of neighbors to consider.
    :param std_ratio: Standard deviation ratio threshold.
    :return: Tuple (inlier_cloud, outlier_cloud)
    """
    inlier_cloud, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                                  std_ratio=std_ratio)
    print(f"Removed {len(pcd.points) - len(inlier_indices)} outlier points")
    
    # Compute outlier indices as the complement of inlier_indices
    all_indices = np.arange(len(pcd.points))
    outlier_indices = np.setdiff1d(all_indices, inlier_indices)
    
    # Select outlier points
    outlier_cloud = pcd.select_by_index(outlier_indices)
    
    return inlier_cloud, outlier_cloud

# ---------------------------
# Main Visualization Function
# ---------------------------
def view_3d(dataset):
    """Load and visualize a point cloud with outliers shown in red."""
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(dataset)
    
    # Remove outliers (using your chosen parameters)
    inlier_cloud, outlier_cloud = remove_statistical_outliers_with_outliers(pcd, nb_neighbors=4, std_ratio=5)
    
    # Paint the outlier cloud in red
    outlier_cloud.paint_uniform_color([1, 0, 0])
    
    # Visualize inliers (original colors) and outliers (red)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      window_name="Inliers (original colors) and Outliers (red)")
    
    return inlier_cloud, outlier_cloud

datasets = ["final_point_cloud_colored.pcd"]
# Load the first point cloud
pcd,_ = view_3d(datasets[0])
#%%
# =========================== #
# 2. Convert Point Cloud to Mesh with Color Transfer #
# =========================== #

def point_cloud_to_mesh(pcd, voxel_size=0.05, iso_level_percentile=50):
    """
    Convert a point cloud to a mesh using marching cubes, then transfer color
    from the original point cloud to the mesh vertices using nearest neighbor lookup.
    """
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    if has_colors:
        orig_colors = np.asarray(pcd.colors)
    else:
        orig_colors = None

    # Compute the bounds of the point cloud
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    # Create a 3D grid over the point cloud bounds
    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Build a KD-tree for the original points (to compute the distance field)
    tree = cKDTree(points)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(x.shape)

    # Determine iso-level from the given percentile of distances
    iso_level = np.percentile(distances, iso_level_percentile)

    # Extract mesh via the Marching Cubes algorithm
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)

    # Scale and translate vertices back to original coordinate system
    verts = verts * voxel_size + mins

    # Create the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Transfer color information: for each vertex, assign the color of its nearest neighbor in the original cloud.
    if has_colors:
        tree_colors = cKDTree(points)
        _, idx = tree_colors.query(verts)
        vertex_colors = orig_colors[idx]
    else:
        vertex_colors = np.zeros((verts.shape[0], 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh

# Convert point cloud to mesh with fixed parameters for testing
#mesh = point_cloud_to_mesh(pcd, voxel_size=0.1, iso_level_percentile=20)
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# Optionally, save the mesh:
# o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

# =========================== #
# 3. Automatic Parameter Estimation #
# =========================== #

def estimate_parameters(pcd):
    """
    Automatically estimate a suitable voxel size and iso-level percentile based on
    the point cloud density.
    """
    points = np.asarray(pcd.points)
    k = min(30, len(points) - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)  # +1 excludes self-distance
    avg_distance = np.mean(distances[:, 1:])   # Ignore self-distance
    voxel_size = avg_distance * 2             # Heuristic multiplier

    all_distances, _ = tree.query(points, k=4)
    nearest_distances = all_distances[:, 1]     # Exclude self-distance
    cv = np.std(nearest_distances) / np.mean(nearest_distances)
    iso_level_percentile = 50 * (1 - cv)          # Map CV [0,1] to percentile [50,0]
    iso_level_percentile = np.clip(iso_level_percentile, 10, 90)  # Limit range

    return voxel_size, iso_level_percentile

# Compute and print automatic parameters
p_voxel_size, p_iso_level_percentile = estimate_parameters(pcd)
print(f"Auto-Estimated Parameters: Voxel Size = {p_voxel_size}, Iso-Level Percentile = {p_iso_level_percentile}")

# =========================== #
# 4. Generate Mesh with Auto-Tuned Parameters #
# =========================== #

optimized_mesh = point_cloud_to_mesh(pcd, voxel_size=(p_voxel_size/1.5), iso_level_percentile=p_iso_level_percentile/6)
o3d.visualization.draw_geometries([optimized_mesh])
#o3d.io.write_triangle_mesh("optimized_mesh.ply", optimized_mesh)

print("Mesh generation complete!")
