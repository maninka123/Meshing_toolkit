"""Known-geometry checks for adaptive reconstruction, independently of captures."""

from dataclasses import replace
import unittest
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from reconstruction.adaptive import Settings, cloud_from_arrays, prepare, reconstruct, sample_surface, split_vertex_fans, supported_mesh, select_candidate


class AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings(max_points=6000, max_depth=8, threads=2, metric_samples=2000)

    def test_baseline_cannot_win_even_with_a_better_score(self):
        winner, _ = select_candidate({"baseline_bpa": {"selection_score": 0, "triangles": 1},
                                     "multi_radius_bpa": {"selection_score": 1, "triangles": 100}})
        self.assertEqual(winner, "multi_radius_bpa")

    def test_both_improved_methods_keep_spatial_colors_after_export(self):
        import open3d as o3d
        rng = np.random.default_rng(44)
        points = rng.normal(size=(1800, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        colors = np.column_stack([points[:, 0] > 0, points[:, 0] <= 0, np.full(len(points), 0.25)]).astype(float)
        with tempfile.TemporaryDirectory() as temp:
            for method in ["improved_ball_pivoting", "improved_poisson"]:
                mesh, report, candidates = reconstruct(cloud_from_arrays(points, colors), self.settings, progress=lambda _: None, selection=method, keep_candidates=True)
                self.assertEqual(report["selected_method"], method)
                self.assertTrue(all(candidate.has_vertex_colors() for candidate in candidates.values()))
                output = Path(temp) / f"{method}.ply"
                self.assertTrue(o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_colors=True))
                loaded = o3d.io.read_triangle_mesh(str(output))
                self.assertTrue(loaded.has_vertex_colors())
                verts, rgb = np.asarray(loaded.vertices), np.asarray(loaded.vertex_colors)
                self.assertGreater(np.mean(rgb[verts[:, 0] > 0.2, 0]), 0.95)
                self.assertGreater(np.mean(rgb[verts[:, 0] < -0.2, 1]), 0.95)
                np.testing.assert_allclose(rgb[:, 2], 0.25, atol=1 / 255)

    def test_denoising_reduces_plane_noise_and_preserves_colors(self):
        rng = np.random.default_rng(12)
        xy = rng.uniform(-1, 1, (4000, 2))
        points = np.column_stack([xy, rng.normal(0, 0.015, len(xy))])
        colors = np.tile([0.2, 0.4, 0.8], (len(points), 1))
        prepared, detail = prepare(cloud_from_arrays(points, colors), replace(self.settings, max_points=2000))
        self.assertLess(np.std(np.asarray(prepared.points)[:, 2]), np.std(points[:, 2]) * 0.8)
        self.assertLessEqual(len(prepared.points), 2000)
        self.assertTrue(prepared.has_colors())
        np.testing.assert_allclose(np.asarray(prepared.colors), np.tile([0.2, 0.4, 0.8], (len(prepared.points), 1)))

    def test_open_plane_hole_is_not_filled(self):
        rng = np.random.default_rng(5)
        xy = rng.uniform(-1, 1, (5000, 2))
        xy = xy[np.linalg.norm(xy, axis=1) > 0.35]
        points = np.column_stack([xy, rng.normal(0, 0.003, len(xy))])
        mesh, report, _ = reconstruct(cloud_from_arrays(points), self.settings, progress=lambda _: None)
        samples, _ = sample_surface(mesh, 10000, 7)
        self.assertEqual(int((np.linalg.norm(samples[:, :2], axis=1) < 0.20).sum()), 0)
        self.assertGreater(report["final_input_consistency"]["boundary_edges"], 0)
        self.assertEqual(report["final_input_consistency"]["nonmanifold_edges"], 0)

    def test_curved_surface_recovers_sphere_without_assuming_a_camera(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(3500, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= 1 + rng.normal(0, 0.009, (len(points), 1))
        mesh, report, _ = reconstruct(cloud_from_arrays(points), self.settings, progress=lambda _: None)
        samples, _ = sample_surface(mesh, 5000, 4)
        self.assertLess(float(np.quantile(np.abs(np.linalg.norm(samples, axis=1) - 1), 0.95)), 0.035)
        self.assertEqual(report["final_input_consistency"]["nonmanifold_edges"], 0)

    def test_scale_and_translation_do_not_change_reconstructed_geometry(self):
        rng = np.random.default_rng(22)
        xy = rng.uniform(-1, 1, (2200, 2))
        points = np.column_stack([xy, 0.1 * xy[:, 0] ** 2 + rng.normal(0, 0.004, len(xy))])
        mesh1, report1, _ = reconstruct(cloud_from_arrays(points), self.settings, progress=lambda _: None)
        factor, offset = 1000.0, np.array([15000.0, -20000.0, 10000.0])
        mesh2, report2, _ = reconstruct(cloud_from_arrays(points * factor + offset), self.settings, progress=lambda _: None)
        restored = (np.asarray(mesh2.vertices) - offset) / factor
        distance, _ = cKDTree(np.asarray(mesh1.vertices)).query(restored)
        self.assertLess(float(np.quantile(distance, 0.95)), 0.01)
        self.assertEqual(report1["selected_method"], report2["selected_method"])

    def test_folded_surfaces_keep_a_sharp_corner(self):
        rng = np.random.default_rng(33)
        floor = np.column_stack([rng.uniform(-1, 0, 2400), rng.uniform(-1, 1, 2400), np.zeros(2400)])
        wall = np.column_stack([np.zeros(2400), rng.uniform(-1, 1, 2400), rng.uniform(0, 1, 2400)])
        points = np.vstack([floor, wall]) + rng.normal(0, 0.003, (4800, 3))
        mesh, _, _ = reconstruct(cloud_from_arrays(points), self.settings, progress=lambda _: None)
        samples, _ = sample_surface(mesh, 8000, 8)
        distance_to_surfaces = np.minimum(np.abs(samples[:, 0]), np.abs(samples[:, 2]))
        self.assertLess(float(np.quantile(distance_to_surfaces, 0.95)), 0.025)

    def test_isolated_noise_is_removed_without_erasing_the_main_surface(self):
        rng = np.random.default_rng(11)
        points = np.column_stack([rng.uniform(-1, 1, (1800, 2)), np.zeros(1800)])
        points = np.vstack([points, [20, 30, 40]])
        prepared, detail = prepare(cloud_from_arrays(points), self.settings)
        self.assertEqual(detail["isolated_removed"], 1)
        self.assertLess(float(np.max(np.linalg.norm(np.asarray(prepared.points), axis=1))), 2)

    def test_triangle_across_a_large_gap_is_removed_even_if_its_vertices_are_supported(self):
        import open3d as o3d
        rng = np.random.default_rng(2)
        first = np.column_stack([rng.uniform(-1.1, -0.9, 400), rng.uniform(-0.2, 0.2, 400), np.zeros(400)])
        second = first.copy()
        second[:, 0] += 2
        reference = cloud_from_arrays(np.vstack([first, second]))
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([first[0], first[1], second[0]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        trimmed = supported_mesh(mesh, reference, 2.5)
        self.assertEqual(len(trimmed.triangles), 0)

    def test_pinched_vertex_is_split_without_changing_triangle_geometry(self):
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 3, 4]])
        original = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].copy()
        self.assertFalse(mesh.is_vertex_manifold())
        mesh = split_vertex_fans(mesh)
        self.assertTrue(mesh.is_vertex_manifold())
        np.testing.assert_array_equal(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)], original)


if __name__ == "__main__":
    unittest.main()
