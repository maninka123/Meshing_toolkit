"""Benchmark baselines stay separate and export their original point colours."""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import open3d as o3d

from benchmark.baselines import BASELINE_METHODS, prepare_baseline, build_baseline
from benchmark.run import write_mesh
from reconstruction.adaptive import Settings, PRODUCTION_METHODS, cloud_from_arrays


class BenchmarkTests(unittest.TestCase):
    def test_baselines_are_separate_from_production_choices(self):
        self.assertEqual(len(BASELINE_METHODS), 3)
        self.assertFalse(set(BASELINE_METHODS).intersection(PRODUCTION_METHODS))

    def test_all_three_baselines_save_vertex_rgb(self):
        rng = np.random.default_rng(77)
        points = rng.normal(size=(1000, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= 1 + rng.normal(0, 0.004, (len(points), 1))
        colors = np.tile([0.1, 0.7, 0.3], (len(points), 1))
        settings = Settings(max_depth=6, threads=2)
        prepared = prepare_baseline(cloud_from_arrays(points, colors), settings)
        with tempfile.TemporaryDirectory() as temp:
            for method in BASELINE_METHODS:
                mesh, _ = build_baseline(prepared, method, settings)
                target = Path(temp) / f"{method}.ply"
                write_mesh(target, mesh)
                rgb = np.asarray(o3d.io.read_triangle_mesh(str(target)).vertex_colors)
                self.assertGreater(len(rgb), 0)
                np.testing.assert_allclose(rgb, np.tile([0.1, 0.7, 0.3], (len(rgb), 1)), atol=1 / 255)
