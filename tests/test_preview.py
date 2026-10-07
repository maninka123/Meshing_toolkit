"""Point previews must align with the actual mesh camera, rather than mirror it."""

import unittest
import numpy as np

from reconstruction.adaptive import cloud_from_arrays
from reconstruction.mesh_preview import camera, render_points


class PreviewTests(unittest.TestCase):
    def test_known_camera_rays_project_back_to_the_same_pixels(self):
        bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
        rays, _, _, _ = camera(bounds, width=80, height=60)
        ray_array = rays.numpy()
        pixels = [(12, 21), (48, 65)]
        points = np.array([ray_array[y, x, :3] + 2 * ray_array[y, x, 3:] for y, x in pixels])
        colors = np.array([[1., 0., 0.], [0., 1., 0.]])
        rendered = render_points(cloud_from_arrays(points, colors), rays)
        for (y, x), color in zip(pixels, colors):
            np.testing.assert_array_equal(rendered[y, x], color)
