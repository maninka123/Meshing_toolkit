"""Production method restrictions, input discovery, and result placement."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconstruction import cli, paths


class LayoutTests(unittest.TestCase):
    def test_outputs_are_grouped_by_dataset_point_cloud_and_method(self):
        source = paths.DATASETS_DIR / "captures" / "capture_1.pcd"
        self.assertEqual(paths.result_directory(source, "improved_poisson"), source.parent / "capture_1" / "meshes" / "improved_poisson")

    def test_nested_dataset_has_same_destination_in_single_and_batch_runs(self):
        source = paths.DATASETS_DIR / "example" / "nested" / "scan.pcd"
        single = paths.result_directory(source, "improved_ball_pivoting")
        batch = paths.result_directory(source, "improved_ball_pivoting", input_root=source.parent.parent)
        self.assertEqual(single, batch)
        self.assertEqual(single, source.parent / "scan" / "meshes" / "improved_ball_pivoting")

    def test_recursive_discovery_excludes_generated_ply_and_benchmark_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["scan.pcd", "nested/scan.pcd", "meshes/old.ply", "benchmark/samples/output.ply", "saved/converted.ply", "scan/meshes/improved_poisson/output.ply"]
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            files = cli.collect_files(root, [".pcd", ".ply"], True, [root / "saved"])
            self.assertEqual(set(map(Path, files)), {root / "scan.pcd", root / "nested/scan.pcd"})

    def test_nonrecursive_discovery_only_collects_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "folder.pcd").mkdir()
            (root / "scan.PCD").touch()
            self.assertEqual(cli.collect_files(root, [".pcd"], False), [str(root / "scan.PCD")])

    def test_custom_config_is_independent_of_cwd_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input").mkdir()
            (root / "input" / "scan.pcd").touch()
            config = root / "custom.ini"
            config.write_text("[general]\ninput_path = input\noutput_dir = saved\n", encoding="utf-8")
            original = Path.cwd()
            try:
                os.chdir(root / "input")
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.main(["--config", str(config), "--dry-run"]), 0)
            finally:
                os.chdir(original)
            self.assertIn(str(root / "saved" / "input" / "scan" / "meshes"), output.getvalue())
            self.assertFalse((root / "saved").exists())

    def test_auto_does_not_skip_a_manually_forced_mesh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "improved_poisson"
            folder.mkdir()
            mesh = folder / "scan_improved_poisson_123.ply"
            mesh.touch()
            report = mesh.with_suffix(".json")
            report.write_text(json.dumps({"requested_method": "improved_poisson"}))
            self.assertEqual(cli.existing_mesh_for_input(root, "scan.pcd", "auto"), "")
            self.assertEqual(cli.existing_mesh_for_input(folder, "scan.pcd", "improved_poisson"), str(mesh))
            report.write_text(json.dumps({"requested_method": "auto"}))
            self.assertEqual(cli.existing_mesh_for_input(root, "scan.pcd", "auto"), str(mesh))

    def test_baselines_are_rejected_by_production_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scan.pcd").touch()
            for method in ["ball_pivoting", "poisson", "alpha_shape", "baseline_poisson"]:
                config = root / "custom.ini"
                config.write_text(f"[general]\ninput_path = scan.pcd\nreconstruction_method = {method}\n")
                with patch.object(cli, "reconstruct_mesh") as reconstruct:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(["--config", str(config)]), 1)
                    reconstruct.assert_not_called()

    def test_cli_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scan.pcd").touch()
            config = root / "custom.ini"
            config.write_text("[general]\ninput_path = scan.pcd\n")
            with patch.object(cli, "reconstruct_mesh", return_value=(False, "failed")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["--config", str(config)]), 1)

    def test_matching_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scan.pcd").touch()
            (root / "scan.ply").touch()
            config = root / "custom.ini"
            config.write_text("[general]\ninput_path = .\nextensions = .pcd, .ply\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--config", str(config), "--dry-run"]), 1)


if __name__ == "__main__":
    unittest.main()
