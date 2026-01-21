# Meshing Toolkit

Meshing Toolkit provides two workflows to reconstruct meshes from point clouds:
1) Open3D pipeline (`Meshing_Open3D.py`) for interactive, single-file runs.
2) PyMeshLab pipeline (`pymeshlab_cli.py` + `pymeshlab_pipeline.py`) for batch-ready, config-driven runs with open-boundary methods.

The default and recommended method for open boundaries is Ball Pivoting.

## Requirements
- Python 3.9+ (recommended)
- `open3d` (required for `.pcd` input)
- `pymeshlab` (required for PyMeshLab pipeline)
- `numpy`, `scipy`, `scikit-image` (Open3D pipeline uses these)

Install:
```bash
pip install open3d pymeshlab numpy scipy scikit-image
```

## Files and What They Do
- `pymeshlab_cli.py`: Config-driven CLI (no prompts). Reads `pymeshlab_config.ini`.
- `pymeshlab_config.ini`: All parameters in sections (general, normals, methods).
- `pymeshlab_pipeline.py`: Core reconstruction logic used by the CLI.
- `pymeshlab_reconstruction.py`: Single-file PyMeshLab script (manual config in code).
- `Meshing_Open3D.py`: Interactive Open3D meshing with multiple methods.
- `Marching Cubes.py`: Experiment for marching cubes from distance fields.

## How to Run (PyMeshLab CLI)
1) Edit `pymeshlab_config.ini`:
   - Set `[general] input_path` to a file or folder.
   - Choose `[general] reconstruction_method`.
2) Run:
```bash
python pymeshlab_cli.py
```

### Input Options
- Single file:
  - `input_path = C:\path\to\scan.pcd`
  - Output goes to `output_dir` (default `meshes`).
- Folder:
  - `input_path = C:\path\to\Dataset_01`
  - All `.pcd` files are processed.
  - Output goes to `Dataset_01\meshes`.

### Methods Available (PyMeshLab)
- `ball_pivoting` (default, open boundaries)
- `alpha_shape` (open boundaries)
- `poisson` (watertight / closed surface)

## Config Guide (pymeshlab_config.ini)
Key sections and what they control:
- `[general]`
  - `input_path`: file or folder path (absolute recommended).
  - `reconstruction_method`: `ball_pivoting`, `alpha_shape`, or `poisson`.
  - `output_dir`: output folder for single-file runs.
  - `extensions`: file extensions for folder mode (comma-separated).
  - `recursive`: scan subfolders (`true/false`).
  - `skip_existing`: skip files when a matching mesh already exists (`true/false`).
- `[normals]`
  - `enabled`, `k`, `smooth_iter`, `flip`.
- `[ball_pivoting]`
  - `radius_auto`, `radius_scale`, `radius`.
- `[alpha_shape]`
  - `alpha`.
- `[poisson]`
  - `depth`, `cg_depth`, `iters`, `scale`, `samples_per_node`, `point_weight`,
    `confidence`, `pre_clean`.

## Open3D Pipeline *(optional)*
Run:
```bash
python Meshing_Open3D.py
```
This script loads a single point cloud, removes outliers, and lets you choose a
meshing algorithm interactively.

## Notes and Tips
- Open boundaries: use `ball_pivoting` or `alpha_shape`.
- Closed surfaces: use `poisson`.
- If ball pivoting produces holes, increase `radius_scale` or set `radius`.
- If edges are too blobby, reduce `radius_scale` or `alpha`.

## Output
Meshes are saved as `.ply` with timestamps under the configured output folder.
