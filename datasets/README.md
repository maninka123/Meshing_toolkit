# Local input datasets

Each dataset contains point clouds and a sibling output folder named after each
cloud. `captures/` contains the three real captures; `scans/` and `combined_clouds/`
contain the other inputs. Add new data under `datasets/<dataset_name>/` and update
the input path in [`config/meshing.ini`](../config/meshing.ini).

For `captures/capture_1.pcd`, production outputs go into:

```text
captures/capture_1/meshes/improved_ball_pivoting/
captures/capture_1/meshes/improved_poisson/
```

Only selected or explicitly requested methods create production folders. Each
run saves a timestamped colour PLY, JSON report, and colour PNG preview. Baselines
are benchmark-only. Full benchmark meshes are saved under
`captures/capture_1/benchmark/<method>/validation_mesh.ply`.

Older baseline outputs are retained under per-input
`benchmark/baseline_<method>/previous_runs/`. Earlier adaptive results and their
comparison meshes are retained in `benchmark/previous_adaptive/`. `legacy/` keeps
eight meshes whose original inputs cannot be identified. Recursive discovery
skips generated and benchmark folders.

Original point clouds and mesh assets are preserved. The 19 obsolete converted
PLY inputs were removed in the earlier cleanup. Original asset paths, current
paths, hashes, and statuses are recorded in
[`docs/asset-migration.csv`](../docs/asset-migration.csv).

Raw inputs and full outputs here remain local and are ignored by Git. Published
results, figures, and compact colour samples from all five methods are in the
separate [`benchmark/`](../benchmark/README.md) folder.
