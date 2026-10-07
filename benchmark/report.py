"""Build the GitHub benchmark report from recorded measurements."""

from collections import Counter
from pathlib import Path
import numpy as np

METHOD_NAMES = {
    "baseline_ball_pivoting": "Baseline Ball Pivoting",
    "baseline_poisson": "Baseline Poisson",
    "baseline_alpha_shape": "Baseline Alpha Shape",
    "improved_ball_pivoting": "Improved Ball Pivoting",
    "improved_poisson": "Improved Poisson",
}


def summary_table(reports, figures=False):
    header = "| Method | Coverage ↑ | p95 / local spacing ↓ | Unsupported area ↓ | Score ↓ |"
    separator = "| --- | ---: | ---: | ---: | ---: |"
    if figures:
        header += " Colour sample |"
        separator += " --- |"
    rows = [header, separator]
    for method, name in METHOD_NAMES.items():
        values = [report["methods"][method] for report in reports if "error" not in report["methods"][method]]
        if not values:
            rows.append(f"| {name} | Failed | — | — | — |" + (" — |" if figures else ""))
            continue
        coverage = np.mean([v["coverage_within_2_local_spacings"] for v in values]) * 100
        p95 = np.mean([v["point_to_surface_p95_spacing"] for v in values])
        unsupported = np.mean([v["unsupported_surface_fraction"] for v in values]) * 100
        score = np.mean([v["selection_score"] for v in values])
        row = f"| {name} | {coverage:.1f}% | {p95:.2f} | {unsupported:.2f}% | {score:.2f} |"
        if figures:
            row += f' <img src="benchmark/figures/capture_1_{method}.png" width="170" alt="{name}: capture 1 with original colours"> |'
        rows.append(row)
    return "\n".join(rows)


def write_report(output, reports):
    counts = Counter(report["recommended_method"] for report in reports)
    recommended = counts.most_common(1)[0][0]
    name = METHOD_NAMES[recommended]
    lines = ["# Meshing benchmark", "",
             f"**Recommendation: {name}**, selected for {counts[recommended]} of {len(reports)} real captures. Production selection considers only the two improved methods.", "",
             '<img src="figures/recommended.svg" width="780" alt="Recommended: Improved Poisson with support trimming">', "",
             "## Five-method comparison", "",
             "These are equal-weight averages across captures. Lower score is better; open boundaries are not penalized.", "",
             summary_table(reports), "", "## Per-capture results", "",
             "| Capture | Method | Coverage | p95 distance (input units) | Unsupported surface | Score | Faces |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        for method, metrics in report["methods"].items():
            if "error" in metrics:
                lines.append(f"| {report['capture']} | {METHOD_NAMES[method]} | Failed | — | — | — | — |")
            else:
                lines.append(f"| {report['capture']} | {METHOD_NAMES[method]} | {100 * metrics['coverage_within_2_local_spacings']:.1f}% | {metrics['point_to_surface_p95_input_units']:.5f} | {100 * metrics['unsupported_surface_fraction']:.2f}% | {metrics['selection_score']:.3f} | {metrics['triangles']:,} |")
    lines.extend(["", "[Download measurements (CSV)](results.csv) · [Parameters, hashes, variants and topology (JSON)](results.json)", "", "## Visual comparisons", "",
                  "Each figure shows the same two cameras for the input and all five training meshes. Colour figures display the retained RGB; geometry figures use uniform shading. Click an image for full resolution.", ""])
    for report in reports:
        capture = report["capture"]
        lines.extend([f"### {capture}", "", f'<a href="figures/{capture}_colours.png"><img src="figures/{capture}_colours.png" width="1200" alt="{capture}: original RGB comparison of all five methods"></a>', "",
                      f"[Uniform geometry comparison](figures/{capture}_geometry.png)", ""])
    lines.extend(["## Downloadable colour samples", "", "Samples use capture_1 and are reduced to at most 60,000 triangles for a small GitHub download. All retain vertex RGB. They are presentation examples; the measurements above use the original, unreduced training meshes.", "",
                  "| Method | PLY sample |", "| --- | --- |"])
    for method, name in METHOD_NAMES.items():
        sample = output / "samples" / f"capture_1_{method}.ply"
        if sample.is_file():
            lines.append(f"| {name} | [Download](samples/{sample.name}) |")
    lines.extend(["", "[Final recommended full-input refit, reduced colour sample](samples/capture_1_final_recommended.ply)", "",
                  '<img src="figures/recommended_capture_1.png" width="900" alt="Final full-input improved Poisson mesh with retained input colours">', "",
                  "## Reproduce", "", "```bash", "python -m benchmark.run", "```", "",
                  "Defaults to the three real files in `datasets/captures/`. Full benchmark meshes are saved locally under `<capture>/benchmark/<method>/validation_mesh.ply`. The selected improved method is refit on all valid points and saved under `<capture>/meshes/<method>/`. Figures, measurements, and compact samples are saved here. Raw input clouds and full meshes remain local.", "",
                  "## Protocol and interpretation", "",
                  "- One fixed seed-42 split: 80% of distinct finite input points for reconstruction, 20% withheld. All five methods use the same split and the same evaluation points. A deterministic 20,000-point subset limits metric cost.",
                  "- Baselines receive conventional KNN=30 normals with tangent-plane orientation; no denoising or support trimming. Ball Pivoting uses one radius, 2.5 times mean nearest-neighbor distance. Baseline Poisson uses depth 9 and scale 1.1. Alpha Shape uses alpha 5 times median nearest-neighbor spacing. These are declared reference settings, not exhaustive tuning of each baseline.",
                  "- Improved methods share scale-aware denoising, locally oriented normals, locally scaled support trimming, topology cleanup, and colour transfer. Ball Pivoting uses three radii; Poisson tests two spacing-derived resolutions. The best Poisson resolution uses the same holdout; these are selection-consistency results, not an independent test set.",
                  "- Coverage: proportion of withheld samples within two local spacings of a triangle. p95: 95th percentile unsigned point-to-triangle distance. Unsupported surface: fraction of area-weighted surface samples farther than 2.5 local spacings from training points.",
                  "- Score = p95 distance/local spacing + 3×uncovered fraction + 4×unsupported fraction + 2×tiny-component area fraction + 0.5×sliver area fraction. Scores within 3% or 0.01 favor fewer triangles. Boundary counts are reported but do not force closed solids.",
                  "- Runtime includes each method's preparation, reconstruction and evaluation. Improved Poisson includes both candidate resolutions; file saving, preview rendering and full-input refitting are excluded. Preparation is reused during execution but charged to each method.",
                  "- Colours are transferred from training points for measured meshes; final refits and presentation samples use the original input RGB. Binary PLY stores RGB at 8-bit precision. There are no external texture files.",
                  "- Input units are not assumed to be metres. These three captures depict a similar scene; measured scan consistency does not establish ground-truth accuracy or performance on every dataset.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    table = summary_table(reports, figures=True)
    from reconstruction.paths import PROJECT_ROOT
    if output.resolve() == (PROJECT_ROOT / "benchmark").resolve():
        readme = PROJECT_ROOT / "README.md"
        if readme.is_file():
            text = readme.read_text(encoding="utf-8")
            start, end = "<!-- BENCHMARK_TABLE_START -->", "<!-- BENCHMARK_TABLE_END -->"
            if start in text and end in text:
                before, tail = text.split(start, 1)
                _, after = tail.split(end, 1)
                readme.write_text(before + start + "\n" + table + "\n" + end + after, encoding="utf-8")
    return table
