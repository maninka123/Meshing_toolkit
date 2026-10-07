"""Repeatable mesh comparison images using CPU ray casting, without GUI windows."""

import numpy as np
import open3d as o3d


def camera(points, variant=0, width=640, height=480):
    low, high = np.quantile(points, [0.002, 0.998], axis=0)
    center = (low + high) / 2
    diagonal = float(np.linalg.norm(high - low))
    direction = np.array([-1.3, -1.6, 1.0] if variant == 0 else [1.2, -1.6, 0.7])
    direction /= np.linalg.norm(direction)
    eye = center + direction * diagonal * 1.3
    up = np.array([0., 0., 1.])
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=45, center=center, eye=eye, up=up, width_px=width, height_px=height
    )
    return rays, eye, center, up


def render_mesh(mesh, rays, colored=True):
    height, width = rays.shape[:2]
    image = np.ones((height, width, 3)) * 0.965
    if mesh is None or not len(mesh.triangles):
        return image
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hits = scene.cast_rays(rays)
    mask = np.isfinite(hits["t_hit"].numpy())
    normal = hits["primitive_normals"].numpy()[mask]
    direction = -rays.numpy()[mask, 3:]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    view_light = np.abs(np.einsum("ij,ij->i", normal, direction))
    light = np.array([-0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    side_light = np.abs(normal @ light)
    shade = 0.24 + 0.55 * view_light + 0.21 * side_light
    if colored and mesh.has_vertex_colors():
        ids = hits["primitive_ids"].numpy()[mask].astype(int)
        uv = hits["primitive_uvs"].numpy()[mask]
        weights = np.column_stack([1 - uv.sum(axis=1), uv])
        colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[ids]]
        base = np.einsum("nk,nkj->nj", weights, colors)
    else:
        base = np.tile([0.35, 0.60, 0.70], (len(normal), 1))
    image[mask] = base * shade[:, None]
    return np.clip(image, 0, 1)


def render_points(cloud, rays):
    """Project points through the same camera, keeping the nearest per pixel."""
    ray_array = rays.numpy()
    height, width = ray_array.shape[:2]
    points = np.asarray(cloud.points)
    # Invert the actual ray basis, including Open3D's handedness/pixel offsets.
    # An independently constructed look-at frame can mirror the point image.
    direction = ray_array[0, 0, 3:].astype(float)
    dx = ray_array[0, 1, 3:].astype(float) - direction
    dy = ray_array[1, 0, 3:].astype(float) - direction
    basis = np.column_stack([dx, dy, direction])
    coefficient = (points - ray_array[0, 0, :3]) @ np.linalg.inv(basis).T
    depth = coefficient[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1e-12)
    x = np.rint(coefficient[:, 0] / safe_depth).astype(int)
    y = np.rint(coefficient[:, 1] / safe_depth).astype(int)
    valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(depth[indices])[::-1]]
    image = np.ones((height, width, 3)) * 0.965
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.tile([0.25, 0.4, 0.5], (len(points), 1))
    image[y[indices], x[indices]] = colors[indices]
    return image


def save_preview(cloud, baseline, improved, report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(cloud.points)
    comparing = baseline is not None
    titles = ["Measured input points"]
    if comparing:
        titles.append(report.get("preview_baseline", "Original pipeline baseline"))
    titles.append("Adaptive: " + report["selected_method"])
    fig, axes = plt.subplots(2, len(titles), figsize=(5 * len(titles), 8), facecolor="white")
    for row in range(2):
        rays, eye, center, up = camera(points, row)
        images = [render_points(cloud, rays)]
        if comparing:
            images.append(render_mesh(baseline, rays))
        images.append(render_mesh(improved, rays))
        for col, (ax, image) in enumerate(zip(axes[row], images)):
            ax.imshow(image)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
    fig.suptitle("Identical cameras and lighting | original coordinates | open boundaries preserved by support trimming", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_gallery(cloud, meshes, titles, output_path, colored=True):
    """All five methods use the exact same two cameras and lighting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = np.asarray(cloud.points)
    fig, axes = plt.subplots(2, len(meshes) + 1, figsize=(3.5 * (len(meshes) + 1), 6.5), facecolor="white")
    for row in range(2):
        rays, eye, center, up = camera(points, row)
        images = [render_points(cloud, rays)]
        images.extend(render_mesh(mesh, rays, colored=colored) for mesh in meshes)
        for ax, title, pixels in zip(axes[row], ["Input points", *titles], images):
            ax.imshow(pixels)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=11)
    fig.suptitle("80% training meshes | identical cameras and lighting | " + ("original RGB colours" if colored else "uniform shading to compare geometry"), fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_thumbnails(cloud, meshes, output_dir, stem):
    """Compact colour figures for the README comparison table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rays, _, _, _ = camera(np.asarray(cloud.points), 0)
    for method, mesh in meshes.items():
        plt.imsave(output_dir / f"{stem}_{method}.png", render_mesh(mesh, rays, colored=True))
