from __future__ import annotations

import numpy as np
import trimesh


def generate_parametric_mesh(
    height_cm: float = 175.0,
    weight_kg: float = 70.0,
    num_vertices: int = 2048,
) -> trimesh.Trimesh:
    aspect = height_cm / 175.0
    girth_factor = weight_kg / 70.0

    radius = 0.25 * (girth_factor**0.5)
    height = 0.5 * aspect

    n_angle = int(np.sqrt(num_vertices))
    theta_vals = np.linspace(0, 2 * np.pi, n_angle)
    phi_vals = np.linspace(0, np.pi, n_angle)
    theta, phi = np.meshgrid(theta_vals, phi_vals)

    r = radius * np.sin(phi) * (1 + 0.3 * np.sin(phi) ** 2)
    x = r * np.sin(theta)
    z = r * np.cos(theta)
    y = height * np.cos(phi)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    faces = []
    for i in range(n_angle - 1):
        for j in range(n_angle - 1):
            a = i * n_angle + j
            b = i * n_angle + (j + 1)
            c = (i + 1) * n_angle + j
            d = (i + 1) * n_angle + (j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.remove_unreferenced_vertices()
    return mesh
