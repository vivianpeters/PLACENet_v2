""" Physical drum geometry and coarse/fine voxel indexing for PLACENet.
    Note that this is specific to the waste drum-in-drum case.  """

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PLACEGeometry:
    """
    Shielded waste-drum geometry aligned with Geant4 PLACE simulation and
    ``convertDataToCSV.C`` voxelization.

    World frame (Simulation/README/RUNNING_README.md):
      - Drum bottom centre at (0, 0, 0) mm.
      - x/y drum axis at world x=y=0.

    Voxel frame (convertDataToCSV.C):
      - 10 mm cubic voxels.
      - Origins at (-500, -500, 0) mm → drum axis at voxel index 50 in x/y.
      - z voxel 0 is the drum bottom (z = 0 mm).
    """

    outer_radius_mm: float = 287.5
    outer_height_mm: float = 865.0
    inner_radius_mm: float = 229.0
    inner_height_mm: float = 639.0
    # Inner air/plastic-fill volume centre height (Geant4 layered drum placement).
    inner_z_center_mm: float = 431.5

    voxel_size_mm: float = 10.0
    origin_x_mm: float = -500.0
    origin_y_mm: float = -500.0
    origin_z_mm: float = 0.0
    drum_xy_center_mm: float = 0.0

    @staticmethod
    def mm_to_voxel(mm: float, origin_mm: float, voxel_size_mm: float) -> float:
        """Continuous voxel coordinate (bin centre), not floored."""
        return (mm - origin_mm) / voxel_size_mm

    @property
    def centre_x_vox(self) -> float:
        return self.mm_to_voxel(self.drum_xy_center_mm, self.origin_x_mm, self.voxel_size_mm)

    @property
    def centre_y_vox(self) -> float:
        return self.mm_to_voxel(self.drum_xy_center_mm, self.origin_y_mm, self.voxel_size_mm)

    @property
    def outer_radius_vox(self) -> float:
        return self.outer_radius_mm / self.voxel_size_mm

    @property
    def inner_radius_vox(self) -> float:
        return self.inner_radius_mm / self.voxel_size_mm

    @property
    def outer_z_min_vox(self) -> float:
        return self.mm_to_voxel(0.0, self.origin_z_mm, self.voxel_size_mm)

    @property
    def outer_z_max_vox(self) -> float:
        return self.mm_to_voxel(self.outer_height_mm, self.origin_z_mm, self.voxel_size_mm)

    @property
    def inner_z_min_vox(self) -> float:
        z_min_mm = self.inner_z_center_mm - 0.5 * self.inner_height_mm
        return self.mm_to_voxel(z_min_mm, self.origin_z_mm, self.voxel_size_mm)

    @property
    def inner_z_max_vox(self) -> float:
        z_max_mm = self.inner_z_center_mm + 0.5 * self.inner_height_mm
        return self.mm_to_voxel(z_max_mm, self.origin_z_mm, self.voxel_size_mm)

    def recommended_grid_shape(self, margin_vox: float = 2.0) -> Tuple[int, int, int]:
        """Histogram shape (binx, biny, binz) covering the outer drum in voxel indices."""
        max_xy_idx = int(np.ceil(self.centre_x_vox + self.outer_radius_vox + margin_vox))
        bin_xy = max(100, max_xy_idx + 1)
        bin_z = int(np.ceil(self.outer_z_max_vox + margin_vox)) + 1
        return bin_xy, bin_xy, bin_z

    def plot_xy_limits(self, margin_vox: float = 2.0) -> Tuple[float, float]:
        centre = self.centre_x_vox
        half = self.outer_radius_vox + margin_vox
        return centre - half, centre + half

    def plot_z_limits(self, margin_vox: float = 0.0) -> Tuple[float, float]:
        return self.outer_z_min_vox, self.outer_z_max_vox + margin_vox


def make_drum_cylinder_surfaces(
    drum: PLACEDrumGeometry,
    *,
    centre_x: float | None = None,
    centre_y: float | None = None,
    n_theta: int = 50,
    n_z: int = 50,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build matplotlib ``plot_surface`` meshes for inner and outer drum cylinders.

    Returns:
        ((xc_i, yc_i, zc_i), (xc_o, yc_o, zc_o))
    """
    cx = drum.centre_x_vox if centre_x is None else centre_x
    cy = drum.centre_y_vox if centre_y is None else centre_y

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    z_inner = np.linspace(drum.inner_z_min_vox, drum.inner_z_max_vox, n_z)
    theta_i, zc_i = np.meshgrid(theta, z_inner)
    xc_i = drum.inner_radius_vox * np.cos(theta_i) + cx
    yc_i = drum.inner_radius_vox * np.sin(theta_i) + cy

    z_outer = np.linspace(drum.outer_z_min_vox, drum.outer_z_max_vox, n_z)
    theta_o, zc_o = np.meshgrid(theta, z_outer)
    xc_o = drum.outer_radius_vox * np.cos(theta_o) + cx
    yc_o = drum.outer_radius_vox * np.sin(theta_o) + cy

    return (xc_i, yc_i, zc_i), (xc_o, yc_o, zc_o)


def add_drum_surfaces_to_axes(ax, drum: PLACEDrumGeometry, *, alpha: float = 0.2) -> None:
    """Overlay inner (lighter) and outer drum cylinders on a 3D axis."""
    inner, outer = make_drum_cylinder_surfaces(drum)
    ax.plot_surface(*inner, color="gold", alpha=alpha, linewidth=0, antialiased=True)
    ax.plot_surface(*outer, color="goldenrod", alpha=alpha * 0.85, linewidth=0, antialiased=True)


def fine_voxel_to_coarse_indices(
    x: float,
    y: float,
    z: float,
    drum: PLACEDrumGeometry,
    grid_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    """Map fine 10 mm label indices to a coarse cell, or None if outside inner drum."""
    nx, ny, nz = grid_shape
    cx, cy = drum.centre_x_vox, drum.centre_y_vox
    r = drum.inner_radius_vox
    z_min, z_max = drum.inner_z_min_vox, drum.inner_z_max_vox

    if (x - cx) ** 2 + (y - cy) ** 2 > r ** 2 or z < z_min or z > z_max:
        return None

    ci = int(np.floor((x - (cx - r)) / (2.0 * r) * nx))
    cj = int(np.floor((y - (cy - r)) / (2.0 * r) * ny))
    ck = int(np.floor((z - z_min) / (z_max - z_min) * nz))
    ci = int(np.clip(ci, 0, nx - 1))
    cj = int(np.clip(cj, 0, ny - 1))
    ck = int(np.clip(ck, 0, nz - 1))
    return ci, cj, ck


def coarse_inner_drum_mask(
    drum: PLACEDrumGeometry,
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Boolean mask (nx, ny, nz): coarse cells whose centres lie inside inner cylinder."""
    nx, ny, nz = grid_shape
    cx, cy = drum.centre_x_vox, drum.centre_y_vox
    r = drum.inner_radius_vox
    mask = np.zeros((nx, ny, nz), dtype=np.float32)

    for ci in range(nx):
        x = cx - r + (ci + 0.5) * (2.0 * r) / nx
        for cj in range(ny):
            y = cy - r + (cj + 0.5) * (2.0 * r) / ny
            if (x - cx) ** 2 + (y - cy) ** 2 > r ** 2:
                continue
            for ck in range(nz):
                mask[ci, cj, ck] = 1.0
    return mask
