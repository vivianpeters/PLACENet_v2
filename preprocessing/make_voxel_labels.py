import numpy as np
from typing import List, Callable

def make_voxel_occupancy_labels(
    positions: List[np.ndarray],
    voxel_grid: tuple[int, int, int],
    coordinate_mapper: Callable,
    label_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Build (N, nx, ny, nz) occupancy maps from raw fine voxels (no DBSCAN).
    All source voxels in one histogram are merged into a single occupancy map.
    """
    nx, ny, nz = voxel_grid
    if coordinate_mapper is None:
        raise ValueError("coordinate_mapper must be provided for voxel mode")
    labels: List[np.ndarray] = []

    for pos in positions:
        occ = np.zeros((nx, ny, nz), dtype=label_dtype)
        if len(pos) == 0:
            labels.append(occ)
            continue
        for row in pos:
            coarse = coordinate_mapper(float(row[0]), float(row[1]), float(row[2]), (nx, ny, nz))

            if coarse is not None:
                occ[coarse] = 1.0
        labels.append(occ)
    return np.stack(labels, axis=0)
