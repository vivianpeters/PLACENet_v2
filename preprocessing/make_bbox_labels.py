import numpy as np
import warnings
from typing import List, Tuple, Literal, Sequence

try:
    from sklearn.cluster import DBSCAN, HDBSCAN
except ImportError:
    pass

ClusterMethod = Literal["dbscan", "hdbscan"]

def cluster_voxels_into_sources(
    pos: np.ndarray,
    method: ClusterMethod = "hdbscan",
    eps: float = 3.0,
    min_samples: int = 1,
    min_cluster_size: int = 1,
) -> List[np.ndarray]:
    """
    Cluster active voxels into separate sources.
    Uses HDBSCAN or DBSCAN based on the `method` argument.
    """
    if len(pos) == 0:
        return []

    coords = pos[:, :3]
    if method == "hdbscan":
        clusterer = HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
        )
        labels = clusterer.fit_predict(coords)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(coords)
    else:
        raise ValueError(f"Unknown cluster method: {method}")

    sources = []
    unique_labels = set(labels)
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        sources.append(pos[mask])
    
    if -1 in unique_labels and not sources:
        sources.append(pos[labels == -1])
        
    return sources

def voxels_to_centre(src_voxels: np.ndarray, centroid_method: str) -> Tuple[float, float, float]:
    if centroid_method == "bbox":
        xmin, xmax = np.min(src_voxels[:, 0]), np.max(src_voxels[:, 0])
        ymin, ymax = np.min(src_voxels[:, 1]), np.max(src_voxels[:, 1])
        zmin, zmax = np.min(src_voxels[:, 2]), np.max(src_voxels[:, 2])
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0

    coords = src_voxels[:, :3]
    if src_voxels.shape[1] >= 4:
        weights = src_voxels[:, 3]
        if np.any(weights > 0):
            xcentre, ycentre, zcentre = np.average(coords, axis=0, weights=weights)
            return float(xcentre), float(ycentre), float(zcentre)
    xcentre, ycentre, zcentre = np.mean(coords, axis=0)
    return float(xcentre), float(ycentre), float(zcentre)

def voxels_to_bbox_label(src_voxels: np.ndarray, centroid_method: str) -> List[float]:
    xmin, xmax = np.min(src_voxels[:, 0]), np.max(src_voxels[:, 0])
    ymin, ymax = np.min(src_voxels[:, 1]), np.max(src_voxels[:, 1])
    zmin, zmax = np.min(src_voxels[:, 2]), np.max(src_voxels[:, 2])
    xwidth = xmax - xmin + 1
    ywidth = ymax - ymin + 1
    zwidth = zmax - zmin + 1
    if centroid_method == "bbox":
        xcentre = (xmin + xmax) / 2.0
        ycentre = (ymin + ymax) / 2.0
        zcentre = (zmin + zmax) / 2.0
    else:
        xcentre, ycentre, zcentre = voxels_to_centre(src_voxels, "mean")
    return [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]

def voxels_to_position_label(src_voxels: np.ndarray, centroid_method: str) -> List[float]:
    xcentre, ycentre, zcentre = voxels_to_centre(src_voxels, centroid_method)
    return [xcentre, ycentre, zcentre]

def assign_source_labels(
    source_labels: List[List[float]],
    max_sources: int,
    label_dim: int,
    pad_fill: float,
    label_dtype: np.dtype,
    randomize_slots: bool,
) -> np.ndarray:
    row = np.full((max_sources, label_dim), pad_fill, dtype=label_dtype)
    n_sources = min(len(source_labels), max_sources)
    if n_sources == 0:
        return row

    if randomize_slots:
        selected_slots = np.random.choice(np.arange(max_sources), size=n_sources, replace=False)
        np.random.shuffle(selected_slots)
        for label, slot_idx in zip(source_labels[:n_sources], selected_slots):
            row[slot_idx] = np.asarray(label, dtype=label_dtype)
    else:
        for src_idx, label in enumerate(source_labels[:n_sources]):
            row[src_idx] = np.asarray(label, dtype=label_dtype)
    return row

def make_simple_labels(
    position_list: List[List[np.ndarray]],
    setnames: List[str],
    max_sources: int,
    cluster_sources: bool,
    eps: float,
    cluster_method: ClusterMethod = "hdbscan",
    cluster_min_samples: int = 1,
    cluster_min_cluster_size: int = 1,
    randomize_slots: bool = False,
    label_dtype: np.dtype = np.float32,
    label_mode: Literal["bbox", "position"] = "bbox",
    centroid_method: Literal["mean", "bbox"] = "mean",
) -> List[np.ndarray]:
    pad_fill = 0.0
    label_dim = 6 if label_mode == "bbox" else 3
    labels: List[np.ndarray] = []

    def _label_from_voxels(voxels: np.ndarray) -> List[float]:
        if label_mode == "position":
            return voxels_to_position_label(voxels, centroid_method)
        return voxels_to_bbox_label(voxels, centroid_method)

    for i in range(len(setnames)):
        dataset_labels: List[np.ndarray] = []
        for pos in position_list[i]:
            if cluster_sources:
                sources = cluster_voxels_into_sources(
                    pos,
                    method=cluster_method,
                    eps=eps,
                    min_samples=cluster_min_samples,
                    min_cluster_size=cluster_min_cluster_size,
                )
                source_labels = [_label_from_voxels(src_voxels) for src_voxels in sources[:max_sources]]
                dataset_labels.append(
                    assign_source_labels(
                        source_labels,
                        max_sources,
                        label_dim,
                        pad_fill,
                        label_dtype,
                        randomize_slots,
                    )
                )
            else:
                source_labels = [_label_from_voxels(pos)]
                dataset_labels.append(
                    assign_source_labels(
                        source_labels,
                        max_sources,
                        label_dim,
                        pad_fill,
                        label_dtype,
                        randomize_slots=False,
                    )
                )
        labels.append(np.array(dataset_labels, dtype=label_dtype))
    return labels
