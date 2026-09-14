"""Data preparation for PLACENet."""

from __future__ import annotations

import csv

from .make_spectra import prep_data, fix_spectra_shapes_with_merge
from .make_bbox_labels import make_simple_labels
from .make_voxel_labels import make_voxel_occupancy_labels
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

ClusterMethod = Literal["dbscan", "hdbscan"]



def label_target_dim(label_mode: str) -> int:
    """Number of regression targets per source slot (6 bbox or 3 position)."""
    if label_mode == "voxel":
        raise ValueError("label_target_dim is undefined for label_mode='voxel'; use label_grid_shape")
    return 3 if label_mode == "position" else 6


def label_output_dim(label_mode: str) -> int:
    """Model output size per slot: targets + confidence."""
    if label_mode == "voxel":
        raise ValueError("label_output_dim is undefined for label_mode='voxel'")
    return label_target_dim(label_mode) + 1


def label_grid_shape(
    label_mode: str,
    voxel_grid: Tuple[int, int, int] = (10, 10, 10),
) -> Optional[Tuple[int, int, int]]:
    """Occupancy grid shape for voxel mode, else None."""
    return voxel_grid if label_mode == "voxel" else None


def infer_label_mode(
    *,
    last_dim: Optional[int] = None,
    label_mode: Optional[str] = None,
    label_ndim: Optional[int] = None,
) -> str:
    """Infer bbox vs position from explicit mode or trailing label dimension."""
    if label_mode is not None:
        return label_mode
    if label_ndim in (4, 5):
        return "voxel"
    if last_dim in (3, 4):
        return "position"
    return "bbox"


@dataclass(frozen=True) # immutable class
class PLACEPrepConfig:
    """Configuration for loading and preparing datasets."""

    data_dir: str  # Path to the data directory
    dets: int = 16  # Number of detectors
    chunk_size: int = 16  # Number of spectra to group together
    max_sources: int = 1  # Maximum number of sources to classify
    cluster_method: ClusterMethod = "hdbscan"  # "hdbscan" or "dbscan"
    cluster_eps: float = 2.0  # DBSCAN eps; HDBSCAN cluster_selection_epsilon (voxel distance)
    cluster_min_samples: int = 1  # DBSCAN min_samples; HDBSCAN min_samples
    cluster_min_cluster_size: int = 2
    randomize_slot_assignment: bool = False  # If True, randomly assign sources to slots instead of sequential
    label_dtype: str = "float32"  # Box label dtype (use float32; int32 truncates centres/widths)
    # "bbox": [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre] per source (6 dims)
    # "position": [xcentre, ycentre, zcentre] per source (3 dims) after HDBSCAN/DBSCAN clustering
    # "voxel": (nx, ny, nz) binary occupancy map; clustering skipped in voxel label mode
    label_mode: Literal["bbox", "position", "voxel"] = "bbox"
    voxel_grid: Tuple[int, int, int] = (10, 10, 10)
    coordinate_mapper: Optional[Callable[[float, float, float, Tuple[int, int, int]], Optional[Tuple[int, int, int]]]] = None

    # How to compute centre(s) from clustered voxels (clustering only assigns membership)
    centroid_method: Literal["mean", "bbox"] = "mean"  # mean: voxel centroid; bbox: centre of AABB


@dataclass(frozen=True) # immutable class
class PLACEFilePair:
    """Container describing the paired files needed for one dataset."""

    setname: str  # Name of the dataset
    energy_path: Path  # Path to the energy data file
    position_path: Path  # Path to the position data file


@dataclass
class PLACEDataset:
    """Structured representation of a prepared dataset."""

    name: str
    data: np.ndarray  # Array of spectra
    labels: np.ndarray  # Array of labels
    positions: List[np.ndarray]  # List of position arrays
    pos_keys: List[str]  # List of filename keys (matching the order of data samples)

    def summary(self) -> str:
        return (
            f"{self.name}: samples={len(self.data)}, "
            f"spectrum_shape={self.data.shape[1:]}, "
            f"label_shape={self.labels.shape[1:]}, "
            f"label_dtype={self.labels.dtype}"
        )


class PLACENetPrep:
    """Data preparation pipeline for datasets."""

    def __init__(self, config: PLACEPrepConfig):
        self.config = config
        self._label_dtype = np.dtype(config.label_dtype)
        if self._label_dtype.kind not in ("f", "c"):
            raise ValueError(
                f"label_dtype must be floating-point (got {config.label_dtype!r}); "
                "int32 labels truncate voxel centres (e.g. 49.5 -> 49)."
            )
        if config.label_mode not in ("bbox", "position", "voxel"):
            raise ValueError(
                f"label_mode must be 'bbox', 'position', or 'voxel' (got {config.label_mode!r})"
            )
        if config.label_mode == "voxel":
            nx, ny, nz = config.voxel_grid
            if min(nx, ny, nz) < 1:
                raise ValueError(f"voxel_grid must be positive (got {config.voxel_grid!r})")
        if config.centroid_method not in ("mean", "bbox"):
            raise ValueError(
                f"centroid_method must be 'mean' or 'bbox' (got {config.centroid_method!r})"
            )
        self._pairs = self._discover_pairs()

    def _discover_pairs(self) -> Dict[str, PLACEFilePair]:
        """
        Discover paired files for each dataset based on filename stem suffixes.
        Supports legacy names (detector_energy_data / position_arrays) and
        newer aliases (spectra / positions), including the common typo
        "postitions".
        Args:
            None
        Returns:
            Dict[str, PLACEFilePair]: A dictionary of dataset names and their paired files.
        """
        root = Path(self.config.data_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {root}")

        energy_suffixes = (
            "detector_energy_data",
            "timediff_data_spectra",
            "timediff_data",
            "energy_data",
            "spectra",
            "spectrum",
            "timediff",
        )
        position_suffixes = ("position_arrays", "positions", "position", "postitions")

        def _match_suffix(stem_lower: str) -> Optional[str]:
            """Return logical file type for known stem suffixes."""
            if stem_lower.endswith(energy_suffixes):
                return "energy"
            if stem_lower.endswith(position_suffixes):
                return "pos"
            return None

        def _extract_setname(file_stem: str, suffix: str) -> str:
            """
            Extract dataset identifier from file stem.
            Prefer splitting at `_gamma` when present; otherwise remove suffix.
            """
            stem_lower = file_stem.lower()
            gamma_idx = stem_lower.find("_gamma")
            if gamma_idx != -1:
                return file_stem[:gamma_idx]

            marker_idx = stem_lower.rfind(suffix)
            prefix = file_stem[:marker_idx] if marker_idx != -1 else file_stem
            return prefix.rstrip("_- ")

        def _pairing_key(raw_setname: str) -> str:
            """
            Build a normalized pairing key so variant stems still match.
            Example: "<set>_timeDiff_data_spectra" and "<set>_positions"
            should map to the same dataset identifier.
            """
            key = raw_setname.lower().replace("-", "_").strip("_ ")
            for trailing in ("_timediff_data", "_timediff"):
                if key.endswith(trailing):
                    key = key[: -len(trailing)]
            return key.strip("_ ")

        buckets: Dict[str, Dict[str, Path]] = {} # Dictionary to store the paired files for each dataset
        display_names: Dict[str, str] = {}
        for file in root.iterdir():
            if not file.is_file():
                continue

            stem_lower = file.stem.lower()
            file_type = _match_suffix(stem_lower)
            if file_type is None:
                continue

            suffixes = energy_suffixes if file_type == "energy" else position_suffixes
            matched_suffix = next(s for s in suffixes if stem_lower.endswith(s))
            raw_setname = _extract_setname(file.stem, matched_suffix)
            if not raw_setname:
                continue
            setname_key = _pairing_key(raw_setname)
            if not setname_key:
                continue

            # Prefer the shorter display name when aliases normalize together.
            current_display = display_names.get(setname_key)
            if current_display is None or len(raw_setname) < len(current_display):
                display_names[setname_key] = raw_setname

            slot = buckets.setdefault(setname_key, {})
            slot[file_type] = file

        pairs = {
            display_names[setname_key]: PLACEFilePair(display_names[setname_key], paths["energy"], paths["pos"])
            for setname_key, paths in buckets.items()
            if "energy" in paths and "pos" in paths
        }

        if not pairs:
            raise RuntimeError(f"No valid dataset pairs found in {root}")

        return pairs

    def list_sets(self) -> List[str]:
        """Return the discovered dataset identifiers (setnames)."""
        return sorted(self._pairs.keys())

    def load_datasets(self) -> List[PLACEDataset]:
        """Load, normalize, and label every discovered dataset."""
        datasets: List[PLACEDataset] = []

        for setname in self.list_sets():
            pair = self._pairs[setname]
            matrices = self._load_csv_as_matrices(str(pair.energy_path))
            position_filenames, positions = self._load_csv_as_pos(str(pair.position_path))
            
            # Diagnostic: Check file sizes for huge files
            pos_file_size_mb = pair.position_path.stat().st_size / (1024 * 1024)
            energy_file_size_mb = pair.energy_path.stat().st_size / (1024 * 1024)
            total_voxels = sum(len(pos) for pos in positions) if positions else 0
            if pos_file_size_mb > 100 or len(positions) > 1000:
                print(f"[INFO] Dataset '{setname}': Position file size: {pos_file_size_mb:.1f} MB, "
                      f"Energy file: {energy_file_size_mb:.1f} MB. "
                      f"Position entries (histograms): {len(positions)}, "
                      f"Total voxels: {total_voxels:,}")
            
            spectra, pos_keys = prep_data(matrices, self.config.chunk_size)
            fixed_spectra = fix_spectra_shapes_with_merge(spectra, self.config.dets)
            data = self._reshape_spectra(fixed_spectra)
            
            # Align positions with pos_keys (filenames) before building labels
            positions_by_filename = {fname: pos for fname, pos in zip(position_filenames, positions)}
            
            aligned_positions = []
            for filename in pos_keys:
                if filename in positions_by_filename:
                    aligned_positions.append(positions_by_filename[filename])
                else:
                    warnings.warn(
                        f"Dataset '{setname}': Filename '{filename}' found in energy file but not in position file. "
                        f"Using empty position array for this entry.",
                        UserWarning
                    )
                    aligned_positions.append(np.empty((0, 4), dtype=float))
            
            if len(aligned_positions) != len(pos_keys):
                raise ValueError(
                    f"Dataset '{setname}': Alignment failed. Expected {len(pos_keys)} positions, "
                    f"got {len(aligned_positions)}."
                )
            
            labels = self._build_labels(setname, aligned_positions)
            self._log_label_precision(setname, labels)

            # Verify alignment
            if len(data) != len(pos_keys):
                raise ValueError(
                    f"Dataset '{setname}': data length ({len(data)}) != pos_keys length ({len(pos_keys)}) "
                    f"after merging. This should not happen."
                )
            
            target_len = len(data)
            if len(labels) != target_len:
                if len(labels) < target_len:
                    raise ValueError(
                        f"Dataset '{setname}': labels length ({len(labels)}) < data length ({target_len}). "
                        f"Not enough labels for all data samples. This may indicate an issue with position alignment."
                    )
                else:
                    warnings.warn(
                        f"Dataset '{setname}': labels length ({len(labels)}) > data length ({target_len}). "
                        f"Truncating labels to match data length. This may indicate a bug in the alignment logic.",
                        UserWarning
                    )
                    labels = labels[:target_len]

            aligned_len = len(data)
            datasets.append(
                PLACEDataset(
                    name=setname,
                    data=data,
                    labels=labels,
                    positions=list(aligned_positions)[:aligned_len],
                    pos_keys=list(pos_keys)[:aligned_len],
                )
            )

        return datasets

    def concatenate(self, datasets: Sequence[PLACEDataset]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenate multiple datasets, keeping only aligned samples.
        Handles overlapping spectra by merging them.
        """
        aligned_data = []
        aligned_labels = []

        for dataset in datasets:
            if len(dataset.data) != len(dataset.labels):
                min_len = min(len(dataset.data), len(dataset.labels))
                aligned_data.append(dataset.data[:min_len])
                aligned_labels.append(dataset.labels[:min_len])
            else:
                aligned_data.append(dataset.data)
                aligned_labels.append(dataset.labels)

        if not aligned_data:
            raise ValueError("No datasets available for concatenation")

        return np.concatenate(aligned_data), np.concatenate(aligned_labels)

    def _reshape_spectra(self, spectra: Sequence[np.ndarray]) -> np.ndarray:
        """
        Reshape the spectra to the correct shape.
        If empty, return empty array.
        Ensures the number of detectors is correct and adds a channel dimension.
        """
        if not spectra:
            return np.empty((0, self.config.dets, 0, 1), dtype=np.float32)

        array = np.asarray(spectra, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D spectra, received shape {array.shape}")

        n_detectors = array.shape[1]
        if n_detectors != self.config.dets:
            raise ValueError(
                f"Detector count mismatch (expected {self.config.dets}, got {n_detectors})"
            )

        return array.reshape(array.shape[0], array.shape[1], array.shape[2], 1)

    def _build_labels(self, setname: str, positions: List[np.ndarray]) -> np.ndarray:
        """
        Build the labels for the dataset.
        Pads and reshapes label arrays for training, removing count values.
        Calls the _make_simple_labels function to create the labels.
        Args:
            setname: The name of the dataset.
            positions: The positions of the sources in the dataset.
        Returns:
            np.ndarray: The labels for the dataset.
        """
        if self.config.label_mode == "voxel":
            labels = make_voxel_occupancy_labels(positions, self.config.voxel_grid, self.config.coordinate_mapper, self._label_dtype)
            self._log_voxel_occupancy(setname, labels)
            return labels

        raw_labels = make_simple_labels(
            [positions],
            [setname],
            max_sources=self.config.max_sources,
            cluster_sources=True,
            eps=self.config.cluster_eps,
            cluster_method=self.config.cluster_method,
            cluster_min_samples=self.config.cluster_min_samples,
            cluster_min_cluster_size=self.config.cluster_min_cluster_size,
            randomize_slots=self.config.randomize_slot_assignment,
            label_dtype=self._label_dtype,
            label_mode=self.config.label_mode,
            centroid_method=self.config.centroid_method,
        )
        labels = np.asarray(raw_labels[0], dtype=self._label_dtype)
        if labels.ndim == 2:
            labels = labels[:, np.newaxis, :]
        return labels

    def _log_voxel_occupancy(self, setname: str, labels: np.ndarray) -> None:
        """Log coarse-grid occupancy statistics for voxel labels."""
        if labels.size == 0:
            return
        occ_frac = float(np.mean(labels > 0.5))
        mean_occ = float(np.mean(np.sum(labels > 0.5, axis=(1, 2, 3))))
        print(
            f"[prep] Dataset '{setname}': label_mode=voxel, grid={self.config.voxel_grid}, "
            f"mean occupied cells/sample={mean_occ:.2f}, global occupancy={occ_frac:.4%}"
        )

    def _log_label_precision(self, setname: str, labels: np.ndarray) -> None:
        """Log how many centres are non-integer (lost if labels are cast to int32)."""
        valid = ~np.all(labels == 0.0, axis=-1)
        if not np.any(valid):
            return
        if self.config.label_mode == "position":
            centres = labels[valid]
        else:
            boxes = labels[..., :6] if labels.shape[-1] >= 6 else labels
            centres = boxes[valid][:, [1, 3, 5]]
        frac_non_int = float(np.mean(np.abs(centres - np.round(centres)) > 1e-3))
        print(
            f"[prep] Dataset '{setname}': label_mode={self.config.label_mode}, "
            f"centroid_method={self.config.centroid_method}, dtype={labels.dtype}, "
            f"non-integer centres on {frac_non_int:.1%} of valid slots"
        )

    @staticmethod
    def _load_csv_as_matrices(file_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load energy data from CSV file.
        
        Returns list of tuples: (filename, source_position, spectrum)
        where source_position is [SrcVoxelX, SrcVoxelY, SrcVoxelZ]
        """
        matrices = []
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 9:
                    continue
                try:
                    filename = row[0].strip()
                    position = np.array([float(x) for x in row[1:4]])
                    spectrum = np.array([float(x) for x in row[8:]])
                    matrices.append((filename, position, spectrum))
                except (ValueError, IndexError):
                    continue
        return matrices

    @staticmethod
    def _load_csv_as_pos(file_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """Load the position data from the CSV file (voxel coordinates).
        Returns a tuple of lists: (filenames, position arrays).
        """
        grouped_positions: Dict[str, List[List[float]]] = {}
        # Group the positions by filename (histogram id)
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                current_hist = row[0].strip()
                if len(row) < 5:
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    count = float(row[4])
                except ValueError:
                    continue
                grouped_positions.setdefault(current_hist, []).append([x, y, z, count])
        # Turn it into list
        filenames = []
        pos_list = []
        for key in sorted(grouped_positions.keys()):
            entries = grouped_positions[key]
            if entries:
                filenames.append(key) # Append the filename to the list
                pos_list.append(np.array(entries, dtype=float)) # Append the position array to the list
        return filenames, pos_list

