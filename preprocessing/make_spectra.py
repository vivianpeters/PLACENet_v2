import numpy as np
from typing import Sequence, Tuple, List, Dict, Optional

def commit_spectrum_group(
    container: Dict[str, List[np.ndarray]],
    key: str,
    spectra: Sequence[np.ndarray],
    chunk_size: int,
) -> None:
    if key not in container:
        container[key] = []
    for i in range(0, len(spectra), chunk_size):
        chunk = np.vstack(spectra[i : i + chunk_size])
        container[key].append(chunk)

def prep_data(
    matrices: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    chunk_size: int,
) -> Tuple[List[np.ndarray], List[str]]:
    filename_spectra: Dict[str, List[np.ndarray]] = {}
    current_filename: Optional[str] = None
    spectrum_group: List[np.ndarray] = []

    for filename, position, spectrum in matrices:
        if filename != current_filename:
            if current_filename is not None:
                commit_spectrum_group(
                    filename_spectra, current_filename, spectrum_group, chunk_size
                )
            current_filename = filename
            spectrum_group = [spectrum]
        else:
            spectrum_group.append(spectrum)

    if current_filename is not None:
        commit_spectrum_group(
            filename_spectra, current_filename, spectrum_group, chunk_size
        )

    for key in filename_spectra:
        arr = np.array(filename_spectra[key])
        arr = arr / np.max(arr)
        filename_spectra[key] = arr

    spectrum_list: List[np.ndarray] = []
    filename_list: List[str] = []
    for key in sorted(filename_spectra.keys()):
        spectrum_list.append(filename_spectra[key])
        filename_list.append(key)

    return spectrum_list, filename_list

def merge_spectra(spectra_array: np.ndarray, target_size: int) -> np.ndarray:
    n_spectra, n_bins = spectra_array.shape
    spectra_per_group = n_spectra / target_size

    merged = []
    for i in range(target_size):
        start_idx = int(i * spectra_per_group)
        end_idx = int((i + 1) * spectra_per_group)
        if i == target_size - 1:
            end_idx = n_spectra
        group = spectra_array[start_idx:end_idx]
        merged.append(np.mean(group, axis=0) if len(group) > 0 else np.zeros(n_bins))
    return np.array(merged)

def fix_spectra_shapes_with_merge(
    spectra_list: Sequence[np.ndarray],
    target_dets: int
) -> List[np.ndarray]:
    fixed: List[np.ndarray] = []
    for spec_array in spectra_list:
        if len(spec_array.shape) == 3:
            n_chunks, dets, bins = spec_array.shape
            flattened = spec_array.reshape(n_chunks * dets, bins)
            if len(flattened) >= target_dets:
                if len(flattened) == target_dets:
                    fixed_chunk = flattened
                else:
                    fixed_chunk = merge_spectra(flattened, target_dets)
                fixed.append(fixed_chunk)
            else:
                padded = np.zeros((target_dets, bins))
                padded[: len(flattened)] = flattened
                fixed.append(padded)
        else:
            fixed.append(spec_array)

    return fixed
