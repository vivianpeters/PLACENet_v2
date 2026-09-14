import numpy as np
from typing import Tuple

class PLACESpectrumAugmenter:
    """Augmenter for spectrum data."""
    def __init__(
        self,
        poisson_noise: bool = True, # If True, add Poisson noise
        energy_shift: bool = True, # If True, shift the energy
        intensity_scale: bool = True, # If True, scale the intensity
        detector_dropout: bool = True, # If True, dropout the detectors
        energy_shift_range: float = 0.02, # Range of the energy shift
        intensity_scale_range: Tuple[float, float] = (0.8, 1.2), # Range of the intensity scale
        detector_dropout_prob: float = 0.1, # Probability of dropping a detector
        poisson_scale: float = 500.0, # Scale for the Poisson noise
    ):
        self.poisson_noise = poisson_noise
        self.energy_shift = energy_shift
        self.intensity_scale = intensity_scale
        self.detector_dropout = detector_dropout
        self.energy_shift_range = energy_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.detector_dropout_prob = detector_dropout_prob
        self.poisson_scale = poisson_scale

    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        """Augment a batch of spectra."""
        if batch.size == 0:
            return batch
        return np.asarray(
            [self._augment_single(sample) for sample in batch],
            dtype=batch.dtype,
        )

    def _augment_single(self, spectrum: np.ndarray) -> np.ndarray:
        """Augment a single spectrum."""
        augmented = spectrum.copy()
        if self.intensity_scale and np.random.random() < 0.5:
            scale = np.random.uniform(*self.intensity_scale_range)
            augmented *= scale
        if self.energy_shift and np.random.random() < 0.5:
            shift_fraction = np.random.uniform(-self.energy_shift_range, self.energy_shift_range)
            augmented = self._shift_spectrum(augmented, shift_fraction)
        if self.poisson_noise:
            augmented = self._add_poisson_noise(augmented)
        if self.detector_dropout and np.random.random() < 0.3:
            augmented = self._dropout_detectors(augmented)
        return augmented

    def _shift_spectrum(self, spectrum: np.ndarray, fraction: float) -> np.ndarray:
        """Shift the spectrum by a fraction of the bins."""
        n_detectors, n_bins, _ = spectrum.shape
        shift_bins = int(fraction * n_bins)
        if shift_bins == 0:
            return spectrum
        shifted = np.zeros_like(spectrum)
        if shift_bins > 0:
            shifted[:, shift_bins:, :] = spectrum[:, :-shift_bins, :]
        else:
            shifted[:, :shift_bins, :] = spectrum[:, -shift_bins:, :]
        return shifted

    def _add_poisson_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Add Poisson noise to the spectrum."""
        counts = np.maximum(spectrum * self.poisson_scale, 0)
        noisy = np.random.poisson(counts)
        return noisy / self.poisson_scale

    def _dropout_detectors(self, spectrum: np.ndarray) -> np.ndarray:
        """Dropout the detectors from the spectrum."""
        drop_mask = np.random.rand(spectrum.shape[0]) < self.detector_dropout_prob
        spectrum[drop_mask] = 0
        return spectrum
