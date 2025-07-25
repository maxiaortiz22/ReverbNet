from __future__ import annotations

from typing import Sequence, List

import numpy as np
from scipy.signal import butter, sosfilt


class BandpassFilter:
    """
    Create an octave‑ or third‑octave‑band filter bank and apply it to 1‑D data.

    Parameters
    ----------
    filter_type : {'octave band', 'third octave band'}
        Spectral partition scheme.
    fs : int
        Sampling rate in Hz.
    order : int
        Per‑section filter order (2‑pole biquads are cascaded).
    bands : Sequence[float]
        Centre frequencies in Hz (e.g. ``[125, 250, ...]``).

    Notes
    -----
    * For centre frequencies **below 8 kHz** a band‑pass filter is designed whose
      lower / upper cut‑offs are spaced symmetrically in the chosen scale
      (octave or third‑octave).
    * For the **highest band (≥ 8 kHz)** a high‑pass response is used to avoid
      aliasing against the Nyquist limit.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        filter_type: str,
        fs: int,
        order: int,
        bands: Sequence[float],
    ) -> None:
        if filter_type not in {"octave band", "third octave band"}:
            raise ValueError("`filter_type` must be 'octave band' or 'third octave band'")
        if fs <= 0 or order <= 0:
            raise ValueError("`fs` and `order` must be positive")
        bands = np.asarray(bands, dtype=float)
        if np.any(bands <= 0):
            raise ValueError("All centre frequencies must be > 0 Hz")

        self.type: str = filter_type
        self.fs: int = int(fs)
        self.order: int = int(order)
        self.bands: np.ndarray = bands

        # Guard against placing the high‑pass just at Nyquist.
        nyq_margin = (self.fs / 2) * np.sqrt(2)
        if bands.max() >= nyq_margin:
            raise ValueError(
                f"Highest band ({bands.max():.0f} Hz) violates Nyquist for fs={fs}"
            )

        # Design filter bank (list of SOS arrays).
        self.sos: List[np.ndarray] = self._design_filters()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def filter_signals(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the filter bank to *data*.

        Parameters
        ----------
        data : ndarray
            1‑D mono signal.

        Returns
        -------
        ndarray
            Array of shape ``(n_bands, len(data))`` containing the band‑filtered
            signals.
        """
        data = np.asarray(data, dtype=float, order="C")
        out = np.empty((len(self.sos), len(data)), dtype=float)
        for i, sos in enumerate(self.sos):
            out[i] = sosfilt(sos, data)
        return out

    # Allow the instance itself to be called like a function
    __call__ = filter_signals

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _design_filters(self) -> List[np.ndarray]:
        sos_bank = []
        for band in self.bands:
            low, high, mode = self._cutoff_frequencies(band)
            sos_bank.append(
                butter(
                    self.order,
                    [low, high] if mode == "bp" else low,
                    fs=self.fs,
                    btype="bandpass" if mode == "bp" else "highpass",
                    output="sos",
                )
            )
        return sos_bank

    def _cutoff_frequencies(self, band: float) -> tuple[float, float | None, str]:
        """
        Calculate the lower/upper cut‑off frequencies for *band*.

        Returns
        -------
        (low, high, mode)
            *low* and *high* are normalised cut‑off frequencies.
            If the filter is high‑pass, *high* is ``None`` and *mode* is ``'hp'``;
            otherwise *mode* is ``'bp'`` (band‑pass).
        """
        # Octave ratio: √2 ; third‑octave ratio: 2^(1/6)
        ratio = np.sqrt(2) if self.type == "octave band" else 2 ** (1 / 6)

        if band < 8000:  # band‑pass
            return band / ratio, band * ratio, "bp"

        # ≥ 8 k Hz: high‑pass (upper cut‑off would exceed Nyquist)
        return band / ratio, None, "hp"