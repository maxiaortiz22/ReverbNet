import numpy as np
from scipy.signal import butter, sosfilt


class BandpassFilter:
    """
    Banco de filtros pasabanda para bandas de octava o tercio de octava.

    Parámetros:
    -----------
    filter_type : str
        'octave band' o 'third octave band'.
    fs : int
        Frecuencia de muestreo.
    order : int
        Orden del filtro Butterworth.
    bands : list of float
        Frecuencias centrales de las bandas a filtrar (en Hz).
    """

    def __init__(self, filter_type, fs, order, bands):
        if filter_type not in ['octave band', 'third octave band']:
            raise ValueError("filter_type debe ser 'octave band' o 'third octave band'")

        self.type = filter_type
        self.fs = fs
        self.order = order
        self.bands = bands
        self.nyquist = fs / 2
        self.sos_filters = []

        for band in bands:
            lowcut, highcut = self._get_band_edges(band)

            if highcut >= self.nyquist:
                # Si la banda supera Nyquist, usar pasa altos
                self.sos_filters.append(butter(self.order, lowcut, fs=self.fs, btype='highpass', output='sos'))
            else:
                self.sos_filters.append(butter(self.order, [lowcut, highcut], fs=self.fs, btype='bandpass', output='sos'))

    def _get_band_edges(self, band_center):
        """Calcula los límites inferior y superior de la banda según el tipo."""
        if self.type == 'octave band':
            factor = np.sqrt(2)
        else:  # third octave band
            factor = 2 ** (1 / 6)

        lowcut = band_center / factor
        highcut = band_center * factor
        return lowcut, highcut

    def filtered_signals(self, data):
        """
        Aplica los filtros a una señal mono o multicanal.

        Parámetros:
        -----------
        data : ndarray
            Señal de entrada (1D: mono o 2D: [n_canales, n_muestras]).

        Retorna:
        --------
        filtered : ndarray
            Señales filtradas con forma (n_bandas, n_canales, n_muestras) o (n_bandas, n_muestras) si data es 1D.
        """
        data = np.asarray(data)

        if data.ndim == 1:
            # Señal mono
            filtered = np.empty((len(self.bands), len(data)))
            for i, sos in enumerate(self.sos_filters):
                filtered[i] = sosfilt(sos, data)
            return filtered

        elif data.ndim == 2:
            # Señal multicanal
            n_channels, n_samples = data.shape
            filtered = np.empty((len(self.bands), n_channels, n_samples))
            for i, sos in enumerate(self.sos_filters):
                for ch in range(n_channels):
                    filtered[i, ch] = sosfilt(sos, data[ch])
            return filtered

        else:
            raise ValueError("data debe ser un array 1D o 2D")
