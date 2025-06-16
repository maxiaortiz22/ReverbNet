import numpy as np
import sys
import math
from scipy import stats
import warnings

# Set to True if you have numba installed, False otherwise
USE_NUMBA = False
try:
    from numba import jit, njit
    USE_NUMBA = True
except ImportError:
    # Create dummy decorators if numba is not available
    def njit(func):
        return func
    def jit(func):
        return func

class NoiseError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f'NoiseError: {self.message}'
        else:
            return f'NoiseError has been raised: {self.message}'


@njit if USE_NUMBA else lambda x: x
def leastsquares_numba(x, y):
    """Optimized least squares using numba for better performance"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    # Calculate slope and intercept using normal equations
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    c = (sum_y - m * sum_x) / n
    
    return m, c


def leastsquares(x, y):
    """Given two vectors x and y of equal dimension, calculates
    the slope and y intercept of the y2 = c + m*x slope, obtained
    by least squares linear regression"""
    
    # Use optimized numba version for better performance
    m, c = leastsquares_numba(x, y)
    y2 = m * x + c  # Fitted line
    return m, c, y2


@njit
def schroeder_numba(ir, t, C):
    """Optimized Schroeder integration using numba"""
    ir_truncated = ir[:int(t)]
    # Reverse cumsum implementation optimized for numba
    cumsum_reversed = np.zeros_like(ir_truncated)
    cumsum_reversed[-1] = ir_truncated[-1]
    for i in range(len(ir_truncated) - 2, -1, -1):
        cumsum_reversed[i] = cumsum_reversed[i + 1] + ir_truncated[i]
    
    sum_ir = np.sum(ir_truncated)
    y = (cumsum_reversed + C) / (sum_ir + C)
    return y


def schroeder(ir, t, C):
    """ Smooths a curve (ir) using Schroeder Integration method. "t" and "C" are Lundeby's compensation arguments """
    return schroeder_numba(ir, t, C)


def tr_convencional(raw_signal, fs, rt='t30'):
    """
    Reverberation time from an impulse response.
    Optimized version with vectorized operations.
    """
    rt = rt.lower()
    if rt == 't30':
        init, end, factor = -5.0, -35.0, 2.0
    elif rt == 't20':
        init, end, factor = -5.0, -25.0, 3.0
    elif rt == 't10':
        init, end, factor = -5.0, -15.0, 6.0
    elif rt == 'edt':
        init, end, factor = 0.0, -10.0, 6.0

    # Find maximum and truncate signal (vectorized)
    in_max = np.argmax(np.abs(raw_signal))
    raw_signal = raw_signal[in_max:]
    
    # Normalize signal
    abs_signal = np.abs(raw_signal)
    abs_signal /= np.max(abs_signal)

    # Schroeder integration (optimized)
    signal_squared = abs_signal ** 2
    sch = np.cumsum(signal_squared[::-1])[::-1]
    
    # Convert to dB with epsilon for numerical stability
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sch_db = 10.0 * np.log10(sch / np.max(sch) + sys.float_info.epsilon)

    # Find indices more efficiently
    init_sample = np.argmin(np.abs(sch_db - init))
    end_sample = np.argmin(np.abs(sch_db - end))
    
    # Linear regression
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[:2]

    # Reverberation time calculation
    t60 = factor * ((end - intercept) / slope - (init - intercept) / slope)
    return t60


@njit
def calculate_averages_numba(y_power, v, t):
    """Optimized average calculation using numba"""
    y_promedio = np.zeros(t)
    eje_tiempo = np.zeros(t)
    
    for i in range(t):
        start_idx = i * v
        end_idx = min((i + 1) * v, len(y_power))
        y_promedio[i] = np.mean(y_power[start_idx:end_idx])
        eje_tiempo[i] = (start_idx + end_idx - 1) // 2
    
    return y_promedio, eje_tiempo


def lundeby(y_power, Fs, Ts, max_ruido_dB):
    """Optimized Lundeby method with numba acceleration and vectorized operations"""
    
    # Pre-calculate constants
    samples_per_window = int(Fs * Ts)
    total_windows = len(y_power) // samples_per_window
    
    # Initial averaging using optimized function
    y_promedio, eje_tiempo = calculate_averages_numba(y_power, samples_per_window, total_windows)

    # Vectorized noise calculation
    noise_start_idx = int(0.9 * len(y_power))
    max_power = np.max(y_power)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ruido_dB = 10 * np.log10(
            np.mean(y_power[noise_start_idx:]) / max_power + sys.float_info.epsilon
        )
        y_promediodB = 10 * np.log10(y_promedio / max_power + sys.float_info.epsilon)

    if ruido_dB > max_ruido_dB:
        raise NoiseError(f'Insufficient S/N ratio to perform Lundeby. Need at least {max_ruido_dB} dB')

    # Find regression range more efficiently
    valid_indices = np.where(y_promediodB > ruido_dB + 10)[0]
    if len(valid_indices) == 0:
        raise ValueError('No hay valor de la señal que esté 10 dB por encima del ruido')
    
    r = int(np.max(valid_indices))
    m, c, _ = leastsquares(eje_tiempo[:r], y_promediodB[:r])
    cruce = (ruido_dB - c) / m

    # Lundeby iterations with optimizations
    error = 1.0
    INTMAX = 25
    veces = 1
    
    while error > 0.0001 and veces <= INTMAX:
        # Vectorized calculations
        p = 10
        delta = abs(10 / m)
        v = max(1, int(delta / p))
        
        # Determine number of windows
        cruce_limit = min(int(cruce - delta), len(y_power))
        if cruce_limit <= 0:
            cruce_limit = len(y_power)
        
        t = max(2, cruce_limit // v)
        
        # Use optimized averaging function
        media, eje_tiempo = calculate_averages_numba(y_power[:cruce_limit], v, t)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mediadB = 10 * np.log10(media / max_power + sys.float_info.epsilon)
        
        m, c, _ = leastsquares(eje_tiempo, mediadB)

        # Noise calculation
        noise_start = max(0, int(cruce + delta))
        noise_end = len(y_power)
        
        if noise_end - noise_start < int(0.1 * len(y_power)):
            noise_start = int(0.9 * len(y_power))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms_dB = 10 * np.log10(
                np.mean(y_power[noise_start:noise_end]) / max_power + sys.float_info.epsilon
            )

        # Update convergence
        new_cruce = (rms_dB - c) / m
        error = abs(cruce - new_cruce) / abs(cruce) if cruce != 0 else 1
        cruce = round(new_cruce)
        veces += 1

    # Final calculations
    punto = min(int(cruce), len(y_power))
    
    # Optimized C calculation
    exp_term = math.exp(m / 10 / math.log10(math.e) * cruce)
    C = max_power * 10**(c/10) * exp_term / (-m / 10 / math.log10(math.e))
    
    return punto, C, ruido_dB


def tr_lundeby(y, fs, max_ruido_dB):
    """Optimized T30 parameter calculation using Lundeby method"""
    
    # Normalize and square signal (vectorized)
    y_normalized = y / np.max(np.abs(y))
    y_power = y_normalized ** 2

    # Find maximum index efficiently
    in_max = np.argmax(y_power)
    y_power = y_power[in_max:]

    # Lundeby method
    t, C, ruido_dB = lundeby(y_power, fs, 0.05, max_ruido_dB)

    # Schroeder integration
    sch = schroeder(y_power, t, C)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sch_db = 10 * np.log10(sch / np.max(sch) + sys.float_info.epsilon)

    # T30 calculation optimized
    time_vector = np.arange(len(sch_db)) / fs
    
    # Find maximum and apply mask efficiently
    i_max = np.argmax(sch_db)
    sch_truncated = sch_db[i_max:]
    time_truncated = time_vector[i_max:]
    
    # Find T30 range more efficiently
    max_val = np.max(sch_truncated)
    mask = (sch_truncated <= max_val - 5) & (sch_truncated > max_val - 35)
    
    if not np.any(mask):
        raise ValueError("No se encontraron suficientes puntos para calcular T30")
    
    t_30 = time_truncated[mask]
    y_t30 = sch_truncated[mask]
    
    m_t30, c_t30, _ = leastsquares(t_30, y_t30)
    T30 = -60 / m_t30
    
    return T30, sch_db, ruido_dB