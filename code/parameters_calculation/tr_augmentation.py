import numpy as np
from pandas import Series
from scipy.optimize import curve_fit
from scipy.signal import hann
from sklearn.linear_model import LinearRegression
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor
import os

# Numba import with fallback
USE_NUMBA = False
try:
    from numba import jit, njit, prange
    USE_NUMBA = True
except ImportError:
    def njit(func=None, **kwargs):
        return func if func else lambda x: x
    def jit(func=None, **kwargs):
        return func if func else lambda x: x
    def prange(x):
        return range(x)

class TrAugmentationError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f'TrAugmentationError: {self.message}'
        else:
            return 'TrAugmentationError has been raised'


def get_audio_list(path, file_types=('.wav', '.WAV', '.flac', '.FLAC')):
    """Optimized audio file search using os.walk for better performance"""
    audio_list = []
    
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(file_types):
                    audio_list.append(os.path.join(root, file))
    else:
        # Fallback to glob for pattern matching
        search_path = path + '/**/*'
        for file_type in file_types:
            audio_list.extend(glob.glob(search_path + file_type, recursive=True))
    
    return audio_list


@njit if USE_NUMBA else lambda x: x
def get_noise_level_numba(env, cross_point, slope, DISTANCIA_AL_CRUCE):
    """Optimized noise level calculation using numba"""
    noise_init = int(cross_point + (-DISTANCIA_AL_CRUCE / slope))
    
    if noise_init > int(len(env) * 0.9):
        # Take last 10%
        start_idx = int(len(env) * 0.9)
        noise_floor = np.mean(env[start_idx:])
    else:
        noise_floor = np.mean(env[noise_init:])
    
    return 10 * np.log10(noise_floor + np.finfo(np.float64).eps)


def get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE):
    """Wrapper for noise level calculation"""
    return get_noise_level_numba(env, cross_point, slope, DISTANCIA_AL_CRUCE)


def linear_regression_optimized(n, arr):
    """Optimized linear regression using direct NumPy operations"""
    # Convert to float64 for better numerical stability
    x = n.astype(np.float64).reshape(-1, 1)
    y = arr.astype(np.float64)
    
    # Direct calculation using normal equations (faster for small datasets)
    if len(x) < 1000:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x.flatten() - x_mean) * (y - y_mean))
        denominator = np.sum((x.flatten() - x_mean) ** 2)
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        return np.array([slope]), intercept
    else:
        # Use sklearn for larger datasets
        model = LinearRegression().fit(x, y)
        return model.coef_, model.intercept_


def temporal_decompose(rir, fs, tau=0.0025):
    """Optimized temporal decomposition with vectorized operations"""
    t_d = np.argmax(np.abs(rir))  # Use abs for robustness
    t_o = int(tau * fs)  # tolerance window in samples (2.5 ms)
    
    # Vectorized bounds checking
    init_idx = max(0, t_d - t_o)
    final_idx = min(len(rir), t_d + t_o + 1)

    early = rir[init_idx:final_idx]
    late = rir[final_idx:]
    delay = rir[:init_idx]
    
    return delay, early, late


@njit if USE_NUMBA else lambda x: x
def rolling_operation_numba(arr, window_length, operation='mean'):
    """Optimized rolling operations using numba"""
    result = np.zeros_like(arr, dtype=np.float64)
    half_window = window_length // 2
    
    for i in prange(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window_data = arr[start:end]
        
        if operation == 'mean':
            result[i] = np.mean(window_data)
        elif operation == 'max':
            result[i] = np.max(window_data)
        elif operation == 'median':
            result[i] = np.median(window_data)
    
    return result


def get_abs_envelope(arr, window_length=50, operation='mean'):
    """Optimized envelope calculation"""
    abs_arr = np.abs(arr)
    
    if USE_NUMBA:
        return rolling_operation_numba(abs_arr, window_length, operation)
    else:
        # Fallback using pandas (still optimized)
        arr_series = Series(abs_arr)
        if operation == 'mean':
            result = arr_series.rolling(window=window_length, min_periods=1, center=True).mean()
        elif operation == 'max':
            result = arr_series.rolling(window=window_length, min_periods=1, center=True).max()
        elif operation == 'median':
            result = arr_series.rolling(window=window_length, min_periods=1, center=True).median()
        else:
            result = arr_series.rolling(window=window_length, min_periods=1, center=True).mean()
        
        return result.to_numpy()


def get_envelope(arr, window_length):
    """Optimized envelope using max operation"""
    return get_abs_envelope(arr, window_length, 'max')


def get_abs_max_envelope(arr, window_length=500):
    """Optimized max envelope using median operation"""
    return get_abs_envelope(arr, window_length, 'median')


def normalize_rir(rir):
    """Optimized RIR normalization"""
    # Find maximum index more efficiently
    abs_rir = np.abs(rir)
    index_max = np.argmax(abs_rir)
    
    # Normalize using the maximum value
    rir_normalized = rir / rir[index_max]
    
    return rir_normalized


@njit if USE_NUMBA else lambda x: x
def get_valid_interval_numba(arr, init_value, final_value):
    """Optimized valid interval calculation using numba"""
    # Find onset (first index where arr < init_value)
    onset = -1
    for i in range(len(arr)):
        if arr[i] < init_value:
            onset = i
            break
    
    if onset == -1:
        onset = len(arr) - 1
    
    # Find final point
    final = -1
    for i in range(onset, len(arr)):
        if arr[i] < final_value:
            final = i
            break
    
    if final == -1:
        final = len(arr)
    
    return onset, final


def get_valid_interval(arr, init_value, final_value):
    """Get valid interval with optimized search"""
    if USE_NUMBA:
        onset, final = get_valid_interval_numba(arr, init_value, final_value)
    else:
        # Vectorized approach for non-numba
        valid_indices = np.where(arr < init_value)[0]
        onset = valid_indices[0] if len(valid_indices) > 0 else len(arr) - 1
        
        arr_aux = arr[onset:]
        final_indices = np.where(arr_aux < final_value)[0]
        final_relative = final_indices[0] if len(final_indices) > 0 else len(arr_aux)
        final = onset + final_relative
    
    arr_chunk = arr[onset:final]
    samples = np.arange(onset, final)
    
    return arr_chunk, samples


def Lundeby_method_optimized(rir, fs):
    """Optimized Lundeby method with better constants and vectorization"""
    # Constants
    EPS = np.finfo(np.float64).eps
    TIME_INTERVAL = 160  # 10 ms
    DISTANCIA_AL_PISO = 5
    N_INTERVALOS_10DB = 10
    DISTANCIA_AL_CRUCE = 5
    RANGO_DINAMICO = 10
    
    # Find maximum more efficiently
    abs_rir = np.abs(rir)
    max_idx = np.argmax(abs_rir)
    
    if max_idx == 0:
        ADD_INIT = False
        processed_rir = rir
    else:
        ADD_INIT = True
        delay = rir[:max_idx]
        processed_rir = rir[max_idx:]

    # Optimized envelope calculation
    env = get_envelope(processed_rir, TIME_INTERVAL)
    
    # Vectorized dB conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env_db = 10 * np.log10(env + EPS)
    
    n = np.arange(len(env_db))

    # Optimized noise floor calculation
    noise_segment_size = int(len(env) * 0.1)
    noise_floor = np.mean(env[-noise_segment_size:])
    noise_floor_db = 10 * np.log10(noise_floor + EPS)

    init_value = np.max(env_db)
    final_value = noise_floor_db + DISTANCIA_AL_PISO
    
    env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
    
    slope, intercept = linear_regression_optimized(n_chunk, env_db_chunk)
    slope = slope[0] if isinstance(slope, np.ndarray) else slope

    cross_point = int((noise_floor_db - intercept) / slope)
    time_interval = max(1, int((-10 / slope) / N_INTERVALOS_10DB))

    # Recalculate envelope with new time interval
    env = get_envelope(processed_rir, time_interval)
    env_db = 10 * np.log10(env + EPS)

    # Optimized iteration
    iteracion = 0
    delta_level = 1.0
    max_iterations = 6

    while iteracion < max_iterations and delta_level > 0.2:
        # Limit cross point to audio length
        cross_point = min(cross_point, len(env_db) - 1)
        cross_level_1 = env_db[cross_point]

        noise_floor_db = get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE)

        init_value = noise_floor_db + DISTANCIA_AL_PISO
        final_value = max(env_db[-2], init_value - RANGO_DINAMICO)
        
        env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
        
        if len(n_chunk) > 1:  # Ensure we have enough points for regression
            slope, intercept = linear_regression_optimized(n_chunk, env_db_chunk)
            slope = slope[0] if isinstance(slope, np.ndarray) else slope
            cross_point = int((noise_floor_db - intercept) / slope)
        
        cross_point = min(cross_point, len(env_db) - 1)
        cross_level_2 = env_db[cross_point]

        delta_level = abs(cross_level_1 - cross_level_2)
        iteracion += 1
        
    rir_cut = processed_rir[:cross_point]
    
    if ADD_INIT:
        cross_point_compensado = len(delay) + len(rir_cut)
    else:
        cross_point_compensado = len(rir_cut)
        
    return cross_point_compensado


def Lundeby_method(rir, fs):
    """Wrapper for backwards compatibility"""
    return Lundeby_method_optimized(rir, fs)


@njit if USE_NUMBA else lambda x: x
def curva_modelo_numba(t, Am, decay_rate, noise_floor):
    """Optimized model curve using numba"""
    return Am * np.exp(-t / decay_rate) + noise_floor


def curva_modelo(t, Am, decay_rate, noise_floor):
    """Model curve with numba optimization"""
    if USE_NUMBA:
        return curva_modelo_numba(t, Am, decay_rate, noise_floor)
    else:
        ones = np.ones(len(t))
        return Am * np.exp(-t / decay_rate) * ones + (noise_floor * ones)


def estimated_fullband_decay(rir, fs):
    """Optimized fullband decay estimation"""
    delay, early, late = temporal_decompose(rir, fs)
    late_env = get_abs_envelope(late)  # window length = 40

    # Model late field reverb
    t = np.linspace(0, len(late_env) / fs, len(late_env))
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(curva_modelo, t, late_env, bounds=(0, 1))
        return popt[1]
    except:
        # Fallback to simple estimation
        return 0.5


def estim_params(late, cross_point, fs):
    """Optimized parameter estimation"""
    late_env = get_abs_max_envelope(late)
    late_env_valid = late_env[:cross_point]
    
    t = np.linspace(0, len(late_env_valid) / fs, len(late_env_valid))
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(curva_modelo, t, late_env_valid, bounds=(0, 1))
        
        return {'Am': popt[0], 'decay_rate': popt[1], 'noise_floor': popt[2]}
    except:
        # Fallback parameters
        return {'Am': np.max(late_env_valid), 'decay_rate': 0.5, 'noise_floor': np.min(late_env_valid)}


@njit if USE_NUMBA else lambda x: x
def curva_noiseless_numba(t, Am, decay_rate):
    """Optimized noiseless curve using numba"""
    noise = np.random.normal(0, 1, len(t))
    return Am * np.exp(-t / decay_rate) * noise


def curva_noiseless(t, Am, decay_rate):
    """Noiseless curve with numba optimization"""
    if USE_NUMBA:
        return curva_noiseless_numba(t, Am, decay_rate)
    else:
        noise = np.random.normal(0, 1, len(t))
        return Am * np.exp(-t / decay_rate) * noise


def cross_fade_optimized(señal_1, señal_2, fs, cross_point):
    """Optimized cross-fade with better bounds checking"""
    largo = int(50 * 0.001 * fs)  # Use actual fs instead of hardcoded 16000
    
    # Bounds checking
    if 2 * largo > len(señal_1) - cross_point or cross_point <= 0:
        return señal_1
    
    # Generate Hann window once
    ventana = hann(largo)
    fade_in, fade_out = ventana[:largo//2], ventana[largo//2:]
    
    # Optimized window creation using broadcasting
    len_signal = len(señal_1)
    ventana_atenuante = np.ones(len_signal)
    ventana_amplificadora = np.zeros(len_signal)
    
    # Calculate indices once
    fade_start = cross_point - len(fade_out) // 2
    fade_end = fade_start + len(fade_out)
    fade_in_end = fade_start + len(fade_in)
    
    # Apply fades with bounds checking
    if fade_start >= 0 and fade_end <= len_signal:
        ventana_atenuante[fade_start:fade_end] = fade_out
        ventana_atenuante[fade_end:] = 0
        
        ventana_amplificadora[fade_start:fade_in_end] = fade_in
        ventana_amplificadora[fade_in_end:] = 1
    
    return señal_1 * ventana_atenuante + señal_2 * ventana_amplificadora


def cross_fade(señal_1, señal_2, fs, cross_point):
    """Wrapper for backwards compatibility"""
    return cross_fade_optimized(señal_1, señal_2, fs, cross_point)


def noise_crossfade(rir, estim_params, cross_point, fs):
    """Optimized noise crossfade"""
    t = np.linspace(0, len(rir) / fs, len(rir))
    rir_noiseless = curva_noiseless(t, estim_params['Am'], estim_params['decay_rate'])
    rir_denoised = cross_fade(rir, rir_noiseless, fs, cross_point)
    return rir_denoised


@njit if USE_NUMBA else lambda x: x
def augmentation_numba(rir, t, decay_rate, t_md):
    """Optimized augmentation using numba"""
    exp_factor = -t * ((decay_rate - t_md) / (decay_rate * t_md))
    return rir * np.exp(exp_factor)


def augmentation(rir, estim_params, estim_fullband_decay, TR60_desired, fs):
    """Optimized augmentation with numba acceleration"""
    t = np.linspace(0, len(rir) / fs, len(rir))
    decay_rate_d = TR60_desired / np.log(1000)
    ratio = decay_rate_d / estim_fullband_decay
    t_md = ratio * estim_params['decay_rate']

    if USE_NUMBA:
        return augmentation_numba(rir, t, estim_params['decay_rate'], t_md)
    else:
        # Fallback implementation
        exp_factor = -t * ((estim_params['decay_rate'] - t_md) / 
                          (estim_params['decay_rate'] * t_md))
        return rir * np.exp(exp_factor)


def process_single_band(args):
    """Process a single frequency band - designed for parallel processing"""
    banda, rir_band, fs, estim_fullband_decay, TR_DESEADO = args
    
    try:
        cross_point = Lundeby_method(rir_band, fs)
        parameters = estim_params(rir_band, cross_point, fs)
        rir_band_denoised = noise_crossfade(rir_band, parameters, cross_point, fs)
        rir_band_aug = augmentation(rir_band_denoised, parameters, 
                                   estim_fullband_decay, TR_DESEADO, fs)
        return banda, rir_band_aug
    except Exception as e:
        # Return original band if processing fails
        return banda, rir_band


def tr_augmentation(rir_entrada, fs, TR_DESEADO, bpfilter, use_parallel=True):
    """Optimized TR augmentation with optional parallel processing"""
    try:
        rir_entrada = normalize_rir(rir_entrada)
        delay, early, rir = temporal_decompose(rir_entrada, fs)

        estim_fullband_decay = estimated_fullband_decay(rir, fs)
        rir_bands = bpfilter.filtered_signals(rir)
        
        num_bands = rir_bands.shape[0]
        rir_band_augs = np.empty(rir_bands.shape, dtype=np.float32)
        
        if use_parallel and num_bands > 2:  # Use parallel processing for multiple bands
            # Prepare arguments for parallel processing
            args_list = [(banda, rir_bands[banda, :], fs, estim_fullband_decay, TR_DESEADO) 
                        for banda in range(num_bands)]
            
            # Process bands in parallel
            with ThreadPoolExecutor(max_workers=min(4, num_bands)) as executor:
                results = list(executor.map(process_single_band, args_list))
            
            # Collect results
            for banda, rir_band_aug in results:
                rir_band_augs[banda, :] = rir_band_aug
        else:
            # Sequential processing
            for banda in range(num_bands):
                args = (banda, rir_bands[banda, :], fs, estim_fullband_decay, TR_DESEADO)
                _, rir_band_aug = process_single_band(args)
                rir_band_augs[banda, :] = rir_band_aug
        
        # Sum all bands and concatenate
        rir_aug = np.sum(rir_band_augs, axis=0)
        rir_aug = np.concatenate((delay, early, rir_aug)).astype(np.float32)
        
        return rir_aug

    except Exception as err:
        raise TrAugmentationError(f'No se pudo trabajar con el audio a {np.round(TR_DESEADO, 1)} s: {str(err)}')