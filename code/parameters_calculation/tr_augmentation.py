import numpy as np
from pandas import Series
from scipy.optimize import curve_fit
try:
    from scipy.signal import hann
except ImportError:
    from scipy.signal.windows import hann
from sklearn.linear_model import LinearRegression
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor
import os

# ----------------------------------------------------------- #
#   IMPORTACIÓN OPCIONAL DE NUMBA
# ----------------------------------------------------------- #
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

# ----------------------------------------------------------- #
#   EXCEPCIÓN PERSONALIZADA
# ----------------------------------------------------------- #
class TrAugmentationError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None
    def __str__(self):
        return f"TrAugmentationError: {self.message}" if self.message else \
               "TrAugmentationError has been raised"

# ----------------------------------------------------------- #
#   UTILIDADES DE I/O
# ----------------------------------------------------------- #
def get_audio_list(path, file_types=('.wav', '.WAV', '.flac', '.FLAC')):
    """Recorrido con os.walk (más veloz que glob on-the-fly)."""
    audio_list = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(file_types):
                    audio_list.append(os.path.join(root, f))
    else:  # patrón suelto
        search_path = path + '/**/*'
        for ft in file_types:
            audio_list.extend(glob.glob(search_path + ft, recursive=True))
    return audio_list

# ----------------------------------------------------------- #
#   FUNCIONES NUMÉRICAS ACELERADAS
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda x: x
def get_noise_level_numba(env, cross_point, slope, DIST):
    noise_init = int(cross_point + (-DIST / slope))
    if noise_init > int(len(env) * 0.9):
        noise_floor = np.mean(env[int(len(env) * 0.9):])
    else:
        noise_floor = np.mean(env[noise_init:])
    return 10 * np.log10(noise_floor + np.finfo(np.float64).eps)

def get_noise_level(cross_point, slope, env, DIST):
    return get_noise_level_numba(env, cross_point, slope, DIST)

# ----------------------------------------------------------- #
#   REGRESIÓN LINEAL OPTIMIZADA
# ----------------------------------------------------------- #
def linear_regression_optimized(n, arr):
    x = n.astype(np.float64).reshape(-1, 1)
    y = arr.astype(np.float64)
    if len(x) < 1000:
        xm, ym = x.mean(), y.mean()
        num = np.sum((x.flatten() - xm) * (y - ym))
        den = np.sum((x.flatten() - xm) ** 2)
        slope = num / den if den != 0 else 0.0
        return np.array([slope]), ym - slope * xm
    # para sets grandes usa sklearn
    model = LinearRegression().fit(x, y)
    return model.coef_, model.intercept_

# ----------------------------------------------------------- #
#   DESCOMPOSICIÓN TEMPORAL
# ----------------------------------------------------------- #
def temporal_decompose(rir, fs, tau=0.0025):
    t_d = np.argmax(np.abs(rir))            # ← mantiene cambio (punto 1)
    t_o = int(tau * fs)
    init_idx, final_idx = max(0, t_d - t_o), min(len(rir), t_d + t_o + 1)
    early  = rir[init_idx:final_idx]
    late   = rir[final_idx:]
    delay  = rir[:init_idx]
    return delay, early, late

# ----------------------------------------------------------- #
#   ENVOLVENTES (con fallback a pandas)
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda f: f
def rolling_operation_numba(arr, win, op='mean'):
    res = np.zeros_like(arr, dtype=np.float64)
    half = win // 2
    for i in prange(len(arr)):
        s, e = max(0, i-half), min(len(arr), i+half+1)
        if op == 'mean':
            res[i] = np.mean(arr[s:e])
        elif op == 'max':
            res[i] = np.max(arr[s:e])
        elif op == 'median':
            res[i] = np.median(arr[s:e])
    return res

def get_abs_envelope(arr, win=50, op='mean'):
    abs_arr = np.abs(arr)
    if USE_NUMBA:
        return rolling_operation_numba(abs_arr, win, op)
    ser = Series(abs_arr)
    if op == 'mean':
        return ser.rolling(win, 1, True).mean().to_numpy()
    if op == 'max':
        return ser.rolling(win, 1, True).max().to_numpy()
    return ser.rolling(win, 1, True).median().to_numpy()

def get_envelope(arr, win):            # max envelope
    return get_abs_envelope(arr, win, 'max')

def get_abs_max_envelope(arr, win=500):# median envelope
    return get_abs_envelope(arr, win, 'median')

# ----------------------------------------------------------- #
#   NORMALIZACIÓN DE RIR
# ----------------------------------------------------------- #
def normalize_rir(rir):
    idx_max = np.argmax(np.abs(rir))
    return rir / rir[idx_max]

# ----------------------------------------------------------- #
#   INTERVALO VÁLIDO
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda f: f
def get_valid_interval_numba(arr, init_val, final_val):
    onset = -1
    for i in range(len(arr)):
        if arr[i] < init_val:
            onset = i
            break
    if onset == -1:
        onset = len(arr) - 1
    final = -1
    for i in range(onset, len(arr)):
        if arr[i] < final_val:
            final = i
            break
    if final == -1:
        final = len(arr)
    return onset, final

def get_valid_interval(arr, init_val, final_val):
    if USE_NUMBA:
        onset, final = get_valid_interval_numba(arr, init_val, final_val)
    else:
        idx = np.where(arr < init_val)[0]
        onset = idx[0] if idx.size else len(arr)-1
        arr_aux = arr[onset:]
        idx2 = np.where(arr_aux < final_val)[0]
        final = onset + (idx2[0] if idx2.size else len(arr_aux))
    return arr[onset:final], np.arange(onset, final)

# ----------------------------------------------------------- #
#   MÉTODO DE LUNDEBY OPTIMIZADO (con protección punto 2)
# ----------------------------------------------------------- #
def Lundeby_method_optimized(rir, fs):
    EPS = np.finfo(np.float64).eps
    TIME_INTERVAL = 160   # 10 ms
    DIST_PISO = 5
    N_INT_10DB = 10
    DIST_CRUCE = 5
    RANGO_DIN = 10

    abs_rir = np.abs(rir)
    max_idx = np.argmax(abs_rir)
    ADD_INIT = max_idx != 0
    delay = rir[:max_idx] if ADD_INIT else np.array([], dtype=rir.dtype)
    proc_rir = rir[max_idx:]

    env = get_envelope(proc_rir, TIME_INTERVAL)
    env_db = 10 * np.log10(env + EPS)
    n = np.arange(len(env_db))

    noise_lin  = env[-int(len(env)*0.1):].mean()
    noise_db   = 10 * np.log10(noise_lin + EPS)

    init_val   = env_db.max()
    final_val  = noise_db + DIST_PISO
    env_chunk, n_chunk = get_valid_interval(env_db, init_val, final_val)

    slope, intercept = linear_regression_optimized(n_chunk, env_chunk)

    # --- Protección punto 2 ------------------------------------------
    if isinstance(slope, np.ndarray):
        slope = slope[0] if slope.size else 0.0

    cross_pt = int((noise_db - intercept) / slope) if slope != 0 else len(env_db)-1
    TIME_INTERVAL = max(1, int((-10 / slope) / N_INT_10DB)) if slope != 0 else 1

    env = get_envelope(proc_rir, TIME_INTERVAL)
    env_db = 10 * np.log10(env + EPS)

    iteracion, delta = 0, 1.0
    while iteracion < 6 and delta > 0.2:
        cross_pt = min(cross_pt, len(env_db) - 1)
        cross_lvl1 = env_db[cross_pt]
        noise_db = get_noise_level(cross_pt, slope, env, DIST_CRUCE)

        init_val = noise_db + DIST_PISO
        final_val = max(env_db[-2], init_val - RANGO_DIN)
        env_chunk, n_chunk = get_valid_interval(env_db, init_val, final_val)

        if n_chunk.size > 1:
            slope, intercept = linear_regression_optimized(n_chunk, env_chunk)
            if isinstance(slope, np.ndarray):
                slope = slope[0] if slope.size else 0.0
            cross_pt = int((noise_db - intercept) / slope) if slope != 0 else len(env_db)-1

        cross_pt = min(cross_pt, len(env_db)-1)
        cross_lvl2 = env_db[cross_pt]
        delta = abs(cross_lvl1 - cross_lvl2)
        iteracion += 1

    rir_cut = proc_rir[:cross_pt]
    cross_comp = cross_pt + len(delay) if ADD_INIT else cross_pt
    return cross_comp

def Lundeby_method(rir, fs):
    return Lundeby_method_optimized(rir, fs)

# ----------------------------------------------------------- #
#   CURVAS EXPONENCIALES
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda f: f
def curva_modelo_numba(t, Am, decay_rate, noise_floor):
    return Am * np.exp(-t / decay_rate) + noise_floor

def curva_modelo(t, Am, decay_rate, noise_floor):
    return curva_modelo_numba(t, Am, decay_rate, noise_floor) if USE_NUMBA \
           else Am * np.exp(-t / decay_rate) + noise_floor

# ----------------------------------------------------------- #
#   DECAY FULLBAND – sin valores por defecto (punto 4)
# ----------------------------------------------------------- #
def estimated_fullband_decay(rir, fs):
    delay, early, late = temporal_decompose(rir, fs)
    late_env = get_abs_envelope(late)
    t = np.linspace(0, len(late_env) / fs, len(late_env))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(curva_modelo, t, late_env, bounds=(0, 1))
    return popt[1]           # ← si falla, exception se propaga

# ----------------------------------------------------------- #
#   PARÁMETROS DE LA CURVA – sin fallback (punto 4)
# ----------------------------------------------------------- #
def estim_params(late, cross_point, fs):
    late_env = get_abs_max_envelope(late)
    late_valid = late_env[:cross_point]
    t = np.linspace(0, len(late_valid) / fs, len(late_valid))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(curva_modelo, t, late_valid, bounds=(0, 1))

    return {'Am': popt[0], 'decay_rate': popt[1], 'noise_floor': popt[2]}

# ----------------------------------------------------------- #
#   CURVA SIN RUIDO
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda f: f
def curva_noiseless_numba(t, Am, decay_rate):
    noise = np.random.normal(0, 1, len(t))
    return Am * np.exp(-t / decay_rate) * noise

def curva_noiseless(t, Am, decay_rate):
    return curva_noiseless_numba(t, Am, decay_rate) if USE_NUMBA \
           else np.random.normal(0, 1, len(t)) * np.exp(-t / decay_rate) * Am

# ----------------------------------------------------------- #
#   CROSS-FADE
# ----------------------------------------------------------- #
def cross_fade_optimized(sig1, sig2, fs, cross_pt):
    largo = int(0.05 * fs)      # 50 ms
    if 2 * largo > len(sig1) - cross_pt or cross_pt <= 0:
        return sig1
    win = hann(largo)
    fade_in, fade_out = win[:largo//2], win[largo//2:]
    len_sig = len(sig1)

    atenuante = np.ones(len_sig)
    amplific  = np.zeros(len_sig)
    fst = cross_pt - len(fade_out)//2
    fend = fst + len(fade_out)
    fin_end = fst + len(fade_in)

    if fst >= 0 and fend <= len_sig:
        atenuante[fst:fend] = fade_out
        atenuante[fend:] = 0
        amplific[fst:fin_end] = fade_in
        amplific[fin_end:] = 1

    return sig1 * atenuante + sig2 * amplific

def cross_fade(s1, s2, fs, cp):
    return cross_fade_optimized(s1, s2, fs, cp)

# ----------------------------------------------------------- #
#   CROSSFADING DE RUIDO
# ----------------------------------------------------------- #
def noise_crossfade(rir, params, cross_pt, fs):
    t = np.linspace(0, len(rir) / fs, len(rir))
    noiseless = curva_noiseless(t, params['Am'], params['decay_rate'])
    return cross_fade(rir, noiseless, fs, cross_pt)

# ----------------------------------------------------------- #
#   AUGMENTATION
# ----------------------------------------------------------- #
@njit if USE_NUMBA else lambda f: f
def augmentation_numba(rir, t, decay_rate, t_md):
    return rir * np.exp(-t * ((decay_rate - t_md) / (decay_rate * t_md)))

def augmentation(rir, params, fullband_decay, TR_des, fs):
    t = np.linspace(0, len(rir) / fs, len(rir))
    decay_d = TR_des / np.log(1000)
    ratio   = decay_d / fullband_decay
    t_md    = ratio * params['decay_rate']
    if USE_NUMBA:
        return augmentation_numba(rir, t, params['decay_rate'], t_md)
    exp_f = -t * ((params['decay_rate'] - t_md) /
                  (params['decay_rate'] * t_md))
    return rir * np.exp(exp_f)

# ----------------------------------------------------------- #
#   TR AUGMENTATION PRINCIPAL
# ----------------------------------------------------------- #
def process_single_band(args):
    banda, rir_band, fs, full_dec, TR_des = args
    try:
        cross_pt = Lundeby_method(rir_band, fs)
        params = estim_params(rir_band, cross_pt, fs)
        denoised = noise_crossfade(rir_band, params, cross_pt, fs)
        aug = augmentation(denoised, params, full_dec, TR_des, fs)
        return banda, aug
    except Exception as e:
        return banda, rir_band  # si falla, vuelve original

def tr_augmentation(rir_in, fs, TR_des, bpfilter, use_parallel=True):
    try:
        rir_norm = normalize_rir(rir_in)
        delay, early, rir = temporal_decompose(rir_norm, fs)

        fullband_decay = estimated_fullband_decay(rir, fs)
        rir_bands = bpfilter.process(rir)
        num_bands = rir_bands.shape[0]
        rir_aug_bands = np.empty(rir_bands.shape, dtype=np.float32)  # punto 5

        if use_parallel and num_bands > 2:
            args = [(b, rir_bands[b], fs, fullband_decay, TR_des)
                    for b in range(num_bands)]
            with ThreadPoolExecutor(max_workers=min(4, num_bands)) as ex:
                for b, aug in ex.map(process_single_band, args):
                    rir_aug_bands[b] = aug
        else:
            for b in range(num_bands):
                _, aug = process_single_band(
                    (b, rir_bands[b], fs, fullband_decay, TR_des))
                rir_aug_bands[b] = aug

        rir_aug = np.sum(rir_aug_bands, axis=0)
        return np.concatenate((delay, early, rir_aug)).astype(np.float32)

    except Exception as err:
        raise TrAugmentationError(
            f"No se pudo trabajar con el audio a {np.round(TR_des,1)} s: {err}"
        )
