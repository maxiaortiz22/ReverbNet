import numpy as np
from scipy.signal.windows import hann
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d, median_filter
import glob

class TrAugmentationError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'MyCustomError, {0} '.format(self.message)
        else:
            return 'MyCustomError has been raised'

def get_audio_list(path, file_types=('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path + file_type, recursive=True))
    return audio_list

def get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE):
    noise_init = cross_point + int(-DISTANCIA_AL_CRUCE / slope)
    if noise_init > len(env) * 0.9:
        noise_floor = env[-int(len(env) * 0.1):].mean()
        noise_floor_db = 10 * np.log10(noise_floor)
    else:
        noise_floor = env[noise_init:].mean()
        noise_floor_db = 10 * np.log10(noise_floor)
    return noise_floor_db

def linear_regression(n, arr):
    slope, intercept = np.polyfit(n, arr, 1)
    return slope, intercept

def temporal_decompose(rir, fs, tau=0.0025):
    t_d = np.argmax(rir)
    t_o = int(tau * fs)
    init_idx = t_d - t_o
    final_idx = t_d + t_o + 1
    if init_idx < 0:
        init_idx = 0
    if final_idx > len(rir) - 1:
        final_idx = len(rir) - 1
    early = rir[init_idx:final_idx]
    late = rir[final_idx:]
    delay = rir[:init_idx]
    return delay, early, late

def get_abs_envelope(arr, window_length=50):
    arr = np.abs(arr)
    window = np.ones(window_length) / window_length
    arr_mean = np.convolve(arr, window, mode='same')
    return arr_mean

def normalize_rir(rir):
    index_max = np.argmax(np.abs(rir))
    max_val = rir[index_max]
    rir = rir / np.abs(max_val)
    return rir

def estimated_fullband_decay(rir, fs):
    delay, early, late = temporal_decompose(rir, fs)
    late_env = get_abs_envelope(late)
    t = np.linspace(0, len(late_env) / fs, len(late_env))
    popt, popv = curve_fit(curva_modelo, t, late_env, method="trf", bounds=(0, 1), maxfev=15000)
    return popt[1]

def get_envelope(arr, window_length):
    arr = np.abs(arr)
    arr_max = maximum_filter1d(arr, size=window_length, mode='nearest')
    return arr_max

def get_valid_interval(arr, init_value, final_value):
    samples = np.arange(len(arr))
    onset = np.where(arr < init_value)[0][0]
    arr_aux = arr[onset:]
    final = np.where(arr_aux < final_value)[0][0]
    arr_aux2 = arr_aux[:final]
    offset = len(arr) - len(arr_aux)
    samples = samples[onset:(final + offset)]
    return arr_aux2, samples

def Lundeby_method(rir, fs):
    EPS = np.finfo(float).eps
    TIME_INTERVAL = 160
    DISTANCIA_AL_PISO = 5
    N_INTERVALOS_10DB = 10
    DISTANCIA_AL_CRUCE = 5
    RANGO_DINAMICO = 10
    max_idx = np.argmax(np.abs(rir))
    if max_idx == 0:
        ADD_INIT = False
        rir = rir
    else:
        ADD_INIT = True
        delay = rir[:max_idx]
        rir = rir[max_idx:]
    env = get_envelope(rir, TIME_INTERVAL)
    rir_squared = rir ** 2
    rir_db = 10 * np.log10(rir_squared + EPS)
    env_db = 10 * np.log10(env + EPS)
    n = np.arange(len(env_db))
    noise_floor = env[-int(len(env) * 0.1):].mean()
    noise_floor_db = 10 * np.log10(noise_floor)
    init_value = env_db.max()
    final_value = noise_floor_db + DISTANCIA_AL_PISO
    env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
    slope, intercept = linear_regression(n_chunk, env_db_chunk)
    cross_point = int((noise_floor_db - intercept) / slope)
    time_interval = int((-10 / slope) / N_INTERVALOS_10DB)
    env = get_envelope(rir, time_interval)
    iteracion = 0
    delta_level = 1.0
    while iteracion < 6 and delta_level > 0.2:
        if cross_point > len(env_db) - 1:
            cross_level_1 = env_db[-1]
        else:
            cross_level_1 = env_db[cross_point]
        noise_floor_db = get_noise_level(cross_point, slope, env, DISTANCIA_AL_CRUCE)
        init_value = noise_floor_db + DISTANCIA_AL_PISO
        if init_value - RANGO_DINAMICO < np.min(env_db):
            final_value = env_db[-2]
        else:
            final_value = init_value - RANGO_DINAMICO
        env_db_chunk, n_chunk = get_valid_interval(env_db, init_value, final_value)
        slope, intercept = linear_regression(n_chunk, env_db_chunk)
        cross_point = int((noise_floor_db - intercept) / slope)
        if cross_point > len(env_db) - 1:
            cross_level_2 = env_db[-1]
        else:
            cross_level_2 = env_db[cross_point]
        delta_level = abs(cross_level_1 - cross_level_2)
        iteracion += 1
    rir_cut = rir[:cross_point]
    if ADD_INIT:
        rir_salida = np.concatenate((delay, rir_cut))
        rir_plot = np.concatenate((delay, rir))
    else:
        rir_salida = rir_cut
        rir_plot = rir
    cross_point_compensado = len(rir_salida)
    return cross_point_compensado

def curva_modelo(t, Am, decay_rate, noise_floor):
    ones = np.ones(len(t))
    modelo = Am * np.exp(-t / decay_rate) * ones + (noise_floor * ones)
    return modelo

def get_abs_max_envelope(arr, window_length=500):
    arr = np.abs(arr)
    arr_median = median_filter(arr, size=window_length, mode='nearest')
    return arr_median

def estim_params(late, cross_point, fs):
    late_env = get_abs_max_envelope(late)
    late_env_valid = late_env[:cross_point]
    t = np.linspace(0, len(late_env_valid) / fs, len(late_env_valid))
    popt, popv = curve_fit(curva_modelo, t, late_env_valid, method="trf", bounds=(0, 1), maxfev=15000)
    estim_params = {'Am': popt[0], 'decay_rate': popt[1], 'noise_floor': popt[2]}
    return estim_params

def curva_noiseless(t, Am, decay_rate):
    noise = np.random.normal(0, 1, len(t))
    modelo = Am * np.exp(-t / decay_rate) * noise
    return modelo

def cross_fade(señal_1, señal_2, fs, cross_point):
    largo = int(50 * 0.001 * fs)
    if 2 * largo > len(señal_1) - cross_point:
        return señal_1
    ventana = hann(largo)
    fade_in, fade_out = ventana[:int(largo / 2)], ventana[int(largo / 2):]
    ventana_atenuante = np.concatenate((np.ones(cross_point - int(fade_out.size / 2)),
                                        fade_out,
                                        np.zeros(len(señal_1) - cross_point - int(fade_out.size / 2))))
    ventana_amplificadora = np.concatenate((np.zeros(cross_point - int(fade_out.size / 2)),
                                            fade_in,
                                            np.ones(len(señal_2) - cross_point - int(fade_out.size / 2))))
    return (señal_1 * ventana_atenuante) + (señal_2 * ventana_amplificadora)

def noise_crossfade(rir, estim_params, cross_point, fs):
    t = np.linspace(0, len(rir) / fs, len(rir))
    rir_noiseless = curva_noiseless(t, estim_params['Am'], estim_params['decay_rate'])
    rir_denoised = cross_fade(rir, rir_noiseless, fs, cross_point)
    return rir_denoised

def augmentation(rir, estim_params, estim_fullband_decay, TR60_desired, fs):
    t = np.linspace(0, len(rir) / fs, len(rir))
    decay_rate_d = TR60_desired / (np.log(1000))
    ratio = decay_rate_d / estim_fullband_decay
    t_md = ratio * estim_params['decay_rate']
    rir_aug = rir * np.exp(-t * ((estim_params['decay_rate'] - t_md) / (estim_params['decay_rate'] * t_md)))
    return rir_aug

def tr_augmentation(rir_entrada, fs, TR_DESEADO, bpfilter):
    try:
        rir_entrada = normalize_rir(rir_entrada)
        delay, early, rir = temporal_decompose(rir_entrada, fs)
        estim_fullband_decay = estimated_fullband_decay(rir, fs)
        rir_bands = bpfilter.process(rir.astype(np.float32))
        rir_band_augs = np.empty(rir_bands.shape)
        for banda in range(rir_bands.shape[0]):
            cross_point = Lundeby_method(rir_bands[banda, :], fs)
            parameters = estim_params(rir_bands[banda, :], cross_point, fs)
            rir_band_denoised = noise_crossfade(rir_bands[banda, :],
                                                parameters,
                                                cross_point,
                                                fs)
            rir_band_augs[banda, :] = augmentation(rir_band_denoised,
                                                   parameters,
                                                   estim_fullband_decay,
                                                   TR_DESEADO,
                                                   fs)
        rir_aug = np.sum(rir_band_augs, axis=0)
        rir_aug = np.concatenate((delay, early, rir_aug)).astype(np.float32)
        return rir_aug
    except Exception as err:
        raise TrAugmentationError(f'{err.__class__.__name__}: {err}')