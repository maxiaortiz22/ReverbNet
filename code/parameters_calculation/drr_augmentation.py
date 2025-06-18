import numpy as np
from numba import njit


def get_DRR(rir, fs, window_length=0.0025):
    """Calcula la Relación Directo-Reverberado (DRR) de una respuesta al impulso."""
    t_d = np.argmax(rir)                                                   
    t_o = int(window_length * fs)                                          
    init_idx = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    early = rir[init_idx:final_idx]
    late = rir[final_idx:]

    early_energy = np.sum(early ** 2)
    late_energy = np.sum(late ** 2)

    DRR = 10 * np.log10(early_energy / late_energy) if late_energy > 0 else np.inf
    return DRR, early, late


def drr_aug(rir, fs, DRR_buscado, window_length=0.0025, verbose=False):
    """Genera una nueva respuesta al impulso modificando su DRR."""
    t_d = np.argmax(rir)                                                   
    t_o = int(window_length * fs)                                          
    init_idx = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    delay = rir[:init_idx]
    early = rir[init_idx:final_idx]
    late = rir[final_idx:]

    # Ventana de Hamming
    w = np.hamming((t_o * 2) + 1)

    early_squared = early ** 2
    w_sq = w ** 2
    one_minus_w = 1 - w
    one_minus_w_sq = one_minus_w ** 2

    a = np.sum(w_sq * early_squared)
    b = 2 * np.sum(one_minus_w * w * early_squared)
    late_energy = np.sum(late ** 2)
    target_ratio = 10 ** (DRR_buscado / 10)
    c = np.sum(one_minus_w_sq * early_squared) - (target_ratio * late_energy)

    alpha = bhaskara_fast(a, b, c)
    if alpha < 0 or np.isnan(alpha) or np.isinf(alpha):
        if verbose:
            print("Alpha inválido. Usando early sin modificar.")
        rir_aug = np.concatenate((delay, early, late)).astype(np.float32)
    else:
        new_early = early * (alpha * w + (1 - w))
        if np.max(np.abs(new_early)) < np.max(np.abs(late)):
            if verbose:
                print("El nivel deseado es demasiado bajo. Usando early sin modificar.")
            new_early = early
        rir_aug = np.concatenate((delay, new_early, late)).astype(np.float32)

    # Normalización final
    max_val = np.max(np.abs(rir_aug))
    if max_val > 0:
        rir_aug /= max_val

    return rir_aug


@njit
def bhaskara_fast(a, b, c):
    """Resuelve la fórmula cuadrática y devuelve la raíz positiva más grande."""
    r = b**2 - 4 * a * c
    if r < 0 or a == 0.0:
        return -1.0
    sqrt_r = np.sqrt(r)
    x1 = (-b + sqrt_r) / (2 * a)
    x2 = (-b - sqrt_r) / (2 * a)
    return max(x1, x2)
