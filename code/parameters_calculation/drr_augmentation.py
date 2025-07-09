"""
drr_augmentation.py
-------------------
• Cálculo de la Relación Directo-Reverberante (get_DRR)
• Aumentación del DRR (drr_aug) mediante solución analítica y cross-fading

Cambios clave:
1. Fallback robusto si `numba` no está disponible (sirve tanto para
   usarlo como `@njit` o `@njit(cache=True)`).
2. Pico directo localizado con `np.argmax(np.abs(rir))`.
3. Si α no es válido, se conserva la parte early original (igual que en
   el script antiguo) para no descartar muestras.
4. Ventana **Hamming simétrica** (`sym=True`) como en el código original.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.signal.windows import hamming  # ventana idéntica a np.hamming

# ------------------------------------------------------------------ #
#   NUMBA opcional
# ------------------------------------------------------------------ #
try:
    from numba import njit
except ImportError:                              # fallback sin Numba
    def njit(*args, **kwargs):                   # type: ignore
        # Uso @njit
        if args and callable(args[0]):
            return args[0]
        # Uso @njit(...)
        def decorator(func):
            return func
        return decorator

# ------------------------------------------------------------------ #
#   Bhaskara rápido (con opcional JIT)
# ------------------------------------------------------------------ #
@njit(cache=True)
def bhaskara_fast(a: float, b: float, c: float) -> float:
    """
    Devuelve la raíz positiva más grande de a·x² + b·x + c = 0,
    o −1 si no existe solución real/positiva o a == 0.
    """
    disc = b * b - 4.0 * a * c
    if a == 0.0 or disc < 0.0:
        return -1.0
    sqrt_disc = np.sqrt(disc)
    x1 = (-b + sqrt_disc) / (2.0 * a)
    x2 = (-b - sqrt_disc) / (2.0 * a)
    return max(x1, x2)

# ------------------------------------------------------------------ #
#   DRR BASE
# ------------------------------------------------------------------ #
def get_DRR(
    rir: np.ndarray,
    fs: int,
    window_length: float = 0.0025
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calcula la Relación Directo-Reverberante (DRR) de una RIR.

    Parameters
    ----------
    rir : np.ndarray
        Impulso de respuesta mono.
    fs : int
        Frecuencia de muestreo (Hz).
    window_length : float, optional
        Semiventana (s) para la parte early (default 2.5 ms).

    Returns
    -------
    DRR : float
        Relación en dB.
    early : np.ndarray
        Segmento temprano.
    late : np.ndarray
        Segmento tardío.

    Raises
    ------
    ValueError
        Si la energía tardía es cero.
    """
    t_d = int(np.argmax(np.abs(rir)))            # pico absoluto
    t_o = int(window_length * fs)
    init_idx = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    early = rir[init_idx:final_idx]
    late  = rir[final_idx:]

    early_energy = np.sum(early ** 2)
    late_energy  = np.sum(late ** 2)
    if late_energy == 0.0:
        raise ValueError("Energía 'late' nula; no puede calcularse DRR.")

    DRR = 10.0 * np.log10(early_energy / late_energy)
    return DRR, early, late

# ------------------------------------------------------------------ #
#   AUMENTACIÓN DE DRR
# ------------------------------------------------------------------ #
def drr_aug(
    rir: np.ndarray,
    fs: int,
    DRR_delta_dB: float,
    window_length: float = 0.0025,
    verbose: bool = False
) -> np.ndarray:
    """
    Genera una nueva RIR cuyo DRR se incrementa en `DRR_delta_dB`.

    Si la ecuación cuadrática no arroja α válido, se conserva la parte
    early original (no se descarta la muestra, comportamiento del código
    antiguo).

    Parameters
    ----------
    rir : np.ndarray
        Respuesta al impulso mono.
    fs : int
        Frecuencia de muestreo (Hz).
    DRR_delta_dB : float
        Incremento deseado en dB (p.ej. +3.0).
    window_length : float, optional
        Semiventana (s) para la parte early (default 2.5 ms).
    verbose : bool, optional
        Imprime mensajes de depuración.

    Returns
    -------
    np.ndarray
        RIR con DRR modificado (dtype original).
    """
    t_d = int(np.argmax(np.abs(rir)))
    t_o = int(window_length * fs)

    init_idx  = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    delay = rir[:init_idx]
    early = rir[init_idx:final_idx]
    late  = rir[final_idx:]

    # Ventana Hamming simétrica (como np.hamming)
    w = hamming((t_o * 2) + 1, sym=True)
    w = w[: len(early)]                         # por si early se trunca

    # Coeficientes para Bhaskara
    early_sq    = early ** 2
    w_sq        = w ** 2
    one_m_w     = 1.0 - w
    one_m_w_sq  = one_m_w ** 2

    a = np.sum(w_sq * early_sq)
    b = 2.0 * np.sum(one_m_w * w * early_sq)
    late_energy = np.sum(late ** 2)
    target_lin  = 10.0 ** (DRR_delta_dB / 10.0)
    c = np.sum(one_m_w_sq * early_sq) - (target_lin * late_energy)

    alpha = bhaskara_fast(a, b, c)

    # --- fallback si α no es válido -----------------------------------
    if alpha <= 0.0 or np.isnan(alpha) or np.isinf(alpha):
        if verbose:
            print(f"[drr_aug] α inválido ({alpha}); se mantiene early original")
        new_early = early
    else:
        new_early = early * (alpha * w + one_m_w)
        # Protección del código antiguo: evitar que new_early quede por
        # debajo del pico tardío
        if np.max(np.abs(new_early)) < np.max(np.abs(late)):
            if verbose:
                print("[drr_aug] α válido pero new_early < late; usando early original")
            new_early = early

    rir_aug = np.concatenate((delay, new_early, late))

    # Normalización global
    max_val = np.max(np.abs(rir_aug))
    if max_val > 0.0:
        rir_aug = rir_aug / max_val

    return rir_aug.astype(rir.dtype)

# ------------------------------------------------------------------ #
#   Pequeña prueba manual
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    rir_test = np.exp(-t / 0.2)
    rir_test[int(0.15 * fs):] += 0.001 * np.random.randn(len(rir_test) - int(0.15 * fs))

    print("DRR original:", get_DRR(rir_test, fs)[0], "dB")
    rir_out = drr_aug(rir_test, fs, DRR_delta_dB=6.0, verbose=True)
    print("DRR tras augment:", get_DRR(rir_out, fs)[0], "dB")
