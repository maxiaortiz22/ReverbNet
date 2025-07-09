"""
tr_lundeby.py
-------------
Estimación del tiempo de reverberación (T30) usando el método de Lundeby
+ Schroeder, con optimizaciones opcionales vía Numba y varias protecciones
numéricas:

* Regresión lineal: fórmula cerrada con _fallback_ a `np.linalg.lstsq`
  cuando el denominador se acerca a cero (--> evita cancelación).
* Control de índices negativos / fuera de rango.
* Ventanas mínimas para que nunca haya divisiones por cero.
* Iteración de Lundeby con límite de 6 pasos y comprobación de convergencia.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import warnings

# --------------------------------------------------------------------- #
#   Dependencia opcional: NUMBA
# --------------------------------------------------------------------- #
USE_NUMBA = False
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:  # dummy decorators si Numba no está
    def njit(*args, **kwargs):  # type: ignore
        def wrap(func):
            return func
        return wrap
    def prange(x):  # type: ignore
        return range(x)

# --------------------------------------------------------------------- #
#   Constantes
# --------------------------------------------------------------------- #
EPS = np.finfo(np.float64).eps        # precisión máquina
MAX_ITERS = 6                         # iteraciones Lundeby
WIN_DB   = 10.0                       # ms de ventana para la envolvente
RNG_DYN  = 20.0                       # rango dinámico p/ regresión (dB)

# --------------------------------------------------------------------- #
#   Utilidades con y sin NUMBA
# --------------------------------------------------------------------- #
@njit(cache=True) if USE_NUMBA else lambda f: f
def _schroeder_integral(signal_sq: np.ndarray) -> np.ndarray:
    """Integral de Schroeder acumulada hacia atrás (vector 1-D)."""
    out = np.empty_like(signal_sq)
    acc: float = 0.0
    for i in range(len(signal_sq) - 1, -1, -1):
        acc += signal_sq[i]
        out[i] = acc
    return out


def _safe_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Ajuste lineal y = m·x + b en doble precisión.

    • Usa fórmula cerrada (O(n))  
    • _Fallback_ a `np.linalg.lstsq` cuando el denominador ≈ 0
      (--> estabilidad numérica)
    """
    n = x.size
    if n < 2:
        return 0.0, float(y.mean() if n else 0.0)

    sum_x  = float(x.sum(dtype=np.float64))
    sum_y  = float(y.sum(dtype=np.float64))
    sum_x2 = float(np.dot(x, x))
    sum_xy = float(np.dot(x, y))

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:                       # casi singular
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(b)

    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n
    return float(m), float(b)

# --------------------------------------------------------------------- #
#   Función principal
# --------------------------------------------------------------------- #
def tr_lundeby(
    rir: np.ndarray,
    fs: int,
    max_noise_dB: float = 45.0,
    tau_ms: float = WIN_DB,
    dyn_range_dB: float = RNG_DYN
) -> Tuple[float, int, float]:
    """
    Calcula T30 (segundos) mediante Lundeby.

    Parameters
    ----------
    rir : np.ndarray
        Impulso de respuesta (RIR) en mono.
    fs : int
        Frecuencia de muestreo (Hz).
    max_noise_dB : float, optional
        Diferencia máxima inicial entre señal y ruido (default: 45 dB).
    tau_ms : float, optional
        Longitud de la ventana de la envolvente en milisegundos.
    dyn_range_dB : float, optional
        Rango dinámico para la regresión (default: 20 dB).

    Returns
    -------
    t30 : float
        Tiempo de reverberación T30 [s].
    cross_idx : int
        Índice de muestra donde se trunca la RIR tras Lundeby.
    noise_floor_dB : float
        Nivel de piso de ruido estimado [dB].

    Raises
    ------
    ValueError
        Si no hay suficiente rango dinámico o la RIR es demasiado corta.
    """
    rir = rir.astype(np.float64, copy=False)
    if rir.size == 0:
        raise ValueError("RIR vacía.")

    # -------------------------------------------------------------- #
    # 1. Eliminar el delay (todo antes del pico directo)
    # -------------------------------------------------------------- #
    idx_direct = int(np.argmax(np.abs(rir)))
    rir_delay  = rir[:idx_direct]
    rir_nodelay = rir[idx_direct:]

    # -------------------------------------------------------------- #
    # 2. Envolvente cuadrática suavizada
    # -------------------------------------------------------------- #
    win_len = max(1, int((tau_ms / 1000.0) * fs))          # ≥ 1 muestra
    # Convolución equivalente a media móvil mediante `uniform_filter1d`
    from scipy.ndimage import uniform_filter1d
    env_sq  = rir_nodelay ** 2
    env_sm  = uniform_filter1d(env_sq, size=win_len, mode="nearest")

    env_dB  = 10.0 * np.log10(env_sm + EPS)

    # -------------------------------------------------------------- #
    # 3. Estimación inicial de piso de ruido (último 10 %)
    # -------------------------------------------------------------- #
    tail_len  = max(1, int(0.10 * env_sm.size))
    noise_lin = env_sm[-tail_len:].mean()
    noise_floor_dB = 10.0 * np.log10(noise_lin + EPS)

    # -------------------------------------------------------------- #
    # 4. Primera regresión para pendiente y punto de cruce
    # -------------------------------------------------------------- #
    init_val  = env_dB.max()
    final_val = noise_floor_dB + max_noise_dB

    mask = np.logical_and(env_dB < init_val, env_dB > final_val)
    if mask.sum() < 2:
        raise ValueError("Rango dinámico insuficiente para la regresión inicial.")

    x_valid = np.where(mask)[0].astype(np.float64)
    y_valid = env_dB[mask]

    m, b = _safe_linear_regression(x_valid, y_valid)
    cross_idx = int((noise_floor_dB - b) / m) if m != 0 else env_dB.size - 1
    cross_idx = np.clip(cross_idx, 0, env_dB.size - 1)

    # -------------------------------------------------------------- #
    # 5. Iteraciones de Lundeby
    # -------------------------------------------------------------- #
    for _ in range(MAX_ITERS):
        # Re-calcular piso de ruido usando DISTANCIA_AL_CRUCE = win_len
        start_noise = min(cross_idx + win_len, env_sm.size - 1)
        noise_lin = env_sm[start_noise:].mean()
        new_noise_dB = 10.0 * np.log10(noise_lin + EPS)

        init_val  = new_noise_dB + max_noise_dB
        final_val = init_val - dyn_range_dB
        mask = np.logical_and(env_dB < init_val, env_dB > final_val)

        if mask.sum() < 2:
            break

        x_valid = np.where(mask)[0].astype(np.float64)
        y_valid = env_dB[mask]
        m, b = _safe_linear_regression(x_valid, y_valid)
        if m == 0:
            break

        new_cross = int((new_noise_dB - b) / m)
        new_cross = np.clip(new_cross, 0, env_dB.size - 1)

        # Convergencia (≤ 1 muestra de diferencia)
        if abs(new_cross - cross_idx) <= 1:
            cross_idx = new_cross
            noise_floor_dB = new_noise_dB
            break

        cross_idx = new_cross
        noise_floor_dB = new_noise_dB

    # -------------------------------------------------------------- #
    # 6. Truncate RIR and compute Schroeder
    # -------------------------------------------------------------- #
    rir_cut = rir_nodelay[:cross_idx]
    if rir_cut.size < int(0.003 * fs):  # al menos 3 ms de cola
        raise ValueError("RIR demasiado corta tras el corte de Lundeby.")

    sch = _schroeder_integral(rir_cut ** 2)
    sch_dB = 10.0 * np.log10(sch / sch[0] + EPS)

    # -------------------------------------------------------------- #
    # 7. T30: regresión entre −5 dB y −35 dB
    # -------------------------------------------------------------- #
    try:
        idx_start = np.where(sch_dB <= -5.0)[0][0]
        idx_end   = np.where(sch_dB <= -35.0)[0][0]
    except IndexError as err:
        raise ValueError("No se encontró el rango −5 dB a −35 dB.") from err

    x_reg = np.arange(idx_start, idx_end + 1, dtype=np.float64)
    y_reg = sch_dB[idx_start:idx_end + 1]
    m_t30, b_t30 = _safe_linear_regression(x_reg, y_reg)
    if m_t30 == 0:
        raise ValueError("Pendiente nula en la regresión T30.")

    # T30 = −60 dB / pendiente  (muestras) → segundos
    t30 = (-60.0 / m_t30) / fs

    # Índice de cruce con delay compensado
    cross_idx_total = cross_idx + len(rir_delay)

    return float(t30), int(cross_idx_total), float(noise_floor_dB)