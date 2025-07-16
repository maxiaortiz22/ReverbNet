import numpy as np

def get_DRR(rir, fs, window_length=0.0025):
    """Calcula la relación directo-reverberado (DRR) de una respuesta al impulso (RIR).
    
    Args:
        rir (numpy array): Respuesta al impulso.
        fs (float): Frecuencia de muestreo en Hz.
        window_length (float): Longitud de la ventana en segundos (default: 0.0025).
    
    Returns:
        tuple: (DRR en dB, parte temprana, parte tardía).
    """
    if len(rir) == 0:
        raise ValueError("La RIR no puede estar vacía")
    
    t_d = np.argmax(rir)  # Camino directo
    t_o = int(window_length * fs)  # Ventana en muestras
    init_idx = max(t_d - t_o, 0)
    final_idx = min(t_d + t_o + 1, len(rir))

    early = rir[init_idx:final_idx]
    late = rir[final_idx:]

    energia_late = np.sum(late**2)
    if energia_late == 0:
        raise ValueError("La energía de la parte tardía es cero")

    DRR = 10 * np.log10(np.sum(early**2) / energia_late)
    return DRR, early, late

def drr_aug(rir, fs, DRR_buscado, window_length=0.0025):
    """Genera una nueva RIR con un DRR deseado ajustando la parte temprana.
    
    Args:
        rir (numpy array): Respuesta al impulso original.
        fs (float): Frecuencia de muestreo en Hz.
        DRR_buscado (float): DRR deseado en dB.
        window_length (float): Longitud de la ventana en segundos (default: 0.0025).
    
    Returns:
        numpy array or None: Nueva RIR normalizada si se pudo generar, None si no.
    """
    if len(rir) == 0:
        raise ValueError("La RIR no puede estar vacía")

    DRR_original, early, late = get_DRR(rir, fs, window_length)
    delay = rir[:np.argmax(rir) - int(window_length * fs)]

    w = np.hamming(len(early))  # Ventana de Hamming
    energia_late = np.sum(late**2)
    if energia_late == 0:
        raise ValueError("La energía de la parte tardía es cero")

    # Cálculo de alpha
    a = np.sum((w**2) * (early**2))
    b = 2 * np.sum((1 - w) * w * (early**2))
    c = np.sum(((1 - w)**2) * (early**2)) - (10**(DRR_buscado / 10) * energia_late)
    alpha = bhaskara(a, b, c)

    # Si no hay alpha válido, no generamos el audio
    if alpha is None:
        print(f"No se pudo generar un audio con DRR = {DRR_buscado:.2f} dB. Omitiendo...")
        return None

    # Nueva parte temprana
    new_early = (alpha * w * early) + ((1 - w) * early)
    if np.max(np.abs(new_early)) < np.max(np.abs(late)):
        print("El nivel deseado es demasiado bajo, no se puede generar el audio.")
        return None

    # Construcción y normalización
    rir_aug = np.concatenate((delay, new_early, late)).astype(np.float32)
    DRR_obtenido = 10 * np.log10(np.sum(new_early**2) / energia_late)
    print(f"DRR buscado: {DRR_buscado:.2f}, DRR obtenido: {DRR_obtenido:.2f}")

    return rir_aug / np.max(np.abs(rir_aug))

def bhaskara(a, b, c):
    """Resuelve una ecuación cuadrática y devuelve la raíz mayor o None si no hay solución real.
    
    Args:
        a, b, c (float): Coeficientes de la ecuación ax^2 + bx + c = 0.
    
    Returns:
        float or None: Valor de alpha si hay solución, None si no la hay.
    """
    r = b**2 - 4 * a * c
    if r > 0:
        x1 = ((-b) + np.sqrt(r)) / (2 * a)
        x2 = ((-b) - np.sqrt(r)) / (2 * a)
        return max(x1, x2)
    elif r == 0:
        return -b / (2 * a)
    else:
        return None  # No hay solución real