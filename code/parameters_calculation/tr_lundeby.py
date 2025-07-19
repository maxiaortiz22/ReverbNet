"""
Lundeby-based reverberation time utilities and supporting helpers.

This module implements several routines commonly used in room acoustics
post‑processing:

* :class:`NoiseError` – custom exception raised when the signal‑to‑noise ratio
  is insufficient for a reliable Lundeby analysis.
* :func:`leastsquares` – simple linear least‑squares fit returning slope,
  intercept, and fitted line.
* :func:`schroeder` – Schroeder backward integration with Lundeby compensation.
* :func:`tr_convencional` – Conventional reverberation‑time estimator
  (T30/T20/T10/EDT) from an impulse response.
* :func:`lundeby` – Determine the upper integration limit and noise level used
  for Lundeby compensation.
* :func:`tr_lundeby` – Compute T30 using Lundeby‑corrected Schroeder integration.

**Important:** Variable names are preserved in Spanish to maintain compatibility
with downstream code (e.g., ``ruido_dB`` for noise level in dB). Logic and
interfaces remain unchanged.
"""

import numpy as np
import sys
import math
from scipy import stats


class NoiseError(Exception):
    """Raised when the S/N ratio is too low to perform Lundeby processing."""

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        # Debug message when stringifying the exception.
        print('calling str')
        if self.message:
            return f'NoiseError: {self.message} '
        else:
            return f'NoiseError has been raised: {self.message}'


def leastsquares(x, y):
    """
    Perform a simple linear least‑squares regression.

    Given two equal‑length vectors ``x`` and ``y``, fit ``y2 = c + m * x`` and
    return the slope, intercept, and fitted values.

    Documentation for the NumPy routine used internally:
    https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.linalg.lstsq.html

    Parameters
    ----------
    x, y : array_like
        Input vectors of equal length.

    Returns
    -------
    m : float
        Slope.
    c : float
        Intercept.
    y2 : ndarray
        Fitted line values.
    """
    # Rewriting the line equation as y = A p, where A = [[x 1]] and p = [[m], [c]].
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=-1)[0]  # Finding coefficients m and c.
    y2 = m * x + c  # Fitted line.
    return m, c, y2


def schroeder(ir, t, C):
    """
    Smooth a curve using Schroeder integration.

    Parameters
    ----------
    ir : array_like
        Input energy (or squared IR) vector.
    t : int
        Upper integration limit (sample index) determined by Lundeby.
    C : float
        Lundeby compensation constant.

    Returns
    -------
    y : ndarray
        Schroeder integrated (smoothed) curve, normalized by total energy + C.
    """
    ir = ir[0:int(t)]
    y = np.flip((np.cumsum(np.flip(ir)) + C) / (np.sum(ir) + C))
    return y


def tr_convencional(raw_signal, fs, rt='t30'):
    """
    Estimate reverberation time from an impulse response (conventional method).

    Parameters
    ----------
    raw_signal : ndarray
        Impulse response (time‑domain).
    fs : int or float
        Sampling rate (Hz).
    rt : {'t30', 't20', 't10', 'edt'}, default='t30'
        Desired estimator window. The returned value is extrapolated to T60
        using the standard scaling factors.

    Returns
    -------
    t60 : float
        Estimated T60 (seconds) using the selected decay range.

    Notes
    -----
    * The IR is windowed starting at its maximum absolute value.
    * Schroeder integration is computed on |IR|^2.
    * Linear regression is performed over the decay range specified by ``rt``.
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    # Window the signal starting at its maximum.
    in_max = np.where(np.abs(raw_signal) == np.max(np.abs(raw_signal)))[0]  # Windows signal from its maximum onwards.
    in_max = int(in_max[0])
    raw_signal = raw_signal[(in_max):]
    
    abs_signal = np.abs(raw_signal) / np.max(np.abs(raw_signal))

    # Schroeder integration.
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(np.abs(sch)) + sys.float_info.epsilon)

    # Linear regression over the selected decay range.
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT) extrapolated to T60.
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)

    return t60


def lundeby(y_power, Fs, Ts, max_ruido_dB):
    """
    Perform Lundeby analysis to determine integration limit and noise level.

    Parameters
    ----------
    y_power : ndarray
        Squared (energy) impulse response.
    Fs : int or float
        Sampling rate (Hz).
    Ts : float
        Window length (s) for initial averaging; Lundeby recommends 10–50 ms.
    max_ruido_dB : float
        Minimum required S/N in dB (threshold). If the measured S/N is worse,
        :class:`NoiseError` is raised.

    Returns
    -------
    punto : float
        Upper integration limit (sample index) for Schroeder integration.
    C : float
        Lundeby compensation factor.
    ruido_dB : float
        Estimated noise level relative to peak (dB).

    Raises
    ------
    NoiseError
        If the measured S/N ratio is insufficient.
    ValueError
        If no interval at least 10 dB above noise exists.
    """
    y_promedio = np.zeros(int(len(y_power) / Fs / Ts))
    eje_tiempo = np.zeros(int(len(y_power) / Fs / Ts))

    t = math.floor(len(y_power) / Fs / Ts)
    v = math.floor(len(y_power) / t)

    for i in range(0, t):
        y_promedio[i] = np.sum(y_power[i * v:(i + 1) * v]) / v
        eje_tiempo[i] = math.ceil(v / 2) + (i * v)

    # First estimate of the noise level determined from the energy present in the last 10% of input signal.
    ruido_dB = 10 * np.log10(
        np.sum(y_power[round(0.9 * len(y_power)):len(y_power)]) / (0.1 * len(y_power)) / np.max(y_power)
                + sys.float_info.epsilon )
    
    #ruido_dB2 = 10*np.log10(np.mean(y_power[-int(y_power.size/10):]))
    #print(f'ruido_dB: {ruido_dB}')
    #print(f'ruido_dB: {ruido_dB2}')
    y_promediodB = 10 * np.log10(y_promedio / np.max(y_power) + sys.float_info.epsilon)

    if ruido_dB > max_ruido_dB:  # Insufficient S/N ratio to perform Lundeby.
        raise NoiseError(f'Insufficient S/N ratio to perform Lundeby. Need at least {max_ruido_dB} dB')
        #raise ValueError(f'Insufficient S/N ratio to perform Lundeby. Need at least {max_ruido_dB} dB')

    # Decay slope estimated from linear regression between the interval containing the maximum (0 dB)
    # and the first interval 10 dB above the initial noise level.
    r = int(np.max(np.argwhere(y_promediodB > ruido_dB + 10)))
    #print(f'r: {r}')

    if r <= 0:
        raise ValueError('No hay valor de la señal que esté 10 dB por encima del ruido')

    m, c, rectacuadmin = leastsquares(eje_tiempo[0:r], y_promediodB[0:r])
    cruce = (ruido_dB - c) / m
    #print(f'cruce: {cruce}')

    # Begin Lundeby's iterations.
    error = 1
    INTMAX = 25
    veces = 1
    while error > 0.0001 and veces <= INTMAX:

        # Calculate new time intervals for median, with approx. p-steps per 10 dB.
        p = 10  # Number of steps every 10 dB.
        delta = np.abs(10 / m)  # Number of samples for the 10 dB decay slope.
        #print(f'delta: {delta}')
        v = math.floor(delta / p)  # Median calculation window.
        if (cruce - delta) > len(y_power):
            t = math.floor(len(y_power) / v)
        else:
            t = math.floor(len(y_power[0:round(cruce - delta)]) / v)
        if t < 2:
            t = 2

        media = np.zeros(t)
        eje_tiempo = np.zeros(t)
        for i in range(0, t):
            media[i] = np.sum(y_power[i * v:(i + 1) * v]) / len(y_power[i * v:(i + 1) * v])
            eje_tiempo[i] = math.ceil(v / 2) + (i * v)
        mediadB = 10 * np.log10(media / np.max(y_power) + sys.float_info.epsilon)
        m, c, rectacuadmin = leastsquares(eje_tiempo, mediadB)

        # New median of the noise energy calculated, starting from the point of the decay line 10 dB under the cross-point.
        noise = y_power[(round(abs(cruce + delta))):]
        if len(noise) < round(0.1 * len(y_power)):
            noise = y_power[round(0.9 * len(y_power)):]
        rms_dB = 10 * np.log10(sum(noise) / len(noise) / np.max(y_power) + sys.float_info.epsilon)
        #print(f'rms_dB: {rms_dB}')

        # New cross-point.
        error = np.abs(cruce - (rms_dB - c) / m) / cruce
        cruce = np.round((rms_dB - c) / m)
        veces += 1
    # Output.
    if cruce > len(y_power):
        punto = len(y_power)
    else:
        punto = cruce
    C = np.max(y_power) * 10 ** (c / 10) * np.exp(m / 10 / np.log10(np.exp(1)) * cruce) / (
                -m / 10 / np.log10(math.exp(1)))
    
    return punto, C, ruido_dB


def tr_lundeby(y, fs, max_ruido_dB):
    """
    Compute T30 using Lundeby‑corrected Schroeder integration.

    Parameters
    ----------
    y : ndarray
        Impulse response (time‑domain).
    fs : int or float
        Sampling rate (Hz).
    max_ruido_dB : float
        Minimum allowable S/N (dB) for Lundeby processing.

    Returns
    -------
    T30 : float
        Estimated T30 (s) extrapolated to T60 (‑60 dB) using the fitted slope.
    sch : ndarray
        Schroeder decay curve (dB) after Lundeby compensation.
    ruido_dB : float
        Estimated noise level (dB) relative to peak.
    """
    # Normalize and square the signal.
    y = y / np.max(np.abs(y))
    y **= 2

    # Window from the maximum onward.
    in_max = np.where(abs(y) == np.max(abs(y)))[0]  # Windows signal from its maximum onwards.
    in_max = int(in_max[0])
    y = y[in_max:]

    # Find Lundeby limits.
    t, C, ruido_dB = lundeby(y, fs, 0.05, max_ruido_dB)

    # Schroeder integration (dB).
    sch = schroeder(y, t, C)
    sch = 10 * np.log10(sch / np.max(np.abs(sch)) + sys.float_info.epsilon)

    # Compute T30.
    t = np.arange(0, len(sch) / fs, 1 / fs)

    i_max = np.where(sch == np.max(sch))  # Finds maximum of input vector.
    sch = sch[int(i_max[0][0]):]
    
    i_30 = np.where((sch <= np.max(sch) - 5) & (sch > (np.max(sch) - 35)))  # Index of values between -5 and -35 dB.
    t_30 = t[i_30]
    y_t30 = sch[i_30]
    m_t30, c_t30, f_t30 = leastsquares(t_30, y_t30)  # Find slope/intercept/line for T30 segment.
                              
    T30 = -60 / m_t30  # T30 calculation.
    
    return T30, sch, ruido_dB