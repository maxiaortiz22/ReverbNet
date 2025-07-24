"""
Design Butterworth octave‑band filters (2nd & 4th order) at 16 kHz sample rate
and emit C++ initialization code for a filter‑bank structure.

Overview
--------
This script:

1. Defines nominal octave‑band center frequencies (125 Hz … 8 kHz).
2. Computes band edges (±1/2 octave). The last band (8 kHz) is treated as a
   high‑pass to avoid exceeding Nyquist.
3. Designs Butterworth filters (order 2 and order 4) in SOS form for each band.
4. Normalizes each SOS cascade to unity gain at its band center frequency.
5. Converts SOS sections to a Python structure matching the target C++ biquad
   layout.
6. Writes a ready‑to‑paste C++ function (`initializeCoefficients()`) that loads
   the coefficients into a `pFilterBank` data structure.
7. Prints sample coefficients to stdout for quick verification.

Outputs
-------
A text file is written to:

``../code/cpp/filter/cpp_octave_coefficient_initialization.txt``

You can copy/paste its contents into the corresponding C++ source file.

Notes
-----
- Signal design uses ``scipy.signal.butter`` in SOS form for numerical stability.
- Gain is normalized only in the *first* SOS section to keep structure simple;
  downstream code should expect unity gain at the band center (within design tolerance).
"""

import numpy as np
from scipy import signal

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------

# Sample rate (Hz).
SAMPLE_RATE = 16000.0

# Nominal center frequencies (Hz) - octave bands.
NOMINAL_CENTER_FREQS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def compute_octave_band_edges(center_freqs):
    """
    Compute lower/upper band edges and filter type for octave bands.

    Parameters
    ----------
    center_freqs : sequence of float
        Nominal band center frequencies in Hz.

    Returns
    -------
    list of tuple
        Each entry is ``(f_c, f_lower, f_upper, filter_type)`` where
        ``filter_type`` is ``'bandpass'`` except for the last band, which is
        returned as ``'highpass'`` (upper edge is set to ``None``).
    """
    bands = []
    for i, f_c in enumerate(center_freqs):
        if i == len(center_freqs) - 1:  # Last band (8 kHz) -> high-pass.
            f_lower = f_c * (2 ** (-1/2))   # Lower edge for an octave.
            f_upper = None                  # High-pass: no upper edge.
            filter_type = 'highpass'
        else:  # Bandpass filters.
            f_lower = f_c * (2 ** (-1/2))   # Lower edge for an octave.
            f_upper = f_c * (2 ** (1/2))    # Upper edge for an octave.
            filter_type = 'bandpass'
        bands.append((f_c, f_lower, f_upper, filter_type))
    return bands


def design_cascaded_filters(low_freq, high_freq, order, fs, filter_type='bandpass'):
    """
    Design a Butterworth filter (SOS form) for the specified band.

    Parameters
    ----------
    low_freq : float
        Lower cutoff in Hz.
    high_freq : float or None
        Upper cutoff in Hz (ignored for ``highpass`` type).
    order : int
        Filter order (2 or 4 supported).
    fs : float
        Sampling rate in Hz.
    filter_type : {'bandpass', 'highpass'}, optional
        Filter response type.

    Returns
    -------
    sos : ndarray, shape (n_sections, 6)
        Second‑order sections representation suitable for stable filtering.
    """
    if filter_type == 'highpass':
        # High-pass filter.
        low = max(low_freq / (fs / 2), 1e-6)
        if order == 2:
            sos = signal.butter(2, low, btype='highpass', output='sos')
        else:  # order == 4
            sos = signal.butter(4, low, btype='highpass', output='sos')
    else:  # bandpass
        # Normalized frequencies with safety bounds.
        low = max(low_freq / (fs / 2), 1e-6)
        high = min(high_freq / (fs / 2), 0.99)

        if order == 2:
            sos = signal.butter(2, [low, high], btype='bandpass', output='sos')
        else:  # order == 4
            sos = signal.butter(4, [low, high], btype='bandpass', output='sos')

    return sos


def normalize_sos_gain(sos, center_freq, fs):
    """
    Normalize SOS cascade to unity gain at the band center frequency.

    This routine evaluates the complex frequency response of the cascaded
    sections at ``center_freq`` (on the unit circle) and scales the **first**
    section's numerator coefficients to achieve unity magnitude.

    Parameters
    ----------
    sos : ndarray
        SOS array as returned by :func:`scipy.signal.butter`.
    center_freq : float
        Band center frequency in Hz.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sos : ndarray
        Gain‑normalized SOS array (modified in place and returned).
    """
    w = 2 * np.pi * center_freq / fs

    # Calculate total gain through all sections.
    total_gain = 1.0
    for section in sos:
        b = section[0:3]
        a = section[3:6]

        # Evaluate transfer function at the center frequency.
        z = np.exp(1j * w)
        H_num = b[0] + b[1] * z**(-1) + b[2] * z**(-2)
        H_den = a[0] + a[1] * z**(-1) + a[2] * z**(-2)
        H = H_num / H_den
        total_gain *= abs(H)

    # Normalize the first section's numerator coefficients.
    if total_gain > 0:
        sos[0, 0:3] /= total_gain

    return sos


def sos_to_cascaded_biquads(sos):
    """
    Convert an SOS array to a Python structure matching the C++ biquad layout.

    Parameters
    ----------
    sos : ndarray
        Second‑order sections array.

    Returns
    -------
    list of dict
        List of ``{'b': array_like(3), 'a': array_like(3)}`` dictionaries.
        ``a0`` is normalized to 1.0 if necessary.
    """
    biquads = []
    for section in sos:
        biquad = {
            'b': section[0:3].copy(),  # b0, b1, b2
            'a': section[3:6].copy()   # a0, a1, a2 (a0 should be 1.0)
        }
        # Ensure a0 = 1.0.
        if biquad['a'][0] != 1.0 and biquad['a'][0] != 0.0:
            biquad['b'] /= biquad['a'][0]
            biquad['a'] /= biquad['a'][0]
        biquads.append(biquad)

    return biquads


# -------------------------------------------------------------------------
# Design all bands (executed at import time)
# -------------------------------------------------------------------------
bands = compute_octave_band_edges(NOMINAL_CENTER_FREQS)
NUM_BANDS = len(bands)

# Containers for designed filters.
filters_order_2 = []
filters_order_4 = []

print(f"Designing {NUM_BANDS} octave filters at {SAMPLE_RATE} Hz sample rate...")
print("Nyquist frequency:", SAMPLE_RATE/2, "Hz")
print()

for band_idx, (center_freq, low_freq, high_freq, filter_type) in enumerate(bands):
    if filter_type == 'highpass':
        print(f"Band {band_idx}: {center_freq} Hz (High-pass from {low_freq:.1f} Hz)")
    else:
        print(f"Band {band_idx}: {center_freq} Hz ({low_freq:.1f} - {high_freq:.1f} Hz)")

    # ---- Order 2 ----
    try:
        sos_2 = design_cascaded_filters(low_freq, high_freq, 2, SAMPLE_RATE, filter_type)
        sos_2_norm = normalize_sos_gain(sos_2.copy(), center_freq, SAMPLE_RATE)
        biquads_2 = sos_to_cascaded_biquads(sos_2_norm)
        filters_order_2.append(biquads_2)
    except Exception as e:
        print(f"Error designing order 2 filter for {center_freq} Hz: {e}")
        filters_order_2.append([])

    # ---- Order 4 ----
    try:
        sos_4 = design_cascaded_filters(low_freq, high_freq, 4, SAMPLE_RATE, filter_type)
        sos_4_norm = normalize_sos_gain(sos_4.copy(), center_freq, SAMPLE_RATE)
        biquads_4 = sos_to_cascaded_biquads(sos_4_norm)
        filters_order_4.append(biquads_4)
    except Exception as e:
        print(f"Error designing order 4 filter for {center_freq} Hz: {e}")
        filters_order_4.append([])


# -------------------------------------------------------------------------
# Emit C++ coefficient initialization code
# -------------------------------------------------------------------------
def generate_cpp_initialization():
    """
    Write a C++ function body that initializes a filter bank with the computed
    octave‑band coefficients.

    The output is written to:
    ``../code/cpp/cpp_octave_coefficient_initialization.txt``.
    """
    with open("../code/cpp/filter/cpp_octave_coefficient_initialization.txt", "w") as f:
        print("// Octave band filter bank initialization for 16 kHz sample rate", file=f)
        print("// Bands: 125, 250, 500, 1000, 2000, 4000, 8000 Hz", file=f)
        print("// Note: 8000 Hz band is high-pass filter", file=f)
        print("void initializeCoefficients() {", file=f)
        print("    // Center frequencies for octave bands", file=f)
        print("    const double center_frequencies[NUM_BANDS] = {", file=f)
        freq_str = ", ".join([f"{freq}" for freq in NOMINAL_CENTER_FREQS])
        print(f"        {freq_str}", file=f)
        print("    };", file=f)
        print("", file=f)
        print("    // Initialize all bands with their center frequencies", file=f)
        print("    for (int band = 0; band < NUM_BANDS; band++) {", file=f)
        print("        pFilterBank->bands[band].center_freq = center_frequencies[band];", file=f)
        print("    }", file=f)
        print("", file=f)

        # Order 2 coefficients.
        print("    if (pFilterBank->filter_order == 2) {", file=f)
        for band_idx, biquads in enumerate(filters_order_2):
            if not biquads:
                continue
            center_freq = bands[band_idx][0]
            filter_type = bands[band_idx][3]
            print(f"        // Band {band_idx}: {center_freq} Hz ({filter_type})", file=f)
            print(f"        pFilterBank->bands[{band_idx}].num_sections = {len(biquads)};", file=f)

            for sec_idx, biquad in enumerate(biquads):
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[0] = {biquad['b'][0]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[1] = {biquad['b'][1]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[2] = {biquad['b'][2]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[0] = {biquad['a'][0]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[1] = {biquad['a'][1]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[2] = {biquad['a'][2]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].z[0] = 0.0;", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].z[1] = 0.0;", file=f)
            print("", file=f)

        # Order 4 coefficients.
        print("    } else { // filter_order == 4", file=f)
        for band_idx, biquads in enumerate(filters_order_4):
            if not biquads:
                continue
            center_freq = bands[band_idx][0]
            filter_type = bands[band_idx][3]
            print(f"        // Band {band_idx}: {center_freq} Hz ({filter_type})", file=f)
            print(f"        pFilterBank->bands[{band_idx}].num_sections = {len(biquads)};", file=f)

            for sec_idx, biquad in enumerate(biquads):
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[0] = {biquad['b'][0]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[1] = {biquad['b'][1]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].b[2] = {biquad['b'][2]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[0] = {biquad['a'][0]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[1] = {biquad['a'][1]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].a[2] = {biquad['a'][2]:.12e};", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].z[0] = 0.0;", file=f)
                print(f"        pFilterBank->bands[{band_idx}].sections[{sec_idx}].z[1] = 0.0;", file=f)
            print("", file=f)

        print("    }", file=f)
        print("}", file=f)


generate_cpp_initialization()
print(f"\nC++ coefficient initialization code generated in 'cpp_octave_coefficient_initialization.txt'")
print("Copy and paste this into your C++ file to replace the initializeCoefficients() function.")

# -------------------------------------------------------------------------
# Verification / sample output
# -------------------------------------------------------------------------
print("\nSample coefficients for verification:")
print("Order 2, Band 0 (125 Hz - bandpass):")
if filters_order_2[0]:
    biquad = filters_order_2[0][0]
    print(f"  b: [{biquad['b'][0]:.6e}, {biquad['b'][1]:.6e}, {biquad['b'][2]:.6e}]")
    print(f"  a: [{biquad['a'][0]:.6e}, {biquad['a'][1]:.6e}, {biquad['a'][2]:.6e}]")

print("Order 2, Band 6 (8000 Hz - highpass):")
if filters_order_2[6]:
    biquad = filters_order_2[6][0]
    print(f"  b: [{biquad['b'][0]:.6e}, {biquad['b'][1]:.6e}, {biquad['b'][2]:.6e}]")
    print(f"  a: [{biquad['a'][0]:.6e}, {biquad['a'][1]:.6e}, {biquad['a'][2]:.6e}]")

print("Order 4, Band 3 (1000 Hz - bandpass):")
if filters_order_4[3]:
    for i, biquad in enumerate(filters_order_4[3]):
        print(f"  Section {i+1}:")
        print(f"    b: [{biquad['b'][0]:.6e}, {biquad['b'][1]:.6e}, {biquad['b'][2]:.6e}]")
        print(f"    a: [{biquad['a'][0]:.6e}, {biquad['a'][1]:.6e}, {biquad['a'][2]:.6e}]")

# Print band information for verification.
print("\nBand information:")
for i, (center_freq, low_freq, high_freq, filter_type) in enumerate(bands):
    if filter_type == 'highpass':
        print(f"Band {i}: {center_freq} Hz - High-pass from {low_freq:.1f} Hz")
    else:
        print(f"Band {i}: {center_freq} Hz - Bandpass {low_freq:.1f} - {high_freq:.1f} Hz")

print(f"\nNote: With Nyquist frequency at {SAMPLE_RATE/2} Hz, the 8000 Hz band uses a high-pass filter to avoid aliasing issues.")