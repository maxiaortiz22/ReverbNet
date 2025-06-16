import numpy as np
from scipy import signal

# Sample rate (Hz)
SAMPLE_RATE = 16000.0

# Nominal center frequencies (Hz) - octave bands
NOMINAL_CENTER_FREQS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]

def compute_octave_band_edges(center_freqs):
    """Compute band edges for octave bands"""
    bands = []
    for i, f_c in enumerate(center_freqs):
        if i == len(center_freqs) - 1:  # Last band (8000 Hz) - high-pass
            f_lower = f_c * (2 ** (-1/2))  # Lower bandedge for octave
            f_upper = None  # High-pass filter
            filter_type = 'highpass'
        else:  # Bandpass filters
            f_lower = f_c * (2 ** (-1/2))  # Lower bandedge for octave
            f_upper = f_c * (2 ** (1/2))   # Upper bandedge for octave
            filter_type = 'bandpass'
        bands.append((f_c, f_lower, f_upper, filter_type))
    return bands

def design_cascaded_filters(low_freq, high_freq, order, fs, filter_type='bandpass'):
    """Design cascaded biquad filters for better numerical stability"""
    if filter_type == 'highpass':
        # High-pass filter
        low = max(low_freq / (fs / 2), 1e-6)
        if order == 2:
            sos = signal.butter(2, low, btype='highpass', output='sos')
        else:  # order == 4
            sos = signal.butter(4, low, btype='highpass', output='sos')
    else:  # bandpass
        # Normalized frequencies with safety bounds
        low = max(low_freq / (fs / 2), 1e-6)
        high = min(high_freq / (fs / 2), 0.99)
        
        if order == 2:
            sos = signal.butter(2, [low, high], btype='bandpass', output='sos')
        else:  # order == 4
            sos = signal.butter(4, [low, high], btype='bandpass', output='sos')
    
    return sos

def normalize_sos_gain(sos, center_freq, fs):
    """Normalize SOS filter to have unity gain at center frequency"""
    w = 2 * np.pi * center_freq / fs
    
    # Calculate total gain through all sections
    total_gain = 1.0
    for section in sos:
        b = section[0:3]
        a = section[3:6]
        
        # Evaluate transfer function at center frequency
        z = np.exp(1j * w)
        H_num = b[0] + b[1] * z**(-1) + b[2] * z**(-2)
        H_den = a[0] + a[1] * z**(-1) + a[2] * z**(-2)
        H = H_num / H_den
        total_gain *= abs(H)
    
    # Normalize first section's b coefficients
    if total_gain > 0:
        sos[0, 0:3] /= total_gain
    
    return sos

def sos_to_cascaded_biquads(sos):
    """Convert SOS to format suitable for C++ implementation"""
    biquads = []
    for section in sos:
        biquad = {
            'b': section[0:3].copy(),  # b0, b1, b2
            'a': section[3:6].copy()   # a0, a1, a2 (a0 should be 1.0)
        }
        # Ensure a0 = 1.0
        if biquad['a'][0] != 1.0 and biquad['a'][0] != 0.0:
            biquad['b'] /= biquad['a'][0]
            biquad['a'] /= biquad['a'][0]
        biquads.append(biquad)
    
    return biquads

# Compute band edges
bands = compute_octave_band_edges(NOMINAL_CENTER_FREQS)
NUM_BANDS = len(bands)

# Calculate coefficients
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
    
    # Order 2
    try:
        sos_2 = design_cascaded_filters(low_freq, high_freq, 2, SAMPLE_RATE, filter_type)
        sos_2_norm = normalize_sos_gain(sos_2.copy(), center_freq, SAMPLE_RATE)
        biquads_2 = sos_to_cascaded_biquads(sos_2_norm)
        filters_order_2.append(biquads_2)
    except Exception as e:
        print(f"Error designing order 2 filter for {center_freq} Hz: {e}")
        filters_order_2.append([])
    
    # Order 4
    try:
        sos_4 = design_cascaded_filters(low_freq, high_freq, 4, SAMPLE_RATE, filter_type)
        sos_4_norm = normalize_sos_gain(sos_4.copy(), center_freq, SAMPLE_RATE)
        biquads_4 = sos_to_cascaded_biquads(sos_4_norm)
        filters_order_4.append(biquads_4)
    except Exception as e:
        print(f"Error designing order 4 filter for {center_freq} Hz: {e}")
        filters_order_4.append([])

# Generate C++ coefficient initialization code
def generate_cpp_initialization():
    with open("cpp_octave_coefficient_initialization.txt", "w") as f:
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
        
        # Order 2 coefficients
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
        
        # Order 4 coefficients
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

# Verification: Print some sample coefficients
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

# Print band information for verification
print("\nBand information:")
for i, (center_freq, low_freq, high_freq, filter_type) in enumerate(bands):
    if filter_type == 'highpass':
        print(f"Band {i}: {center_freq} Hz - High-pass from {low_freq:.1f} Hz")
    else:
        print(f"Band {i}: {center_freq} Hz - Bandpass {low_freq:.1f} - {high_freq:.1f} Hz")
        
print(f"\nNote: With Nyquist frequency at {SAMPLE_RATE/2} Hz, the 8000 Hz band uses a high-pass filter to avoid aliasing issues.")