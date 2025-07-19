"""
Quick interactive test for the C++ ``OctaveFilterBank`` extension.

This script performs a simple end‑to‑end sanity check:

1. Extends ``sys.path`` to include the compiled C++ extension (expected under
   ``code/cpp/build/Release`` relative to the project root).
2. Instantiates an ``OctaveFilterBank`` object.
3. Generates 1 second of white Gaussian noise at the sampling rate ``fs``.
4. Processes the noise through the filter bank.
5. For each band, plays back the filtered audio and plots its magnitude
   spectrum (single‑sided FFT).

Run this script from ./ReverbNet/scripts; it resolves the build directory relative to the
script location. Audio playback uses the default output device configured in
``sounddevice``.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Append the compiled C++ extension path to sys.path so Python can import it.
build_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "code", "cpp", "build", "Release"
)
sys.path.append(build_path)

from audio_processing import OctaveFilterBank


if __name__ == '__main__':
    filter_type = 'octave band'  # Descriptor for the filter design (not programmatically used below).
    fs = 16000
    order = 4
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]

    # Instantiate the C++ octave filter bank (order fixed here).
    filter_bank_cpp = OctaveFilterBank(filter_order=4)

    # Generate 1 second of white noise for testing.
    t = np.arange(0, 1, 1/fs)
    data = np.random.randn(len(t))

    # Filter the noise through the C++ filter bank; returns one signal per band.
    filtered_signals_cpp = filter_bank_cpp.process(data)

    # Plot frequency response of each filtered band and play it back.
    for i, band in enumerate(bands):
        # Play the filtered signal corresponding to this band.
        sd.play(filtered_signals_cpp[i], fs)
        sd.wait()

        # Compute magnitude spectrum (single‑sided) of the filtered signal.
        freq = np.fft.fftfreq(len(filtered_signals_cpp[i]), 1/fs)
        freq = freq[:len(freq)//2]
        freq_response = np.abs(np.fft.fft(filtered_signals_cpp[i]))
        freq_response = freq_response[:len(freq_response)//2]

        plt.plot(freq, freq_response, label=f'Band {band} Hz (C++)')
        plt.title(f'Band {band} Hz (C++)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()