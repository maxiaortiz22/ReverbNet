import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Añadir la ruta del proyecto a sys.path
# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)

from audio_processing import OctaveFilterBank

if __name__ == '__main__':
    filter_type = 'octave band'
    fs = 16000
    order = 4
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]

    filter_bank_cpp = OctaveFilterBank(filter_order=4)

    #Generate with noise for testing
    t = np.arange(0, 1, 1/fs)
    data = np.random.randn(len(t))

    filtered_signals_cpp = filter_bank_cpp.process(data)

    #Plot the frequency response
    for i, band in enumerate(bands):
        # Play the filtered data of the cpp filter  
        sd.play(filtered_signals_cpp[i], fs)
        sd.wait()

        # Plot the frequency response of the cpp filter
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
        