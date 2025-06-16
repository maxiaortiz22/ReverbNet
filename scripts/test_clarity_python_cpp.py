import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)
sys.path.append('../code/parameters_calculation')

import audio_processing
from clarity import clarity
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('*.wav')
    fs = 16000
    order = 4

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        clarity_python = clarity(50, data, fs)
        clarity_cpp = audio_processing.ClarityCalculator.calculate(50, data, fs)

        print(f'Clarity 50 Python: {clarity_python}')
        print(f'Clarity 50 CPP: {clarity_cpp}')

        clartity_80_python = clarity(80, data, fs)
        clartity_80_cpp = audio_processing.ClarityCalculator.calculate(80, data, fs)

        print(f'Clarity 80 Python: {clartity_80_python}')
        print(f'Clarity 80 CPP: {clartity_80_cpp}')
