import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)
sys.path.append('../code/parameters_calculation')

import audio_processing
from definition import definition
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('*.wav')
    fs = 16000
    order = 4

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        definition_python = definition(data, fs)
        definition_cpp = audio_processing.DefinitionCalculator.calculate(data, fs)

        print(f'Definition Python: {definition_python}')
        print(f'Definition CPP: {definition_cpp}')
