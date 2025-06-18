import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)

from audio_processing import DefinitionCalculator
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('../data/RIRs/*.wav')
    fs = 16000
    order = 4

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        definition_cpp = DefinitionCalculator.calculate(data, fs)

        print(f'Definition CPP: {definition_cpp}')
