import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)

from audio_processing import ClarityCalculator
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('../data/RIRs/*.wav')
    files = [audio for audio in files if 'sintetica' in audio]
    fs = 16000
    audios = 10  # Número de audios a procesar
    files = files[:audios]  # Limitar a los primeros 'audios' archivos
    

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        clarity_cpp = ClarityCalculator.calculate(50, data, fs)

        print(f'Clarity 50 CPP: {clarity_cpp}')

        clartity_80_cpp = ClarityCalculator.calculate(80, data, fs)

        print(f'Clarity 80 CPP: {clartity_80_cpp}')
