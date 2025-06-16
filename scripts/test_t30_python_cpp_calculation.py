import sys
import os

# Añadir la ruta del proyecto a sys.path
build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cpp", "build", "Release")
sys.path.append(build_path)
sys.path.append('../code/parameters_calculation')

import audio_processing
from tr_lundeby import tr_lundeby, NoiseError
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('*.wav')
    max_ruido_dB = -45
    fs = 16000
    order = 4

    # Create an instance of TRLundebyCalculator
    lundeby_calc = audio_processing.TRLundebyCalculator()

    for file in files:
        data, fs = load(file, sr=fs)
        print(file)

        try:
            # Python implementation
            t30_python, _, ruidodB_python = tr_lundeby(data, fs, max_ruido_dB)
            
            print(f'T30 Python: {t30_python:.3f}')
            print(f'ruidodB Python: {ruidodB_python:.3f}')
            print('-' * 50)
            
        except (ValueError, NoiseError) as err:
            print(f"Error processing {file}: {err}")
            continue