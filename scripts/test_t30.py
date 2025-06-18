import sys
import os

# Añadir la ruta del proyecto a sys.path
sys.path.append('../code/parameters_calculation')

from tr_lundeby import tr_lundeby, NoiseError
import glob
from librosa import load

if __name__ == '__main__':
    files = glob.glob('../data/RIRs/*.wav')
    max_ruido_dB = -45
    fs = 16000
    order = 4


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