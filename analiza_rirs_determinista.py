import sys; sys.path.append('code')
import os
import numpy as np
import random
import pandas as pd
from datetime import datetime
from librosa import load
from scipy.signal import fftconvolve
from code.cpp import audio_processing
from code.parameters_calculation import (drr_aug, get_DRR, drr_aug,
                                         tr_augmentation, tr_lundeby)

# Seteo la semilla para que los resultados sean reproducibles
random.seed(1)
np.random.seed(1)

# === CONFIGURACIÓN ===
# Listas fijas de audios y RIRs (ajusta a tus necesidades)
AUDIO_DIR = 'data/Speech/train'
RIR_DIR = 'data/RIRs'

AUDIOS = [
    'F1s3.wav',
    'M1s3.wav',
    'F1s5.wav',
    'M1s5.wav',
    'F5s4.wav',
    'M8s4.wav',
    'F5s3.wav',
]
RIRS = [
    'classroom_00x05y.wav',
    'classroom_05x25y.wav',
    'classroom_25x40y.wav',
    'great_hall_x00y05.wav',
    'great_hall_x01y05.wav',
    'great_hall_x02y07.wav',
    'octagon_x04y05.wav',
    'octagon_x05y03.wav',
    'octagon_x05y05.wav',
    'sintetica_Seed11888550_Tr0.3.wav',
    'sintetica_Seed17721549_Tr2.7.wav',
    'sintetica_Seed30588681_Tr1.5.wav',
]

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
FILTER_TYPE = 'octave band'
FS = 16000
ORDER = 4
MAX_RUIDO_DB = -45
TR_AUGS = [0.2, 0.4, 1.0, 2.0, 3.0]  # Ejemplo, ajusta según quieras
DRR_AUGS = [-6, -3, 0, 6, 9, 15, 19] # Ejemplo, ajusta según quieras

# === FUNCIONES AUXILIARES ===
def safe_descriptor(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return np.nan

def process_rir(audio_path, rir_path, bpfilter, tipo, valor_aug=None):
    try:
        audio, _ = load(audio_path, sr=FS, duration=5.0)
        audio = audio / np.max(np.abs(audio))
        rir, _ = load(rir_path, sr=FS)
        rir = rir / np.max(np.abs(rir))
    except Exception as e:
        print(f"Error cargando {audio_path} o {rir_path}: {e}")
        return []

    results = []
    rir_to_use = rir
    if tipo == 'TR_aug' and valor_aug is not None:
        try:
            rir_to_use = tr_augmentation(rir, FS, valor_aug, bpfilter)
            rir_to_use = rir_to_use / np.max(np.abs(rir_to_use))
        except Exception as e:
            print(f"Error en TR_aug ({valor_aug}) para {rir_path}: {e}")
            rir_to_use = None
    elif tipo == 'DRR_aug' and valor_aug is not None:
        try:
            rir_to_use = drr_aug(rir, FS, valor_aug)
            rir_to_use = rir_to_use / np.max(np.abs(rir_to_use))
        except Exception as e:
            print(f"Error en DRR_aug ({valor_aug}) para {rir_path}: {e}")
            rir_to_use = None

    if rir_to_use is None:
        for band in BANDS:
            results.append({
                'audio': os.path.basename(audio_path),
                'rir': os.path.basename(rir_path),
                'tipo': tipo,
                'valor_aug': valor_aug,
                'band': band,
                't30': np.nan,
                'c50': np.nan,
                'c80': np.nan,
                'd50': np.nan,
                'drr': np.nan
            })
        return results

    filtered_rir = bpfilter.process(rir_to_use.astype(np.float32))
    for i, band in enumerate(BANDS):
        t30 = safe_descriptor(tr_lundeby, filtered_rir[i], FS, MAX_RUIDO_DB)
        c50 = safe_descriptor(audio_processing.ClarityCalculator.calculate, 50, filtered_rir[i], FS)
        c80 = safe_descriptor(audio_processing.ClarityCalculator.calculate, 80, filtered_rir[i], FS)
        d50 = safe_descriptor(audio_processing.DefinitionCalculator.calculate, filtered_rir[i], FS)
        drr = safe_descriptor(lambda x, y: get_DRR(x, y)[0], filtered_rir[i], FS)
        results.append({
            'audio': os.path.basename(audio_path),
            'rir': os.path.basename(rir_path),
            'tipo': tipo,
            'valor_aug': valor_aug,
            'band': band,
            't30': t30 if not isinstance(t30, tuple) else t30[0],
            'c50': c50,
            'c80': c80,
            'd50': d50,
            'drr': drr
        })
    return results

# === MAIN ===
def main():
    bpfilter = audio_processing.OctaveFilterBank(ORDER)
    all_results = []
    total = len(AUDIOS) * len(RIRS)
    count = 0
    for audio in AUDIOS:
        audio_path = os.path.join(AUDIO_DIR, audio)
        for rir in RIRS:
            rir_path = os.path.join(RIR_DIR, rir)
            is_sintetica = 'sintetica' in rir
            count += 1
            print(f"[{count}/{total}] Procesando: audio={audio}, rir={rir} (original)")
            all_results.extend(process_rir(audio_path, rir_path, bpfilter, 'original'))
            if not is_sintetica:
                for tr_aug in TR_AUGS:
                    print(f"[{count}/{total}] Procesando: audio={audio}, rir={rir} (TR_aug={tr_aug})")
                    all_results.extend(process_rir(audio_path, rir_path, bpfilter, 'TR_aug', tr_aug))
                for drr_aug in DRR_AUGS:
                    print(f"[{count}/{total}] Procesando: audio={audio}, rir={rir} (DRR_aug={drr_aug})")
                    all_results.extend(process_rir(audio_path, rir_path, bpfilter, 'DRR_aug', drr_aug))
    df = pd.DataFrame(all_results)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = f"analisis_rirs_{now}_nuevo.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n¡Listo! Resultados guardados en {out_csv}")

if __name__ == "__main__":
    import sys
    from contextlib import redirect_stdout

    log_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_nuevo.txt"
    with open(log_name, "w", encoding="utf-8") as f, redirect_stdout(f):
        main() 