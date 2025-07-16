import os
import random
from datetime import datetime
from math import nan
from pathlib import Path

import numpy as np
import pandas as pd
from librosa import load
from scipy.signal import butter, fftconvolve

from .cpp import audio_processing
from .parameters_calculation import (TAE, NoiseError,
                                    TrAugmentationError,
                                    drr_aug, get_DRR, pink_noise,
                                    tr_augmentation, tr_lundeby)


class DataBaseGenerator:
    """
    Clase refactorizada para generar una base de datos de descriptores acústicos
    a partir de señales de voz y respuestas al impulso (RIRs).
    """

    # Usar __slots__ es bueno para la memoria, ¡bien hecho!
    __slots__ = (
        'speech_files_train', 'speech_files_test', 'rir_files', 'to_augmentate',
        'rirs_for_training', 'rirs_for_testing', 'bands', 'filter_type', 'fs',
        'max_ruido_dB', 'order', 'add_noise', 'snr', 'tr_aug_params', 'drr_aug_params',
        'db_name', 'sos_lowpass_filter', 'tr_variations', 'drr_variations', 'bp_filter',
        'data_path', 'cache_path'
    )

    def __init__(self, speech_files_train, speech_files_test, rir_files, to_augmentate,
                 rirs_for_training, rirs_for_testing, bands, filter_type, fs,
                 max_ruido_dB, order, add_noise, snr, tr_aug_params, drr_aug_params):

        # --- Asignación de parámetros ---
        self.speech_files_train = speech_files_train
        self.speech_files_test = speech_files_test
        self.rir_files = rir_files
        self.to_augmentate = to_augmentate
        self.rirs_for_training = rirs_for_training
        self.rirs_for_testing = rirs_for_testing
        self.bands = bands
        self.filter_type = filter_type
        self.fs = fs
        self.max_ruido_dB = max_ruido_dB
        self.order = order
        self.add_noise = add_noise
        self.snr = snr
        self.tr_aug_params = tr_aug_params
        self.drr_aug_params = drr_aug_params
        
        # --- Configuración de rutas (más robusto con pathlib) ---
        self.data_path = Path('data')
        self.cache_path = Path('cache')
        self.cache_path.mkdir(exist_ok=True) # Asegura que el directorio cache exista

        # --- Nombre de la base de datos ---
        self.db_name = (
            f'base_de_datos_{self.max_ruido_dB}_noise_{self.add_noise}_'
            f'traug_{"_".join(map(str, self.tr_aug_params))}_'
            f'drraug_{"_".join(map(str, self.drr_aug_params))}_'
            f'snr_{self.snr[0]}_{self.snr[-1]}'
        )

        # --- Inicialización de filtros y variaciones (sin cambios, estaba bien) ---
        cutoff = 20  # Hz
        self.sos_lowpass_filter = butter(self.order, cutoff, fs=self.fs, btype='lowpass', output='sos')
        self.tr_variations = np.arange(self.tr_aug_params[0], self.tr_aug_params[1], self.tr_aug_params[2])
        self.drr_variations = np.arange(self.drr_aug_params[0], self.drr_aug_params[1], self.drr_aug_params[2])
        self.bp_filter = None

    # ----------  protocolo de pickling ----------
    def __getstate__(self):
        # copia todos los atributos salvo bp_filter
        return {slot: getattr(self, slot) 
                for slot in self.__slots__ if slot != 'bp_filter'}

    def __setstate__(self, state):
        for slot in self.__slots__:
            if slot == 'bp_filter':
                continue
            setattr(self, slot, state.get(slot, None))
        # el filtro se recreará perezosamente
        self.bp_filter = None
    # ---------------------------

    def _ensure_filter(self):
        if self.bp_filter is None:
            self.bp_filter = audio_processing.OctaveFilterBank(self.order)
    
    def _load_and_normalize_audio(self, file_path: Path, duration: float = 5.0) -> np.ndarray:
        """Carga un archivo de audio y lo normaliza."""
        audio_data, _ = load(file_path, sr=self.fs, duration=duration)
        max_val = np.max(np.abs(audio_data))
        return audio_data / max_val if max_val > 0 else audio_data

    def _calculate_descriptors(self, rir_band: np.ndarray) -> dict:
        """Calcula todos los descriptores para una banda de una RIR."""
        t30, _, _ = tr_lundeby(rir_band, self.fs, self.max_ruido_dB)
        c50 = audio_processing.ClarityCalculator.calculate(50, rir_band, self.fs)
        c80 = audio_processing.ClarityCalculator.calculate(80, rir_band, self.fs)
        d50 = audio_processing.DefinitionCalculator.calculate(rir_band, self.fs)
        drr, _, _ = get_DRR(rir_band, self.fs)
        return {'T30': t30, 'C50': c50, 'C80': c80, 'D50': d50, 'DRR': drr}
        
    def _get_tae_with_noise(self, reverbed_audio_band: np.ndarray) -> tuple:
        """Calcula el TAE, añadiendo ruido si está configurado."""
        if not self.add_noise:
            tae = TAE(reverbed_audio_band, self.fs, self.sos_lowpass_filter)
            return list(tae), nan

        noise_data = pink_noise(len(reverbed_audio_band))
        rms_signal = audio_processing.AudioProcessor.rms(reverbed_audio_band)
        rms_noise = audio_processing.AudioProcessor.rms(noise_data)
        
        snr_required = np.random.uniform(self.snr[0], self.snr[-1])
        comp = audio_processing.AudioProcessor.rms_comp(rms_signal, rms_noise, snr_required)
        noise_data_comp = noise_data * comp
        
        reverbed_noisy_audio = reverbed_audio_band + noise_data_comp
        # Normalización opcional aquí si se desea, aunque el TAE suele ser robusto a la escala.
        
        tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter)
        return list(tae), snr_required

    def process_single_rir(self, rir_file: str) -> list:
        """
        Método trabajador diseñado para ser usado con multiprocesamiento.
        Procesa una única RIR con todos sus archivos de voz y aumentos,
        y devuelve una lista completa de sus registros.
        """
        print(f"Iniciando proceso para RIR: {rir_file}...")
        
        # Instacion el filtro pasabandas para cada proceso
        self._ensure_filter()

        # --- Inicialización de variables para esta RIR ---
        rir_name = Path(rir_file).stem
        rir_data = self._load_and_normalize_audio(self.data_path / 'RIRs' / rir_file)
        records_for_this_rir = []
        random.seed(int(datetime.now().timestamp())) # Reiniciar la semilla aleatoria para cada proceso

        # --- Determinar si es training o testing ---
        if rir_file in self.rirs_for_training:
            speech_files = self.speech_files_train
            speech_type = 'train'
            speech_path = self.data_path / 'Speech' / 'train'
        else:
            speech_files = self.speech_files_test
            speech_type = 'test'
            speech_path = self.data_path / 'Speech' / 'test'

        # --- Bucle sobre los archivos de voz correspondientes ---
        for speech_file in speech_files:
            speech_name = Path(speech_file).stem
            speech_data = self._load_and_normalize_audio(speech_path / speech_file, duration=5.0)

            # 1. Procesar la RIR original siempre
            records_for_this_rir.extend(self._process_entry(
                speech_data, rir_data, speech_name, rir_name, speech_type, 'original'
            ))

            # 2. Verificar si se debe aumentar
            should_augment = not ('sintetica' in rir_name or not any(rir_name in s for s in self.to_augmentate))
            if not should_augment:
                continue
            
            # 3. Aumento de TR
            for tr_var in self.tr_variations:
                try:
                    rir_tr_aug = tr_augmentation(rir_data, self.fs, tr_var, self.bp_filter)
                    rir_tr_aug /= np.max(np.abs(rir_tr_aug))
                    
                    aug_tag = f'TR_var_{tr_var:.2f}'
                    records_for_this_rir.extend(self._process_entry(
                        speech_data, rir_tr_aug, speech_name, rir_name, speech_type, aug_tag
                    ))

                    # 4. Aumento de DRR anidado
                    if tr_var in random.sample(list(self.tr_variations), k=5):
                        for drr_var in self.drr_variations:
                            try:
                                rir_drr_aug = drr_aug(rir_tr_aug, self.fs, drr_var)
                                rir_drr_aug /= np.max(np.abs(rir_drr_aug))
                                
                                aug_tag = f'TR_{tr_var:.2f}_DRR_var_{drr_var:.2f}'
                                records_for_this_rir.extend(self._process_entry(
                                    speech_data, rir_drr_aug, speech_name, rir_name, speech_type, aug_tag
                                ))
                            except Exception:
                                continue
                except (TrAugmentationError, Exception):
                    continue
        
        print(f"Proceso para RIR: {rir_file} terminado. ✅")
        return records_for_this_rir
    
    def _process_entry(self, speech_data: np.ndarray, rir_data: np.ndarray,
                       speech_name: str, rir_name: str, speech_type: str, aug_tag: str) -> list:
        """
        Procesa una combinación única de voz y RIR (original o aumentada)
        y devuelve una lista de registros para el DataFrame.
        """
        reverbed_audio = fftconvolve(speech_data, rir_data, mode='same')
        reverbed_audio /= np.max(np.abs(reverbed_audio))

        filtered_speech_bands = self.bp_filter.process(reverbed_audio.astype(np.float32))
        filtered_rir_bands = self.bp_filter.process(rir_data.astype(np.float32))
        
        entry_records = []
        name = f'{speech_name}|{rir_name}|{aug_tag}'

        for i, band in enumerate(self.bands):
            try:
                descriptors = self._calculate_descriptors(filtered_rir_bands[i])
                tae, snr = self._get_tae_with_noise(filtered_speech_bands[i])

                record = {
                    'ReverbedAudio': name,
                    'type_data': speech_type,
                    'banda': band,
                    'descriptors': [descriptors['T30'], descriptors['C50'], descriptors['C80'], descriptors['D50']],
                    'drr': descriptors['DRR'],
                    'tae': tae,
                    'snr': snr
                }
                entry_records.append(record)
            except (ValueError, NoiseError, Exception) as e:
                # print(f"Error procesando {name} en banda {band}: {e}")
                continue
        return entry_records

    def generate_database(self):
        """
        Genera y guarda la base de datos completa. Este método reemplaza
        la lógica de multiprocessing para ser ejecutado por un gestor externo.
        """
        # --- Comprobación de caché ---
        db_file = self.cache_path / f'{self.db_name}.pkl'
        if db_file.exists():
            print(f"La base de datos ya existe en: {db_file}")
            return pd.read_pickle(db_file)

        all_records = []
        
        # --- Bucle principal sobre las RIRs ---
        for rir_file in self.rir_files:
            print(f"Procesando RIR: {rir_file}...")
            rir_name = Path(rir_file).stem
            rir_data = self._load_and_normalize_audio(self.data_path / 'RIRs' / rir_file)

            # Determina si es para training o testing para evitar duplicar código
            if rir_file in self.rirs_for_training:
                speech_files = self.speech_files_train
                speech_type = 'train'
                speech_path = self.data_path / 'Speech' / 'train'
            else:
                speech_files = self.speech_files_test
                speech_type = 'test'
                speech_path = self.data_path / 'Speech' / 'test'

            # Bucle sobre los archivos de voz correspondientes
            for speech_file in speech_files:
                speech_name = Path(speech_file).stem
                speech_data = self._load_and_normalize_audio(speech_path / speech_file, duration=5.0)

                # 1. Procesar la RIR original siempre
                all_records.extend(self._process_entry(
                    speech_data, rir_data, speech_name, rir_name, speech_type, 'original'
                ))

                # 2. Si la RIR es candidata para aumento de datos
                should_augment = not ('sintetica' in rir_name or not any(rir_name in s for s in self.to_augmentate))
                if not should_augment:
                    continue
                
                # 3. Aumento de TR
                for tr_var in self.tr_variations:
                    try:
                        rir_tr_aug = tr_augmentation(rir_data, self.fs, tr_var, self.bp_filter)
                        rir_tr_aug /= np.max(np.abs(rir_tr_aug))
                        
                        aug_tag = f'TR_var_{tr_var:.2f}'
                        all_records.extend(self._process_entry(
                            speech_data, rir_tr_aug, speech_name, rir_name, speech_type, aug_tag
                        ))

                        # 4. Aumento de DRR (anidado, como en el original)
                        if tr_var in random.sample(list(self.tr_variations), k=5):
                            for drr_var in self.drr_variations:
                                try:
                                    rir_drr_aug = drr_aug(rir_tr_aug, self.fs, drr_var)
                                    rir_drr_aug /= np.max(np.abs(rir_drr_aug))
                                    
                                    aug_tag = f'TR_{tr_var:.2f}_DRR_var_{drr_var:.2f}'
                                    all_records.extend(self._process_entry(
                                        speech_data, rir_drr_aug, speech_name, rir_name, speech_type, aug_tag
                                    ))
                                except Exception as e:
                                    # print(f"Error en aumento de DRR {drr_var}: {e}")
                                    continue
                    except (TrAugmentationError, Exception) as e:
                        # print(f"Error en aumento de TR {tr_var}: {e}")
                        continue
        
        # --- Creación y guardado del DataFrame ---
        if not all_records:
            print("No se generaron registros. La base de datos está vacía.")
            return None
            
        final_db = pd.DataFrame(all_records)
        final_db.to_pickle(db_file)
        print(f"Base de datos generada y guardada en: {db_file}")
        
        return final_db