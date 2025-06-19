from .parameters_calculation import tr_lundeby, NoiseError
from .parameters_calculation import TAE, pink_noise
from .parameters_calculation import tr_augmentation, TrAugmentationError
from .parameters_calculation import drr_aug, get_DRR
from .cpp import audio_processing
from librosa import load
import pandas as pd
from os import listdir
from math import nan
from scipy.signal import butter, fftconvolve
import numpy as np
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ProcessingResult:
    """Estructura para los resultados del procesamiento"""
    names: List[str]
    bands: List[str]
    taes: List[List[float]]
    descriptors: List[List[float]]
    snrs: List[float]
    drrs: List[float]
    data_types: List[str]

class DataBase:
    """Clase optimizada para generar la base de datos para entrenar la red."""

    def __init__(self, speech_files_train: List[str], speech_files_test: List[str], 
                 rir_files: List[str], tot_sinteticas: int, to_augmentate: List[str],
                 rirs_for_training: List[str], rirs_for_testing: List[str], 
                 bands: List[str], filter_type: str, fs: int, max_ruido_dB: float,
                 order: int, add_noise: bool, snr: List[float], 
                 TR_aug: List[float], DRR_aug: List[float]):
        
        # Almacenar parámetros
        self.speech_files_train = speech_files_train
        self.speech_files_test = speech_files_test
        self.rir_files = rir_files
        self.tot_sinteticas = tot_sinteticas
        self.to_augmentate = set(to_augmentate)  # Usar set para búsquedas O(1)
        self.rirs_for_training = set(rirs_for_training)  # Usar set para búsquedas O(1)
        self.rirs_for_testing = set(rirs_for_testing)
        self.bands = bands
        self.filter_type = filter_type
        self.fs = fs
        self.max_ruido_dB = max_ruido_dB
        self.order = order
        self.add_noise = add_noise
        self.snr = snr
        self.TR_aug = TR_aug
        self.DRR_aug = DRR_aug

        # Generar nombre de base de datos
        self.db_name = (f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_'
                       f'traug_{TR_aug[0]}_{TR_aug[1]}_{TR_aug[2]}_'
                       f'drraug_{DRR_aug[0]}_{DRR_aug[1]}_{DRR_aug[2]}_'
                       f'snr_{snr[0]}_{snr[-1]}')

        # Configurar filtros una sola vez
        self._setup_filters()
        
        # Pre-calcular variaciones
        self.TR_variations = np.arange(TR_aug[0], TR_aug[1], TR_aug[2])
        self.DRR_variations = np.arange(DRR_aug[0], DRR_aug[1], DRR_aug[2])

    def _setup_filters(self):
        """Configurar filtros una sola vez"""
        self.cutoff = 20  # Frecuencia de corte a 20 Hz
        self.sos_lowpass_filter = butter(self.order, self.cutoff, fs=self.fs, 
                                       btype='lowpass', output='sos')
        self.bpfilter = audio_processing.OctaveFilterBank(filter_order=self.order)

    @lru_cache(maxsize=128)
    def _load_audio_cached(self, file_path: str, duration: Optional[float] = None) -> np.ndarray:
        """Cache para cargar audios ya procesados"""
        data, _ = load(file_path, sr=self.fs, duration=duration)
        return data / np.max(np.abs(data))

    def _process_audio_with_rir(self, speech_data: np.ndarray, rir_data: np.ndarray, 
                               speech_name: str, rir_name: str, variant_name: str,
                               data_type: str) -> Optional[ProcessingResult]:
        """Procesar un audio con una RIR específica"""
        try:
            # Reverberar el audio
            reverbed_audio = fftconvolve(speech_data, rir_data, mode='same')
            reverbed_audio = reverbed_audio / np.max(np.abs(reverbed_audio))

            # Filtrar señales
            filtered_speech = self.bpfilter.process(reverbed_audio)
            filtered_rir = self.bpfilter.process(rir_data)

            name = f'{speech_name}|{rir_name}|{variant_name}'
            
            result = ProcessingResult([], [], [], [], [], [], [])

            for i, band in enumerate(self.bands):
                try:
                    # Calcular descriptores
                    descriptors = self._calculate_descriptors(filtered_rir[i])
                    
                    # Calcular TAE
                    tae, snr_used = self._calculate_tae(filtered_speech[i], reverbed_audio)
                    
                    # Agregar resultados
                    result.names.append(name)
                    result.bands.append(band)
                    result.taes.append(list(tae))
                    result.descriptors.append(descriptors)
                    result.snrs.append(snr_used)
                    result.drrs.append(descriptors[4])  # DRR es el 5to descriptor
                    result.data_types.append(data_type)
                    
                except (ValueError, NoiseError, Exception) as err:
                    continue  # Pasar a la siguiente banda si hay error

            return result if result.names else None

        except Exception as err:
            return None

    def _calculate_descriptors(self, filtered_rir: np.ndarray) -> List[float]:
        """Calcular todos los descriptores acústicos"""
        t30, _, _ = tr_lundeby(filtered_rir, self.fs, self.max_ruido_dB)
        c50 = audio_processing.ClarityCalculator.calculate(50, filtered_rir, self.fs)
        c80 = audio_processing.ClarityCalculator.calculate(80, filtered_rir, self.fs)
        d50 = audio_processing.DefinitionCalculator.calculate(filtered_rir, self.fs)
        drr, _, _ = get_DRR(filtered_rir, self.fs)
        return [t30, c50, c80, d50, drr]

    def _calculate_tae(self, filtered_speech: np.ndarray, 
                      reverbed_audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calcular TAE con o sin ruido"""
        if self.add_noise:
            # Generar ruido rosa
            noise_data = pink_noise(len(filtered_speech))
            
            # Calcular SNR requerido
            rms_signal = audio_processing.AudioProcessor.rms(filtered_speech)
            rms_noise = audio_processing.AudioProcessor.rms(noise_data)
            snr_required = np.random.uniform(self.snr[0], self.snr[-1])
            
            # Compensar ruido
            comp = audio_processing.AudioProcessor.rms_comp(rms_signal, rms_noise, snr_required)
            noise_data_comp = noise_data * comp
            
            # Añadir ruido
            reverbed_noisy_audio = reverbed_audio + noise_data_comp
            reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))
            
            tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter)
            return tae, snr_required
        else:
            tae = TAE(filtered_speech, self.fs, self.sos_lowpass_filter)
            return tae, nan

    def _should_augment_rir(self, rir_name: str) -> bool:
        """Verificar si una RIR debe ser aumentada"""
        return ('sintetica' not in rir_name) and any(rir_name in s for s in self.to_augmentate)

    def calc_database_multiprocess(self, rir_file: str) -> Optional[List]:
        """Versión optimizada del cálculo de base de datos"""
        # Verificar si la base de datos ya existe
        if self._database_exists():
            return None

        rir_name = rir_file.split('.wav')[0]
        
        # Cargar RIR una sola vez
        try:
            rir_data = self._load_audio_cached(f'data/RIRs/{rir_file}')
        except Exception as err:
            print(f"Error cargando RIR {rir_file}: {err}")
            return None

        # Determinar tipo de datos y archivos de voz
        if rir_file in self.rirs_for_training:
            speech_files = self.speech_files_train
            data_type = 'train'
            speech_path = 'data/Speech/train/'
        else:
            speech_files = self.speech_files_test
            data_type = 'test'
            speech_path = 'data/Speech/test/'

        all_results = []
        
        # Procesar cada archivo de voz
        for speech_file in speech_files:
            speech_name = speech_file.split('.wav')[0]
            
            try:
                speech_data = self._load_audio_cached(f'{speech_path}{speech_file}', duration=5.0)
            except Exception as err:
                print(f"Error cargando audio {speech_file}: {err}")
                continue

            # Procesar RIR original
            result = self._process_audio_with_rir(
                speech_data, rir_data, speech_name, rir_name, 'original', data_type
            )
            if result:
                all_results.append(result)

            # Procesar aumentaciones si es necesario
            if self._should_augment_rir(rir_name):
                augmentation_results = self._process_augmentations(
                    speech_data, rir_data, speech_name, rir_name, data_type
                )
                all_results.extend(augmentation_results)

        # Combinar todos los resultados
        return self._combine_results(all_results, rir_name)

    def _process_augmentations(self, speech_data: np.ndarray, rir_data: np.ndarray,
                             speech_name: str, rir_name: str, data_type: str) -> List[ProcessingResult]:
        """Procesar todas las aumentaciones de una RIR"""
        results = []
        
        # Seleccionar TR variations para DRR augmentation
        random.seed(datetime.now())
        DRR_tr_aug = random.sample(list(self.TR_variations), k=min(5, len(self.TR_variations)))

        # Procesar variaciones de TR
        for TR_var in self.TR_variations:
            try:
                rir_aug = tr_augmentation(rir_data, self.fs, TR_var, self.bpfilter)
                rir_aug = rir_aug / np.max(np.abs(rir_aug))

                # Procesar TR augmentation
                result = self._process_audio_with_rir(
                    speech_data, rir_aug, speech_name, rir_name, 
                    f'TR_var_{TR_var}', data_type
                )
                if result:
                    results.append(result)

                # Procesar DRR augmentation si corresponde
                if TR_var in DRR_tr_aug:
                    drr_results = self._process_drr_augmentations(
                        speech_data, rir_aug, speech_name, rir_name, data_type
                    )
                    results.extend(drr_results)

            except TrAugmentationError:
                continue

        return results

    def _process_drr_augmentations(self, speech_data: np.ndarray, rir_tr_aug: np.ndarray,
                                 speech_name: str, rir_name: str, data_type: str) -> List[ProcessingResult]:
        """Procesar aumentaciones de DRR"""
        results = []
        
        for DRR_var in self.DRR_variations:
            try:
                rir_aug = drr_aug(rir_tr_aug, self.fs, DRR_var)
                rir_aug = rir_aug / np.max(np.abs(rir_aug))

                result = self._process_audio_with_rir(
                    speech_data, rir_aug, speech_name, rir_name,
                    f'DRR_var_{DRR_var}', data_type
                )
                if result:
                    results.append(result)

            except Exception:
                continue

        return results

    def _combine_results(self, results: List[ProcessingResult], rir_name: str) -> List:
        """Combinar todos los resultados en el formato esperado"""
        if not results:
            return [[], [], [], [], [], [], []]

        combined = ProcessingResult([], [], [], [], [], [], [])
        
        for result in results:
            combined.names.extend(result.names)
            combined.bands.extend(result.bands)
            combined.taes.extend(result.taes)
            combined.descriptors.extend(result.descriptors)
            combined.snrs.extend(result.snrs)
            combined.drrs.extend(result.drrs)
            combined.data_types.extend(result.data_types)

        print(f'RIR procesada: {rir_name}')
        return [combined.names, combined.bands, combined.taes, combined.descriptors,
                combined.snrs, combined.drrs, combined.data_types]

    def _database_exists(self) -> bool:
        """Verificar si la base de datos ya existe"""
        try:
            available_cache_files = listdir('cache')
            return any(self.db_name in available for available in available_cache_files)
        except FileNotFoundError:
            return False

    def save_database_multiprocess(self, results: List) -> None:
        """Versión optimizada para guardar la base de datos"""
        if not results or not any(results):
            print("No hay resultados para guardar")
            return

        # Crear directorio
        db_path = Path(f'cache/{self.db_name}')
        db_path.mkdir(parents=True, exist_ok=True)

        print('Guardando base de datos...')
        
        # Inicializar listas combinadas
        combined_data = {
            'ReverbedAudio': [],
            'banda': [],
            'tae': [],
            'descriptors': [],
            'snr': [],
            'drr': [],
            'type_data': []
        }

        # Combinar todos los resultados
        for result in results:
            if result and len(result) == 7:  # Verificar que el resultado tenga el formato correcto
                for i in range(len(result[0])):
                    combined_data['ReverbedAudio'].append(result[0][i])
                    combined_data['banda'].append(result[1][i])
                    combined_data['tae'].append(result[2][i])
                    combined_data['descriptors'].append(result[3][i])
                    combined_data['snr'].append(result[4][i])
                    combined_data['drr'].append(result[5][i])
                    combined_data['type_data'].append(result[6][i])

        if not combined_data['ReverbedAudio']:
            print("No hay datos para guardar")
            return

        # Guardar en chunks para manejar datasets grandes
        self._save_in_chunks(combined_data, db_path)
        print('Base de datos guardada!')

    def _save_in_chunks(self, data: dict, db_path: Path, chunk_size: int = 50000) -> None:
        """Guardar datos en chunks para manejar datasets grandes"""
        total_items = len(data['ReverbedAudio'])
        
        if total_items <= chunk_size:
            # Guardar todo en un archivo
            df = pd.DataFrame(data)
            df.to_pickle(db_path / '0.pkl', protocol=5)
        else:
            # Guardar en múltiples archivos
            num_chunks = (total_items + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_items)
                
                chunk_data = {
                    key: values[start_idx:end_idx] 
                    for key, values in data.items()
                }
                
                df = pd.DataFrame(chunk_data)
                df.to_pickle(db_path / f'{chunk_idx}.pkl', protocol=5)

    def get_database_name(self) -> str:
        """Obtener el nombre de la base de datos"""
        return self.db_name