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
import json
import glob
from tqdm import tqdm

@dataclass
class ProcessingResult:
    """Estructura para los resultados del procesamiento"""
    names: List[str]
    bands: List[str]
    taes: List[List[float]]  # Ya está correcto como List[List[float]]
    descriptors: List[List[float]]
    snrs: List[float]
    drrs: List[float]
    data_types: List[str]

class DataBase:
    """Clase optimizada para generar la base de datos para entrenar la red con guardado incremental."""

    def __init__(self, speech_files_train: List[str], speech_files_test: List[str], 
                 rir_files: List[str], tot_sinteticas: int, to_augmentate: List[str],
                 rirs_for_training: List[str], rirs_for_testing: List[str], 
                 bands: List[str], filter_type: str, fs: int, max_ruido_dB: float,
                 order: int, add_noise: bool, snr: List[float], 
                 TR_aug: List[float], DRR_aug: List[float], batch_size: int = 10):
        
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
        self.batch_size = batch_size  # Tamaño del lote para guardado incremental

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

        # Paths para guardado incremental
        self.temp_dir = Path(f'cache/{self.db_name}_temp')
        self.final_dir = Path(f'cache/{self.db_name}')

    def _setup_filters(self):
        """Configurar filtros una sola vez"""
        self.cutoff = 20  # Frecuencia de corte a 20 Hz
        self.sos_lowpass_filter = butter(self.order, self.cutoff, fs=self.fs, 
                                       btype='lowpass', output='sos')

    @lru_cache(maxsize=128)
    def _load_audio_cached(self, file_path: str, duration: Optional[float] = None) -> np.ndarray:
        """Cache para cargar audios ya procesados"""
        data, _ = load(file_path, sr=self.fs, duration=duration)
        return data / np.max(np.abs(data))

    def _process_audio_with_rir(self, speech_data: np.ndarray, rir_data: np.ndarray, 
                               speech_name: str, rir_name: str, variant_name: str,
                               data_type: str, bpfilter) -> Optional[ProcessingResult]:
        """Procesar un audio con una RIR específica"""
        try:
            # Reverberar el audio
            reverbed_audio = fftconvolve(speech_data, rir_data, mode='same')
            reverbed_audio = reverbed_audio / np.max(np.abs(reverbed_audio))

            # Filtrar señales
            filtered_speech = bpfilter.process(reverbed_audio)
            filtered_rir = bpfilter.process(rir_data)

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
                      reverbed_audio: np.ndarray) -> Tuple[List[float], float]:
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
            return tae.tolist(), snr_required  # Convertir a lista para serialización
        else:
            tae = TAE(filtered_speech, self.fs, self.sos_lowpass_filter)
            return tae.tolist(), nan  # Convertir a lista para serialización

    def _should_augment_rir(self, rir_name: str) -> bool:
        """Verificar si una RIR debe ser aumentada"""
        return ('sintetica' not in rir_name) and any(rir_name in s for s in self.to_augmentate)

    def _save_batch_to_temp(self, batch_data: dict, batch_id: str) -> None:
        """Guardar un lote de datos en archivos temporales"""
        if not batch_data['ReverbedAudio']:
            return

        try:
            # Crear directorio temporal si no existe
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            # Crear DataFrame y guardar
            df = pd.DataFrame(batch_data)
            batch_file = self.temp_dir / f'batch_{batch_id}.pkl'
            df.to_pickle(batch_file, protocol=5)

            # Guardar metadatos
            metadata = {
                'batch_id': batch_id,
                'num_samples': len(batch_data['ReverbedAudio']),
                'timestamp': datetime.now().isoformat(),
                'process_id': os.getpid()
            }

            metadata_file = self.temp_dir / f'batch_{batch_id}_meta.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error guardando lote {batch_id}: {e}")
            raise

    # <--- INICIO DE BLOQUE MODIFICADO
    def _combine_temp_batches(self) -> None:
        """
        Combinar todos los lotes temporales en un archivo HDF5 final.
        Este método es eficiente en memoria, ya que no carga todos los lotes a la vez.
        """
        if not self.temp_dir.exists():
            print("No hay lotes temporales para combinar")
            return
            
        batch_files = sorted(glob.glob(str(self.temp_dir / 'batch_*.pkl')))
        
        if not batch_files:
            print("No se encontraron lotes temporales")
            return
            
        print(f"Combinando {len(batch_files)} lotes temporales en un archivo HDF5...")
        
        self.final_dir.mkdir(parents=True, exist_ok=True)
        final_file_path = self.final_dir / 'database.h5' # <-- El archivo final ahora es .h5
        
        # Abrir el archivo HDFStore para añadir datos por lotes
        # Se requiere la librería 'tables' (pip install tables)
        try:
            store = pd.HDFStore(final_file_path, mode='w', complevel=5, complib='blosc')
            
            total_samples = 0
            successful_batches = 0
            for batch_file in tqdm(batch_files, desc="Combinando lotes en HDF5"):
                try:
                    df = pd.read_pickle(batch_file)
                    
                    # Asegurar que las columnas problemáticas sean del tipo correcto
                    if 'tae' in df.columns:
                        # Convertir arrays a listas si es necesario
                        df['tae'] = df['tae'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') and not isinstance(x, list) else x)
                    
                    if 'descriptors' in df.columns:
                        # Convertir arrays a listas si es necesario
                        df['descriptors'] = df['descriptors'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') and not isinstance(x, list) else x)
                    
                    store.append('data', df, format='table', data_columns=True, min_itemsize={'ReverbedAudio': 255})
                    total_samples += len(df)
                    successful_batches += 1
                except Exception as e:
                    print(f"Error leyendo o añadiendo el lote {batch_file}: {e}")
                    continue
                    
            store.close()
            print(f"Base de datos HDF5 combinada y guardada: {total_samples} muestras en {final_file_path}")
            print(f"Lotes procesados exitosamente: {successful_batches}/{len(batch_files)}")
            
            # Limpiar archivos temporales solo si la combinación fue exitosa
            if successful_batches > 0:
                self._cleanup_temp_files()
            else:
                print("No se pudo procesar ningún lote. Los archivos temporales se mantienen para depuración.")

        except Exception as e:
            print(f"Error fatal durante la combinación a HDF5: {e}")
            # No limpiar los archivos temporales si hay un error fatal para poder depurar
    # <--- FIN DE BLOQUE MODIFICADO

    def _cleanup_temp_files(self) -> None:
        """Limpiar archivos temporales después de combinar"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print("Archivos temporales limpiados")
        except Exception as e:
            print(f"Error limpiando archivos temporales: {e}")

    def _get_processed_rirs(self) -> set:
        """Obtener la lista de RIRs ya procesadas"""
        if not self.temp_dir.exists():
            return set()
            
        processed = set()
        metadata_files = glob.glob(str(self.temp_dir / '*_meta.json'))
        
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                continue
                
        return processed
    
    # <--- INICIO DE BLOQUE MODIFICADO
    def calc_database_multiprocess(self, rir_file: str) -> Optional[bool]:
        """
        Procesa una RIR, sus aumentaciones y audios de voz, guardando los resultados
        incrementalmente para mantener bajo el uso de memoria.
        """
        if self._database_exists():
            print("La base de datos final ya existe. Saltando procesamiento.")
            return None

        rir_name = rir_file.split('.wav')[0]

        try:
            rir_data = self._load_audio_cached(f'data/RIRs/{rir_file}')
        except Exception as err:
            print(f"Error cargando RIR {rir_file}: {err}")
            return None

        bpfilter = audio_processing.OctaveFilterBank(filter_order=self.order)

        if rir_file in self.rirs_for_training:
            speech_files = self.speech_files_train
            data_type = 'train'
            speech_path = 'data/Speech/train/'
        else:
            speech_files = self.speech_files_test
            data_type = 'test'
            speech_path = 'data/Speech/test/'

        # --- Lógica de bacheo "al vuelo" ---
        current_batch = {
            'ReverbedAudio': [], 'banda': [], 'tae': [], 'descriptors': [],
            'snr': [], 'drr': [], 'type_data': []
        }
        samples_in_batch = 0
        local_batch_index = 0
        total_samples_processed = 0

        def add_to_batch(result, batch_dict, samples_count, batch_idx):
            if result is None:
                return batch_dict, samples_count, batch_idx, 0
            
            num_added = 0
            for i in range(len(result.names)):
                for key, value_list in zip(batch_dict.keys(), [result.names, result.bands, result.taes, result.descriptors, result.snrs, result.drrs, result.data_types]):
                     batch_dict[key].append(value_list[i])
                
                samples_count += 1
                num_added += 1

                if samples_count >= self.batch_size:
                    batch_id = f"{os.getpid()}_{batch_idx:04d}"
                    self._save_batch_to_temp(batch_dict, batch_id)
                    # print(f"Proceso {os.getpid()}: Lote {batch_idx} guardado ({samples_count} muestras)")
                    
                    batch_idx += 1
                    batch_dict = {key: [] for key in batch_dict}
                    samples_count = 0
            return batch_dict, samples_count, batch_idx, num_added
        
        # --- Bucle de procesamiento principal ---
        for speech_file in speech_files:
            speech_name = speech_file.split('.wav')[0]
            try:
                speech_data = self._load_audio_cached(f'{speech_path}{speech_file}', duration=5.0)
            except Exception as err:
                print(f"Error cargando audio {speech_file}: {err}")
                continue

            # Procesar la RIR original
            result = self._process_audio_with_rir(speech_data, rir_data, speech_name, rir_name, 'original', data_type, bpfilter)
            current_batch, samples_in_batch, local_batch_index, added = add_to_batch(result, current_batch, samples_in_batch, local_batch_index)
            total_samples_processed += added

            # Procesar aumentaciones si corresponde
            if self._should_augment_rir(rir_name):
                augmentation_results = self._process_augmentations(speech_data, rir_data, speech_name, rir_name, data_type, bpfilter)
                for aug_result in augmentation_results:
                    current_batch, samples_in_batch, local_batch_index, added = add_to_batch(aug_result, current_batch, samples_in_batch, local_batch_index)
                    total_samples_processed += added

        # Guardar el último lote si tiene datos
        if samples_in_batch > 0:
            batch_id = f"{os.getpid()}_{local_batch_index:04d}"
            self._save_batch_to_temp(current_batch, batch_id)
            # print(f"Proceso {os.getpid()}: Último lote {local_batch_index} guardado ({samples_in_batch} muestras)")

        # print(f"Proceso {os.getpid()}: Total de muestras procesadas para {rir_name}: {total_samples_processed}")
        return True
    # <--- FIN DE BLOQUE MODIFICADO

    def _process_augmentations(self, speech_data: np.ndarray, rir_data: np.ndarray,
                             speech_name: str, rir_name: str, data_type: str, bpfilter) -> List[ProcessingResult]:
        """Procesar todas las aumentaciones de una RIR"""
        results = []
        
        # Seleccionar TR variations para DRR augmentation
        random.seed(datetime.now().timestamp())  # Usar timestamp como semilla válida
        DRR_tr_aug = random.sample(list(self.TR_variations), k=min(5, len(self.TR_variations)))

        # Procesar variaciones de TR
        for TR_var in self.TR_variations:
            try:
                rir_aug = tr_augmentation(rir_data, self.fs, TR_var, bpfilter)
                rir_aug = rir_aug / np.max(np.abs(rir_aug))

                # Procesar TR augmentation
                result = self._process_audio_with_rir(
                    speech_data, rir_aug, speech_name, rir_name, 
                    f'TR_var_{TR_var}', data_type, bpfilter
                )
                if result:
                    results.append(result)

                # Procesar DRR augmentation si corresponde
                if TR_var in DRR_tr_aug:
                    drr_results = self._process_drr_augmentations(
                        speech_data, rir_aug, speech_name, rir_name, data_type, bpfilter
                    )
                    results.extend(drr_results)

            except TrAugmentationError:
                continue

        return results

    def _process_drr_augmentations(self, speech_data: np.ndarray, rir_tr_aug: np.ndarray,
                                 speech_name: str, rir_name: str, data_type: str, bpfilter) -> List[ProcessingResult]:
        """Procesar aumentaciones de DRR"""
        results = []
        
        for DRR_var in self.DRR_variations:
            try:
                rir_aug = drr_aug(rir_tr_aug, self.fs, DRR_var)
                rir_aug = rir_aug / np.max(np.abs(rir_aug))

                result = self._process_audio_with_rir(
                    speech_data, rir_aug, speech_name, rir_name,
                    f'DRR_var_{DRR_var}', data_type, bpfilter
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

        return [combined.names, combined.bands, combined.taes, combined.descriptors,
                combined.snrs, combined.drrs, combined.data_types]

    def _database_exists(self) -> bool:
        """Verificar si la base de datos ya existe"""
        # <--- MODIFICACIÓN: Comprueba si existe el archivo HDF5 final.
        final_file = self.final_dir / 'database.h5'
        if final_file.exists():
            return True
        return False

    def get_database_name(self) -> str:
        """Obtener el nombre de la base de datos"""
        return self.db_name