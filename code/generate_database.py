from __future__ import annotations
from .parameters_calculation import tr_lundeby
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

# --------------------------------------------------------------------------- #
#   Estructuras auxiliares
# --------------------------------------------------------------------------- #

@dataclass
class ProcessingResult:
    """Resultados de procesar UNA combinación speech + RIR (+ augmentación)."""
    names:        List[str]
    bands:        List[str]
    taes:         List[List[float]]
    descriptors:  List[List[float]]
    snrs:         List[float]
    drrs:         List[float]
    data_types:   List[str]


# --------------------------------------------------------------------------- #
#   Clase principal
# --------------------------------------------------------------------------- #

class DataBase:
    """
    Generador/gestor de la base de datos con guardado incremental y soporte
    multiprocessing.
    """

    # ------------------------------------------------------------------- #
    #   Inicialización
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        speech_files_train: List[str],
        speech_files_test:  List[str],
        rir_files:          List[str],
        tot_sinteticas:     int,
        to_augmentate:      List[str],
        rirs_for_training:  List[str],
        rirs_for_testing:   List[str],
        bands:              List[int],
        filter_type:        str,
        fs:                 int,
        max_ruido_dB:       float,
        order:              int,
        add_noise:          bool,
        snr:                List[float],
        TR_aug:             List[float],
        DRR_aug:            List[float],
        batch_size:         int = 10
    ) -> None:

        # --- Listas de ficheros ------------------------------------------------
        self.speech_files_train = speech_files_train
        self.speech_files_test  = speech_files_test
        self.rir_files          = rir_files

        # --- Listas normalizadas a minúsculas, sin extensión -------------------
        self.to_augmentate     = {Path(f).stem.lower() for f in to_augmentate}
        self.rirs_for_training = {Path(f).stem.lower() for f in rirs_for_training}
        self.rirs_for_testing  = {Path(f).stem.lower() for f in rirs_for_testing}

        # --- Otros parámetros --------------------------------------------------
        self.tot_sinteticas = tot_sinteticas
        self.bands          = bands
        self.filter_type    = filter_type
        self.fs             = fs
        self.max_ruido_dB   = max_ruido_dB
        self.order          = order
        self.add_noise      = add_noise
        self.snr            = snr
        self.batch_size     = batch_size

        # --- Rangos de aumentación --------------------------------------------
        self.TR_variations  = np.arange(TR_aug[0],  TR_aug[1],  TR_aug[2])
        self.DRR_variations = np.arange(DRR_aug[0], DRR_aug[1], DRR_aug[2])

        # --- Nombre de la base y rutas ----------------------------------------
        self.db_name  = (
            f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_'
            f'traug_{TR_aug[0]}_{TR_aug[1]}_{TR_aug[2]}_'
            f'drraug_{DRR_aug[0]}_{DRR_aug[1]}_{DRR_aug[2]}_'
            f'snr_{snr[0]}_{snr[-1]}'
        )
        self.temp_dir  = Path(f'cache/{self.db_name}_temp')
        self.final_dir = Path(f'cache/{self.db_name}')

        # --- Filtro pasabajos para el TAE --------------------------------------
        self._setup_filters()

    # ------------------------------------------------------------------- #
    #   Métodos auxiliares
    # ------------------------------------------------------------------- #
    def _setup_filters(self) -> None:
        """Configurar el filtro pasabajos de la envolvente TAE (una sola vez)."""
        self.cutoff = 20  # Hz
        self.sos_lowpass_filter = butter(
            self.order, self.cutoff, fs=self.fs,
            btype='lowpass', output='sos'
        )

    @lru_cache(maxsize=128)
    def _load_audio_cached(
        self,
        file_path: str,
        duration:  Optional[float] = None
    ) -> np.ndarray:
        """Carga de audio con caché (librosa)."""
        data, _ = load(file_path, sr=self.fs, duration=duration)
        return data / np.max(np.abs(data))

    # ------------------------------------------------------------------- #
    #   Decidir si una RIR se debe aumentar
    # ------------------------------------------------------------------- #
    def _should_augment_rir(self, rir_name: str) -> bool:
        """
        Devolver `True` si la RIR (stem, ya sin extensión) debe pasar por
        aumentación.  Criterios:
        - NO contiene la palabra 'sintetica'
        - Coincide (por subcadena) con alguno de los nombres en
          `self.to_augmentate`.
        """
        name_lower = Path(rir_name).stem.lower()

        if "sintetica" in name_lower:
            return False

        return any(aug in name_lower for aug in self.to_augmentate)

    # ------------------------------------------------------------------- #
    #   Guardado incremental (batches temporales)
    # ------------------------------------------------------------------- #
    def _save_batch_to_temp(self, batch_dict: dict, batch_id: str) -> None:
        """Guardar un lote en la carpeta temporal."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        batch_file = self.temp_dir / f"{batch_id}.pkl"
        pd.DataFrame(batch_dict).to_pickle(batch_file, protocol=5)

    def _combine_temp_batches(self) -> None:
        """Unir todos los .pkl temporales y guardarlos en HDF5."""
        self.final_dir.mkdir(parents=True, exist_ok=True)
        final_file_path = self.final_dir / "database.h5"

        batch_files = sorted(self.temp_dir.glob("*.pkl"))
        if not batch_files:
            print("No hay lotes temporales para combinar.")
            return

        all_dfs, total_samples, ok_batches = [], 0, 0
        for bf in tqdm(batch_files, desc="Combinando lotes en HDF5"):
            try:
                df = pd.read_pickle(bf)

                # Convertir columnas que puedan traer ndarrays
                for col in ("tae", "descriptors"):
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: x.tolist()
                            if hasattr(x, "tolist") and not isinstance(x, list)
                            else x
                        )

                all_dfs.append(df)
                total_samples += len(df)
                ok_batches += 1
            except Exception as err:
                print(f"Error leyendo lote {bf}: {err}")

        if not all_dfs:
            print("No se pudo ensamblar ningún lote.")
            return

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_hdf(final_file_path, key="data", format="fixed")
        print(
            f"Base combinada: {total_samples} muestras "
            f"({ok_batches}/{len(batch_files)} lotes correctos) → {final_file_path}"
        )

        self._cleanup_temp_files()

    def _cleanup_temp_files(self) -> None:
        """Borrar la carpeta temporal tras combinar."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as err:
            print(f"Error limpiando temporales: {err}")

    # ------------------------------------------------------------------- #
    #   Helpers varios
    # ------------------------------------------------------------------- #
    def _database_exists(self) -> bool:
        """¿Ya existe el HDF5 final?"""
        return (self.final_dir / "database.h5").exists()

    def get_database_name(self) -> str:
        return self.db_name

    # ------------------------------------------------------------------- #
    #   Procesamiento principal (multiprocessing friendly)
    # ------------------------------------------------------------------- #
    def calc_database_multiprocess(self, rir_file: str) -> Optional[bool]:
        """
        Procesar **una** RIR (incluyendo sus aumentaciones) y todos los audios
        de voz asociados.  Guarda los resultados por lotes temporales.
        """
        if self._database_exists():
            # Otro proceso ya terminó toda la BD
            return None

        rir_stem = Path(rir_file).stem.lower()

        # --- Cargar RIR ---------------------------------------------------
        try:
            rir_data = self._load_audio_cached(f"data/RIRs/{rir_file}")
        except Exception as err:
            print(f"Error cargando RIR {rir_file}: {err}")
            return None

        # --- Determinar split train/test ---------------------------------
        if rir_stem in self.rirs_for_training:
            speech_files = self.speech_files_train
            data_type    = "train"
            speech_path  = "data/Speech/train/"
        elif rir_stem in self.rirs_for_testing:
            speech_files = self.speech_files_test
            data_type    = "test"
            speech_path  = "data/Speech/test/"
        else:
            # Si no está listada explícitamente, envíala a train por defecto
            speech_files = self.speech_files_train
            data_type    = "train"
            speech_path  = "data/Speech/train/"

        # --- Filtro de bandas C++ ----------------------------------------
        bpfilter = audio_processing.OctaveFilterBank(filter_order=self.order)

        # --- Inicializar lote local --------------------------------------
        batch = {
            "ReverbedAudio": [], "banda": [], "tae": [], "descriptors": [],
            "snr": [], "drr": [], "type_data": []
        }
        samples_in_batch = 0
        local_batch_idx  = 0

        # --- Función interna para agregar resultados ---------------------
        def _add_to_batch(result: Optional[ProcessingResult]) -> None:
            nonlocal batch, samples_in_batch, local_batch_idx
            if result is None:
                return
            for i in range(len(result.names)):
                for key, src in zip(
                    batch.keys(),
                    (result.names, result.bands, result.taes,
                     result.descriptors, result.snrs, result.drrs,
                     result.data_types)
                ):
                    batch[key].append(src[i])
                samples_in_batch += 1

                # ¿Toca volcar a disco?
                if samples_in_batch >= self.batch_size:
                    batch_id = f"{os.getpid()}_{local_batch_idx:04d}"
                    self._save_batch_to_temp(batch, batch_id)
                    batch  = {k: [] for k in batch}  # reiniciar
                    samples_in_batch = 0
                    local_batch_idx += 1

        # -----------------------------------------------------------------
        #   BUCLE principal por cada audio de voz
        # -----------------------------------------------------------------
        for speech_file in speech_files:
            speech_name = Path(speech_file).stem
            try:
                speech_data = self._load_audio_cached(
                    f"{speech_path}{speech_file}", duration=5.0
                )
            except Exception as err:
                print(f"Error cargando speech {speech_file}: {err}")
                continue

            # --- RIR original -------------------------------------------
            res_orig = self._process_audio_with_rir(
                speech_data, rir_data, speech_name, rir_stem,
                "original", data_type, bpfilter
            )
            _add_to_batch(res_orig)

            # --- Aumentaciones (TR + DRR) --------------------------------
            if self._should_augment_rir(rir_stem):
                for res in self._process_augmentations(
                    speech_data, rir_data, speech_name, rir_stem,
                    data_type, bpfilter
                ):
                    _add_to_batch(res)

        # --- Volcar el último lote --------------------------------------
        if samples_in_batch:
            batch_id = f"{os.getpid()}_{local_batch_idx:04d}"
            self._save_batch_to_temp(batch, batch_id)

        return True

    # ------------------------------------------------------------------- #
    #   Procesado de una señal con una RIR concreta
    # ------------------------------------------------------------------- #
    def _process_audio_with_rir(
        self,
        speech_data: np.ndarray,
        rir_data:    np.ndarray,
        speech_name: str,
        rir_name:    str,
        variant:     str,
        data_type:   str,
        bpfilter
    ) -> Optional[ProcessingResult]:
        """Convolución, filtrado por bandas y cálculo de descriptores."""
        try:
            reverbed = fftconvolve(speech_data, rir_data, mode="same")
            reverbed = reverbed / np.max(np.abs(reverbed))

            # Filtrado (C++ → float32)
            speech_bands = bpfilter.process(reverbed.astype(np.float32))
            rir_bands    = bpfilter.process(rir_data.astype(np.float32))

            name_base = f"{speech_name}|{rir_name}|{variant}"

            result = ProcessingResult([], [], [], [], [], [], [])

            for band_idx, band in enumerate(self.bands):
                try:
                    # --- Descriptores acústicos --------------------------
                    t30, _, _ = tr_lundeby(rir_bands[band_idx], self.fs, self.max_ruido_dB)
                    c50 = audio_processing.ClarityCalculator.calculate(50, rir_bands[band_idx], self.fs)
                    c80 = audio_processing.ClarityCalculator.calculate(80, rir_bands[band_idx], self.fs)
                    d50 = audio_processing.DefinitionCalculator.calculate(rir_bands[band_idx], self.fs)
                    drr, _, _ = get_DRR(rir_bands[band_idx], self.fs)

                    # --- TAE (+ SNR opcional) ----------------------------
                    if self.add_noise:
                        # Ruido rosa
                        noise_data = pink_noise(len(speech_bands[band_idx]))
                        rms_sig  = np.sqrt(np.mean(speech_bands[band_idx] ** 2))
                        rms_noise= np.sqrt(np.mean(noise_data ** 2))
                        snr_req  = np.random.uniform(self.snr[0], self.snr[-1])
                        comp     = 10 ** ((rms_sig - rms_noise - snr_req) / 20)
                        noisy    = reverbed + noise_data * comp
                        tae      = TAE(noisy, self.fs, self.sos_lowpass_filter)
                        snr_val  = snr_req
                    else:
                        tae     = TAE(speech_bands[band_idx], self.fs, self.sos_lowpass_filter)
                        snr_val = nan

                    # --- Guardar ----------------------------------------
                    result.names.append(name_base)
                    result.bands.append(band)
                    result.taes.append(list(tae))
                    result.descriptors.append([t30, c50, c80, d50])
                    result.snrs.append(snr_val)
                    result.drrs.append(drr)
                    result.data_types.append(data_type)

                except (ValueError):
                    continue  # pasa a la siguiente banda

            return result if result.names else None

        except Exception:
            return None

    # ------------------------------------------------------------------- #
    #   Aumentaciones TR y DRR
    # ------------------------------------------------------------------- #
    def _process_augmentations(
        self,
        speech_data: np.ndarray,
        rir_data:    np.ndarray,
        speech_name: str,
        rir_name:    str,
        data_type:   str,
        bpfilter
    ) -> List[ProcessingResult]:
        """Procesa todas las variaciones de TR y un subconjunto de DRR."""
        results = []

        # Escoger aleatoriamente qué TR usar para generar DRR
        random.seed(datetime.now().timestamp())
        tr_for_drr = random.sample(list(self.TR_variations), k=min(5, len(self.TR_variations)))

        # --- Variaciones TR ----------------------------------------------
        for TR_var in self.TR_variations:
            try:
                rir_tr = tr_augmentation(rir_data, self.fs, TR_var, bpfilter)
                rir_tr /= np.max(np.abs(rir_tr))

                res_tr = self._process_audio_with_rir(
                    speech_data, rir_tr, speech_name, rir_name,
                    f"TR_var_{TR_var}", data_type, bpfilter
                )
                if res_tr:
                    results.append(res_tr)

                # --- Variaciones DRR sobre esta TR -----------------------
                if TR_var in tr_for_drr:
                    for DRR_var in self.DRR_variations:
                        try:
                            rir_drr = drr_aug(rir_tr, self.fs, DRR_var)
                            rir_drr /= np.max(np.abs(rir_drr))

                            res_drr = self._process_audio_with_rir(
                                speech_data, rir_drr, speech_name, rir_name,
                                f"DRR_var_{DRR_var}", data_type, bpfilter
                            )
                            if res_drr:
                                results.append(res_drr)

                        except Exception:
                            continue
            except TrAugmentationError:
                continue

        return results